"""
Cache-backed map-style ``Dataset`` for ELECTRAFY.

Reads ``<mpid>.pt`` records produced by ``perf/build_chgcar_cache.py`` lazily
on every ``__getitem__`` — no upfront materialisation. Intended for use with
metatrain's standard ``DataLoader`` + ``CollateFn`` + ``DistributedSampler``
pipeline (see ``metatrain.experimental.electrafy.trainer.Trainer``).

Each cache record contains positions, types, cell, pbc, and the flattened
density on the original VASP ``(N1, N2, N3)`` grid. The neighbor list is
intentionally **not** cached — the trainer's ``CollateFn`` rebuilds it
per-batch via ``get_system_with_neighbor_lists_transform``, so the cache stays
invariant under cutoff changes.

Sample layout returned by ``__getitem__``:

    Sample(
        system: metatomic.torch.System,
        charge_density: TensorMap (samples=[("system", i)],
                                   properties=[("grid_point", k)],
                                   values=(1, N1*N2*N3)),
        grid_shape: TensorMap (samples=[("system", i)],
                               properties=[("axis", a)],
                               values=(1, 3)),  # encodes (N1, N2, N3) as float64
    )

The ``grid_shape`` field is needed because the model uses each structure's
native ``(N1, N2, N3)`` grid (paper Appendix C) and the trainer must call
``model.set_override_grid_shapes(...)`` per batch. Encoded as a TensorMap so
metatensor's ``group_and_join`` can collate it into a ``(n_systems, 3)`` block
the same way it batches targets.
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .chgcar import (
    chgcar_to_system_and_density_native,
    density_to_single_sample_tmap,
)


PathLike = Union[str, Path]
GridShape = Tuple[int, int, int]


def _grid_shape_tmap(
    shape: Sequence[int],
    sample_index: int = 0,
    *,
    dtype: torch.dtype = torch.float64,
) -> TensorMap:
    """Encode an ``(N1, N2, N3)`` grid shape as a single-sample TensorMap.

    Stored as float (``DiskDataset`` does the same trick — integer block
    values aren't yet supported by metatensor's serializer).
    """
    values = torch.tensor([[int(shape[0]), int(shape[1]), int(shape[2])]], dtype=dtype)
    samples = Labels(
        names=["system"],
        values=torch.tensor([[int(sample_index)]], dtype=torch.int32),
    )
    properties = Labels(
        names=["axis"],
        values=torch.tensor([[0], [1], [2]], dtype=torch.int32),
    )
    block = TensorBlock(
        values=values, samples=samples, components=[], properties=properties
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def decode_grid_shapes(grid_shape_tmap: TensorMap) -> List[GridShape]:
    """Inverse of :func:`_grid_shape_tmap` over a collated batch.

    Reads the (n_systems, 3) values block of the ``grid_shape`` extra-data
    TensorMap that ``CollateFn`` produces and returns one ``(N1, N2, N3)`` per
    system in batch order.
    """
    block = grid_shape_tmap.block()
    values = block.values.detach().to(dtype=torch.int64).cpu()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"grid_shape TensorMap has block.values shape {tuple(values.shape)}, "
            f"expected (n_systems, 3)"
        )
    return [(int(r[0]), int(r[1]), int(r[2])) for r in values]


def _system_from_record(record: dict, dtype: torch.dtype) -> System:
    return System(
        types=record["types"].to(torch.int32),
        positions=record["positions"].to(dtype),
        cell=record["cell"].to(dtype),
        pbc=record["pbc"].to(torch.bool),
    )


class CachedChgcarDataset(torch.utils.data.Dataset):
    """Lazy map-style ``Dataset`` over a directory of ``<mpid>.pt`` cache files.

    Per ``__getitem__`` the dataset:

    1. ``torch.load`` the record (~70 ms cold, ~3-12 ms warm; bench job 3160000)
    2. reconstruct a :class:`metatomic.torch.System` (positions, types, cell, pbc)
    3. wrap the flat density in a single-sample ``charge_density`` TensorMap
    4. encode the native ``(N1, N2, N3)`` grid shape as a TensorMap

    Returns a NamedTuple compatible with metatensor.learn's ``group_and_join``
    collation, so the standard metatrain ``CollateFn`` batches it correctly
    without modification.

    :param paths: Iterable of ``.pt`` file paths. The order defines sample IDs
        (idx ↔ paths[idx]); ``DistributedSampler`` handles per-rank striding.
    :param fallback_root: Optional directory of ``mp-XXXX/CHGCAR.gz`` to fall
        back to when a cache file is missing (or fails to load). Useful while
        the cache build trails the live download. Set to ``None`` (default) to
        raise on cache misses.
    :param dtype: Tensor dtype for the reconstructed System and density.
        **Keep this at ``torch.float64``** when this dataset is used with
        metatrain's standard ``CollateFn`` — the buffer serializer
        (``metatomic.torch.serialization.save_buffer``) only supports fp64
        and raises ``ValueError: cannot save TensorBlock with dtype
        torch.float32, only float64 is supported`` otherwise. The actual
        training dtype is applied later via ``batch_to`` inside the trainer.
    """

    SAMPLE_FIELDS = ("system", "charge_density", "grid_shape")

    def __init__(
        self,
        paths: Sequence[PathLike],
        *,
        fallback_root: Optional[PathLike] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.paths: List[Path] = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("CachedChgcarDataset requires a non-empty paths list")
        self.fallback_root: Optional[Path] = (
            Path(fallback_root) if fallback_root is not None else None
        )
        self.dtype = dtype
        self._sample_class = namedtuple("Sample", self.SAMPLE_FIELDS)

    def __len__(self) -> int:
        return len(self.paths)

    def _live_load(self, mpid: str):
        if self.fallback_root is None:
            raise FileNotFoundError(
                f"cache miss for {mpid} and no fallback_root configured"
            )
        src = self.fallback_root / mpid / "CHGCAR.gz"
        if not src.is_file():
            raise FileNotFoundError(
                f"cache miss for {mpid} and fallback CHGCAR.gz not at {src}"
            )
        return chgcar_to_system_and_density_native(src, dtype=self.dtype)

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= len(self.paths):
            raise IndexError(f"index {idx} out of range [0, {len(self.paths)})")
        path = self.paths[idx]
        try:
            record = torch.load(path, map_location="cpu", weights_only=False)
            system = _system_from_record(record, dtype=self.dtype)
            flat = record["density"].to(self.dtype).reshape(-1)
            shape: GridShape = tuple(int(x) for x in record["shape"])  # type: ignore[assignment]
        except FileNotFoundError:
            system, flat, shape = self._live_load(path.stem)
            flat = flat.to(self.dtype)
        target = density_to_single_sample_tmap(flat, sample_index=idx)
        grid_shape = _grid_shape_tmap(shape, sample_index=idx, dtype=self.dtype)
        return self._sample_class(system=system, charge_density=target, grid_shape=grid_shape)

    def grid_sizes(self) -> List[int]:
        """Per-sample grid point count (``N1 * N2 * N3``) in dataset index order.

        Used by :class:`~metatrain.experimental.electrafy.modules.samplers.SortedBucketSampler`
        and :class:`~metatrain.experimental.electrafy.modules.samplers.GridBudgetBatchSampler`
        to group same-size structures into the same DDP step.

        Reads ``<cache_root>/_shapes.json`` (a precomputed index produced by
        ``perf/build_shape_index.py``). Falls back to per-file ``torch.load``
        if the index is unavailable -- expensive on large datasets (~25 min for
        15k files), so prebuild the index for production use.
        """
        if not self.paths:
            return []
        cache_root = self.paths[0].parent
        index_path = cache_root / "_shapes.json"
        if index_path.is_file():
            with open(index_path) as f:
                idx: Dict[str, Sequence[int]] = json.load(f)
            missing = [p.name for p in self.paths if p.name not in idx]
            if missing:
                raise RuntimeError(
                    f"_shapes.json at {index_path} is missing entries for "
                    f"{len(missing)} paths (first: {missing[:3]}). "
                    f"Rebuild the index with perf/build_shape_index.py."
                )
            return [
                int(idx[p.name][0]) * int(idx[p.name][1]) * int(idx[p.name][2])
                for p in self.paths
            ]
        out: List[int] = []
        for p in self.paths:
            rec = torch.load(p, map_location="cpu", weights_only=False)
            sh = rec["shape"]
            out.append(int(sh[0]) * int(sh[1]) * int(sh[2]))
        return out


def scan_atomic_types(
    paths: Sequence[PathLike],
    *,
    sample_n: Optional[int] = None,
    workers: int = 0,
) -> List[int]:
    """Scan ``.pt`` cache records and return the sorted union of atomic types.

    Reads only ``record["types"]`` (the smallest field), so each file load is
    cheap; serial loading covers ~15k files in ~1 minute on warm cache. Set
    ``workers > 0`` to fan out across processes.

    :param paths: Iterable of ``.pt`` paths.
    :param sample_n: If set, scan a deterministic stride sample of size at most
        ``sample_n`` instead of all files. Useful when the dataset is huge and
        the periodic table coverage is essentially complete in any subset.
    :param workers: ``ProcessPoolExecutor`` size. ``0`` runs serially in-process.
    :return: Sorted list of unique atomic numbers seen across all scanned files.
    """
    paths = [Path(p) for p in paths]
    if sample_n is not None and 0 < sample_n < len(paths):
        step = max(1, len(paths) // sample_n)
        paths = paths[::step][:sample_n]

    if workers <= 0:
        seen: set = set()
        for p in paths:
            record = torch.load(p, map_location="cpu", weights_only=False)
            seen.update(int(t) for t in record["types"].tolist())
        return sorted(seen)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_scan_one, paths, chunksize=64))
    seen = set()
    for r in results:
        seen.update(r)
    return sorted(seen)


def _scan_one(path: Path) -> List[int]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    return [int(t) for t in record["types"].tolist()]


__all__ = [
    "CachedChgcarDataset",
    "decode_grid_shapes",
    "scan_atomic_types",
]
