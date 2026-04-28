"""
Tests for the cache-backed map-style dataset and the trainer integration.

Builds a tiny ``<mpid>.pt`` cache on-the-fly from the repository's CHGCAR
fixture (Mn3ZnN, 60x60x60 grid), then exercises:

- ``CachedChgcarDataset.__getitem__`` returns the right shapes / dtypes
- ``decode_grid_shapes`` round-trips through the TensorMap encoding
- ``scan_atomic_types`` finds all atomic numbers across the cache
- The trainer's ``_apply_grid_shapes`` helper lifts shapes out of a collated
  batch and routes them to ``ELECTRAFY.set_override_grid_shapes``
- The full ``DataLoader`` path with metatrain's ``CollateFn`` produces a
  batch that survives ``unpack_batch`` + ``decode_grid_shapes``
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch

from metatomic.torch import NeighborListOptions
from torch.utils.data import DataLoader

from metatrain.experimental.electrafy.modules.cache_dataset import (
    CachedChgcarDataset,
    decode_grid_shapes,
    scan_atomic_types,
)
from metatrain.experimental.electrafy.modules.chgcar import (
    chgcar_to_system_and_density_native,
)
from metatrain.experimental.electrafy.trainer import (
    _apply_grid_shapes,
    _unwrap_to_electrafy,
)
from metatrain.utils.data import CollateFn, unpack_batch
from metatrain.utils.neighbor_lists import (
    get_system_with_neighbor_lists_transform,
)


# Try a few candidate locations: original layout (electrafy/CHGCAR) and the
# worktree symlink (electrafy-streaming/CHGCAR). Skips the test if none exist.
_CANDIDATE_FIXTURES = [
    Path(__file__).resolve().parents[6] / "CHGCAR",  # original repo layout
    Path(__file__).resolve().parents[5] / "CHGCAR",  # worktree layout
]


def _find_chgcar() -> Path:
    for c in _CANDIDATE_FIXTURES:
        if c.is_file() or c.is_symlink():
            return c.resolve()
    pytest.skip(
        "CHGCAR fixture not found at any of: "
        + ", ".join(str(c) for c in _CANDIDATE_FIXTURES)
    )


@pytest.fixture(scope="module")
def chgcar_path() -> Path:
    return _find_chgcar()


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory, chgcar_path):
    """Build a 3-entry .pt cache from the same CHGCAR (different sample IDs)."""
    cache = tmp_path_factory.mktemp("electrafi_cache")
    sys_, flat, shape = chgcar_to_system_and_density_native(
        chgcar_path, dtype=torch.float64
    )
    record = {
        "positions": sys_.positions.detach().clone(),
        "types": sys_.types.detach().clone(),
        "cell": sys_.cell.detach().clone(),
        "pbc": sys_.pbc.detach().clone(),
        "density": flat.to(torch.float32).contiguous(),
        "shape": tuple(int(x) for x in shape),
    }
    for mpid in ["mp-1", "mp-2", "mp-3"]:
        torch.save(record, cache / f"{mpid}.pt")
    return cache


def _list_cache(cache_dir: Path) -> List[Path]:
    return sorted(cache_dir.glob("*.pt"))


def _MINIMAL_HYPERS():
    """Tiny hypers for instantiating an ELECTRAFY shell — never fed forward."""
    return {
        "cutoff": 4.0,
        "cutoff_function": "Cosine",
        "cutoff_width": 0.5,
        "d_pet": 8,
        "d_node": 16,
        "d_feedforward": 16,
        "num_heads": 2,
        "num_attention_layers": 1,
        "num_gnn_layers": 1,
        "normalization": "RMSNorm",
        "activation": "SiLU",
        "transformer_type": "PreLN",
        "attention_temperature": 1.0,
        "gaussians_per_electron": 2,
        "gamma": 0.1,
        "grid_shape": [4, 4, 4],
        "fourier_chunk_size": 64,
    }


class TestCachedChgcarDataset:
    def test_len(self, cache_dir):
        ds = CachedChgcarDataset(_list_cache(cache_dir))
        assert len(ds) == 3

    def test_getitem_fields(self, cache_dir):
        ds = CachedChgcarDataset(_list_cache(cache_dir))
        sample = ds[0]
        # NamedTuple with three fields in declared order
        assert sample._fields == ("system", "charge_density", "grid_shape")
        # System and TensorMap are torch.ScriptClass — duck-type instead.
        assert hasattr(sample.system, "positions")
        assert hasattr(sample.system, "types")
        assert sample.system.positions.shape[0] == 5  # Mn3ZnN = 5 atoms
        assert hasattr(sample.charge_density, "block")
        assert hasattr(sample.grid_shape, "block")

    def test_getitem_density_shape_matches_grid(self, cache_dir):
        ds = CachedChgcarDataset(_list_cache(cache_dir))
        sample = ds[1]
        block = sample.charge_density.block()
        # Single-sample target: values are (1, N1*N2*N3)
        assert block.values.shape == (1, 60 * 60 * 60)
        assert block.samples.values.tolist() == [[1]]

    def test_getitem_grid_shape_encoding(self, cache_dir):
        ds = CachedChgcarDataset(_list_cache(cache_dir))
        sample = ds[2]
        shapes = decode_grid_shapes(sample.grid_shape)
        assert shapes == [(60, 60, 60)]
        # sample row index is 2 (matches __getitem__ idx)
        assert sample.grid_shape.block().samples.values.tolist() == [[2]]

    def test_dtype(self, cache_dir):
        ds = CachedChgcarDataset(_list_cache(cache_dir), dtype=torch.float32)
        sample = ds[0]
        assert sample.system.positions.dtype == torch.float32
        assert sample.charge_density.block().values.dtype == torch.float32

    def test_index_bounds(self, cache_dir):
        ds = CachedChgcarDataset(_list_cache(cache_dir))
        with pytest.raises(IndexError):
            ds[3]
        with pytest.raises(IndexError):
            ds[-1]

    def test_empty_paths_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            CachedChgcarDataset([])


class TestScanAtomicTypes:
    def test_finds_all_types(self, cache_dir):
        types = scan_atomic_types(_list_cache(cache_dir))
        # Mn(25), Zn(30), N(7) — same fixture replicated
        assert types == [7, 25, 30]


class TestCollateRoundtrip:
    def test_collate_rejects_fp32_dataset(self, cache_dir):
        """metatomic's buffer serializer only supports fp64. Confirm we get
        a clear error if someone passes fp32 to the dataset and then tries to
        run it through metatrain's CollateFn — caught the bug behind kuma
        bench job 3170038."""
        ds = CachedChgcarDataset(_list_cache(cache_dir), dtype=torch.float32)
        nl_opts = NeighborListOptions(cutoff=4.0, full_list=True, strict=True)
        collate = CollateFn(
            target_keys=["charge_density"],
            callables=[get_system_with_neighbor_lists_transform([nl_opts])],
        )
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
        )
        with pytest.raises(ValueError, match="float64"):
            next(iter(loader))

    def test_collate_extra_data_grid_shape(self, cache_dir):
        """The full path:
        Dataset → DataLoader(collate_fn=CollateFn(...)) → unpack_batch → decode.
        Confirms that ``grid_shape`` survives metatrain's collation.
        """
        ds = CachedChgcarDataset(_list_cache(cache_dir))
        nl_opts = NeighborListOptions(cutoff=4.0, full_list=True, strict=True)
        collate = CollateFn(
            target_keys=["charge_density"],
            callables=[get_system_with_neighbor_lists_transform([nl_opts])],
        )
        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        batch = next(iter(loader))
        systems, targets, extra_data = unpack_batch(batch)
        assert len(systems) == 2
        assert "charge_density" in targets
        assert "grid_shape" in extra_data
        shapes = decode_grid_shapes(extra_data["grid_shape"])
        assert shapes == [(60, 60, 60), (60, 60, 60)]


class TestApplyGridShapes:
    def test_helper_routes_shapes_to_model(self, cache_dir):
        """Use an actual ELECTRAFY instance to confirm the helper applies and
        clears overrides. Doesn't run forward."""
        from metatrain.experimental.electrafy.model import DENSITY_KEY, ELECTRAFY
        from metatrain.utils.data import DatasetInfo
        from metatrain.utils.data.target_info import TargetInfo
        from metatensor.torch import Labels, TensorBlock, TensorMap

        # Tiny model just to instantiate the API; we never call forward.
        layout = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.empty((0, 1), dtype=torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.zeros((0, 1), dtype=torch.int32),
                    ),
                    components=[],
                    properties=Labels(
                        names=["grid_point"],
                        values=torch.zeros((1, 1), dtype=torch.int32),
                    ),
                )
            ],
        )
        dataset_info = DatasetInfo(
            length_unit="angstrom",
            atomic_types=[7, 25, 30],
            targets={
                DENSITY_KEY: TargetInfo(layout=layout, quantity="", unit="")
            },
        )
        model_hypers = _MINIMAL_HYPERS()
        model = ELECTRAFY(model_hypers, dataset_info)

        ds = CachedChgcarDataset(_list_cache(cache_dir))
        nl_opts = NeighborListOptions(cutoff=4.0, full_list=True, strict=True)
        collate = CollateFn(
            target_keys=["charge_density"],
            callables=[get_system_with_neighbor_lists_transform([nl_opts])],
        )
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate
        )
        batch = next(iter(loader))
        systems, _, extra_data = unpack_batch(batch)

        inner = _unwrap_to_electrafy(model)
        assert inner._override_grid_shapes is None
        applied = _apply_grid_shapes(inner, extra_data, len(systems))
        assert applied is True
        assert inner._override_grid_shapes == [(60, 60, 60), (60, 60, 60)]
        inner.clear_override_grid_shapes()
        assert inner._override_grid_shapes is None

    def test_helper_no_op_without_grid_shape(self):
        from metatrain.experimental.electrafy.model import DENSITY_KEY, ELECTRAFY
        from metatrain.utils.data import DatasetInfo
        from metatrain.utils.data.target_info import TargetInfo
        from metatensor.torch import Labels, TensorBlock, TensorMap

        layout = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.empty((0, 1), dtype=torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.zeros((0, 1), dtype=torch.int32),
                    ),
                    components=[],
                    properties=Labels(
                        names=["grid_point"],
                        values=torch.zeros((1, 1), dtype=torch.int32),
                    ),
                )
            ],
        )
        dataset_info = DatasetInfo(
            length_unit="angstrom",
            atomic_types=[7],
            targets={DENSITY_KEY: TargetInfo(layout=layout, quantity="", unit="")},
        )
        model_hypers = _MINIMAL_HYPERS()
        model = ELECTRAFY(model_hypers, dataset_info)
        # No 'grid_shape' key in extra_data → helper is a no-op.
        applied = _apply_grid_shapes(model, {}, 0)
        assert applied is False
        applied = _apply_grid_shapes(model, None, 0)
        assert applied is False


class TestTrainerEndToEnd:
    """End-to-end smoke: ``CachedChgcarDataset`` + metatrain ``Trainer.train()``
    with native grid shapes flowing through ``extra_data``. Verifies the full
    plumbing: dataset → CollateFn → unpack_batch → set_override_grid_shapes
    → forward → loss → backward → checkpoint.

    Uses a tiny resampled (4³) cache record so the per-step Fourier cost is
    cheap on CPU. Two train + 1 val samples, 1 epoch.
    """

    def test_train_one_epoch(self, tmp_path, chgcar_path):
        from metatrain.experimental.electrafy.model import DENSITY_KEY, ELECTRAFY
        from metatrain.experimental.electrafy.modules.chgcar import (
            chgcar_to_system_and_density,
        )
        from metatrain.experimental.electrafy.trainer import Trainer
        from metatrain.utils.data import DatasetInfo
        from metatrain.utils.data.target_info import TargetInfo
        from metatensor.torch import Labels, TensorBlock, TensorMap

        grid_shape = (4, 4, 4)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Build a tiny cache: 3 records, each with the resampled (4,4,4) density.
        sys_, flat = chgcar_to_system_and_density(
            chgcar_path, grid_shape, dtype=torch.float64
        )
        record = {
            "positions": sys_.positions.detach().clone(),
            "types": sys_.types.detach().clone(),
            "cell": sys_.cell.detach().clone(),
            "pbc": sys_.pbc.detach().clone(),
            "density": flat.to(torch.float32).contiguous(),
            "shape": grid_shape,
        }
        for mpid in ["mp-1", "mp-2", "mp-3"]:
            torch.save(record, cache_dir / f"{mpid}.pt")

        all_paths = sorted(cache_dir.glob("*.pt"))
        train_ds = CachedChgcarDataset(all_paths[:2])
        val_ds = CachedChgcarDataset(all_paths[2:])

        # 1-grid layout matching what ELECTRAFY emits.
        n_grid = grid_shape[0] * grid_shape[1] * grid_shape[2]
        layout = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.empty((0, n_grid), dtype=torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.zeros((0, 1), dtype=torch.int32),
                    ),
                    components=[],
                    properties=Labels(
                        names=["grid_point"],
                        values=torch.arange(n_grid, dtype=torch.int32).unsqueeze(1),
                    ),
                )
            ],
        )
        dataset_info = DatasetInfo(
            length_unit="angstrom",
            atomic_types=[7, 25, 30],  # N, Mn, Zn
            targets={
                DENSITY_KEY: TargetInfo(layout=layout, quantity="", unit="")
            },
        )
        hypers = _MINIMAL_HYPERS()
        hypers["grid_shape"] = list(grid_shape)
        model = ELECTRAFY(hypers, dataset_info).to(torch.float64)

        params_before = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }

        trainer = Trainer(
            {
                "distributed": False,
                "distributed_port": 39591,
                "batch_size": 1,
                "num_epochs": 1,
                "warmup_fraction": 0.0,
                "learning_rate": 1e-3,
                "weight_decay": None,
                "log_interval": 1.0,
                "validation_interval": 1.0,
                "checkpoint_interval": 1,
                "grad_clip_norm": 10.0,
                "num_workers": 0,
                "compile": False,
                "best_model_metric": "loss",
                "loss": {
                    DENSITY_KEY: {
                        "type": "nmae",
                        "weight": 1.0,
                        "reduction": "mean",
                        "gradients": {},
                    }
                },
            }
        )
        trainer.train(
            model=model,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=[train_ds],
            val_datasets=[val_ds],
            checkpoint_dir=str(tmp_path),
        )

        # The model's override must be cleared after each forward — otherwise
        # subsequent forwards (e.g. inference) would see stale shapes.
        assert model._override_grid_shapes is None

        # At least one parameter moved.
        moved = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, dict(model.named_parameters())[name])
        ]
        assert moved, "no parameter moved after one epoch"

        # Checkpoint was written.
        assert list(tmp_path.glob("model_*.ckpt")), "no checkpoint produced"

        # Trainer state consistent.
        assert trainer.epoch == 0
        assert trainer.best_metric is not None and trainer.best_metric < float("inf")


class TestUnwrapModel:
    def test_unwraps_ddp(self):
        # Construct an ELECTRAFY then fake-wrap with a Module that has .module.
        from metatrain.experimental.electrafy.model import DENSITY_KEY, ELECTRAFY
        from metatrain.utils.data import DatasetInfo
        from metatrain.utils.data.target_info import TargetInfo
        from metatensor.torch import Labels, TensorBlock, TensorMap

        layout = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.empty((0, 1), dtype=torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.zeros((0, 1), dtype=torch.int32),
                    ),
                    components=[],
                    properties=Labels(
                        names=["grid_point"],
                        values=torch.zeros((1, 1), dtype=torch.int32),
                    ),
                )
            ],
        )
        dataset_info = DatasetInfo(
            length_unit="angstrom",
            atomic_types=[7],
            targets={DENSITY_KEY: TargetInfo(layout=layout, quantity="", unit="")},
        )
        model_hypers = _MINIMAL_HYPERS()
        m = ELECTRAFY(model_hypers, dataset_info)

        class _FakeDDP(torch.nn.Module):
            def __init__(self, mod): super().__init__(); self.module = mod

        wrapped = _FakeDDP(m)
        assert _unwrap_to_electrafy(wrapped) is m
        # Plain ELECTRAFY also works.
        assert _unwrap_to_electrafy(m) is m
