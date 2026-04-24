"""
CHGCAR reader and dataset helpers for ELECTRAFY.

Reads VASP CHGCAR files (optionally ``.lz4`` or ``.gz`` compressed), returning
the structure (as :class:`ase.Atoms`) and the charge density on its native grid
(in electrons / Å³).

Also provides utilities to resample the density onto a fixed grid shape (the
model uses a single grid shape per run so densities can be batched as a single
``(n_systems, N_grid)`` TensorMap) and to pack a list of densities into the
``charge_density`` TensorMap layout produced by :meth:`ELECTRAFY.forward`.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import ase
import ase.io.vasp as aiv
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System, systems_to_torch

PathLike = Union[str, Path]


def _open_chgcar(path: Path):
    """Open a CHGCAR for text reading, transparently decompressing ``.lz4`` /
    ``.gz`` variants."""
    suffix = path.suffix.lower()
    if suffix == ".lz4":
        import lz4.frame  # optional dependency, only required for .lz4

        return io.TextIOWrapper(lz4.frame.open(path, "rb"))
    if suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path)


def read_chgcar(path: PathLike) -> Tuple[ase.Atoms, np.ndarray]:
    """
    Read a VASP CHGCAR (plain, ``.gz`` or ``.lz4``).

    VASP stores ``chg * V`` in Fortran order ``(nx, ny, nz)`` — the innermost
    (x) index varies fastest. We reshape as ``(nz, ny, nx)`` then transpose to
    ``(nx, ny, nz)`` and divide by the cell volume, giving electrons/Å³.

    :param path: Path to CHGCAR file (optionally compressed).
    :return: ``(atoms, density)`` where ``density`` has shape ``(N1, N2, N3)``
        matching ``atoms.cell`` in electrons/Å³.
    """
    path = Path(path)
    with _open_chgcar(path) as fd:
        # POSCAR block: title, scale, 3 cell vectors, elements, counts = 7 lines
        poscar_lines: List[str] = [fd.readline() for _ in range(7)]
        counts = [int(x) for x in poscar_lines[-1].split()]
        n_atoms = sum(counts)
        # "Direct" / "Cartesian" + n_atoms position lines
        poscar_lines.append(fd.readline())
        poscar_lines.extend(fd.readline() for _ in range(n_atoms))
        atoms = aiv.read_vasp(io.StringIO("".join(poscar_lines)))

        fd.readline()  # blank line between positions and grid
        ng = fd.readline().split()
        n1, n2, n3 = int(ng[0]), int(ng[1]), int(ng[2])
        n_total = n1 * n2 * n3

        # ``np.fromfile`` only works on real file descriptors, not on wrapped
        # streams (gzip / lz4), so we read the remainder as text and parse it.
        # Stop at the PAW augmentation block if present.
        rest = fd.read()
        aug_at = rest.find("augmentation")
        if aug_at >= 0:
            rest = rest[:aug_at]
        chg = np.fromstring(rest, sep=" ", dtype=np.float64)
        if chg.size < n_total:
            raise ValueError(
                f"{path}: expected at least {n_total} density values, got {chg.size}"
            )
        chg = chg[:n_total]

    density = chg.reshape(n3, n2, n1).transpose(2, 1, 0) / atoms.get_volume()
    return atoms, np.ascontiguousarray(density)


def resample_density(
    density: np.ndarray,
    target_shape: Sequence[int],
    *,
    method: str = "fourier",
) -> np.ndarray:
    """
    Resample a periodic density onto a different grid shape.

    :param density: ``(N1, N2, N3)`` density on the source grid.
    :param target_shape: Target ``(M1, M2, M3)`` grid shape.
    :param method: ``"fourier"`` (default) — truncate / zero-pad in Fourier
        space, exact for band-limited densities and preserves the integral
        exactly. ``"linear"`` — trilinear interpolation with periodic boundary
        (``scipy.ndimage.zoom`` with ``mode='grid-wrap'``, ``order=1``).
    :return: ``(M1, M2, M3)`` resampled density in the same units.
    """
    target = tuple(int(s) for s in target_shape)
    if density.shape == target:
        return density.copy()

    if method == "fourier":
        # Truncate or zero-pad centered FFT coefficients. Exact for signals
        # band-limited by both grids; preserves the DC component (integral).
        hat = np.fft.fftshift(np.fft.fftn(density))
        src = hat.shape
        out = np.zeros(target, dtype=np.complex128)
        # Copy the min(src, target) window around the center.
        slices_src: List[slice] = []
        slices_dst: List[slice] = []
        for s, t in zip(src, target):
            n = min(s, t)
            s0 = (s - n) // 2
            t0 = (t - n) // 2
            slices_src.append(slice(s0, s0 + n))
            slices_dst.append(slice(t0, t0 + n))
        out[tuple(slices_dst)] = hat[tuple(slices_src)]
        # Rescale so DC amplitude stays consistent with the new grid size:
        # np.fft.fftn has no 1/N factor, so the integral ≈ DC / N_total.
        scale = np.prod(target) / np.prod(src)
        resampled = np.fft.ifftn(np.fft.ifftshift(out)).real * scale
        return np.ascontiguousarray(resampled)

    if method == "linear":
        from scipy.ndimage import zoom

        factors = tuple(t / s for t, s in zip(target, density.shape))
        return np.ascontiguousarray(
            zoom(density, factors, order=1, mode="grid-wrap")
        )

    raise ValueError(f"Unknown resample method: {method!r}")


def density_integral(density: np.ndarray, cell: np.ndarray) -> float:
    """Return ``∫ ρ dV`` on a uniform grid spanning ``cell`` — number of
    electrons in the cell."""
    volume = float(abs(np.linalg.det(cell)))
    return float(density.sum()) * volume / density.size


def chgcar_to_system_and_density(
    path: PathLike,
    grid_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    resample_method: str = "fourier",
) -> Tuple[System, torch.Tensor]:
    """
    Read a CHGCAR, resample to ``grid_shape``, and return the pair
    ``(system, density_flat)`` where ``density_flat`` has shape
    ``(N1*N2*N3,)`` — the flattened row that goes into the per-system sample
    of the ``charge_density`` TensorMap.
    """
    atoms, density = read_chgcar(path)
    resampled = resample_density(density, grid_shape, method=resample_method)
    system = systems_to_torch([atoms], dtype=dtype)[0]
    flat = torch.from_numpy(resampled.astype(np.float64)).reshape(-1).to(dtype)
    return system, flat


def pack_density_tensormap(
    densities: Sequence[torch.Tensor],
    *,
    device: Union[torch.device, str] = "cpu",
) -> TensorMap:
    """
    Pack per-system flattened densities into a TensorMap matching
    :meth:`ELECTRAFY.forward`'s output layout:

    - keys: ``Labels.single()``
    - samples: ``[("system", i)]`` for ``i in range(n_systems)``
    - components: ``[]``
    - properties: ``[("grid_point", k)]`` for ``k in range(N_grid)``
    - values: ``(n_systems, N_grid)``
    """
    if len(densities) == 0:
        raise ValueError("densities must be non-empty")
    n_grid = densities[0].numel()
    for i, d in enumerate(densities):
        if d.numel() != n_grid:
            raise ValueError(
                f"density {i} has {d.numel()} grid points, expected {n_grid}"
            )
    values = torch.stack([d.reshape(-1) for d in densities], dim=0).to(device)
    samples = Labels(
        names=["system"],
        values=torch.arange(len(densities), dtype=torch.int32, device=device).unsqueeze(1),
    )
    properties = Labels(
        names=["grid_point"],
        values=torch.arange(n_grid, dtype=torch.int32, device=device).unsqueeze(1),
    )
    block = TensorBlock(values=values, samples=samples, components=[], properties=properties)
    return TensorMap(keys=Labels.single().to(device), blocks=[block])


def density_to_single_sample_tmap(
    flat_density: torch.Tensor,
    *,
    sample_index: int = 0,
    device: Union[torch.device, str] = "cpu",
) -> TensorMap:
    """
    Wrap a single flattened density tensor in the same TensorMap layout as
    :meth:`ELECTRAFY.forward` but with a single sample row — suitable for use
    as a per-structure target in ``Dataset.from_dict(...)``, where metatrain's
    collator joins the per-sample TensorMaps along the sample dimension.
    """
    flat = flat_density.reshape(1, -1).to(device)
    n_grid = flat.shape[1]
    samples = Labels(
        names=["system"],
        values=torch.tensor([[int(sample_index)]], dtype=torch.int32, device=device),
    )
    properties = Labels(
        names=["grid_point"],
        values=torch.arange(n_grid, dtype=torch.int32, device=device).unsqueeze(1),
    )
    block = TensorBlock(values=flat, samples=samples, components=[], properties=properties)
    return TensorMap(keys=Labels.single().to(device), blocks=[block])


def load_chgcar_dataset(
    paths: Sequence[PathLike],
    grid_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu",
    resample_method: str = "fourier",
) -> Tuple[List[System], TensorMap]:
    """
    Load a list of CHGCARs and return ``(systems, charge_density_tensormap)``
    ready to be fed into metatrain's loss pipeline. All densities are
    resampled to ``grid_shape`` so the target TensorMap has fixed
    ``(n_systems, N_grid)`` shape.
    """
    systems: List[System] = []
    flats: List[torch.Tensor] = []
    for p in paths:
        sys_, flat = chgcar_to_system_and_density(
            p, grid_shape, dtype=dtype, resample_method=resample_method
        )
        systems.append(sys_)
        flats.append(flat)
    tmap = pack_density_tensormap(flats, device=device)
    return systems, tmap


def load_chgcar_per_sample(
    paths: Sequence[PathLike],
    grid_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu",
    resample_method: str = "fourier",
) -> Tuple[List[System], List[TensorMap]]:
    """
    Variant of :func:`load_chgcar_dataset` that returns the density as a list
    of per-sample TensorMaps — the format expected by
    ``metatensor.learn.data.Dataset.from_dict``.
    """
    systems: List[System] = []
    tmaps: List[TensorMap] = []
    for i, p in enumerate(paths):
        sys_, flat = chgcar_to_system_and_density(
            p, grid_shape, dtype=dtype, resample_method=resample_method
        )
        systems.append(sys_)
        tmaps.append(density_to_single_sample_tmap(flat, sample_index=i, device=device))
    return systems, tmaps
