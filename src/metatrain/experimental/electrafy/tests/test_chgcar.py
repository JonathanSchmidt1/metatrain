"""
Tests for the CHGCAR reader and resampling helpers.

Uses the repository's ``CHGCAR`` fixture (Mn3ZnN, 60x60x60 grid, 56 valence
electrons with Mn_pv / Zn / N PAW ZVALs).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from metatrain.experimental.electrafy.modules.chgcar import (
    chgcar_to_system_and_density,
    density_integral,
    load_chgcar_dataset,
    pack_density_tensormap,
    read_chgcar,
    resample_density,
)


CHGCAR_PATH = Path(__file__).resolve().parent / "resources" / "CHGCAR.gz"


@pytest.fixture(scope="module")
def chgcar_fixture():
    if not CHGCAR_PATH.is_file():
        pytest.skip(f"CHGCAR fixture not found at {CHGCAR_PATH}")
    return CHGCAR_PATH


class TestReadChgcar:
    def test_structure(self, chgcar_fixture):
        atoms, density = read_chgcar(chgcar_fixture)
        assert len(atoms) == 5
        assert atoms.get_chemical_symbols() == ["Mn", "Mn", "Mn", "Zn", "N"]
        np.testing.assert_allclose(
            np.diag(atoms.cell.array), [3.902, 3.902, 3.902], atol=1e-9
        )
        assert density.shape == (60, 60, 60)
        assert density.dtype == np.float64
        assert np.all(np.isfinite(density))

    def test_electron_count(self, chgcar_fixture):
        """Integral over cell equals total valence electrons (Mn_pv=13 x3 + Zn=12 + N=5 = 56)."""
        atoms, density = read_chgcar(chgcar_fixture)
        n_electrons = density_integral(density, atoms.cell.array)
        assert n_electrons == pytest.approx(56.0, abs=1e-3)

    def test_density_nonnegative(self, chgcar_fixture):
        _, density = read_chgcar(chgcar_fixture)
        # VASP charge density is non-negative up to noise.
        assert density.min() > -1e-6


class TestResample:
    def test_shape_and_integral_fourier(self, chgcar_fixture):
        atoms, density = read_chgcar(chgcar_fixture)
        target = (32, 32, 32)
        resampled = resample_density(density, target, method="fourier")
        assert resampled.shape == target
        n1 = density_integral(density, atoms.cell.array)
        n2 = density_integral(resampled, atoms.cell.array)
        # Fourier truncation preserves DC exactly, so integrals match closely.
        assert n2 == pytest.approx(n1, rel=1e-6)

    def test_shape_linear(self, chgcar_fixture):
        # Linear interpolation aliases sharp density peaks and can shift the
        # integral by ~5-10% when downsampling. Prefer ``method="fourier"`` for
        # training targets; this test only verifies the shape.
        _, density = read_chgcar(chgcar_fixture)
        target = (32, 32, 32)
        resampled = resample_density(density, target, method="linear")
        assert resampled.shape == target
        assert np.all(np.isfinite(resampled))

    def test_identity_resample(self, chgcar_fixture):
        _, density = read_chgcar(chgcar_fixture)
        out = resample_density(density, density.shape, method="fourier")
        np.testing.assert_allclose(out, density, atol=1e-9)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown resample method"):
            resample_density(np.zeros((4, 4, 4)), (2, 2, 2), method="bogus")


class TestSystemAndDensity:
    def test_matches_model_target_layout(self, chgcar_fixture):
        grid_shape = (16, 16, 16)
        system, flat = chgcar_to_system_and_density(
            chgcar_fixture, grid_shape, dtype=torch.float64
        )
        # System / TensorMap are torch custom classes (not regular Python types),
        # so we verify structure via duck-typed attributes.
        assert system.positions.dtype == torch.float64
        assert system.positions.shape == (5, 3)
        assert system.cell.shape == (3, 3)
        assert flat.dtype == torch.float64
        assert flat.shape == (16 * 16 * 16,)


class TestDatasetLoading:
    def test_single_chgcar_to_tensormap(self, chgcar_fixture):
        grid_shape = (16, 16, 16)
        systems, tmap = load_chgcar_dataset([chgcar_fixture], grid_shape)
        assert len(tmap.keys) == 1
        block = tmap.block(0)
        assert block.values.shape == (1, 16 * 16 * 16)
        assert block.samples.names == ["system"]
        assert block.properties.names == ["grid_point"]
        assert len(systems) == 1

    def test_repeated_chgcar_stacks(self, chgcar_fixture):
        grid_shape = (8, 8, 8)
        systems, tmap = load_chgcar_dataset(
            [chgcar_fixture, chgcar_fixture, chgcar_fixture], grid_shape
        )
        assert len(systems) == 3
        block = tmap.block(0)
        assert block.values.shape == (3, 8 * 8 * 8)
        # All three rows are the same structure/density → identical.
        torch.testing.assert_close(block.values[0], block.values[1])
        torch.testing.assert_close(block.values[0], block.values[2])


class TestPackDensityTensormap:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            pack_density_tensormap([])

    def test_mismatched_grid_raises(self):
        with pytest.raises(ValueError, match="grid points"):
            pack_density_tensormap([torch.zeros(8), torch.zeros(16)])
