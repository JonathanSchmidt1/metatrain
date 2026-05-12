"""Unit tests for the electrafy NMAE metric accumulator."""

from __future__ import annotations

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.electrafy.modules.metrics import NMAEAccumulator


def _tmap_from_values(values: torch.Tensor, name: str = "charge_density") -> TensorMap:
    """Wrap a (n_samples, n_props) tensor in the minimal TensorMap layout the
    accumulator's update() expects."""
    n_systems, n_props = values.shape
    samples = Labels(
        names=["system"],
        values=torch.arange(n_systems, dtype=torch.int32).unsqueeze(1),
    )
    properties = Labels(
        names=["grid_point"],
        values=torch.arange(n_props, dtype=torch.int32).unsqueeze(1),
    )
    block = TensorBlock(values=values, samples=samples, components=[], properties=properties)
    return TensorMap(keys=Labels.single(), blocks=[block])


class TestNMAEAccumulator:
    def test_single_update_matches_closed_form(self):
        ref = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        pred = ref + 0.1
        acc = NMAEAccumulator(separate_blocks=False)
        acc.update({"charge_density": _tmap_from_values(pred)},
                   {"charge_density": _tmap_from_values(ref)})
        out = acc.finalize(not_per_atom=["charge_density"], is_distributed=False)
        # |pred-ref| = 0.1 each x 3 = 0.3; |ref| = 1+2+3 = 6 -> 0.05
        assert "charge_density NMAE" in out
        assert abs(out["charge_density NMAE"] - 0.05) < 1e-12

    def test_multiple_updates_accumulate_globally(self):
        # Two batches: total |err| = 0.3 + 0.6 = 0.9; total |ref| = 6 + 12 = 18
        acc = NMAEAccumulator(separate_blocks=False)
        ref1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        ref2 = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float64)
        pred1 = ref1 + 0.1  # |err|=0.3
        pred2 = ref2 + 0.2  # |err|=0.6
        acc.update({"charge_density": _tmap_from_values(pred1)},
                   {"charge_density": _tmap_from_values(ref1)})
        acc.update({"charge_density": _tmap_from_values(pred2)},
                   {"charge_density": _tmap_from_values(ref2)})
        out = acc.finalize(not_per_atom=["charge_density"], is_distributed=False)
        # 0.9 / 18 = 0.05
        assert abs(out["charge_density NMAE"] - 0.05) < 1e-12

    def test_zero_target_with_eps(self):
        # All-zero targets should not divide-by-zero -- eps caps the denominator.
        ref = torch.zeros((1, 3), dtype=torch.float64)
        pred = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
        acc = NMAEAccumulator(separate_blocks=False)
        acc.update({"charge_density": _tmap_from_values(pred)},
                   {"charge_density": _tmap_from_values(ref)})
        out = acc.finalize(
            not_per_atom=["charge_density"], is_distributed=False, eps=1.0
        )
        # |err| = 0.6, denom = max(0, 1.0) = 1.0 -> 0.6
        assert abs(out["charge_density NMAE"] - 0.6) < 1e-12

    def test_output_key_has_no_per_atom_suffix(self):
        ref = torch.ones((2, 4), dtype=torch.float64)
        pred = torch.zeros((2, 4), dtype=torch.float64)
        acc = NMAEAccumulator(separate_blocks=False)
        acc.update({"charge_density": _tmap_from_values(pred)},
                   {"charge_density": _tmap_from_values(ref)})
        # not_per_atom=[] would normally trigger "(per atom)" suffix on RMSE,
        # but NMAE never appends it.
        out = acc.finalize(not_per_atom=[], is_distributed=False)
        keys = list(out.keys())
        assert keys == ["charge_density NMAE"], keys
