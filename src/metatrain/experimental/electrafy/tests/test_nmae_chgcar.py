"""
NMAE regression tests against the real CHGCAR fixture.

Builds a perturbed copy of the reference density (``pred = (1+ε)*target``)
and asserts the loss returns ``|ε|``, since:

    NMAE = sum|pred - target| / sum|target|
         = sum|ε * target| / sum|target|
         = |ε|     (when target is sign-uniform on the support, which holds
                    for VASP charge density — non-negative up to noise)

VASP charge density is non-negative (up to ~1e-6 noise; see
``test_chgcar.py::TestReadChgcar::test_density_nonnegative``), so
``sum|ε*target|`` collapses cleanly to ``|ε| * sum|target|`` and the test
gives an exact relationship.

Covers BOTH:
- the standalone ``NMAELoss`` (``modules.loss``) used by overfit scripts and
  the standalone trainers,
- the ``TensorMapNMAELoss`` (``metatrain.utils.loss``) used by metatrain's
  ``LossAggregator`` — this is the loss class that fires inside the actual
  training run via ``loss: {type: nmae, ...}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.electrafy.modules.chgcar import (
    chgcar_to_system_and_density_native,
)
from metatrain.experimental.electrafy.modules.loss import (
    NMAELoss,
    batch_nmae_loss,
)
from metatrain.utils.loss import TensorMapNMAELoss


_CANDIDATE_FIXTURES = [
    Path(__file__).resolve().parents[6] / "CHGCAR",  # original layout
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
def chgcar_target() -> torch.Tensor:
    """Read the CHGCAR fixture and return the flattened density tensor."""
    chgcar_path = _find_chgcar()
    _, flat, _ = chgcar_to_system_and_density_native(chgcar_path, dtype=torch.float64)
    return flat


def _density_tmap(flat: torch.Tensor, sample_index: int = 0) -> TensorMap:
    n = flat.numel()
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=flat.reshape(1, -1),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[sample_index]], dtype=torch.int32),
                ),
                components=[],
                properties=Labels(
                    names=["grid_point"],
                    values=torch.arange(n, dtype=torch.int32).unsqueeze(1),
                ),
            )
        ],
    )


# ----------------------------------------------------------------------------
# Standalone NMAELoss (modules.loss)
# ----------------------------------------------------------------------------


class TestNMAELossAgainstChgcar:
    def test_zero_error_perfect_prediction(self, chgcar_target):
        """pred == target → NMAE = 0 exactly."""
        loss = NMAELoss()(chgcar_target, chgcar_target)
        assert float(loss) == 0.0

    @pytest.mark.parametrize("epsilon", [0.001, 0.01, 0.05, 0.10])
    def test_multiplicative_perturbation(self, chgcar_target, epsilon):
        """pred = (1+ε)*target → NMAE = |ε|.

        Holds because the CHGCAR density is non-negative, so
        sum|ε * target| = |ε| * sum|target|.
        """
        pred = (1.0 + epsilon) * chgcar_target
        loss = NMAELoss()(pred, chgcar_target)
        assert float(loss) == pytest.approx(abs(epsilon), rel=1e-6, abs=1e-7)

    @pytest.mark.parametrize("epsilon", [-0.001, -0.01, -0.05])
    def test_negative_multiplicative_perturbation(self, chgcar_target, epsilon):
        """pred = (1+ε)*target with ε<0 → NMAE = |ε| (sign drops in abs)."""
        pred = (1.0 + epsilon) * chgcar_target
        loss = NMAELoss()(pred, chgcar_target)
        assert float(loss) == pytest.approx(abs(epsilon), rel=1e-6, abs=1e-7)

    def test_zero_prediction(self, chgcar_target):
        """pred = 0 → NMAE = 1 exactly (sum|0 - target| / sum|target|)."""
        pred = torch.zeros_like(chgcar_target)
        loss = NMAELoss()(pred, chgcar_target)
        assert float(loss) == pytest.approx(1.0, rel=1e-6)

    def test_double_prediction(self, chgcar_target):
        """pred = 2*target → NMAE = 1 exactly."""
        loss = NMAELoss()(2.0 * chgcar_target, chgcar_target)
        assert float(loss) == pytest.approx(1.0, rel=1e-6)

    def test_dtype_preserved(self, chgcar_target):
        """Loss tensor inherits target dtype."""
        out = NMAELoss()(chgcar_target, chgcar_target)
        assert out.dtype == chgcar_target.dtype

    def test_grad_flow(self, chgcar_target):
        """Gradients flow through the predicted tensor."""
        pred = chgcar_target.clone().detach().requires_grad_(True)
        loss = NMAELoss()(pred * 1.01, chgcar_target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestBatchNMAEAgainstChgcar:
    def test_batch_homogeneous_perturbation(self, chgcar_target):
        """Two-system batch, both at ε=1% perturbation → batch mean = 0.01."""
        pred1 = 1.01 * chgcar_target
        pred2 = 0.99 * chgcar_target
        loss = batch_nmae_loss([pred1, pred2], [chgcar_target, chgcar_target])
        assert float(loss) == pytest.approx(0.01, rel=1e-6, abs=1e-7)

    def test_batch_mixed_perturbation(self, chgcar_target):
        """ε=0 + ε=2% → batch mean = 0.01."""
        pred1 = chgcar_target
        pred2 = 1.02 * chgcar_target
        loss = batch_nmae_loss([pred1, pred2], [chgcar_target, chgcar_target])
        assert float(loss) == pytest.approx(0.01, rel=1e-6, abs=1e-7)


# ----------------------------------------------------------------------------
# TensorMapNMAELoss (metatrain.utils.loss) — the loss class actually used in
# training via the LossSpecification(type='nmae', ...) config.
# ----------------------------------------------------------------------------


class TestTensorMapNMAELossAgainstChgcar:
    def _make_loss(self) -> TensorMapNMAELoss:
        return TensorMapNMAELoss(
            name="charge_density", gradient=None, weight=1.0, reduction="mean"
        )

    def test_zero_error_perfect_prediction(self, chgcar_target):
        """TensorMap pred == target → NMAE ≈ 0."""
        target_tm = _density_tmap(chgcar_target, sample_index=0)
        # Clone to ensure no aliasing surprises.
        pred_tm = _density_tmap(chgcar_target.clone(), sample_index=0)
        loss = self._make_loss()
        out = loss.compute({"charge_density": pred_tm}, {"charge_density": target_tm})
        # Not exactly 0 because of the +eps in the denominator, but ≪ 1e-6.
        assert float(out) < 1e-9

    @pytest.mark.parametrize("epsilon", [0.001, 0.01, 0.05, 0.10])
    def test_multiplicative_perturbation(self, chgcar_target, epsilon):
        """pred = (1+ε)*target → TensorMap NMAE = |ε|."""
        target_tm = _density_tmap(chgcar_target, sample_index=0)
        pred_tm = _density_tmap((1.0 + epsilon) * chgcar_target, sample_index=0)
        loss = self._make_loss()
        out = loss.compute({"charge_density": pred_tm}, {"charge_density": target_tm})
        assert float(out) == pytest.approx(abs(epsilon), rel=1e-6, abs=1e-7)

    def test_negative_perturbation(self, chgcar_target):
        """pred = 0.99 * target → NMAE = 0.01 (the abs picks up the sign)."""
        target_tm = _density_tmap(chgcar_target, sample_index=0)
        pred_tm = _density_tmap(0.99 * chgcar_target, sample_index=0)
        loss = self._make_loss()
        out = loss.compute({"charge_density": pred_tm}, {"charge_density": target_tm})
        assert float(out) == pytest.approx(0.01, rel=1e-6, abs=1e-7)

    def test_zero_prediction(self, chgcar_target):
        """pred = 0 → NMAE = 1."""
        target_tm = _density_tmap(chgcar_target, sample_index=0)
        pred_tm = _density_tmap(torch.zeros_like(chgcar_target), sample_index=0)
        loss = self._make_loss()
        out = loss.compute({"charge_density": pred_tm}, {"charge_density": target_tm})
        assert float(out) == pytest.approx(1.0, rel=1e-6)

    def test_grad_flow_through_tensormap(self, chgcar_target):
        """Gradients propagate through the predicted block.values."""
        target_tm = _density_tmap(chgcar_target, sample_index=0)
        pred_values = (1.01 * chgcar_target).clone().detach().requires_grad_(True)
        # Wrap in a TensorMap with the gradient-tracking values.
        pred_tm = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=pred_values.reshape(1, -1),
                    samples=Labels(
                        names=["system"],
                        values=torch.tensor([[0]], dtype=torch.int32),
                    ),
                    components=[],
                    properties=Labels(
                        names=["grid_point"],
                        values=torch.arange(
                            chgcar_target.numel(), dtype=torch.int32
                        ).unsqueeze(1),
                    ),
                )
            ],
        )
        loss = self._make_loss()
        out = loss.compute({"charge_density": pred_tm}, {"charge_density": target_tm})
        out.backward()
        assert pred_values.grad is not None
        assert torch.isfinite(pred_values.grad).all()


# ----------------------------------------------------------------------------
# Cross-check: standalone NMAELoss and TensorMapNMAELoss must agree on the
# same input. Catches any drift between the two implementations.
# ----------------------------------------------------------------------------


class TestNMAEImplementationsAgree:
    @pytest.mark.parametrize("epsilon", [0.0, 0.005, 0.01, 0.05, -0.02])
    def test_standalone_vs_tensormap_agree(self, chgcar_target, epsilon):
        pred = (1.0 + epsilon) * chgcar_target
        standalone = NMAELoss()(pred, chgcar_target)

        target_tm = _density_tmap(chgcar_target, sample_index=0)
        pred_tm = _density_tmap(pred, sample_index=0)
        tm_loss = TensorMapNMAELoss(
            name="charge_density", gradient=None, weight=1.0, reduction="mean"
        ).compute({"charge_density": pred_tm}, {"charge_density": target_tm})

        # Both use eps=1e-10 in the denominator, so they agree to float64 precision.
        assert float(tm_loss) == pytest.approx(float(standalone), rel=1e-9, abs=1e-10)
