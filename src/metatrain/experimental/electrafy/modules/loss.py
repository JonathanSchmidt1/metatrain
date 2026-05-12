"""
NMAE (Normalized Mean Absolute Error) loss for charge density prediction.

L = integral |rho_pred(r) - rho_ref(r)| dV
    ------------------------------------------
             integral rho_ref(r) dV

Both integrals are approximated by the sum over the real-space grid
multiplied by the voxel volume (V / N_total).  The volume factor cancels,
leaving a simple ratio of sums.

Two flavours are exposed:

* :class:`NMAELoss` -- a plain ``torch.nn.Module`` operating on raw density
  tensors. Used in unit tests and helpful for ad-hoc training loops.
* :class:`TensorMapNMAELoss` -- a :class:`metatrain.utils.loss.LossInterface`
  implementation that consumes the same TensorMap dict the standard metatrain
  ``LossAggregator`` does. Registered with ``metatrain.utils.loss.create_loss``
  by :mod:`metatrain.experimental.electrafy` at import time so the type key
  ``"nmae"`` works in ``LossSpecification(type="nmae")``.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from metatensor.torch import TensorMap

from metatrain.utils.loss import LossInterface


class NMAELoss(nn.Module):
    """
    Normalized Mean Absolute Error loss on charge density grids.

    :param eps: Small value added to the denominator for numerical stability.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        rho_pred: torch.Tensor,
        rho_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NMAE loss.

        :param rho_pred: Predicted density, arbitrary shape (e.g. (N1, N2, N3)).
        :param rho_ref:  Reference density, same shape as rho_pred.
        :return: Scalar NMAE loss.
        """
        numerator = torch.abs(rho_pred - rho_ref).sum()
        denominator = rho_ref.abs().sum() + self.eps
        return numerator / denominator


def batch_nmae_loss(
    rho_preds: list[torch.Tensor],
    rho_refs: list[torch.Tensor],
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute mean NMAE over a batch of density grids (which may have different shapes).

    :param rho_preds: List of predicted density tensors.
    :param rho_refs:  List of reference density tensors.
    :param eps: Numerical stability constant.
    :return: Scalar mean NMAE.
    """
    losses = [
        torch.abs(pred - ref).sum() / (ref.abs().sum() + eps)
        for pred, ref in zip(rho_preds, rho_refs)
    ]
    return torch.stack(losses).mean()


class TensorMapNMAELoss(LossInterface):
    """Normalised Mean Absolute Error on per-structure TensorMap targets.

    Computes ``sum|pred - ref| / sum|ref|`` independently for each key-block
    pair, then averages across blocks. Intended for charge-density targets
    where raw MAE is meaningless without normalisation by the total charge.

    Registered with :func:`metatrain.utils.loss.create_loss` under the type
    key ``"nmae"`` by :mod:`metatrain.experimental.electrafy`'s
    ``_register_electrafy_losses`` shim, so ``LossSpecification(type="nmae")``
    works the same way it does for ``"mse"`` or ``"mae"``.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(name, gradient, weight, reduction)
        self.eps = eps

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        pred_tmap = predictions[self.target]
        ref_tmap = targets[self.target]

        block_losses: List[torch.Tensor] = []
        for key in pred_tmap.keys:
            pred_vals = pred_tmap.block(key).values.reshape(-1)
            ref_vals = ref_tmap.block(key).values.reshape(-1)
            nmae = pred_vals.sub(ref_vals).abs().sum() / (
                ref_vals.abs().sum() + self.eps
            )
            block_losses.append(nmae)

        return torch.stack(block_losses).mean()
