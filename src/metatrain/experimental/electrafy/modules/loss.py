"""
NMAE (Normalized Mean Absolute Error) loss for charge density prediction.

L = integral |rho_pred(r) - rho_ref(r)| dV
    ------------------------------------------
             integral rho_ref(r) dV

Both integrals are approximated by the sum over the real-space grid
multiplied by the voxel volume (V / N_total).  The volume factor cancels,
leaving a simple ratio of sums.
"""

import torch
import torch.nn as nn


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
