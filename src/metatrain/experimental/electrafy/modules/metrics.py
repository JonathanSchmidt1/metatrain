"""ELECTRAFY-specific metric accumulators.

Mirrors the API of :class:`metatrain.utils.metrics.RMSEAccumulator` /
:class:`metatrain.utils.metrics.MAEAccumulator` so the existing trainer
plumbing (``update`` / ``finalize`` / distributed all-reduce) drops in
without changes.

NMAE (Normalised Mean Absolute Error) is the canonical metric for charge
density:

    NMAE(rho_pred, rho_ref) = sum|rho_pred - rho_ref| / sum|rho_ref|

The two accumulator slots therefore hold (sum_abs_err, sum_abs_target)
rather than the (sum_squared_err, n_elem) used by RMSEAccumulator.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import TensorMap


class NMAEAccumulator:
    """Accumulates the NMAE between predictions and targets.

    Drop-in shape-compatible substitute for
    :class:`metatrain.utils.metrics.RMSEAccumulator`: same constructor
    signature, ``update`` signature, and ``finalize`` signature/return type
    (a ``Dict[str, float]``) so the trainer can swap one in for the other.

    :param separate_blocks: if true, NMAE is computed separately for each
        block in the target / prediction ``TensorMap`` objects.
    """

    information: Dict[str, Tuple[float, float]]
    separate_blocks: bool

    def __init__(self, separate_blocks: bool = False) -> None:
        self.information = {}
        """Mapping ``{target_key: (sum_abs_err, sum_abs_target)}``."""

        self.separate_blocks = separate_blocks
        """Whether to compute NMAE per-block (False = aggregate over all
        blocks of a target)."""

    def update(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """Update the accumulator with one batch of predictions/targets.

        Mirrors :py:meth:`metatrain.utils.metrics.MAEAccumulator.update` so a
        mask in ``extra_data['{key}_mask']`` is honoured the same way.
        """
        for key, target in targets.items():
            prediction = predictions[key]

            mask = None
            if extra_data is not None:
                mask_key = f"{key}_mask"
                if mask_key in extra_data:
                    mask = extra_data[mask_key]

            for block_key in target.keys:
                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += " ("
                    for name, value in zip(
                        block_key.names, block_key.values, strict=True
                    ):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1] + ")"

                if key_to_write not in self.information:
                    self.information[key_to_write] = (0.0, 0.0)

                if mask is None:
                    mask_as_tensor = ~torch.isnan(target_block.values)
                    pred_vals = prediction_block.values[mask_as_tensor]
                    ref_vals = target_block.values[mask_as_tensor]
                else:
                    mask_as_tensor = mask.block(block_key).values
                    pred_vals = prediction_block.values[mask_as_tensor]
                    ref_vals = target_block.values[mask_as_tensor]

                abs_err = (pred_vals - ref_vals).abs().sum().item()
                abs_ref = ref_vals.abs().sum().item()

                prev_err, prev_ref = self.information[key_to_write]
                self.information[key_to_write] = (
                    prev_err + abs_err,
                    prev_ref + abs_ref,
                )

    def finalize(
        self,
        not_per_atom: List[str],
        is_distributed: bool = False,
        device: Optional[torch.device] = None,
        eps: float = 1e-30,
    ) -> Dict[str, float]:
        """Reduce per-rank state and return the NMAE for each key.

        Output keys are formatted as ``"{key} NMAE"`` -- no per-atom variant
        since NMAE is intrinsically normalised by the integrated reference.

        :param not_per_atom: list of substrings that flag a key as already
            non-per-atom. Accepted for API compatibility with RMSEAccumulator
            but does not change the label (which never uses "(per atom)").
        :param is_distributed: when True, all-reduce ``sum_abs_err`` and
            ``sum_abs_target`` across ranks before forming the ratio.
        :param device: local CUDA device for the all-reduce. Required when
            ``is_distributed=True``.
        :param eps: numerical floor on the denominator to avoid division by
            zero for all-zero targets (e.g. masked-out blocks).
        """
        # Suppress unused-arg lint -- kept for shape-compat with RMSEAccumulator.
        _ = not_per_atom

        if is_distributed:
            from metatrain.utils.metrics import _get_global_keys

            sorted_global_keys = _get_global_keys(list(self.information.keys()))
            for key in sorted_global_keys:
                if key in self.information:
                    abs_err = torch.tensor(
                        self.information[key][0], device=device
                    )
                    abs_ref = torch.tensor(
                        self.information[key][1], device=device
                    )
                else:
                    abs_err = torch.tensor(0.0, device=device)
                    abs_ref = torch.tensor(0.0, device=device)
                torch.distributed.all_reduce(abs_err)
                torch.distributed.all_reduce(abs_ref)
                self.information[key] = (abs_err.item(), abs_ref.item())

        finalized: Dict[str, float] = {}
        for key, (abs_err, abs_ref) in self.information.items():
            finalized[f"{key} NMAE"] = abs_err / max(abs_ref, eps)
        return finalized


__all__ = ["NMAEAccumulator"]
