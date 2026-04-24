"""
Gaussian density head: converts dyadic aggregation outputs (S, V, T)
into per-Gaussian parameters (weights, centers, covariances).

Each atom a with Z_val(a) valence electrons gets n_gauss(a) = M * Z_val(a)
Gaussians, indexed by channels c = 0 .. n_gauss(a)-1 of the (S, V, T) tensors.

Parameter construction (ELECTRAFY Eqs. 13-15)
---------------------------------------------
Weights  w_j = tanh(MLP(S_{i,c}))
              then softmax-normalized per atom so integral = n_electrons_atom

Centers  mu_j = R_atom + V_{i,c}   (direct vector displacement from atom)

Covariance  Sigma_j = gamma * T_{i,c} @ T_{i,c}^T + eps * I
            (Gram factorization ensures positive definiteness)
"""

from typing import Tuple

import torch
import torch.nn as nn


class GaussianDensityHead(nn.Module):
    """
    Converts per-atom (S, V, T) features into Gaussian mixture parameters.

    :param n_channels: C = M * max_zval; total feature channels per atom.
    :param gamma: Global scale factor for covariance matrices (Angstrom^2).
    :param eps_cov: Small diagonal term added to each covariance for stability.
    """

    def __init__(
        self,
        n_channels: int,
        gamma: float = 0.1,
        eps_cov: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.gamma = gamma
        self.eps_cov = eps_cov

        # Weight MLP: scalar channel -> raw weight logit
        self.weight_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        S: torch.Tensor,
        V: torch.Tensor,
        T: torch.Tensor,
        positions: torch.Tensor,
        n_gaussians_per_atom: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Gaussian parameters for the full batch.

        :param S: (N_atoms, C) scalar features.
        :param V: (N_atoms, C, 3) vector features.
        :param T: (N_atoms, C, 3, 3) tensor features.
        :param positions: (N_atoms, 3) atom positions in Angstrom.
        :param n_gaussians_per_atom: (N_atoms,) integer tensor; number of
            Gaussians assigned to each atom (= M * Z_val(a)).
        :return: Tuple (weights, centers, covs, atom_idx):
            - weights: (N_gauss,) real weights (tanh-scaled, segment-softmax
              normalized per atom).
            - centers: (N_gauss, 3) Gaussian centers in Angstrom.
            - covs:    (N_gauss, 3, 3) positive-definite covariance matrices.
            - atom_idx: (N_gauss,) index of the parent atom for each Gaussian.
        """
        device = S.device
        dtype = S.dtype
        N = S.shape[0]
        C = self.n_channels

        total_gauss = int(n_gaussians_per_atom.sum().item())
        if total_gauss == 0:
            empty = torch.zeros(0, device=device, dtype=dtype)
            return (
                empty,
                torch.zeros(0, 3, device=device, dtype=dtype),
                torch.zeros(0, 3, 3, device=device, dtype=dtype),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        # ---- Vectorized channel slicing ----
        # Build a (N, C) mask where mask[a, c] = (c < n_gaussians_per_atom[a])
        channel_indices = torch.arange(C, device=device)  # (C,)
        mask = channel_indices[None, :] < n_gaussians_per_atom[:, None]  # (N, C)

        # Flat gather indices: which (atom, channel) pairs are active
        atom_idx = torch.repeat_interleave(
            torch.arange(N, device=device), n_gaussians_per_atom
        )  # (N_gauss,)

        # Per-atom channel offsets via cumsum trick
        offsets = torch.zeros(N + 1, device=device, dtype=torch.long)
        offsets[1:] = n_gaussians_per_atom.cumsum(0)
        # For each Gaussian, compute its channel index within its atom
        gauss_global_idx = torch.arange(total_gauss, device=device)
        channel_idx = gauss_global_idx - offsets[atom_idx]  # (N_gauss,)

        # Gather features using advanced indexing (no Python loop, no GPU sync)
        s_flat = S[atom_idx, channel_idx]           # (N_gauss,)
        v_flat = V[atom_idx, channel_idx, :]        # (N_gauss, 3)
        t_flat = T[atom_idx, channel_idx, :, :]     # (N_gauss, 3, 3)
        pos_flat = positions[atom_idx]              # (N_gauss, 3)

        # ---- Weights ----
        # tanh allows negative contributions (ELECTRAFY Eq. 13),
        # then segment-softmax normalizes magnitudes per atom.
        w_raw = self.weight_mlp(s_flat.unsqueeze(-1)).squeeze(-1)  # (N_gauss,)
        w_raw = torch.tanh(w_raw)
        weights = _segment_softmax(w_raw, atom_idx, N)  # (N_gauss,)

        # ---- Centers ----
        centers = pos_flat + v_flat  # (N_gauss, 3)

        # ---- Covariances ----
        # Gram factorization: Sigma = gamma * T @ T^T + eps * I
        TT = torch.bmm(t_flat, t_flat.transpose(-1, -2))  # (N_gauss, 3, 3)
        I3 = torch.eye(3, device=device, dtype=dtype)
        covs = self.gamma * TT + self.eps_cov * I3[None, :, :]  # (N_gauss, 3, 3)

        return weights, centers, covs, atom_idx


def _segment_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    """
    Compute softmax of `logits` within each segment defined by `segment_ids`.

    :param logits: (N,) tensor of raw logits.
    :param segment_ids: (N,) integer tensor in [0, n_segments).
    :param n_segments: Total number of segments.
    :return: (N,) tensor with per-segment softmax applied.
    """
    # Subtract per-segment max for numerical stability
    seg_max = torch.full(
        (n_segments,), float("-inf"), device=logits.device, dtype=logits.dtype
    )
    seg_max.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=False)
    logits_shifted = logits - seg_max[segment_ids]

    exp_logits = torch.exp(logits_shifted)

    # Sum per segment
    seg_sum = torch.zeros(n_segments, device=logits.device, dtype=logits.dtype)
    seg_sum.scatter_add_(0, segment_ids, exp_logits)

    return exp_logits / (seg_sum[segment_ids] + 1e-10)
