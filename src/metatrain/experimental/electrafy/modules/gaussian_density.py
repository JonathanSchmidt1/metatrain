"""
Gaussian density head: converts dyadic aggregation outputs (S, V, T)
into per-Gaussian parameters (weights, centers, covariances).

Each atom a with Z_val(a) valence electrons gets n_gauss(a) = M * Z_val(a)
Gaussians, indexed by channels c = 0 .. n_gauss(a)-1 of the (S, V, T) tensors.

Parameter construction (ELECTRAFY Eqs. 13-15)
---------------------------------------------
Weights  w_j = tanh(MLP(S_{i,c}))
              Sign-stable per Eq 13. No per-atom softmax: total-charge
              consistency is enforced downstream by the IFFT-side
              electron-count renormalization (Eq 11).

Centers  mu_j = R_atom + V_{i,c}   (direct vector displacement from atom)

Covariance  Sigma_j = gamma_j * T_{i,c} @ T_{i,c}^T + eps * I
            with gamma_j = gamma_prior * softplus(MLP(S_{i,c})), per-Gaussian
            (ELECTRAFY Eq 15). The MLP last layer is initialized so that
            softplus(output) ≈ 1 at start, i.e. gamma_j ≈ gamma_prior.
"""

from typing import Tuple

import torch
import torch.nn as nn


class GaussianDensityHead(nn.Module):
    """
    Converts per-atom (S, V, T) features into Gaussian mixture parameters.

    :param n_channels: C = M * max_zval; total feature channels per atom.
    :param gamma: Prior scale factor for covariance matrices (Angstrom^2).
        At init, per-Gaussian gamma_j ≈ this value; the MLP can then deviate.
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

        # Weight MLP: scalar channel -> raw weight logit (Eq 13).
        self.weight_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

        # Per-Gaussian gamma MLP: scalar S^(j) -> raw scale (Eq 15).
        # gamma_j = gamma_prior * softplus(raw); init so softplus(raw) ≈ 1.
        self.gamma_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        with torch.no_grad():
            last = self.gamma_mlp[-1]
            last.weight.zero_()
            # softplus^{-1}(1) = ln(e - 1) ≈ 0.5413
            last.bias.fill_(0.5413248546129181)

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
            - weights: (N_gauss,) signed real weights w = tanh(MLP(S)) per
              Eq 13. Total charge is fixed by IFFT-side renormalization, not
              by a per-atom softmax.
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

        # ---- Weights (ELECTRAFY Eq. 13) ----
        # w_j = tanh(MLP(S_{i,c})) — signed, sign-stable. Per-atom or
        # per-system sum is NOT constrained here; total charge is set by
        # IFFT-side renormalization (Eq 11) in periodic_density_from_gaussians.
        w_raw = self.weight_mlp(s_flat.unsqueeze(-1)).squeeze(-1)  # (N_gauss,)
        weights = torch.tanh(w_raw)  # (N_gauss,)

        # ---- Centers ----
        centers = pos_flat + v_flat  # (N_gauss, 3)

        # ---- Covariances (ELECTRAFY Eq 15) ----
        # Sigma_j = gamma_j * T_j T_j^T + eps * I
        # gamma_j = gamma_prior * softplus(MLP(S_j))   — per-Gaussian, positive
        TT = torch.bmm(t_flat, t_flat.transpose(-1, -2))  # (N_gauss, 3, 3)
        gamma_raw = self.gamma_mlp(s_flat.unsqueeze(-1)).squeeze(-1)  # (N_gauss,)
        gamma_j = self.gamma * torch.nn.functional.softplus(gamma_raw)  # (N_gauss,)
        I3 = torch.eye(3, device=device, dtype=dtype)
        covs = gamma_j[:, None, None] * TT + self.eps_cov * I3[None, :, :]

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
