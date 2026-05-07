"""
Analytic Gaussian-to-Fourier density pipeline (ELECTRAFY core).

Given a set of 3D anisotropic Gaussians (weights, centers, covariances),
computes the periodic real-space charge density via the analytic Fourier
transform of Gaussians + IFFT (Poisson summation formula).

Math
----
For a Gaussian N(r; mu_j, Sigma_j) the continuous Fourier transform is:
    F[N](G) = exp(-1/2  G^T Sigma_j G) * exp(-i G . mu_j)

The periodic density Fourier coefficients are therefore:
    rho_hat(G) = sum_j  w_j * exp(-1/2  G^T Sigma_j G) * exp(-i G . mu_j)

Applying IFFT with the correct volume prefactor recovers the real-space grid:
    rho(r_n) = (N_total / V) * IFFT[rho_hat](n)

The density is finally renormalized so that its integral matches the true
number of valence electrons.
"""

import math
import os
from typing import Tuple

import torch


def _build_g_vectors(
    grid_shape: Tuple[int, int, int],
    cell: torch.Tensor,
    rfft: bool = False,
) -> torch.Tensor:
    """
    Build the 3D array of reciprocal-space G-vectors for an FFT grid.

    :param grid_shape: (N1, N2, N3) — number of grid points along each axis.
    :param cell: (3, 3) tensor whose rows are the lattice vectors a1, a2, a3
        (VASP convention, in Angstrom).
    :param rfft: If True, only build the half-grid suitable for ``irfftn`` —
        i.e. the last axis runs over k3 ∈ [0, N3//2] instead of all N3
        FFT frequencies. Output shape becomes (N1, N2, N3//2 + 1, 3).
    :return: G-vector tensor of shape (N1, N2, N3, 3) (or
        (N1, N2, N3//2 + 1, 3) if ``rfft=True``), in Angstrom^{-1}.
    """
    N1, N2, N3 = grid_shape
    device = cell.device
    dtype = cell.dtype

    # Reciprocal lattice matrix B; rows are b1, b2, b3 with a_i . b_j = 2pi delta_ij
    B = 2 * math.pi * torch.linalg.inv(cell).T  # (3, 3)

    # Integer FFT frequencies. fftfreq gives [0, 1, ..., N/2-1, -N/2, ..., -1]/N;
    # rfftfreq gives [0, 1, ..., N//2]/N. Multiplying by N recovers integers.
    f1 = torch.fft.fftfreq(N1, device=device, dtype=dtype) * N1
    f2 = torch.fft.fftfreq(N2, device=device, dtype=dtype) * N2
    if rfft:
        f3 = torch.fft.rfftfreq(N3, device=device, dtype=dtype) * N3
    else:
        f3 = torch.fft.fftfreq(N3, device=device, dtype=dtype) * N3

    # Reciprocal vectors on the FFT grid: G_k = k1*b1 + k2*b2 + k3*b3.
    # For a grid r_n = (n1/N1)*a1 + (n2/N2)*a2 + (n3/N3)*a3, the IFFT phase
    # is exp(i G_k · r_n) = exp(2pi i (k1*n1/N1 + k2*n2/N2 + k3*n3/N3)),
    # which requires G_k = k_i * b_i (no N division).
    b1 = B[0]  # (3,)
    b2 = B[1]
    b3 = B[2]

    # Broadcast to (N1, N2, N3, 3) — or (N1, N2, N3//2+1, 3) when rfft=True.
    G = (
        f1[:, None, None, None] * b1
        + f2[None, :, None, None] * b2
        + f3[None, None, :, None] * b3
    )
    return G


def _fourier_chunk_body(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    G_chunk: torch.Tensor,
) -> torch.Tensor:
    """Compute Fourier coefficients for one chunk of G-vectors.

    Returned shape: (chunk,) complex. Pure-functional, suitable for
    use under :func:`torch.utils.checkpoint.checkpoint`.

    The quadratic form ``-1/2 G^T Σ_j G`` is split into two ops:
    a `(jde, ge -> gjd)` einsum to form `SigmaG`, then an elementwise
    multiply with `G_chunk` reduced over the spatial axis. This is
    deliberately NOT collapsed into a single 3-arg einsum: tranche T5
    showed that fusing into `"gd, jde, ge -> gj"` (with `G_chunk`
    appearing twice) interacts badly with `torch.compile(dynamic=True)`
    + `torch.utils.checkpoint(use_reentrant=False)` — the duplicate
    aliased input drops one branch of the backward graph and gradients
    into `covs` (and thence the γ-MLP / dyadic T branch) collapse, so
    the model never trains. The two-step form keeps the duplicate
    across separate Functions, so autograd wires up each contribution
    independently.
    """
    SigmaG = torch.einsum("jde, ge -> gjd", covs, G_chunk)        # (chunk, J, 3)
    exponent = -0.5 * (G_chunk[:, None, :] * SigmaG).sum(-1)      # (chunk, J)
    phase = torch.einsum("gd, jd -> gj", G_chunk, centers)        # (chunk, J)
    amp = weights[None, :] * torch.exp(exponent)                  # (chunk, J)
    return torch.complex(
        (amp * torch.cos(phase)).sum(-1),
        -(amp * torch.sin(phase)).sum(-1),
    )


# Surgical torch.compile of the Fourier chunk body.
#
# This is the smallest closed graph in the model that contains zero
# Python-int specialization (no `.item()`, no Python branches on tensor
# values, no shape-derived ints) — ideal for compile under dynamic=True.
# Inputs are (weights, centers, covs, G_chunk); shapes vary as
# `(N_gauss,)`, `(N_gauss, 3)`, `(N_gauss, 3, 3)`, `(chunk, 3)`. Dynamo
# emits a single dynamic-shape graph that handles every batch.
#
# Bench-historical: full-model compile (kuma 3170683 / 3171246 / 3172376)
# hung on the model's forward because of `int(n_gaussians_per_atom.sum().item())`
# in gaussian_density.py and `(N1, N2, N3)` Python tuples in model.py.
# Compiling INSIDE those Python boundaries — only the chunk body — keeps
# all those specializations out of the compiled graph.
#
# The two-step quadratic form ABOVE (SigmaG → elementwise → reduce, NOT a
# fused `gd, jde, ge -> gj` einsum) is intentionally preserved; T5 (kuma
# 2865159) showed the fused form drops covs.grad under
# compile + checkpoint(use_reentrant=False).
#
# Set ELECTRAFI_COMPILE_FOURIER=0 to disable (debugging / eager comparison).
_COMPILE_FOURIER = os.environ.get("ELECTRAFI_COMPILE_FOURIER", "1") != "0"
if _COMPILE_FOURIER:
    _fourier_chunk_body_call = torch.compile(_fourier_chunk_body, dynamic=True)
else:
    _fourier_chunk_body_call = _fourier_chunk_body


def gaussian_fourier_coefficients(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    G_flat: torch.Tensor,
    chunk_size: int = 4096,
    checkpoint: bool = True,
) -> torch.Tensor:
    """
    Compute Fourier coefficients rho_hat(G) for a Gaussian mixture.

    rho_hat(G) = sum_j  w_j * exp(-1/2 G^T Sigma_j G) * exp(-i G . mu_j)

    Each chunk's heavy intermediates (SigmaG, exponent, phase, amp) are wrapped
    in :func:`torch.utils.checkpoint.checkpoint` when ``checkpoint=True`` (the
    default during training): without checkpointing, autograd retains every
    chunk's activations across the entire G-vector loop, scaling memory as
    O(N_G * N_gauss * 3) — which OOMs the H100 at paper-scale models. With
    checkpointing, memory scales as O(chunk_size * N_gauss * 3), at the cost
    of one extra forward pass per chunk during backward (~30% wall-clock).

    :param weights: (N_gauss,) real weights.
    :param centers: (N_gauss, 3) Gaussian centers mu_j in Angstrom.
    :param covs: (N_gauss, 3, 3) positive-definite covariance matrices Sigma_j.
    :param G_flat: (N_G, 3) G-vectors in Angstrom^{-1}.
    :param chunk_size: Number of G-vectors processed per chunk.
    :param checkpoint: If True (default), apply gradient checkpointing per
        chunk during backward to bound activation memory. Disabled when no
        input requires grad (eval / no-grad).
    :return: Complex tensor of shape (N_G,).
    """
    N_G = G_flat.shape[0]
    use_ckpt = checkpoint and (
        weights.requires_grad or centers.requires_grad or covs.requires_grad
    )

    chunks: list[torch.Tensor] = []
    for start in range(0, N_G, chunk_size):
        end = min(start + chunk_size, N_G)
        G_chunk = G_flat[start:end]  # (chunk, 3)
        if use_ckpt:
            out = torch.utils.checkpoint.checkpoint(
                _fourier_chunk_body_call, weights, centers, covs, G_chunk,
                use_reentrant=False,
            )
        else:
            out = _fourier_chunk_body_call(weights, centers, covs, G_chunk)
        chunks.append(out)

    return torch.cat(chunks, dim=0)


def periodic_density_from_gaussians(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    cell: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    n_electrons: float,
    chunk_size: int = 4096,
    use_rfft: bool = False,
) -> torch.Tensor:
    """
    Compute the periodic real-space charge density from Gaussian parameters.

    Default path uses the full complex ``ifftn`` then takes ``.real`` (this is
    what the ELECTRAFI paper's Eq 10 specifies). An ``irfftn``-based half-grid
    fast path is implemented behind ``use_rfft=True`` and gives ~2x speedup,
    BUT introduces ~1-2% relative error at typical paper-scale Gaussian
    widths because the analytic ρ̂(G) is only Hermitian-symmetric on the
    *continuous* G — on the discrete FFT grid it is NOT (the Nyquist of any
    even axis is sampled at -N/2 by ``fftfreq`` versus +N/2 by ``rfftfreq``,
    and these are conjugate-but-distinct complex values). Empirically that
    error is large enough to cap training above the paper's 0.58% NMAE floor,
    so ``use_rfft`` is OFF by default. Toggle on for ablations only.

    :param weights: (N_gauss,) Gaussian weights (need not be normalized).
    :param centers: (N_gauss, 3) Gaussian centers in Angstrom.
    :param covs: (N_gauss, 3, 3) positive-definite covariance matrices in Angstrom^2.
    :param cell: (3, 3) lattice matrix with rows = lattice vectors (Angstrom).
    :param grid_shape: (N1, N2, N3) real-space FFT grid dimensions.
    :param n_electrons: Total number of valence electrons (used for normalization).
    :param chunk_size: G-vector chunk size for memory-efficient computation.
    :param use_rfft: See note above. Default False.
    :return: Real-valued density tensor of shape (N1, N2, N3) in electrons/Angstrom^3.
    """
    N1, N2, N3 = grid_shape
    N_total = N1 * N2 * N3

    if use_rfft:
        # Build half-grid G-vectors (last axis runs k3 ∈ [0, N3//2]).
        G = _build_g_vectors(grid_shape, cell, rfft=True)  # (N1, N2, N3//2+1, 3)
        N3_r = N3 // 2 + 1
        G_flat = G.reshape(-1, 3)  # (N1*N2*N3_r, 3)
        rho_hat_flat = gaussian_fourier_coefficients(
            weights, centers, covs, G_flat, chunk_size=chunk_size
        )
        rho_hat_half = rho_hat_flat.reshape(N1, N2, N3_r)
        # irfftn: complex (N1, N2, N3//2+1) -> real (N1, N2, N3); applies the
        # implicit Hermitian extension along the last axis.
        rho = torch.fft.irfftn(rho_hat_half, s=grid_shape) * (N_total / 1.0)
    else:
        G = _build_g_vectors(grid_shape, cell, rfft=False)  # (N1, N2, N3, 3)
        G_flat = G.reshape(-1, 3)
        rho_hat_flat = gaussian_fourier_coefficients(
            weights, centers, covs, G_flat, chunk_size=chunk_size
        )
        rho_hat = rho_hat_flat.reshape(N1, N2, N3)
        rho = torch.fft.ifftn(rho_hat).real * (N_total / 1.0)

    # Continuous: rho(r_n) = (1/V) sum_G rho_hat(G) exp(iG.r_n)
    # Discrete:   rho(r_n) = (N_total/V) * IFFT[rho_hat](n)  — see prefactor below.
    V = torch.abs(torch.linalg.det(cell))
    rho = rho / V

    # Normalize to correct electron count (paper Eq 11: rescale all weights
    # by a single POSITIVE factor so ∫ρ dr = N_e). We rescale ρ post-IFFT,
    # which is mathematically equivalent. The factor must be positive to
    # avoid sign-flipping ρ everywhere when ∫ρ < 0 (common at init, since
    # tanh weights are random near zero). We take the absolute value of the
    # integral with a small floor to keep gradients defined.
    integral = rho.sum() * (V / N_total)
    ratio = n_electrons / torch.clamp(integral.abs(), min=1e-8)
    ratio = torch.clamp(ratio, max=1e4)
    rho = rho * ratio

    return rho
