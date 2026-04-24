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
from typing import Tuple

import torch


def _build_g_vectors(
    grid_shape: Tuple[int, int, int],
    cell: torch.Tensor,
) -> torch.Tensor:
    """
    Build the 3D array of reciprocal-space G-vectors for an FFT grid.

    :param grid_shape: (N1, N2, N3) — number of grid points along each axis.
    :param cell: (3, 3) tensor whose rows are the lattice vectors a1, a2, a3
        (VASP convention, in Angstrom).
    :return: G-vector tensor of shape (N1, N2, N3, 3), in Angstrom^{-1}.
    """
    N1, N2, N3 = grid_shape
    device = cell.device
    dtype = cell.dtype

    # Reciprocal lattice matrix B; rows are b1, b2, b3 with a_i . b_j = 2pi delta_ij
    B = 2 * math.pi * torch.linalg.inv(cell).T  # (3, 3)

    # Integer FFT frequencies: [0, 1, ..., N/2-1, -N/2, ..., -1]
    f1 = torch.fft.fftfreq(N1, device=device, dtype=dtype) * N1
    f2 = torch.fft.fftfreq(N2, device=device, dtype=dtype) * N2
    f3 = torch.fft.fftfreq(N3, device=device, dtype=dtype) * N3

    # Reciprocal vectors on the FFT grid: G_k = k1*b1 + k2*b2 + k3*b3.
    # For a grid r_n = (n1/N1)*a1 + (n2/N2)*a2 + (n3/N3)*a3, the IFFT phase
    # is exp(i G_k · r_n) = exp(2pi i (k1*n1/N1 + k2*n2/N2 + k3*n3/N3)),
    # which requires G_k = k_i * b_i (no N division).
    b1 = B[0]  # (3,)
    b2 = B[1]
    b3 = B[2]

    # Broadcast to (N1, N2, N3, 3)
    G = (
        f1[:, None, None, None] * b1
        + f2[None, :, None, None] * b2
        + f3[None, None, :, None] * b3
    )
    return G  # (N1, N2, N3, 3)


def gaussian_fourier_coefficients(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    G_flat: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute Fourier coefficients rho_hat(G) for a Gaussian mixture.

    rho_hat(G) = sum_j  w_j * exp(-1/2 G^T Sigma_j G) * exp(-i G . mu_j)

    :param weights: (N_gauss,) real weights.
    :param centers: (N_gauss, 3) Gaussian centers mu_j in Angstrom.
    :param covs: (N_gauss, 3, 3) positive-definite covariance matrices Sigma_j.
    :param G_flat: (N_G, 3) G-vectors in Angstrom^{-1}.
    :param chunk_size: Number of G-vectors processed per chunk (memory control).
    :return: Complex tensor of shape (N_G,).
    """
    N_G = G_flat.shape[0]
    dtype = weights.dtype
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    rho_hat = torch.zeros(N_G, dtype=cdtype, device=weights.device)

    for start in range(0, N_G, chunk_size):
        end = min(start + chunk_size, N_G)
        G_chunk = G_flat[start:end]  # (chunk, 3)

        # Quadratic form: -1/2 G^T Sigma_j G for all (G, j) pairs
        # SigmaG: (chunk, N_gauss, 3)
        SigmaG = torch.einsum("gd, jde -> gje", G_chunk, covs)
        # exponent: (chunk, N_gauss)
        exponent = -0.5 * (G_chunk[:, None, :] * SigmaG).sum(-1)

        # Phase: G . mu_j  -> (chunk, N_gauss)
        phase = torch.einsum("gd, jd -> gj", G_chunk, centers)

        # Amplitude per (G, j): w_j * exp(exponent)
        amp = weights[None, :] * torch.exp(exponent)  # (chunk, N_gauss)

        rho_hat[start:end] = torch.complex(
            (amp * torch.cos(phase)).sum(-1),
            -(amp * torch.sin(phase)).sum(-1),
        )

    return rho_hat


def periodic_density_from_gaussians(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    cell: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    n_electrons: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute the periodic real-space charge density from Gaussian parameters.

    :param weights: (N_gauss,) Gaussian weights (need not be normalized).
    :param centers: (N_gauss, 3) Gaussian centers in Angstrom.
    :param covs: (N_gauss, 3, 3) positive-definite covariance matrices in Angstrom^2.
    :param cell: (3, 3) lattice matrix with rows = lattice vectors (Angstrom).
    :param grid_shape: (N1, N2, N3) real-space FFT grid dimensions.
    :param n_electrons: Total number of valence electrons (used for normalization).
    :param chunk_size: G-vector chunk size for memory-efficient computation.
    :return: Real-valued density tensor of shape (N1, N2, N3) in electrons/Angstrom^3.
    """
    N1, N2, N3 = grid_shape
    N_total = N1 * N2 * N3

    # Build G-vector grid and flatten
    G = _build_g_vectors(grid_shape, cell)  # (N1, N2, N3, 3)
    G_flat = G.reshape(-1, 3)  # (N_total, 3)

    # Compute Fourier coefficients
    rho_hat_flat = gaussian_fourier_coefficients(
        weights, centers, covs, G_flat, chunk_size=chunk_size
    )
    rho_hat = rho_hat_flat.reshape(N1, N2, N3)

    # Inverse FFT + volume prefactor
    # Continuous: rho(r_n) = (1/V) sum_G rho_hat(G) exp(iG.r_n)
    # Discrete:   rho(r_n) = (N_total/V) * IFFT[rho_hat](n)
    V = torch.abs(torch.linalg.det(cell))
    rho = torch.fft.ifftn(rho_hat).real * (N_total / V)

    # Normalize to correct electron count
    # integral rho dV ≈ sum_n rho(r_n) * (V / N_total)
    integral = rho.sum() * (V / N_total)
    # Clamp the ratio to prevent explosion when integral is near-zero
    # (common early in training when weights are small)
    ratio = n_electrons / (integral + 1e-8 * torch.sign(integral + 1e-30))
    ratio = torch.clamp(ratio, min=-1e4, max=1e4)
    rho = rho * ratio

    return rho
