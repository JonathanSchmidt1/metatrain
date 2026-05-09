"""
Triton fused kernel for the Fourier-chunk inner loop.

Replaces the four `(chunk, J)`-sized intermediates that the PyTorch reference
materializes (SigmaG, exponent, phase, amp) with a single fused kernel that
keeps everything in registers and writes only the (chunk,) complex output to
HBM. Targets the memory-bandwidth-bound regime: at d_pet=512 + gpe=128 + 50
atoms, the (chunk, J) tensors are ~100 MB each, written and re-read multiple
times across PyTorch ops; the Triton fusion avoids them entirely.

This first revision implements FORWARD ONLY. Backward is a follow-up — for
now this module is for benchmarking + numerical-equivalence proof.
Inputs MUST be contiguous fp32 on CUDA.

API
---
``triton_fourier_chunk_fwd(weights, centers, covs, G_chunk) -> complex``

    Pure functional. Equivalent to ``modules/fourier_density.py::
    _fourier_chunk_body`` (same math, T5-trap-safe — the cov quadratic is
    expanded scalar-by-scalar, so no autograd-aliasing concerns).

T5-trap status
--------------
The cov quadratic ``-1/2 G^T Σ_j G`` is unrolled into nine scalar mul+adds
per (g, j) inside the kernel. There is no fused 3-arg einsum and no Python-
level autograd graph at all, so the trap that fired in T5 (gradient through
``covs`` collapsing under torch.compile + checkpoint(use_reentrant=False)
when G_chunk appears twice) is structurally inapplicable.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _fourier_bwd_kernel(
        weights_ptr,
        centers_ptr,
        covs_ptr,
        G_ptr,
        grad_re_ptr,    # (G_size,) fp32
        grad_im_ptr,    # (G_size,) fp32
        dw_ptr,         # (J,) fp32  out
        dc_ptr,         # (J, 3) fp32  out
        dK_ptr,         # (J, 3, 3) fp32  out
        G_size,
        J,
        BLOCK_G: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """One program per j-tile. Inner loop over g-tiles, recomputing fwd."""
        pid_j = tl.program_id(0)
        j_off = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        j_mask = j_off < J

        w = tl.load(weights_ptr + j_off, mask=j_mask, other=0.0)
        c0 = tl.load(centers_ptr + j_off * 3 + 0, mask=j_mask, other=0.0)
        c1 = tl.load(centers_ptr + j_off * 3 + 1, mask=j_mask, other=0.0)
        c2 = tl.load(centers_ptr + j_off * 3 + 2, mask=j_mask, other=0.0)
        K00 = tl.load(covs_ptr + j_off * 9 + 0, mask=j_mask, other=0.0)
        K01 = tl.load(covs_ptr + j_off * 9 + 1, mask=j_mask, other=0.0)
        K02 = tl.load(covs_ptr + j_off * 9 + 2, mask=j_mask, other=0.0)
        K10 = tl.load(covs_ptr + j_off * 9 + 3, mask=j_mask, other=0.0)
        K11 = tl.load(covs_ptr + j_off * 9 + 4, mask=j_mask, other=0.0)
        K12 = tl.load(covs_ptr + j_off * 9 + 5, mask=j_mask, other=0.0)
        K20 = tl.load(covs_ptr + j_off * 9 + 6, mask=j_mask, other=0.0)
        K21 = tl.load(covs_ptr + j_off * 9 + 7, mask=j_mask, other=0.0)
        K22 = tl.load(covs_ptr + j_off * 9 + 8, mask=j_mask, other=0.0)

        dw_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dc0_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dc1_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dc2_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK00_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK01_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK02_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK10_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK11_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK12_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK20_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK21_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
        dK22_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)

        for g_start in range(0, G_size, BLOCK_G):
            g_off = g_start + tl.arange(0, BLOCK_G)
            g_mask = g_off < G_size

            G0 = tl.load(G_ptr + g_off * 3 + 0, mask=g_mask, other=0.0)
            G1 = tl.load(G_ptr + g_off * 3 + 1, mask=g_mask, other=0.0)
            G2 = tl.load(G_ptr + g_off * 3 + 2, mask=g_mask, other=0.0)
            grad_re = tl.load(grad_re_ptr + g_off, mask=g_mask, other=0.0)
            grad_im = tl.load(grad_im_ptr + g_off, mask=g_mask, other=0.0)

            # Recompute phase, exponent, amp (matches fwd math).
            phase = (
                G0[:, None] * c0[None, :]
                + G1[:, None] * c1[None, :]
                + G2[:, None] * c2[None, :]
            )
            SG0 = (
                K00[None, :] * G0[:, None]
                + K01[None, :] * G1[:, None]
                + K02[None, :] * G2[:, None]
            )
            SG1 = (
                K10[None, :] * G0[:, None]
                + K11[None, :] * G1[:, None]
                + K12[None, :] * G2[:, None]
            )
            SG2 = (
                K20[None, :] * G0[:, None]
                + K21[None, :] * G1[:, None]
                + K22[None, :] * G2[:, None]
            )
            exponent = -0.5 * (
                G0[:, None] * SG0
                + G1[:, None] * SG1
                + G2[:, None] * SG2
            )
            e_exp = tl.exp(exponent)
            amp = w[None, :] * e_exp
            cos_p = tl.cos(phase)
            sin_p = tl.sin(phase)

            # Pullbacks. out_re = Σ_j amp*cos, out_im = -Σ_j amp*sin.
            # d_amp = grad_re*cos - grad_im*sin
            # d_phase = -amp*(grad_re*sin + grad_im*cos)
            d_amp = grad_re[:, None] * cos_p - grad_im[:, None] * sin_p
            d_phase = -amp * (grad_re[:, None] * sin_p + grad_im[:, None] * cos_p)
            # d_exponent = d_amp * amp (because d/dx (w*exp(x)) = w*exp(x) = amp)
            d_exp_acc = d_amp * amp

            valid = j_mask[None, :] & g_mask[:, None]
            d_amp = tl.where(valid, d_amp, 0.0)
            d_phase = tl.where(valid, d_phase, 0.0)
            d_exp_acc = tl.where(valid, d_exp_acc, 0.0)

            # d_weights[j] = Σ_g d_amp[g,j] * e_exp[g,j]
            dw_acc += tl.sum(d_amp * e_exp, axis=0)
            # d_centers[j, k] = Σ_g d_phase[g,j] * G[g,k]
            dc0_acc += tl.sum(d_phase * G0[:, None], axis=0)
            dc1_acc += tl.sum(d_phase * G1[:, None], axis=0)
            dc2_acc += tl.sum(d_phase * G2[:, None], axis=0)
            # d_covs[j, d, e] = -0.5 Σ_g d_exp_acc[g,j] * G[g,d] * G[g,e]
            half_de = -0.5 * d_exp_acc
            dK00_acc += tl.sum(half_de * (G0[:, None] * G0[:, None]), axis=0)
            dK01_acc += tl.sum(half_de * (G0[:, None] * G1[:, None]), axis=0)
            dK02_acc += tl.sum(half_de * (G0[:, None] * G2[:, None]), axis=0)
            dK10_acc += tl.sum(half_de * (G1[:, None] * G0[:, None]), axis=0)
            dK11_acc += tl.sum(half_de * (G1[:, None] * G1[:, None]), axis=0)
            dK12_acc += tl.sum(half_de * (G1[:, None] * G2[:, None]), axis=0)
            dK20_acc += tl.sum(half_de * (G2[:, None] * G0[:, None]), axis=0)
            dK21_acc += tl.sum(half_de * (G2[:, None] * G1[:, None]), axis=0)
            dK22_acc += tl.sum(half_de * (G2[:, None] * G2[:, None]), axis=0)

        tl.store(dw_ptr + j_off, dw_acc, mask=j_mask)
        tl.store(dc_ptr + j_off * 3 + 0, dc0_acc, mask=j_mask)
        tl.store(dc_ptr + j_off * 3 + 1, dc1_acc, mask=j_mask)
        tl.store(dc_ptr + j_off * 3 + 2, dc2_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 0, dK00_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 1, dK01_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 2, dK02_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 3, dK10_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 4, dK11_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 5, dK12_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 6, dK20_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 7, dK21_acc, mask=j_mask)
        tl.store(dK_ptr + j_off * 9 + 8, dK22_acc, mask=j_mask)

    @triton.jit
    def _fourier_fwd_kernel(
        weights_ptr,    # (J,) fp32
        centers_ptr,    # (J, 3) fp32 row-major
        covs_ptr,       # (J, 3, 3) fp32 row-major
        G_ptr,          # (G_size, 3) fp32 row-major
        out_re_ptr,     # (G_size,) fp32
        out_im_ptr,     # (G_size,) fp32
        G_size,
        J,
        BLOCK_G: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        pid = tl.program_id(0)
        g_off = pid * BLOCK_G + tl.arange(0, BLOCK_G)        # (BLOCK_G,)
        g_mask = g_off < G_size

        # Load this g-tile's G-vectors.
        G0 = tl.load(G_ptr + g_off * 3 + 0, mask=g_mask, other=0.0)
        G1 = tl.load(G_ptr + g_off * 3 + 1, mask=g_mask, other=0.0)
        G2 = tl.load(G_ptr + g_off * 3 + 2, mask=g_mask, other=0.0)

        re_acc = tl.zeros((BLOCK_G,), dtype=tl.float32)
        im_acc = tl.zeros((BLOCK_G,), dtype=tl.float32)

        for j_start in range(0, J, BLOCK_J):
            j_off = j_start + tl.arange(0, BLOCK_J)            # (BLOCK_J,)
            j_mask = j_off < J

            w = tl.load(weights_ptr + j_off, mask=j_mask, other=0.0)
            c0 = tl.load(centers_ptr + j_off * 3 + 0, mask=j_mask, other=0.0)
            c1 = tl.load(centers_ptr + j_off * 3 + 1, mask=j_mask, other=0.0)
            c2 = tl.load(centers_ptr + j_off * 3 + 2, mask=j_mask, other=0.0)
            K00 = tl.load(covs_ptr + j_off * 9 + 0, mask=j_mask, other=0.0)
            K01 = tl.load(covs_ptr + j_off * 9 + 1, mask=j_mask, other=0.0)
            K02 = tl.load(covs_ptr + j_off * 9 + 2, mask=j_mask, other=0.0)
            K10 = tl.load(covs_ptr + j_off * 9 + 3, mask=j_mask, other=0.0)
            K11 = tl.load(covs_ptr + j_off * 9 + 4, mask=j_mask, other=0.0)
            K12 = tl.load(covs_ptr + j_off * 9 + 5, mask=j_mask, other=0.0)
            K20 = tl.load(covs_ptr + j_off * 9 + 6, mask=j_mask, other=0.0)
            K21 = tl.load(covs_ptr + j_off * 9 + 7, mask=j_mask, other=0.0)
            K22 = tl.load(covs_ptr + j_off * 9 + 8, mask=j_mask, other=0.0)

            # Outer products (BLOCK_G, BLOCK_J).
            # phase[g, j] = G[g] · centers[j]
            phase = (
                G0[:, None] * c0[None, :]
                + G1[:, None] * c1[None, :]
                + G2[:, None] * c2[None, :]
            )

            # SigmaG[g, j, d] = covs[j, d, e] · G[g, e].   Three (BLOCK_G, BLOCK_J).
            SG0 = (
                K00[None, :] * G0[:, None]
                + K01[None, :] * G1[:, None]
                + K02[None, :] * G2[:, None]
            )
            SG1 = (
                K10[None, :] * G0[:, None]
                + K11[None, :] * G1[:, None]
                + K12[None, :] * G2[:, None]
            )
            SG2 = (
                K20[None, :] * G0[:, None]
                + K21[None, :] * G1[:, None]
                + K22[None, :] * G2[:, None]
            )

            # exponent[g, j] = -1/2 G[g] · SigmaG[g, j]
            exponent = -0.5 * (
                G0[:, None] * SG0
                + G1[:, None] * SG1
                + G2[:, None] * SG2
            )

            amp = w[None, :] * tl.exp(exponent)
            cos_p = tl.cos(phase)
            sin_p = tl.sin(phase)

            valid = j_mask[None, :] & g_mask[:, None]
            amp = tl.where(valid, amp, 0.0)

            re_acc += tl.sum(amp * cos_p, axis=1)
            im_acc += -tl.sum(amp * sin_p, axis=1)

        tl.store(out_re_ptr + g_off, re_acc, mask=g_mask)
        tl.store(out_im_ptr + g_off, im_acc, mask=g_mask)


def _check_inputs(weights, centers, covs, G_chunk):
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available")
    assert weights.is_cuda and centers.is_cuda and covs.is_cuda and G_chunk.is_cuda
    assert weights.dtype == torch.float32
    assert centers.dtype == torch.float32
    assert covs.dtype == torch.float32
    assert G_chunk.dtype == torch.float32
    G_size = G_chunk.shape[0]
    J = weights.shape[0]
    assert centers.shape == (J, 3)
    assert covs.shape == (J, 3, 3)
    assert G_chunk.shape == (G_size, 3)
    return G_size, J


def _launch_fwd(weights, centers, covs, G_chunk, block_g, block_j):
    G_size, J = _check_inputs(weights, centers, covs, G_chunk)
    out_re = torch.empty(G_size, device=G_chunk.device, dtype=torch.float32)
    out_im = torch.empty(G_size, device=G_chunk.device, dtype=torch.float32)
    grid = (triton.cdiv(G_size, block_g),)
    _fourier_fwd_kernel[grid](
        weights, centers, covs, G_chunk,
        out_re, out_im,
        G_size, J,
        BLOCK_G=block_g, BLOCK_J=block_j,
    )
    return out_re, out_im


def _launch_bwd(weights, centers, covs, G_chunk, grad_re, grad_im, block_g, block_j):
    G_size, J = _check_inputs(weights, centers, covs, G_chunk)
    dw = torch.empty(J, device=G_chunk.device, dtype=torch.float32)
    dc = torch.empty(J, 3, device=G_chunk.device, dtype=torch.float32)
    dK = torch.empty(J, 3, 3, device=G_chunk.device, dtype=torch.float32)
    grid = (triton.cdiv(J, block_j),)
    _fourier_bwd_kernel[grid](
        weights, centers, covs, G_chunk,
        grad_re.contiguous(), grad_im.contiguous(),
        dw, dc, dK,
        G_size, J,
        BLOCK_G=block_g, BLOCK_J=block_j,
    )
    return dw, dc, dK


class _FourierChunkTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, centers, covs, G_chunk, block_g, block_j):
        weights = weights.contiguous()
        centers = centers.contiguous()
        covs = covs.contiguous()
        G_chunk = G_chunk.contiguous()
        out_re, out_im = _launch_fwd(weights, centers, covs, G_chunk, block_g, block_j)
        # Save inputs for backward (we recompute fwd inside the bwd kernel, so
        # we don't need to save phase/amp/exponent — just the inputs).
        ctx.save_for_backward(weights, centers, covs, G_chunk)
        ctx.block_g = block_g
        ctx.block_j = block_j
        return torch.complex(out_re, out_im)

    @staticmethod
    def backward(ctx, grad_out):
        weights, centers, covs, G_chunk = ctx.saved_tensors
        grad_re = grad_out.real
        grad_im = grad_out.imag
        dw, dc, dK = _launch_bwd(
            weights, centers, covs, G_chunk,
            grad_re, grad_im,
            ctx.block_g, ctx.block_j,
        )
        # Returns one None per non-tensor positional arg (block_g, block_j)
        return dw, dc, dK, None, None, None


def triton_fourier_chunk(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    G_chunk: torch.Tensor,
    block_g: int = 64,
    block_j: int = 64,
) -> torch.Tensor:
    """Fully autograd-aware fused chunk kernel (fwd + bwd). Returns complex (chunk,)."""
    return _FourierChunkTriton.apply(weights, centers, covs, G_chunk, block_g, block_j)


def triton_fourier_chunk_fwd(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    G_chunk: torch.Tensor,
    block_g: int = 64,
    block_j: int = 64,
) -> torch.Tensor:
    """Forward-only entry point (no autograd). Useful for inference benches."""
    weights = weights.contiguous()
    centers = centers.contiguous()
    covs = covs.contiguous()
    G_chunk = G_chunk.contiguous()
    out_re, out_im = _launch_fwd(weights, centers, covs, G_chunk, block_g, block_j)
    return torch.complex(out_re, out_im)


# ---------------------------------------------------------------------------
# Reference + equivalence test (for direct execution as a script)
# ---------------------------------------------------------------------------


def _reference_chunk(
    weights: torch.Tensor,
    centers: torch.Tensor,
    covs: torch.Tensor,
    G_chunk: torch.Tensor,
) -> torch.Tensor:
    """v0 reference from fourier_density.py — kept here for self-contained tests."""
    SigmaG = torch.einsum("jde, ge -> gjd", covs, G_chunk)
    exponent = -0.5 * (G_chunk[:, None, :] * SigmaG).sum(-1)
    phase = torch.einsum("gd, jd -> gj", G_chunk, centers)
    amp = weights[None, :] * torch.exp(exponent)
    return torch.complex(
        (amp * torch.cos(phase)).sum(-1),
        -(amp * torch.sin(phase)).sum(-1),
    )


def _self_test() -> None:
    """Numerical-equivalence + speed sanity for fwd and full fwd+bwd."""
    import time

    if not HAVE_TRITON:
        print("Triton not installed — skip self-test")
        return
    if not torch.cuda.is_available():
        print("No CUDA — skip self-test")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)
    print(f"# device: {torch.cuda.get_device_name(0)}")

    def make_inputs(J, G_size):
        w = torch.randn(J, device=device, dtype=torch.float32)
        c = torch.randn(J, 3, device=device, dtype=torch.float32)
        L = torch.randn(J, 3, 3, device=device, dtype=torch.float32) * 0.1
        eye = torch.eye(3, device=device, dtype=torch.float32).expand(J, 3, 3) * 0.5
        K = torch.einsum("jab, jcb -> jac", L, L) + eye
        G = torch.randn(G_size, 3, device=device, dtype=torch.float32)
        return w, c, K, G

    print("\n=== forward equivalence + timing ===")
    for J, G_size in [(64, 256), (1024, 4096), (6400, 4096), (6400, 16384)]:
        w, c, K, G = make_inputs(J, G_size)
        ref = _reference_chunk(w, c, K, G)
        tri = triton_fourier_chunk_fwd(w, c, K, G)
        re_diff = (ref.real - tri.real).abs().max().item()
        im_diff = (ref.imag - tri.imag).abs().max().item()
        re_rel = re_diff / (ref.real.abs().max().item() + 1e-12)
        im_rel = im_diff / (ref.imag.abs().max().item() + 1e-12)

        # Warm up
        for _ in range(5):
            _reference_chunk(w, c, K, G)
            triton_fourier_chunk_fwd(w, c, K, G)
        torch.cuda.synchronize()

        ref_t, tri_t = [], []
        for _ in range(10):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _reference_chunk(w, c, K, G)
            torch.cuda.synchronize(); ref_t.append(time.perf_counter() - t0)
        for _ in range(10):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            triton_fourier_chunk_fwd(w, c, K, G)
            torch.cuda.synchronize(); tri_t.append(time.perf_counter() - t0)
        ref_med = sorted(ref_t)[5] * 1000
        tri_med = sorted(tri_t)[5] * 1000
        print(
            f"  J={J:>5} G={G_size:>6}: "
            f"ref={ref_med:7.2f}ms triton={tri_med:7.2f}ms "
            f"ratio={ref_med/tri_med:.2f}x "
            f"|Δre|={re_diff:.2e} (rel {re_rel:.2e}) "
            f"|Δim|={im_diff:.2e} (rel {im_rel:.2e})"
        )

    print("\n=== full fwd+bwd equivalence + timing ===")
    for J, G_size in [(64, 256), (1024, 4096), (6400, 4096), (6400, 16384)]:
        w0, c0, K0, G0 = make_inputs(J, G_size)
        w0.requires_grad_(True); c0.requires_grad_(True); K0.requires_grad_(True)
        out_ref = _reference_chunk(w0, c0, K0, G0)
        loss_ref = out_ref.real.square().sum() + out_ref.imag.square().sum()
        gw_ref, gc_ref, gK_ref = torch.autograd.grad(loss_ref, [w0, c0, K0])

        w1, c1, K1, G1 = make_inputs(J, G_size)
        w1.requires_grad_(True); c1.requires_grad_(True); K1.requires_grad_(True)
        out_tri = triton_fourier_chunk(w1, c1, K1, G1)
        loss_tri = out_tri.real.square().sum() + out_tri.imag.square().sum()
        gw_tri, gc_tri, gK_tri = torch.autograd.grad(loss_tri, [w1, c1, K1])

        # NOTE: we re-seed make_inputs each call, so w0/w1 are different.
        # Rebuild with shared inputs instead.
        w, c, K, G = make_inputs(J, G_size)
        w_a = w.clone().requires_grad_(True); c_a = c.clone().requires_grad_(True); K_a = K.clone().requires_grad_(True)
        w_b = w.clone().requires_grad_(True); c_b = c.clone().requires_grad_(True); K_b = K.clone().requires_grad_(True)
        out_a = _reference_chunk(w_a, c_a, K_a, G)
        loss_a = out_a.real.square().sum() + out_a.imag.square().sum()
        gw_a, gc_a, gK_a = torch.autograd.grad(loss_a, [w_a, c_a, K_a])
        out_b = triton_fourier_chunk(w_b, c_b, K_b, G)
        loss_b = out_b.real.square().sum() + out_b.imag.square().sum()
        gw_b, gc_b, gK_b = torch.autograd.grad(loss_b, [w_b, c_b, K_b])

        dw = (gw_a - gw_b).abs().max().item() / (gw_a.abs().max().item() + 1e-12)
        dc = (gc_a - gc_b).abs().max().item() / (gc_a.abs().max().item() + 1e-12)
        dK = (gK_a - gK_b).abs().max().item() / (gK_a.abs().max().item() + 1e-12)

        # timing of the full fwd+bwd
        def step_ref():
            wx = w.clone().detach().requires_grad_(True)
            cx = c.clone().detach().requires_grad_(True)
            Kx = K.clone().detach().requires_grad_(True)
            o = _reference_chunk(wx, cx, Kx, G)
            (o.real.square().sum() + o.imag.square().sum()).backward()

        def step_tri():
            wx = w.clone().detach().requires_grad_(True)
            cx = c.clone().detach().requires_grad_(True)
            Kx = K.clone().detach().requires_grad_(True)
            o = triton_fourier_chunk(wx, cx, Kx, G)
            (o.real.square().sum() + o.imag.square().sum()).backward()

        for _ in range(5):
            step_ref(); step_tri()
        torch.cuda.synchronize()
        rt, tt = [], []
        for _ in range(10):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step_ref()
            torch.cuda.synchronize(); rt.append(time.perf_counter() - t0)
        for _ in range(10):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step_tri()
            torch.cuda.synchronize(); tt.append(time.perf_counter() - t0)
        rmed = sorted(rt)[5] * 1000
        tmed = sorted(tt)[5] * 1000
        print(
            f"  J={J:>5} G={G_size:>6}: "
            f"ref={rmed:7.2f}ms triton={tmed:7.2f}ms "
            f"ratio={rmed/tmed:.2f}x "
            f"rel|dw|={dw:.2e} rel|dc|={dc:.2e} rel|dK|={dK:.2e}"
        )


if __name__ == "__main__":
    _self_test()
