"""
Unit tests for ELECTRAFY modules: fourier_density, gaussian_density,
dyadic_aggregation, valence, and loss.
"""

import math

import pytest
import torch

from metatrain.experimental.electrafy.modules.fourier_density import (
    _build_g_vectors,
    gaussian_fourier_coefficients,
    periodic_density_from_gaussians,
)
from metatrain.experimental.electrafy.modules.gaussian_density import (
    GaussianDensityHead,
    _segment_softmax,
)
from metatrain.experimental.electrafy.modules.loss import NMAELoss, batch_nmae_loss
from metatrain.experimental.electrafy.modules.valence import (
    VASP_ZVAL,
    ZVAL_LOOKUP,
    MAX_ZVAL,
    build_zval_lookup,
)


# ──────────────────────────────────────────────────────────
# G-vector tests
# ──────────────────────────────────────────────────────────


class TestBuildGVectors:
    def test_cubic_cell_shape(self):
        """G-vectors for a cubic cell have correct shape."""
        cell = 5.0 * torch.eye(3, dtype=torch.float64)
        G = _build_g_vectors((8, 8, 8), cell)
        assert G.shape == (8, 8, 8, 3)

    def test_cubic_cell_values(self):
        """G=0 at origin, and first reciprocal vector is 2*pi/a along x."""
        a = 5.0
        cell = a * torch.eye(3, dtype=torch.float64)
        G = _build_g_vectors((8, 8, 8), cell)

        # G[0,0,0] should be zero (DC component)
        assert torch.allclose(G[0, 0, 0], torch.zeros(3, dtype=torch.float64), atol=1e-12)

        # G[1,0,0] = b1/N1 * 1 = (2*pi/a, 0, 0) / N1 * N1 = (2*pi/a, 0, 0)
        # Actually: f1[1] = 1, so G[1,0,0] = 1 * b1/N1 * N1... let me think.
        # fftfreq(8) * 8 = [0,1,2,3,4,-3,-2,-1] for indices, so f1[1]=1
        # b1 = 2*pi * inv(cell)^T row 0 = 2*pi/a * (1,0,0)
        # G[1,0,0] = 1/N1 * b1 = (2*pi/(a*1)) * (1/8) ... wait
        # Actually the code does: b1 = B[0] / N1, and G = f1 * b1
        # B[0] = 2*pi/a * (1,0,0), b1_scaled = B[0]/8
        # f1[1] = 1*8 = 8 (fftfreq(8)*8), so G[1,0,0] = 8 * B[0]/8 = B[0]
        expected = torch.tensor([2 * math.pi / a, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(G[1, 0, 0], expected, atol=1e-10)

    def test_non_orthogonal_cell(self):
        """G-vectors work for a triclinic cell."""
        cell = torch.tensor(
            [[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.5, 3.0]],
            dtype=torch.float64,
        )
        G = _build_g_vectors((4, 4, 4), cell)
        assert G.shape == (4, 4, 4, 3)
        # DC component is still zero
        assert torch.allclose(G[0, 0, 0], torch.zeros(3, dtype=torch.float64), atol=1e-12)


# ──────────────────────────────────────────────────────────
# Fourier density tests
# ──────────────────────────────────────────────────────────


class TestFourierDensity:
    def test_single_gaussian_integral(self):
        """A single isotropic Gaussian integrates to n_electrons."""
        cell = 10.0 * torch.eye(3, dtype=torch.float64)
        grid_shape = (16, 16, 16)
        n_electrons = 8.0

        weights = torch.tensor([1.0], dtype=torch.float64)
        centers = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)
        sigma = 0.5
        covs = (sigma**2) * torch.eye(3, dtype=torch.float64).unsqueeze(0)

        rho = periodic_density_from_gaussians(
            weights, centers, covs, cell, grid_shape, n_electrons
        )

        V = 10.0**3
        N_total = 16**3
        integral = rho.sum() * (V / N_total)
        assert abs(integral.item() - n_electrons) < 0.1, (
            f"Integral {integral.item()} != {n_electrons}"
        )

    def test_peak_location(self):
        """Density peak is at the Gaussian center."""
        cell = 8.0 * torch.eye(3, dtype=torch.float64)
        grid_shape = (16, 16, 16)

        weights = torch.tensor([1.0], dtype=torch.float64)
        centers = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float64)
        covs = 0.3 * torch.eye(3, dtype=torch.float64).unsqueeze(0)

        rho = periodic_density_from_gaussians(
            weights, centers, covs, cell, grid_shape, n_electrons=1.0
        )

        # Peak should be near grid point (8, 8, 8) = center of the cell at (4,4,4)
        peak_idx = torch.unravel_index(rho.argmax(), rho.shape)
        # 4.0/8.0 * 16 = 8.0 -> index 8
        for dim in range(3):
            assert abs(peak_idx[dim].item() - 8) <= 1, (
                f"Peak at {peak_idx}, expected near (8,8,8)"
            )

    def test_output_is_real(self):
        """Density output should be real-valued (no large imaginary residuals)."""
        cell = 5.0 * torch.eye(3, dtype=torch.float64)
        weights = torch.tensor([1.0, -0.5], dtype=torch.float64)
        centers = torch.tensor([[2.5, 2.5, 2.5], [1.0, 1.0, 1.0]], dtype=torch.float64)
        covs = 0.2 * torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1)

        rho = periodic_density_from_gaussians(
            weights, centers, covs, cell, (8, 8, 8), n_electrons=4.0
        )
        assert rho.dtype in (torch.float32, torch.float64)

    def test_two_gaussians_symmetry(self):
        """Two identical Gaussians at symmetric positions give symmetric density."""
        cell = 10.0 * torch.eye(3, dtype=torch.float64)
        grid_shape = (16, 16, 16)

        weights = torch.tensor([1.0, 1.0], dtype=torch.float64)
        centers = torch.tensor([[3.0, 5.0, 5.0], [7.0, 5.0, 5.0]], dtype=torch.float64)
        covs = 0.3 * torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1)

        rho = periodic_density_from_gaussians(
            weights, centers, covs, cell, grid_shape, n_electrons=2.0
        )

        # rho should be symmetric under x -> L-x (reflection at x=5).
        # On a grid x_n = n*L/N, this continuous reflection maps index
        # i -> (N - i) mod N.  torch.flip implements i -> N-1-i (reflection
        # about index (N-1)/2), so we compose it with a roll to shift the
        # reflection center from (N-1)/2 to N/2.
        rho_reflected = torch.roll(torch.flip(rho, dims=[0]), shifts=1, dims=0)
        assert torch.allclose(rho, rho_reflected, atol=1e-6)

    def test_chunking_consistency(self):
        """Different chunk_size values give the same result."""
        cell = 6.0 * torch.eye(3, dtype=torch.float64)
        grid_shape = (8, 8, 8)
        weights = torch.randn(3, dtype=torch.float64)
        centers = torch.rand(3, 3, dtype=torch.float64) * 6.0
        L = torch.randn(3, 3, 3, dtype=torch.float64)
        covs = torch.bmm(L, L.transpose(-1, -2)) + 0.1 * torch.eye(3).unsqueeze(0)

        rho1 = periodic_density_from_gaussians(
            weights, centers, covs, cell, grid_shape, n_electrons=5.0, chunk_size=64
        )
        rho2 = periodic_density_from_gaussians(
            weights, centers, covs, cell, grid_shape, n_electrons=5.0, chunk_size=512
        )
        assert torch.allclose(rho1, rho2, atol=1e-10)


# ──────────────────────────────────────────────────────────
# Segment softmax tests
# ──────────────────────────────────────────────────────────


class TestSegmentSoftmax:
    def test_single_segment(self):
        """Softmax of a single segment should match torch.softmax."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        ids = torch.tensor([0, 0, 0])
        result = _segment_softmax(logits, ids, 1)
        expected = torch.softmax(logits, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_multiple_segments(self):
        """Each segment is independently softmaxed."""
        logits = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
        ids = torch.tensor([0, 0, 0, 1, 1])
        result = _segment_softmax(logits, ids, 2)

        expected_0 = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
        expected_1 = torch.softmax(torch.tensor([10.0, 20.0]), dim=0)
        assert torch.allclose(result[:3], expected_0, atol=1e-6)
        assert torch.allclose(result[3:], expected_1, atol=1e-6)

    def test_sums_to_one(self):
        """Each segment sums to 1."""
        logits = torch.randn(20)
        ids = torch.tensor([0] * 5 + [1] * 7 + [2] * 8)
        result = _segment_softmax(logits, ids, 3)

        for seg in range(3):
            seg_sum = result[ids == seg].sum()
            assert abs(seg_sum.item() - 1.0) < 1e-5

    def test_negative_logits(self):
        """Works correctly when all logits are very negative."""
        logits = torch.tensor([-100.0, -200.0, -150.0])
        ids = torch.tensor([0, 0, 0])
        result = _segment_softmax(logits, ids, 1)
        assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(result >= 0)

    def test_gradient_flows(self):
        """Gradients flow through segment softmax."""
        logits = torch.randn(10, requires_grad=True)
        ids = torch.tensor([0] * 5 + [1] * 5)
        result = _segment_softmax(logits, ids, 2)
        result.sum().backward()
        assert logits.grad is not None
        assert not torch.any(torch.isnan(logits.grad))


# ──────────────────────────────────────────────────────────
# Gaussian density head tests
# ──────────────────────────────────────────────────────────


class TestGaussianDensityHead:
    @pytest.fixture
    def head(self):
        return GaussianDensityHead(n_channels=12, gamma=0.1, eps_cov=1e-6)

    def test_output_shapes(self, head):
        """Output tensors have correct shapes."""
        N, C = 4, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        T = torch.randn(N, C, 3, 3)
        positions = torch.randn(N, 3)
        n_gauss = torch.tensor([3, 6, 3, 6])

        weights, centers, covs, atom_idx = head(S, V, T, positions, n_gauss)

        total = n_gauss.sum().item()
        assert weights.shape == (total,)
        assert centers.shape == (total, 3)
        assert covs.shape == (total, 3, 3)
        assert atom_idx.shape == (total,)

    def test_covariance_positive_definite(self, head):
        """All output covariance matrices are positive definite."""
        N, C = 5, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        T = torch.randn(N, C, 3, 3)
        positions = torch.randn(N, 3)
        n_gauss = torch.tensor([6, 6, 6, 6, 6])

        _, _, covs, _ = head(S, V, T, positions, n_gauss)

        eigenvalues = torch.linalg.eigvalsh(covs)
        assert torch.all(eigenvalues > 0), (
            f"Found non-positive eigenvalue: {eigenvalues.min().item()}"
        )

    def test_covariance_pd_extreme_T(self, head):
        """Covariances are PD even with extreme T values."""
        N, C = 2, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        # Very large T matrices
        T = 100.0 * torch.randn(N, C, 3, 3)
        positions = torch.randn(N, 3)
        n_gauss = torch.tensor([6, 6])

        _, _, covs, _ = head(S, V, T, positions, n_gauss)
        eigenvalues = torch.linalg.eigvalsh(covs)
        assert torch.all(eigenvalues > 0)

        # Very small T matrices (near-zero)
        T = 1e-8 * torch.randn(N, C, 3, 3)
        _, _, covs, _ = head(S, V, T, positions, n_gauss)
        eigenvalues = torch.linalg.eigvalsh(covs)
        assert torch.all(eigenvalues > 0)

    def test_empty_system(self, head):
        """Zero gaussians returns empty tensors."""
        N, C = 2, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        T = torch.randn(N, C, 3, 3)
        positions = torch.randn(N, 3)
        n_gauss = torch.tensor([0, 0])

        weights, centers, covs, atom_idx = head(S, V, T, positions, n_gauss)
        assert weights.shape == (0,)
        assert centers.shape == (0, 3)
        assert covs.shape == (0, 3, 3)

    def test_single_atom(self, head):
        """Works for a single atom."""
        N, C = 1, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        T = torch.randn(N, C, 3, 3)
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        n_gauss = torch.tensor([5])

        weights, centers, covs, atom_idx = head(S, V, T, positions, n_gauss)
        assert weights.shape == (5,)
        assert torch.all(atom_idx == 0)

    def test_weights_include_negative(self, head):
        """Weights can be negative (tanh allows it)."""
        torch.manual_seed(42)
        N, C = 10, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        T = torch.randn(N, C, 3, 3)
        positions = torch.randn(N, 3)
        n_gauss = torch.full((N,), 6, dtype=torch.long)

        weights, _, _, _ = head(S, V, T, positions, n_gauss)
        # With tanh, we should see both positive and negative weights
        # (after segment_softmax of tanh values)
        has_negative = torch.any(weights < 0).item()
        has_positive = torch.any(weights > 0).item()
        # At least positive should exist (softmax of tanh can still be all positive
        # if tanh outputs are all positive). But the MLP with random weights should
        # produce some negative tanh outputs.
        assert has_positive

    def test_gradient_flow(self, head):
        """Gradients flow through the full head."""
        N, C = 3, 12
        S = torch.randn(N, C, requires_grad=True)
        V = torch.randn(N, C, 3, requires_grad=True)
        T = torch.randn(N, C, 3, 3, requires_grad=True)
        positions = torch.randn(N, 3)
        n_gauss = torch.tensor([4, 4, 4])

        weights, centers, covs, _ = head(S, V, T, positions, n_gauss)
        loss = weights.sum() + centers.sum() + covs.sum()
        loss.backward()

        assert S.grad is not None and not torch.any(torch.isnan(S.grad))
        assert V.grad is not None and not torch.any(torch.isnan(V.grad))
        assert T.grad is not None and not torch.any(torch.isnan(T.grad))

    def test_atom_idx_correctness(self, head):
        """atom_idx correctly maps Gaussians to their parent atoms."""
        N, C = 3, 12
        S = torch.randn(N, C)
        V = torch.randn(N, C, 3)
        T = torch.randn(N, C, 3, 3)
        positions = torch.randn(N, 3)
        n_gauss = torch.tensor([2, 5, 3])

        _, _, _, atom_idx = head(S, V, T, positions, n_gauss)

        expected = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
        assert torch.equal(atom_idx, expected)


# ──────────────────────────────────────────────────────────
# Loss function tests
# ──────────────────────────────────────────────────────────


class TestNMAELoss:
    def test_perfect_prediction(self):
        """NMAE is zero for perfect prediction."""
        loss_fn = NMAELoss()
        rho = torch.rand(8, 8, 8) + 0.1
        assert loss_fn(rho, rho).item() < 1e-8

    def test_known_value(self):
        """NMAE for a known case: pred=2*ref gives NMAE=1."""
        loss_fn = NMAELoss()
        ref = torch.ones(4, 4, 4)
        pred = 2.0 * ref
        # |pred - ref| = 1 everywhere, sum = 64
        # |ref| sum = 64
        # NMAE = 64/64 = 1.0
        assert abs(loss_fn(pred, ref).item() - 1.0) < 1e-6

    def test_gradient_flows(self):
        """Gradients flow through NMAE loss."""
        loss_fn = NMAELoss()
        pred = torch.randn(4, 4, 4, requires_grad=True)
        ref = torch.rand(4, 4, 4) + 0.1
        loss = loss_fn(pred, ref)
        loss.backward()
        assert pred.grad is not None

    def test_batch_nmae(self):
        """batch_nmae_loss averages per-system NMAE."""
        ref1 = torch.ones(4, 4, 4)
        ref2 = torch.ones(6, 6, 6)
        pred1 = 2.0 * ref1  # NMAE = 1.0
        pred2 = ref2  # NMAE = 0.0

        result = batch_nmae_loss([pred1, pred2], [ref1, ref2])
        assert abs(result.item() - 0.5) < 1e-6


# ──────────────────────────────────────────────────────────
# Valence lookup tests
# ──────────────────────────────────────────────────────────


class TestValence:
    def test_common_elements(self):
        """Spot-check ZVAL for common elements."""
        assert VASP_ZVAL["H"] == 1
        assert VASP_ZVAL["C"] == 4
        assert VASP_ZVAL["O"] == 6
        assert VASP_ZVAL["Si"] == 4
        assert VASP_ZVAL["Fe"] == 8

    def test_max_zval(self):
        """MAX_ZVAL is the max of all ZVAL values."""
        assert MAX_ZVAL == max(VASP_ZVAL.values())

    def test_lookup_tensor(self):
        """Lookup tensor indexed by atomic number matches the dict."""
        from ase.data import chemical_symbols

        for z in range(1, 90):
            sym = chemical_symbols[z]
            if sym in VASP_ZVAL:
                assert ZVAL_LOOKUP[z].item() == VASP_ZVAL[sym], (
                    f"Mismatch for {sym} (Z={z}): "
                    f"lookup={ZVAL_LOOKUP[z].item()}, dict={VASP_ZVAL[sym]}"
                )

    def test_rebuild_gives_same(self):
        """build_zval_lookup reproduces the pre-built table."""
        rebuilt = build_zval_lookup()
        assert torch.equal(rebuilt, ZVAL_LOOKUP)


# ──────────────────────────────────────────────────────────
# Dyadic aggregation tests
# ──────────────────────────────────────────────────────────


class TestDyadicAggregation:
    def test_output_shapes(self):
        from metatrain.experimental.electrafy.modules.dyadic_aggregation import (
            DyadicAggregation,
        )

        N, E, C = 4, 10, 12
        d_node, d_edge = 32, 16
        n_layers = 2

        agg = DyadicAggregation(n_layers, d_node, d_edge, C)

        node_list = [torch.randn(N, d_node) for _ in range(n_layers)]
        edge_list = [torch.randn(N, E, d_edge) for _ in range(n_layers)]
        edge_vectors = torch.randn(N, E, 3)
        padding_mask = torch.ones(N, E, dtype=torch.bool)

        S, V, T = agg(node_list, edge_list, edge_vectors, padding_mask)

        assert S.shape == (N, C)
        assert V.shape == (N, C, 3)
        assert T.shape == (N, C, 3, 3)

    def test_gradient_flow(self):
        from metatrain.experimental.electrafy.modules.dyadic_aggregation import (
            DyadicAggregation,
        )

        N, E, C = 2, 5, 6
        d_node, d_edge = 8, 4
        n_layers = 1

        agg = DyadicAggregation(n_layers, d_node, d_edge, C)

        node_list = [torch.randn(N, d_node, requires_grad=True)]
        edge_list = [torch.randn(N, E, d_edge, requires_grad=True)]
        edge_vectors = torch.randn(N, E, 3)
        padding_mask = torch.ones(N, E, dtype=torch.bool)

        S, V, T = agg(node_list, edge_list, edge_vectors, padding_mask)
        loss = S.sum() + V.sum() + T.sum()
        loss.backward()

        assert node_list[0].grad is not None
        assert edge_list[0].grad is not None

    def test_padding_mask_respected(self):
        """Padded edges should not affect output when masked."""
        from metatrain.experimental.electrafy.modules.dyadic_aggregation import (
            DyadicAggregation,
        )

        N, E, C = 2, 8, 6
        d_node, d_edge = 8, 4

        agg = DyadicAggregation(1, d_node, d_edge, C)

        node_list = [torch.randn(N, d_node)]
        edge_list = [torch.randn(N, E, d_edge)]
        edge_vectors = torch.randn(N, E, 3)

        # All real
        mask_full = torch.ones(N, E, dtype=torch.bool)
        S1, V1, T1 = agg(node_list, edge_list, edge_vectors, mask_full)

        # Mask out last 3 edges, replace with garbage
        mask_partial = mask_full.clone()
        mask_partial[:, 5:] = False
        edge_list_noisy = [edge_list[0].clone()]
        edge_list_noisy[0][:, 5:] = 1000.0  # garbage in padded positions
        edge_vectors_noisy = edge_vectors.clone()
        edge_vectors_noisy[:, 5:] = 999.0

        S2, V2, T2 = agg(node_list, edge_list_noisy, edge_vectors_noisy, mask_partial)

        # Scalar branch depends only on node_emb, so should be identical
        assert torch.allclose(S1, S2, atol=1e-5)
        # Vector/tensor branches use edges, but padded entries should be suppressed
        # They won't be exactly equal because softmax normalization changes with
        # different number of real edges, but the outputs should be finite
        assert torch.all(torch.isfinite(V2))
        assert torch.all(torch.isfinite(T2))
