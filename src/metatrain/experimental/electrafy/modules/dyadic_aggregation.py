"""
Dyadic neighbor aggregation layer (ELECTRAFY Appendix B.1, Eqs 19-26).

Converts PET's per-layer node/edge features into three geometric channels:
  - S (N_atoms, C)       : scalar features (for Gaussian weights)
  - V (N_atoms, C, 3)   : vector features (for Gaussian displacements)
  - T (N_atoms, C, 3, 3): symmetric tensor features (for Gaussian covariances)

Architecture
------------
Scalar branch (Eq 19)
  S_{i,c} = MLP(node_emb_i)

Vector branch (Eqs 20-22) — with learned carrier vector
  d_bar^v_{i,e} = Linear_v(edge_emb_{i,e})              (free per-edge vector)
  a^v_{i,e}     = sigmoid(Linear_g_v(edge_emb_{i,e}))   (per-edge gate)
  n^v_{i,e}     = (1 - a^v) n_{i,e} + a^v d_bar^v_{i,e}  (Eq 20)
  n_hat^v       = n^v / ||n^v||                          (Eq 21)
  alpha_{i,c,e} = softmax_e( Linear_a_v(edge_emb)_{i,e,c} )
  v_{i,c}       = sum_e alpha_{i,c,e} n_hat^v_{i,e}      (Eq 22a)
  m_{i,c}       = softplus(Linear_m(node_emb_i))         (positive magnitude)
  V_{i,c}       = (v_{i,c} / ||v_{i,c}||) * m_{i,c}      (Eq 22b)

Tensor branch (Eqs 23-24) — with its own carrier vector
  d_bar^t_{i,e} = Linear_t(edge_emb_{i,e})
  a^t_{i,e}     = sigmoid(Linear_g_t(edge_emb_{i,e}))
  n^t_{i,e}     = (1 - a^t) n_{i,e} + a^t d_bar^t_{i,e}
  n_hat^t       = n^t / ||n^t||
  Q_{i,e}       = n_hat^t n_hat^t^T - (1/3) I            (traceless dyad)
  beta_{i,c,e}  = softmax_e( Linear_a_t(edge_emb)_{i,e,c} )
  T_aniso_{i,c} = sum_e beta_{i,c,e} Q_{i,e}             (Eq 24a)
  kappa_{i,c}   = Linear_iso(node_emb_i)                 (scalar trace)
  T_{i,c}       = T_aniso_{i,c} + kappa_{i,c} I          (Eq 24b)

Cross-layer aggregation (Eqs 25-27)
  A^[l]_{i,c}     = Linear(d_node -> C)(node_emb^[l]_i)     (logits)
  alpha^[l]_{i,c} = softmax_l( A^[l]_{i,c} )
  S_final = sum_l alpha^[l] * S^[l]      (same for V and T)
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class DyadicAggregationLayer(nn.Module):
    """
    Single-layer dyadic aggregation: maps PET node/edge features -> (S, V, T).

    :param d_node: Dimension of PET node embeddings.
    :param d_edge: Dimension of PET edge embeddings.
    :param n_channels: Number of output channels C (= M * max_zval).
    :param eps: Small constant for numerical stability in vector normalization.
    """

    def __init__(
        self,
        d_node: int,
        d_edge: int,
        n_channels: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps

        # Scalar branch (Eq 19)
        self.scalar_proj = nn.Linear(d_node, n_channels)

        # Vector branch (Eqs 20-22): own carrier-vector construction
        self.vec_carrier_proj = nn.Linear(d_edge, 3)        # d̄^v
        self.vec_gate_proj = nn.Linear(d_edge, 1)           # a^v (pre-sigmoid)
        self.vec_attn_proj = nn.Linear(d_edge, n_channels)  # alpha logits
        self.vec_mag_proj = nn.Linear(d_node, n_channels)   # m (pre-softplus)

        # Tensor branch (Eqs 23-24): SEPARATE carrier-vector construction
        self.tens_carrier_proj = nn.Linear(d_edge, 3)        # d̄^t
        self.tens_gate_proj = nn.Linear(d_edge, 1)           # a^t (pre-sigmoid)
        self.tens_attn_proj = nn.Linear(d_edge, n_channels)  # beta logits
        self.tens_iso_proj = nn.Linear(d_node, n_channels)   # kappa (trace)

        # Cross-layer logits (used by DyadicAggregation to softmax over layers)
        self.layer_logit_proj = nn.Linear(d_node, n_channels)

        # Initialize gates so the carrier defaults to ~ pure edge direction
        # (a ≈ 0 at start). This keeps early-training behavior close to the
        # bare-edge implementation while giving the gate room to ramp up.
        with torch.no_grad():
            self.vec_gate_proj.weight.zero_()
            self.vec_gate_proj.bias.fill_(-2.0)   # sigmoid(-2) ≈ 0.12
            self.tens_gate_proj.weight.zero_()
            self.tens_gate_proj.bias.fill_(-2.0)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param node_emb: (N, d_node) node feature tensor.
        :param edge_emb: (N, E, d_edge) edge feature tensor (NEF format).
        :param edge_vectors: (N, E, 3) Cartesian edge vectors (not normalized).
        :param padding_mask: (N, E) bool tensor, True where neighbor is real.
        :return: Tuple (S, V, T, logits):
            - S: (N, C) scalar features
            - V: (N, C, 3) vector features
            - T: (N, C, 3, 3) tensor features
            - logits: (N, C) cross-layer aggregation logits
        """
        N, E, _ = edge_emb.shape
        C = self.n_channels
        device = node_emb.device
        dtype = node_emb.dtype

        # ---- Scalar branch (Eq 19) ----
        S = self.scalar_proj(node_emb)  # (N, C)

        # ---- Shared geometry ----
        edge_norms = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp(
            min=self.eps
        )  # (N, E, 1)
        n_hat = edge_vectors / edge_norms  # (N, E, 3) unit edge direction

        pad_mask_float = padding_mask.to(dtype=dtype)  # (N, E)
        neg_inf_mask = (1.0 - pad_mask_float) * (-1e9)  # (N, E)
        # Zero out padded edges so they contribute nothing to carrier sums.
        pad3 = pad_mask_float[:, :, None]  # (N, E, 1)

        # ---- Vector-branch carrier (Eqs 20-21) ----
        d_bar_v = self.vec_carrier_proj(edge_emb)            # (N, E, 3)
        a_v = torch.sigmoid(self.vec_gate_proj(edge_emb))    # (N, E, 1)
        n_carrier_v = ((1.0 - a_v) * n_hat + a_v * d_bar_v) * pad3
        n_carrier_v_norm = torch.linalg.vector_norm(
            n_carrier_v, dim=-1, keepdim=True
        ).clamp(min=self.eps)
        n_hat_v = n_carrier_v / n_carrier_v_norm             # (N, E, 3)

        # ---- Vector branch aggregation (Eq 22) ----
        alpha_raw = self.vec_attn_proj(edge_emb)             # (N, E, C)
        alpha_raw = alpha_raw + neg_inf_mask[:, :, None]
        alpha = torch.softmax(alpha_raw, dim=1)              # (N, E, C)

        v_raw = torch.einsum("nec, neo -> nco", alpha, n_hat_v)  # (N, C, 3)
        v_norm = torch.linalg.vector_norm(v_raw, dim=-1, keepdim=True).clamp(
            min=self.eps
        )  # (N, C, 1)
        # Magnitude must be positive (paper: m ∈ R_{>0}).
        m = torch.nn.functional.softplus(self.vec_mag_proj(node_emb))  # (N, C)
        V = (v_raw / v_norm) * m[:, :, None]                  # (N, C, 3)

        # ---- Tensor-branch carrier (Eq 23 setup) ----
        d_bar_t = self.tens_carrier_proj(edge_emb)            # (N, E, 3)
        a_t = torch.sigmoid(self.tens_gate_proj(edge_emb))    # (N, E, 1)
        n_carrier_t = ((1.0 - a_t) * n_hat + a_t * d_bar_t) * pad3
        n_carrier_t_norm = torch.linalg.vector_norm(
            n_carrier_t, dim=-1, keepdim=True
        ).clamp(min=self.eps)
        n_hat_t = n_carrier_t / n_carrier_t_norm              # (N, E, 3)

        # ---- Tensor branch aggregation (Eq 23-24) ----
        I3 = torch.eye(3, device=device, dtype=dtype)
        Q = (
            torch.einsum("neo, nep -> neop", n_hat_t, n_hat_t)
            - I3[None, None, :, :] / 3.0
        )  # (N, E, 3, 3)

        beta_raw = self.tens_attn_proj(edge_emb)              # (N, E, C)
        beta_raw = beta_raw + neg_inf_mask[:, :, None]
        beta = torch.softmax(beta_raw, dim=1)                 # (N, E, C)
        T_aniso = torch.einsum("nec, neop -> ncop", beta, Q)  # (N, C, 3, 3)

        kappa = self.tens_iso_proj(node_emb)                  # (N, C)
        T = T_aniso + kappa[:, :, None, None] * I3[None, None, :, :]

        # ---- Cross-layer logits (Eq 25) ----
        logits = self.layer_logit_proj(node_emb)              # (N, C)

        return S, V, T, logits


class DyadicAggregation(nn.Module):
    """
    Multi-layer dyadic aggregation with cross-layer softmax aggregation.

    One DyadicAggregationLayer per GNN layer.  The final (S, V, T) are
    a per-channel softmax-weighted sum across layers (ELECTRAFY Eq. 25-26).

    :param n_gnn_layers: Number of GNN layers (= len(node_features_list)).
    :param d_node: Node embedding dimension from PET.
    :param d_edge: Edge embedding dimension from PET.
    :param n_channels: Output channels C = M * max_zval.
    :param eps: Numerical stability constant.
    """

    def __init__(
        self,
        n_gnn_layers: int,
        d_node: int,
        d_edge: int,
        n_channels: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DyadicAggregationLayer(d_node, d_edge, n_channels, eps)
                for _ in range(n_gnn_layers)
            ]
        )

    def forward(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param node_features_list: List of (N, d_node) tensors, one per GNN layer.
        :param edge_features_list: List of (N, E, d_edge) tensors, one per GNN layer.
        :param edge_vectors: (N, E, 3) Cartesian edge vectors (shared across layers).
        :param padding_mask: (N, E) bool mask for real neighbors.
        :return: (S, V, T) after cross-layer aggregation:
            - S: (N, C)
            - V: (N, C, 3)
            - T: (N, C, 3, 3)
        """
        S_list: List[torch.Tensor] = []
        V_list: List[torch.Tensor] = []
        T_list: List[torch.Tensor] = []
        logit_list: List[torch.Tensor] = []

        for layer, node_emb, edge_emb in zip(
            self.layers, node_features_list, edge_features_list
        ):
            S_l, V_l, T_l, logits_l = layer(node_emb, edge_emb, edge_vectors, padding_mask)
            S_list.append(S_l)
            V_list.append(V_l)
            T_list.append(T_l)
            logit_list.append(logits_l)

        # Cross-layer aggregation: alpha^[l] = softmax over l dim
        # logits: (L, N, C) -> softmax over L
        logits = torch.stack(logit_list, dim=0)  # (L, N, C)
        alpha = torch.softmax(logits, dim=0)  # (L, N, C)

        S_stack = torch.stack(S_list, dim=0)  # (L, N, C)
        V_stack = torch.stack(V_list, dim=0)  # (L, N, C, 3)
        T_stack = torch.stack(T_list, dim=0)  # (L, N, C, 3, 3)

        S = (alpha * S_stack).sum(0)                    # (N, C)
        V = (alpha[:, :, :, None] * V_stack).sum(0)     # (N, C, 3)
        T = (alpha[:, :, :, None, None] * T_stack).sum(0)  # (N, C, 3, 3)

        return S, V, T
