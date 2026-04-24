"""
Dyadic neighbor aggregation layer (ELECTRAFY Appendix B).

Converts PET's per-layer node/edge features into three geometric channels:
  - S (N_atoms, C)       : scalar features (for Gaussian weights)
  - V (N_atoms, C, 3)   : vector features (for Gaussian displacements)
  - T (N_atoms, C, 3, 3): symmetric tensor features (for Gaussian covariances)

Architecture
------------
Scalar branch
  S_{i,c} = Linear(d_node -> C)(node_emb_i)

Vector branch
  alpha_{i,e,c} = softmax_e( Linear(d_edge -> C)(edge_emb_{i,e}) )
  n_hat_{i,e}   = edge_vec_{i,e} / ||edge_vec_{i,e}||          (unit vector)
  v_raw_{i,c}   = sum_e alpha_{i,e,c} * n_hat_{i,e}            (N,C,3)
  m_{i,c}       = Linear(d_node -> C)(node_emb_i)              (magnitude)
  V_{i,c}       = (v_raw_{i,c} / ||v_raw_{i,c}||) * m_{i,c}

Tensor branch
  Q_{i,e}       = n_hat n_hat^T - (1/3) I                      (traceless dyad, N,E,3,3)
  beta_{i,e,c}  = softmax_e( Linear(d_edge -> C)(edge_emb_{i,e}) )
  T_aniso_{i,c} = sum_e beta_{i,e,c} * Q_{i,e}                 (N,C,3,3)
  kappa_{i,c}   = Linear(d_node -> C)(node_emb_i)
  T_{i,c}       = T_aniso_{i,c} + kappa_{i,c} * I

Cross-layer aggregation (ELECTRAFY Eq. 25-26)
  A^{[l]}_{i,c}   = Linear(d_node -> C)(node_emb^{[l]}_i)     (logits)
  alpha^{[l]}_{i,c} = softmax_l( A^{[l]}_{i,c} )
  S_final         = sum_l alpha^{[l]} * S^{[l]}
  (same for V and T)
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

        # Scalar branch
        self.scalar_proj = nn.Linear(d_node, n_channels)

        # Vector branch: per-edge attention weights + per-atom magnitude
        self.vec_attn_proj = nn.Linear(d_edge, n_channels)
        self.vec_mag_proj = nn.Linear(d_node, n_channels)

        # Tensor branch: per-edge attention weights + per-atom isotropic term
        self.tens_attn_proj = nn.Linear(d_edge, n_channels)
        self.tens_iso_proj = nn.Linear(d_node, n_channels)

        # Cross-layer logits (used by DyadicAggregation to do softmax over layers)
        self.layer_logit_proj = nn.Linear(d_node, n_channels)

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

        # ---- Scalar branch ----
        S = self.scalar_proj(node_emb)  # (N, C)

        # ---- Shared geometry ----
        edge_norms = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp(
            min=self.eps
        )  # (N, E, 1)
        n_hat = edge_vectors / edge_norms  # (N, E, 3), unit edge directions

        # Softmax mask: set padding edges to -inf before softmax
        pad_mask_float = padding_mask.float()  # (N, E), 1=real, 0=pad
        neg_inf_mask = (1.0 - pad_mask_float) * (-1e9)  # (N, E)

        # ---- Vector branch ----
        # Attention weights: (N, E, C)
        alpha_raw = self.vec_attn_proj(edge_emb)  # (N, E, C)
        alpha_raw = alpha_raw + neg_inf_mask[:, :, None]
        alpha = torch.softmax(alpha_raw, dim=1)  # (N, E, C), sums to 1 over E

        # Aggregate: v_raw = sum_e alpha_{e,c} * n_hat_e
        v_raw = torch.einsum("nec, neo -> nco", alpha, n_hat)  # (N, C, 3)

        # Normalize direction and scale by learned magnitude
        v_norm = torch.linalg.vector_norm(v_raw, dim=-1, keepdim=True).clamp(
            min=self.eps
        )  # (N, C, 1)
        m = self.vec_mag_proj(node_emb)  # (N, C)
        V = (v_raw / v_norm) * m[:, :, None]  # (N, C, 3)

        # ---- Tensor branch ----
        # Traceless dyad: Q_e = n_hat n_hat^T - (1/3) I
        I3 = torch.eye(3, device=device, dtype=node_emb.dtype)
        Q = (
            torch.einsum("neo, nep -> neop", n_hat, n_hat)
            - I3[None, None, :, :] / 3.0
        )  # (N, E, 3, 3)

        # Tensor attention weights: (N, E, C)
        beta_raw = self.tens_attn_proj(edge_emb)  # (N, E, C)
        beta_raw = beta_raw + neg_inf_mask[:, :, None]
        beta = torch.softmax(beta_raw, dim=1)  # (N, E, C)

        # Aggregate anisotropic part: T_aniso = sum_e beta_{e,c} * Q_e
        T_aniso = torch.einsum("nec, neop -> ncop", beta, Q)  # (N, C, 3, 3)

        # Add learned isotropic term: kappa * I
        kappa = self.tens_iso_proj(node_emb)  # (N, C)
        T = T_aniso + kappa[:, :, None, None] * I3[None, None, :, :]  # (N, C, 3, 3)

        # ---- Cross-layer logits ----
        logits = self.layer_logit_proj(node_emb)  # (N, C)

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
