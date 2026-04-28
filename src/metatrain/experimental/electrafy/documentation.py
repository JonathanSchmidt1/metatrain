"""
ELECTRAFY
=========

Periodic charge density prediction via Gaussian splatting + analytic Fourier
transform, using PET's CartesianTransformer as the GNN backbone.

Reference: Elsborg et al. "Global Plane Waves From Local Gaussians: Periodic
Charge Densities in a Blink" (2026), https://arxiv.org/abs/2501.09146

{{SECTION_DEFAULT_HYPERS}}
"""

from typing import List, Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.loss import LossSpecification


class ModelHypers(TypedDict):
    """Hyperparameters for the ELECTRAFY model."""

    cutoff: float = 6.0
    """Neighbor list cutoff radius in Angstrom."""

    cutoff_function: Literal["Cosine", "Bump"] = "Cosine"
    """Smoothing function applied at the cutoff."""

    cutoff_width: float = 0.5
    """Width of the cutoff smoothing region in Angstrom."""

    d_pet: int = 128
    """Dimension of the PET edge embeddings (also the attention d_model)."""

    d_node: int = 256
    """Dimension of the PET node embeddings."""

    d_feedforward: int = 256
    """FFN hidden dimension inside each CartesianTransformer block."""

    num_heads: int = 8
    """Number of multi-head attention heads per transformer layer."""

    num_attention_layers: int = 2
    """Number of transformer sub-layers per GNN layer."""

    num_gnn_layers: int = 2
    """Number of CartesianTransformer (GNN) layers."""

    normalization: Literal["RMSNorm", "LayerNorm"] = "RMSNorm"
    """Layer normalization variant."""

    activation: Literal["SiLU", "SwiGLU"] = "SiLU"
    """Activation function used in feed-forward blocks."""

    transformer_type: Literal["PreLN", "PostLN"] = "PreLN"
    """Pre- or post-layer-norm transformer variant."""

    attention_temperature: float = 1.0
    """Additional temperature scaling for attention scores."""

    gaussians_per_electron: int = 12
    """M — number of Gaussians per valence electron per atom.
    The paper uses M=120 for production; use smaller values for debugging."""

    gamma: float = 0.1
    """Global scale factor (Angstrom^2) for Gram-factored covariance matrices."""

    grid_shape: List[int] = [32, 32, 32]
    """Fixed FFT grid shape (N1, N2, N3) used for all structures.

    All predicted and reference density grids are evaluated / stored at this
    resolution.  Using a fixed shape enables standard TensorMap batching across
    structures with different cell geometries.

    Note: real-space resolution (Angstrom/voxel) therefore varies across
    structures with different cell sizes.  For production, the reference
    densities in the dataset should be resampled to this grid.
    """

    fourier_chunk_size: int = 4096
    """Number of G-vectors processed per chunk in the analytic Fourier transform.
    Reduce to save GPU memory at the cost of speed."""


class TrainerHypers(TypedDict):
    """Hyperparameters for training ELECTRAFY models."""

    distributed: bool = False
    """Whether to use distributed training."""

    distributed_port: int = 39591
    """Port for distributed communication."""

    batch_size: int = 4
    """Batch size (number of structures per step).

    Density grids can be memory-intensive; start small and increase as VRAM allows."""

    num_epochs: int = 1000
    """Total number of training epochs."""

    warmup_fraction: float = 0.01
    """Fraction of total steps used for linear LR warmup."""

    learning_rate: float = 3e-4
    """Peak learning rate (matches ELECTRAFY paper)."""

    weight_decay: Optional[float] = None
    """Weight decay for Adam (non-Muon) parameters."""

    log_interval: float = 1.0
    """Logging interval in epochs (fractional = sub-epoch logging)."""

    validation_interval: float = 1.0
    """Validation interval in epochs."""

    checkpoint_interval: float = 10.0
    """Checkpoint-saving interval in epochs."""

    grad_clip_norm: float = 10.0
    """Max global gradient norm. Tighter values (e.g. 1.0) stabilize early
    training; T7 used 1.0 with 5%% warmup vs T4's 10.0 with 1%%."""

    num_workers: Optional[int] = None
    """DataLoader workers per rank. ``None`` auto-detects via
    ``get_num_workers()`` (≈ ``min(cpu_count - 4, 8)``)."""

    compile: bool = False
    """If true, wrap the model in ``torch.compile(dynamic=True)`` after the
    DDP wrap. Validated by kuma jobs T4/T6/T7."""

    best_model_metric: str = "loss"
    """Metric key to track for best-model selection (validation side)."""

    loss: LossSpecification = {
        "type": "nmae",
        "weight": 1.0,
    }
    """Loss specification for the charge density target.

    The default ``"nmae"`` (Normalized Mean Absolute Error) matches the
    ELECTRAFY paper loss:  L = integral|pred - ref| / integral|ref|.
    """
