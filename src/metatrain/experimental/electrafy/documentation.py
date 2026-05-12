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
    """**Fallback** FFT grid shape (N1, N2, N3) used for any forward pass
    where no per-batch override has been installed via
    :py:meth:`ELECTRAFY.set_override_grid_shapes`.

    Native-grid is the standard training path: the metatrain trainer reads
    ``extra_data["grid_shape"]`` (emitted by
    :class:`~metatrain.experimental.electrafy.modules.cache_dataset.CachedChgcarDataset`
    as a TensorMap with one ``(N1, N2, N3)`` row per system) and calls
    ``set_override_grid_shapes`` before every forward, then
    ``clear_override_grid_shapes`` after. Under that flow this hyper is
    never consulted -- the model uses each structure's native
    NGXF/NGYF/NGZF.

    This fallback matters in **non-trainer code paths**:

    * Ad-hoc inference (``model(systems, outputs)`` from a notebook /
      script) when the caller forgets / doesn't need per-system grids.
    * Unit tests that pin a small fixed grid for speed (e.g. ``[4, 4, 4]``
      in ``tests/test_train_smoke.py``).
    * Externally collected datasets that don't carry a ``grid_shape``
      extra-data field.

    Picking the value:

    * For **training on a CHGCAR cache**: leave at any sane default; it
      will not be used. The native grid wins.
    * For **inference on uniform grids**: pick the resolution you want
      every prediction to be evaluated on. Real-space resolution
      (Angstrom/voxel) varies with cell size at fixed ``grid_shape``;
      densities should be resampled to this grid if comparing to a
      reference.
    """

    fourier_chunk_size: int = 4096
    """Number of G-vectors processed per chunk in the analytic Fourier transform.
    Reduce to save GPU memory at the cost of speed."""

    head_mlp_hidden: int = 64
    """Hidden width of the per-Gaussian weight and gamma MLPs in the
    GaussianDensityHead (paper Eq 13/15: f_w, f_gamma)."""

    dyadic_mlp_hidden: int = 0
    """Hidden width of the per-channel scalar/kappa/m MLPs in the dyadic
    aggregation layer (paper Eq 19/22/24, MLPs g_c, phi_m, g_tilde). When 0,
    falls back to a single linear projection (legacy pre-audit behaviour);
    set to e.g. d_node//2 to use a 2-layer MLP."""


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

    optimizer: Literal["adam", "adamw", "muon"] = "adam"
    """Optimizer family. ``"muon"`` uses
    `Moonshot's Muon <https://github.com/KellerJordan/Muon>`_ on the matrix
    parameters of the PET backbone (with AdamW on biases / scalars / 1-D
    parameters), matching the convention used in metatrain's PET muon branch.
    Falls back to AdamW with a clear error if the ``muon`` package is not
    installed."""

    muon_momentum: float = 0.95
    """Muon momentum (``mu``). Ignored when ``optimizer != "muon"``."""

    use_bucketed_sampler: bool = False
    """Replace the default ``DistributedSampler`` with a size-aware sampler
    that groups same-grid-size structures into the same DDP step. Requires
    the train dataset to expose a ``grid_sizes()`` method (e.g.
    :class:`~metatrain.experimental.electrafy.modules.cache_dataset.CachedChgcarDataset`).

    On 2x4 H100 with the MP-CHGCAR corpus (~200k-2M voxels per system) this
    delivered 1.64x e2e speedup (bench 3348578) vs random
    ``DistributedSampler`` because variable system sizes no longer gate the
    max-rank DDP step."""

    bucket_tol: float = 0.10
    """Max within-step grid-size spread for the bucketed sampler
    (``max_grid / min_grid <= 1 + bucket_tol``). ``0.0`` = strict mode
    (consecutive sorted indices); larger values relax the spread and
    reshuffle within-pool each epoch. Ignored when
    ``use_bucketed_sampler`` is ``False``."""

    max_grid_points_per_batch: int = 0
    """If ``> 0``, use the grid-budget batch sampler instead of fixed
    ``batch_size`` (and instead of the bucketed sampler if both are set).
    Greedy-packs structures into batches whose total grid-point count does
    not exceed this budget, then assigns batches to ranks by stride.

    Ignored when ``0``. Useful when memory rather than count is the binding
    constraint (variable-grid datasets)."""

    n_steps_override: int = 0
    """If ``> 0``, override the cosine-LR scheduler's total step count
    (otherwise computed as ``num_epochs * steps_per_epoch``). Use this to
    cleanly taper LR to 0 over a *partial* run (e.g. when resuming a
    crashed run to finish one more epoch with proper LR decay)."""

    train_metric: Literal["rmse", "nmae"] = "rmse"
    """Per-epoch metric printed alongside the loss and surfaced through the
    metric logger.

    * ``"rmse"`` (default, backward-compatible) -- uses
      :class:`metatrain.utils.metrics.RMSEAccumulator`. Output key is
      ``"{target_key} RMSE"`` (or ``"... (per atom)"``).
    * ``"nmae"`` -- uses
      :class:`~metatrain.experimental.electrafy.modules.metrics.NMAEAccumulator`,
      the canonical density-prediction metric:
      ``NMAE = sum|pred - ref| / sum|ref|``. Output key is
      ``"{target_key} NMAE"``. Numerator and denominator are reduced across
      ranks before forming the ratio, so the global NMAE is exact (not a
      mean-of-ratios).
    """
