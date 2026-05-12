"""
ELECTRAFY Trainer — adapted from the PET trainer for charge density prediction.

Key differences from PET trainer:
- No additive (composition) models or scalers — density has no simple baseline.
- No energy/force/stress wrapping — output is a per-structure density grid.
- Uses NMAE loss by default (paper convention).
- Native NGXF/NGYF/NGZF grids per structure: each batch carries a
  ``grid_shape`` extra-data TensorMap which the trainer reads to call
  ``model.set_override_grid_shapes`` before every forward.
- Optional ``torch.compile(dynamic=True)`` wrap (kuma T4/T6/T7 confirmed it
  works with the gradient-checkpointed Fourier loop).
- No fine-tuning support (experimental).
"""

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch._dynamo
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.distributed.batch_utils import should_skip_batch
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import DistributedEnvironment
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger, WandbHandler
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import RMSEAccumulator, MAEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.transfer import batch_to

from .documentation import TrainerHypers
from .model import ELECTRAFY
from .modules.cache_dataset import decode_grid_shapes
from .modules.metrics import NMAEAccumulator
from .modules.samplers import GridBudgetBatchSampler, SortedBucketSampler


# Allow Tensor.item() inside compiled regions (PET nef.py uses it for reverse
# neighbor index sizing). Without this, torch.compile produces noisy
# graph-break warnings each forward — same setting the standalone kuma
# train_mp_*.py scripts use.
torch._dynamo.config.capture_scalar_outputs = True


def _unwrap_to_electrafy(model: torch.nn.Module) -> ELECTRAFY:
    """Strip ``torch.compile`` and ``DistributedDataParallel`` wrappers to get
    the underlying :class:`ELECTRAFY` instance — needed to call
    ``set_override_grid_shapes`` per batch (the API lives on the model, not
    on the wrappers).

    Order matches the wrap order: training wraps DDP first, then compile.
    Unwrapping reverses: compile (``_orig_mod``) outermost, then DDP
    (``module``).
    """
    inner: torch.nn.Module = model
    if hasattr(inner, "_orig_mod"):  # torch.compile
        inner = inner._orig_mod
    if hasattr(inner, "module"):  # DDP
        inner = inner.module
    if not isinstance(inner, ELECTRAFY):
        raise TypeError(
            f"expected ELECTRAFY after unwrapping; got {type(inner).__name__}"
        )
    return inner


def _apply_grid_shapes(
    inner: ELECTRAFY,
    extra_data: Optional[Dict[str, Any]],
    n_systems: int,
) -> bool:
    """Hand the per-batch native NGXF/NGYF/NGZF grids to the ELECTRAFY model.

    Reads ``extra_data["grid_shape"]`` -- a TensorMap with values of shape
    ``(n_systems, 3)`` emitted by
    :class:`~metatrain.experimental.electrafy.modules.cache_dataset.CachedChgcarDataset`
    -- decodes it back into a ``List[(N1, N2, N3)]``, and installs it on
    the model via
    :py:meth:`~metatrain.experimental.electrafy.model.ELECTRAFY.set_override_grid_shapes`.

    This is the **only channel** through which per-system native grids
    reach the Fourier head during training -- ``System`` has no grid
    field. The model's ``ModelHypers.grid_shape`` hyper is only a fallback
    consulted when this override is not set (ad-hoc inference paths). On
    the trainer's hot path, this helper is invoked before every forward
    and ``clear_override_grid_shapes`` is invoked after, so the fallback
    is never read.

    Returns ``True`` when shapes were installed (caller MUST pair with
    ``clear_override_grid_shapes`` in a ``finally`` block), ``False``
    when no ``grid_shape`` extra is present (caller falls back to the
    model's hyper -- legacy / unit-test path).
    """
    if extra_data is None or "grid_shape" not in extra_data:
        return False
    shapes = decode_grid_shapes(extra_data["grid_shape"])
    if len(shapes) != n_systems:
        raise RuntimeError(
            f"grid_shape extra has {len(shapes)} entries but batch has "
            f"{n_systems} systems"
        )
    inner.set_override_grid_shapes(shapes)
    return True


def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """Cosine-annealing LR scheduler with linear warmup.

    Honours ``n_steps_override`` (use to cleanly taper LR over a partial run,
    e.g. when finishing a crashed-then-resumed run in one extra epoch).
    """
    n_steps_override = int(train_hypers.get("n_steps_override", 0) or 0)
    if n_steps_override > 0:
        total_steps = n_steps_override
    else:
        total_steps = train_hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _dataset_grid_sizes(ds) -> Optional[List[int]]:
    """Return per-sample grid sizes for ``ds`` or ``None`` if unavailable.

    Handles ``torch.utils.data.Subset`` by descending to ``.dataset`` and
    re-indexing through ``.indices``. The base dataset must expose a
    ``grid_sizes()`` method (duck-typed; see
    :class:`metatrain.experimental.electrafy.modules.cache_dataset.CachedChgcarDataset`).
    """
    base = ds
    indices: Optional[List[int]] = None
    if isinstance(ds, torch.utils.data.Subset):
        base = ds.dataset
        indices = [int(i) for i in ds.indices]
    if not hasattr(base, "grid_sizes"):
        return None
    sizes = base.grid_sizes()
    if indices is not None:
        sizes = [sizes[i] for i in indices]
    return [int(s) for s in sizes]


def _build_train_sampler(
    ds,
    *,
    world_size: int,
    rank: int,
    use_bucketed: bool,
    bucket_tol: float,
    max_grid_per_batch: int,
    is_distributed: bool,
    seed: int,
):
    """Pick the right (sampler, batch_sampler) pair for a train dataset.

    Returns ``(sampler, None)`` for index-style samplers and ``(None, bs)``
    for batch samplers (``GridBudgetBatchSampler``). Either entry can also
    be ``None`` (non-distributed default-shuffled DataLoader).
    """
    if max_grid_per_batch > 0 or use_bucketed:
        grid_sizes = _dataset_grid_sizes(ds)
        if grid_sizes is None:
            logging.warning(
                "use_bucketed_sampler/max_grid_points_per_batch requested but "
                "train dataset does not expose grid_sizes(); falling back to "
                "DistributedSampler."
            )
        elif max_grid_per_batch > 0:
            bs = GridBudgetBatchSampler(
                grid_sizes=grid_sizes,
                world_size=world_size,
                rank=rank,
                max_grid_per_batch=max_grid_per_batch,
                seed=seed,
            )
            logging.info(
                f"GridBudgetBatchSampler: {bs.stats['n_batches_total']} batches "
                f"({bs.stats['n_batches_per_rank']} per rank), "
                f"items_min/mean/max={bs.stats['items_min']}/"
                f"{bs.stats['items_mean']:.1f}/{bs.stats['items_max']}, "
                f"fill_mean={bs.stats['fill_mean']:.0f}"
            )
            return None, bs
        else:
            s = SortedBucketSampler(
                grid_sizes=grid_sizes,
                world_size=world_size,
                rank=rank,
                seed=seed,
                tol=bucket_tol,
            )
            logging.info(
                f"SortedBucketSampler: tol={bucket_tol}, "
                f"{len(s)} steps per rank from {len(grid_sizes)} samples"
            )
            return s, None
    if is_distributed:
        return (
            DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True,
            ),
            None,
        )
    return None, None


class _CompositeOptimizer(torch.optim.Optimizer):
    """Wrap a list of sub-optimizers so they can be driven through the same
    ``step()`` / ``zero_grad()`` / state-dict surface the metatrain trainer
    expects from a single optimizer.

    Used for the Muon recipe: ``Muon`` on the matrix params of the PET
    backbone + ``AdamW`` on everything else (biases, scalars, embeddings,
    last-layer heads). Param groups from the sub-optimizers are exposed via
    ``param_groups`` so ``LambdaLR`` can sweep the LR multiplier across all
    groups in lockstep.
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("_CompositeOptimizer needs at least one sub-optimizer")
        self.optimizers = list(optimizers)
        # Don't go through Optimizer.__init__: we hold no params ourselves; the
        # sub-optimizers own them. Manually surface the bits the trainer reads.
        self.defaults: Dict[str, Any] = {}
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]
        self.state: Dict[Any, Any] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        # Re-flatten param_groups in case any sub-optimizer mutated them.
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]
        return loss

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"sub_optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:  # type: ignore[override]
        subs = sd["sub_optimizers"]
        if len(subs) != len(self.optimizers):
            raise ValueError(
                f"_CompositeOptimizer state_dict has {len(subs)} sub-optimizers "
                f"but expected {len(self.optimizers)}"
            )
        for opt, sub_sd in zip(self.optimizers, subs):
            opt.load_state_dict(sub_sd)
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]


def _build_optimizer(
    model: torch.nn.Module,
    hypers: TrainerHypers,
) -> torch.optim.Optimizer:
    """Build the optimizer per ``hypers["optimizer"]``.

    Choices:

    * ``"adam"`` (default) -- plain Adam with ``learning_rate``.
    * ``"adamw"`` -- AdamW with ``weight_decay`` (auto-selected when
      ``weight_decay`` is set, for backwards compatibility).
    * ``"muon"`` -- ``torch.optim.Muon`` on matrix params of the PET
      backbone, ``torch.optim.AdamW`` on everything else (biases, scalars,
      1-D params, embeddings). Wrapped in :class:`_CompositeOptimizer` so the
      training loop sees a single optimizer. Requires PyTorch >= 2.9 (when
      ``torch.optim.Muon`` was added); raises a clear error otherwise.
    """
    lr = float(hypers["learning_rate"])
    wd = hypers.get("weight_decay")
    optimizer_choice = str(hypers.get("optimizer", "adam")).lower()

    # Backwards compat: weight_decay set on a default ("adam") config selects
    # AdamW (the previous behavior).
    if optimizer_choice == "adam" and wd is not None:
        optimizer_choice = "adamw"

    if optimizer_choice == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_choice == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=float(wd or 0.0)
        )
    if optimizer_choice == "muon":
        try:
            from torch.optim import Muon  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "optimizer='muon' requires torch.optim.Muon (PyTorch >= 2.9); "
                f"got torch {torch.__version__}. Use optimizer='adamw' instead."
            ) from exc
        muon_params: List[torch.nn.Parameter] = []
        adamw_params: List[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Same convention as the muon-branch PET trainer + the standalone
            # kuma/train_mp_streaming.py: 2-D+ parameters of the GNN backbone
            # go to Muon; biases, scalars, embeddings, last-layer heads go to
            # AdamW.
            is_matrix = p.ndim >= 2
            is_embedding = "embed" in name.lower() or "embedding" in name.lower()
            if is_matrix and not is_embedding:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        muon_mom = float(hypers.get("muon_momentum", 0.95))
        muon = Muon(
            muon_params,
            lr=lr,
            momentum=muon_mom,
            weight_decay=float(wd or 0.0),
            adjust_lr_fn="match_rms_adamw",
        )
        adam = torch.optim.AdamW(
            adamw_params, lr=lr, weight_decay=float(wd or 0.0)
        )
        return _CompositeOptimizer([muon, adam])
    raise ValueError(
        f"unknown optimizer={optimizer_choice!r}; expected "
        f"'adam', 'adamw', or 'muon'"
    )


def _make_metric_calculator(hypers: TrainerHypers):
    """Pick the train/val metric accumulator implied by ``hypers["train_metric"]``.

    Returns an object with the ``RMSEAccumulator`` API
    (``update(predictions, targets, extra_data)`` and
    ``finalize(not_per_atom, is_distributed, device) -> Dict[str, float]``).
    """
    choice = str(hypers.get("train_metric", "rmse")).lower()
    if choice == "rmse":
        return RMSEAccumulator(False)
    if choice == "nmae":
        return NMAEAccumulator(False)
    raise ValueError(
        f"unknown train_metric={choice!r}; expected 'rmse' or 'nmae'"
    )


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None

    def train(
        self,
        model: ELECTRAFY,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in ELECTRAFY.__supported_dtypes__

        is_distributed = self.hypers["distributed"]

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with multi-gpu device. "
                    "For distributed training with ELECTRAFY, set `device` to cuda."
                )
            distr_env = DistributedEnvironment(self.hypers["distributed_port"])
            device_number = distr_env.local_rank % torch.cuda.device_count()
            device = torch.device("cuda", device_number)
            torch.distributed.init_process_group(backend="nccl", device_id=device)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0
            device = devices[0]

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Move model to device and dtype
        model.to(device=device, dtype=dtype)

        logging.info("Setting up data loaders")

        use_bucketed = bool(self.hypers.get("use_bucketed_sampler", False))
        max_grid_per_batch = int(self.hypers.get("max_grid_points_per_batch", 0) or 0)
        bucket_tol = float(self.hypers.get("bucket_tol", 0.10))
        sampler_world_size = world_size if is_distributed else 1
        sampler_rank = rank

        # Validation always uses the standard (deterministic) sampler -- size
        # imbalance during a one-pass eval doesn't matter for throughput.
        if is_distributed:
            val_samplers = [
                DistributedSampler(
                    ds, num_replicas=world_size, rank=rank,
                    shuffle=False, drop_last=False,
                )
                for ds in val_datasets
            ]
        else:
            val_samplers = [None] * len(val_datasets)

        train_samplers: List[Any] = []
        train_batch_samplers: List[Any] = []
        for ds in train_datasets:
            sampler, batch_sampler = _build_train_sampler(
                ds,
                world_size=sampler_world_size,
                rank=sampler_rank,
                use_bucketed=use_bucketed,
                bucket_tol=bucket_tol,
                max_grid_per_batch=max_grid_per_batch,
                is_distributed=is_distributed,
                seed=int(self.hypers.get("distributed_port", 0)),
            )
            train_samplers.append(sampler)
            train_batch_samplers.append(batch_sampler)

        # Collate functions — only neighbor lists, no additive/scaler transforms
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        requested_neighbor_lists = get_requested_neighbor_lists(model)

        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            ],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            ],
        )

        # Num workers
        if self.hypers.get("num_workers") is None:
            num_workers = get_num_workers()
            logging.info(f"Using {num_workers} data-loading workers (auto-detected).")
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)

        # Training dataloaders
        batch_size = self.hypers["batch_size"]
        train_dataloaders = []
        for ds, sampler, batch_sampler in zip(
            train_datasets, train_samplers, train_batch_samplers, strict=True
        ):
            if batch_sampler is None and len(ds) < batch_size:
                raise ValueError(
                    f"Training dataset has fewer samples ({len(ds)}) "
                    f"than batch size ({batch_size}). Reduce batch_size."
                )
            if batch_sampler is not None:
                train_dataloaders.append(
                    DataLoader(
                        dataset=ds,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn_train,
                        num_workers=num_workers,
                    )
                )
            else:
                train_dataloaders.append(
                    DataLoader(
                        dataset=ds,
                        batch_size=batch_size,
                        sampler=sampler,
                        shuffle=(sampler is None),
                        drop_last=(sampler is None),
                        collate_fn=collate_fn_train,
                        num_workers=num_workers,
                    )
                )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Validation dataloaders
        val_dataloaders = []
        for ds, sampler in zip(val_datasets, val_samplers, strict=True):
            val_dataloaders.append(
                DataLoader(
                    dataset=ds,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn_val,
                    num_workers=num_workers,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        # Optional torch.compile wrap. Dynamic shapes because each batch has
        # its own native (N1, N2, N3). Validated end-to-end by kuma jobs
        # 2865154 / 3104195 / 3108060 (T4 / T6 / T7).
        if self.hypers.get("compile", False):
            logging.info("compiling model with torch.compile(dynamic=True)")
            model = torch.compile(model, dynamic=True)

        # Loss function
        loss_hypers = dict(self.hypers.get("loss", {"type": "nmae", "weight": 1.0}))
        loss_fn = LossAggregator(
            targets=train_targets,
            config={k: LossSpecification(**v) if isinstance(v, dict) else v
                    for k, v in loss_hypers.items()} if isinstance(loss_hypers, dict)
            else loss_hypers,
        )

        # Optimizer
        optimizer = _build_optimizer(model, self.hypers)
        logging.info(
            f"Optimizer: {type(optimizer).__name__} "
            f"(choice={self.hypers.get('optimizer', 'adam')!r}, "
            f"lr={self.hypers['learning_rate']}, "
            f"weight_decay={self.hypers.get('weight_decay')})"
        )

        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)

        # LR scheduler
        lr_scheduler = _get_scheduler(optimizer, self.hypers, len(train_dataloader))
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        if self.best_metric is None:
            self.best_metric = float("inf")

        logging.info("Starting training")

        steps_per_epoch = len(train_dataloader)
        log_interval = self.hypers.get("log_interval", 1.0)
        val_interval = self.hypers.get("validation_interval", 1.0)
        log_every_n_steps = max(1, round(log_interval * steps_per_epoch))
        val_every_n_steps = max(1, round(val_interval * steps_per_epoch))
        global_step = start_epoch * steps_per_epoch
        metric_logger = None
        grad_clip = self.hypers.get("grad_clip_norm", 10.0)
        checkpoint_interval = int(self.hypers.get("checkpoint_interval", 10))

        epoch = start_epoch
        for epoch in range(start_epoch, self.hypers["num_epochs"]):
            # Reseed per-epoch shuffling. Distributed + bucketed/grid-budget
            # samplers all expose set_epoch; for single-rank with no sampler
            # the DataLoader's shuffle=True default handles it.
            for sampler in train_samplers:
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
            for bs in train_batch_samplers:
                if bs is not None and hasattr(bs, "set_epoch"):
                    bs.set_epoch(epoch)

            train_rmse_calculator = _make_metric_calculator(self.hypers)
            train_loss = 0.0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True)
            for batch in pbar:
                if should_skip_batch(batch, is_distributed, device):
                    continue

                optimizer.zero_grad()

                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype, device=device
                )

                # Per-batch native grid shape (paper Appendix C). Reset after
                # forward so a stale override can't leak into the next batch.
                inner_model = _unwrap_to_electrafy(model)
                applied = _apply_grid_shapes(inner_model, extra_data, len(systems))
                try:
                    # Forward pass: model produces density TensorMaps
                    predictions = evaluate_model(
                        model,
                        systems,
                        {key: train_targets[key] for key in targets.keys()},
                        is_training=True,
                    )
                finally:
                    if applied:
                        inner_model.clear_override_grid_shapes()

                train_loss_batch = loss_fn(predictions, targets, extra_data)

                if is_distributed:
                    for param in model.parameters():
                        train_loss_batch += 0.0 * param.sum()

                train_loss_batch.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()

                if is_distributed:
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()

                train_rmse_calculator.update(predictions, targets, extra_data)

                postfix: Dict[str, str] = {"loss": f"{train_loss_batch.item():.4e}"}
                pbar.set_postfix(postfix)

                global_step += 1

                # Step-level wandb logging
                if (
                    global_step % log_every_n_steps == 0
                    and global_step % val_every_n_steps != 0
                ):
                    for handler in ROOT_LOGGER.handlers:
                        if isinstance(handler, WandbHandler):
                            handler.run.log(
                                {
                                    "step/loss": train_loss_batch.item(),
                                    "step/learning_rate": optimizer.param_groups[0][
                                        "lr"
                                    ],
                                },
                                step=global_step,
                            )
                            break

                # Step-level validation
                if global_step % val_every_n_steps == 0:
                    val_rmse_calculator = _make_metric_calculator(self.hypers)
                    val_loss = 0.0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            if should_skip_batch(val_batch, is_distributed, device):
                                continue
                            systems_v, targets_v, extra_data_v = unpack_batch(
                                val_batch
                            )
                            systems_v, targets_v, extra_data_v = batch_to(
                                systems_v, targets_v, extra_data_v,
                                dtype=dtype, device=device,
                            )
                            inner_model_v = _unwrap_to_electrafy(model)
                            applied_v = _apply_grid_shapes(
                                inner_model_v, extra_data_v, len(systems_v)
                            )
                            try:
                                predictions_v = evaluate_model(
                                    model, systems_v,
                                    {key: train_targets[key] for key in targets_v},
                                    is_training=False,
                                )
                            finally:
                                if applied_v:
                                    inner_model_v.clear_override_grid_shapes()
                            val_loss_batch = loss_fn(
                                predictions_v, targets_v, extra_data_v
                            )
                            if is_distributed:
                                torch.distributed.all_reduce(val_loss_batch)
                            val_loss += val_loss_batch.item()
                            val_rmse_calculator.update(
                                predictions_v, targets_v, extra_data_v
                            )

                    # Finalize and log metrics
                    finalized_train_info = train_rmse_calculator.finalize(
                        not_per_atom=[],
                        is_distributed=is_distributed,
                        device=device,
                    )
                    finalized_val_info = val_rmse_calculator.finalize(
                        not_per_atom=[],
                        is_distributed=is_distributed,
                        device=device,
                    )
                    finalized_train_info = {"loss": train_loss, **finalized_train_info}
                    finalized_val_info = {"loss": val_loss, **finalized_val_info}

                    if metric_logger is None:
                        metric_logger = MetricLogger(
                            log_obj=ROOT_LOGGER,
                            dataset_info=(
                                model.module if is_distributed else model
                            ).dataset_info,
                            initial_metrics=[
                                finalized_train_info, finalized_val_info
                            ],
                            names=["training", "validation"],
                        )
                    metric_logger.log(
                        metrics=[finalized_train_info, finalized_val_info],
                        epoch=global_step,
                        rank=rank,
                        learning_rate=optimizer.param_groups[0]["lr"],
                    )

                    val_metric = get_selected_metric(
                        finalized_val_info,
                        self.hypers.get("best_model_metric", "loss"),
                    )
                    if val_metric < self.best_metric:
                        self.best_metric = val_metric
                        raw_sd = (
                            model.module if is_distributed else model
                        ).state_dict()
                        self.best_model_state_dict = {
                            k: v.clone() for k, v in raw_sd.items()
                        }
                        self.best_epoch = epoch
                        self.best_optimizer_state_dict = copy.deepcopy(
                            optimizer.state_dict()
                        )

                    # Reset training accumulators
                    train_rmse_calculator = _make_metric_calculator(self.hypers)
                    train_loss = 0.0

            # Epoch-level checkpointing
            if epoch % checkpoint_interval == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # Final state for external checkpoint save
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_model_state_dict": self.best_model_state_dict,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
            }
        )
        torch.save(checkpoint, check_file_extension(path, ".ckpt"))

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        if context == "restart":
            trainer.epoch = checkpoint["epoch"]
        else:
            trainer.epoch = None
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]
        return trainer

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        if checkpoint.get("trainer_ckpt_version", 1) != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade trainer checkpoint: version "
                f"{checkpoint.get('trainer_ckpt_version')} != "
                f"{cls.__checkpoint_version__}."
            )
        return checkpoint
