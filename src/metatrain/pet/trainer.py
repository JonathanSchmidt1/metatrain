import copy
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.augmentation import RotationalAugmenter
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
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .documentation import TrainerHypers
from .model import PET
from .modules.finetuning import apply_finetuning_strategy


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Get a CosineAnnealing learning-rate scheduler with warmup

    :param optimizer: The optimizer for which to create the scheduler.
    :param train_hypers: The training hyperparameters.
    :param steps_per_epoch: The number of steps per epoch.
    :return: The learning rate scheduler.
    """
    total_steps = train_hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)
    min_lr_ratio = 0.0  # hardcoded for now, could be made configurable in the future

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 13

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any] | List[Dict[str, Any]]] = (
            None
        )
        self.scheduler_state_dict: Optional[Dict[str, Any] | List[Dict[str, Any]]] = (
            None
        )
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[
            Dict[str, Any] | List[Dict[str, Any]]
        ] = None

    def train(
        self,
        model: PET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in PET.__supported_dtypes__

        is_distributed = self.hypers["distributed"]
        is_finetune = self.hypers["finetune"]["read_from"] is not None

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with PET, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            distr_env = DistributedEnvironment(self.hypers["distributed_port"])
            device_number = distr_env.local_rank % torch.cuda.device_count()
            device = torch.device("cuda", device_number)
            torch.distributed.init_process_group(backend="nccl", device_id=device)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0
            device = devices[0]
            # only one device, as we don't support non-distributed multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        use_bf16_autocast = dtype == torch.bfloat16 and device.type == "cuda"
        if use_bf16_autocast:
            logging.info(
                "Using CUDA autocast for BF16 training (FP32 model/data, BF16 compute)."
            )
            model_dtype = torch.float32
            batch_dtype = torch.float32
            precision_mode = "bf16-autocast"
        else:
            model_dtype = dtype
            batch_dtype = dtype
            precision_mode = str(dtype).replace("torch.", "")
        logging.info(f"Effective precision mode: {precision_mode}")

        # Apply fine-tuning strategy if provided
        if is_finetune:
            assert self.hypers["finetune"]["read_from"] is not None  # for mypy
            model = apply_finetuning_strategy(model, self.hypers["finetune"])
            method = self.hypers["finetune"]["method"]
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logging.info(f"Applied finetuning strategy: {method}")
            logging.info(
                f"Number of trainable parameters: {num_trainable_params} "
                f"[{num_trainable_params / num_params:.2%} %]"
            )
            inherit_heads = self.hypers["finetune"]["inherit_heads"]
            if inherit_heads:
                logging.info(
                    "Inheriting initial weights for heads and last layers for targets: "
                    f"from {list(inherit_heads.values())} to "
                    f"{list(inherit_heads.keys())}"
                )

        # Move the model to the device and dtype:
        model.to(device=device, dtype=model_dtype)
        # The additive models of PET are always in float64 (to avoid numerical errors in
        # the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)

        logging.info("Calculating composition weights")
        model.additive_models[0].train_model(  # this is the composition model
            train_datasets,
            model.additive_models[1:],
            self.hypers["batch_size"],
            is_distributed,
            self.hypers["atomic_baseline"],
        )

        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                is_distributed,
                self.hypers["fixed_scaling_weights"],
            )

        logging.info("Setting up data loaders")

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                for train_dataset in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for val_dataset in val_datasets
            ]
        else:
            train_samplers = [None] * len(train_datasets)
            val_samplers = [None] * len(val_datasets)

        # Extract additive models and scaler and move them to CPU/float64 so they
        # can be used in the collate function
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        additive_models = copy.deepcopy(
            model.additive_models.to(dtype=torch.float64, device="cpu")
        )
        model.additive_models.to(device)
        model.additive_models[0].weights_to(device=device, dtype=torch.float64)
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        scaler = copy.deepcopy(model.scaler.to(dtype=torch.float64, device="cpu"))
        model.scaler.to(device)
        model.scaler.scales_to(device=device, dtype=torch.float64)

        # Create collate functions:
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data
        rotational_augmenter = RotationalAugmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                rotational_augmenter.apply_random_augmentations,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[  # no augmentation for validation
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )

        # Create dataloader for the training datasets:
        if self.hypers["num_workers"] is None:
            num_workers = get_num_workers()
            logging.info(
                "Number of workers for data-loading not provided and chosen "
                f"automatically. Using {num_workers} workers."
            )
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)
        pin_memory = device.type == "cuda"

        train_dataloaders = []
        for train_dataset, train_sampler in zip(
            train_datasets, train_samplers, strict=True
        ):
            if len(train_dataset) < self.hypers["batch_size"]:
                raise ValueError(
                    f"A training dataset has fewer samples "
                    f"({len(train_dataset)}) than the batch size "
                    f"({self.hypers['batch_size']}). "
                    "Please reduce the batch size."
                )
            train_dataloaders.append(
                DataLoader(
                    dataset=train_dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=train_sampler,
                    shuffle=(
                        # the sampler takes care of this (if present)
                        train_sampler is None
                    ),
                    drop_last=(
                        # the sampler takes care of this (if present)
                        train_sampler is None
                    ),
                    collate_fn=collate_fn_train,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for val_dataset, val_sampler in zip(val_datasets, val_samplers, strict=True):
            val_dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=val_sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn_val,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        # Create a loss function:
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])  # mypy
        loss_fn = LossAggregator(targets=train_targets, config=loss_hypers)
        logging.info("Using the following loss functions:")
        for name, info in loss_fn.metadata.items():
            logging.info(f"{name}:")
            main = {k: v for k, v in info.items() if k != "gradients"}
            logging.info(main)
            if "gradients" not in info or len(info["gradients"]) == 0:
                continue
            logging.info("With gradients:")
            for grad, ginfo in info["gradients"].items():
                logging.info(f"\t{name}::{grad}: {ginfo}")

        if self.hypers["use_muon"]:
            from torch.optim import Muon

            muon_params = []
            adamw_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    param.ndim >= 2
                    and "embed" not in name
                    and "norm" not in name
                    and "scaler" not in name
                    and "additive" not in name
                    and "last_layer" not in name
                ):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
            n_muon = sum(p.numel() for p in muon_params)
            n_adamw = sum(p.numel() for p in adamw_params)
            logging.info(
                f"Muon optimizer: {len(muon_params)} params ({n_muon} elements), "
                f"AdamW: {len(adamw_params)} params ({n_adamw} elements)"
            )
            wd = self.hypers["weight_decay"] or 0.0
            optimizers = [
                Muon(
                    muon_params,
                    lr=self.hypers["learning_rate"],
                    momentum=self.hypers["muon_momentum"],
                    adjust_lr_fn="match_rms_adamw",
                    weight_decay=wd,
                ),
                torch.optim.AdamW(
                    adamw_params,
                    lr=self.hypers["learning_rate"],
                    weight_decay=wd,
                ),
            ]
        elif self.hypers["weight_decay"] is not None:
            optimizers = [
                torch.optim.AdamW(
                    model.parameters(),
                    lr=self.hypers["learning_rate"],
                    weight_decay=self.hypers["weight_decay"],
                )
            ]
        else:
            optimizers = [
                torch.optim.Adam(model.parameters(), lr=self.hypers["learning_rate"])
            ]

        if self.optimizer_state_dict is not None and not is_finetune:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not (model.module if is_distributed else model).has_new_targets:
                if isinstance(self.optimizer_state_dict, list):
                    for opt, state in zip(
                        optimizers, self.optimizer_state_dict, strict=True
                    ):
                        opt.load_state_dict(state)
                else:
                    optimizers[0].load_state_dict(self.optimizer_state_dict)

        # Create learning rate schedulers (one per optimizer)
        lr_schedulers = [
            get_scheduler(opt, self.hypers, len(train_dataloader)) for opt in optimizers
        ]

        if self.scheduler_state_dict is not None and not is_finetune:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                if isinstance(self.scheduler_state_dict, list):
                    for sched, state in zip(
                        lr_schedulers, self.scheduler_state_dict, strict=True
                    ):
                        sched.load_state_dict(state)
                else:
                    lr_schedulers[0].load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch

        # Compute step-level intervals for logging and validation
        steps_per_epoch = len(train_dataloader)
        log_interval = self.hypers["log_interval"]
        val_interval = self.hypers["validation_interval"]
        log_every_n_steps = max(1, round(log_interval * steps_per_epoch))
        val_every_n_steps = max(1, round(val_interval * steps_per_epoch))
        global_step = start_epoch * steps_per_epoch
        metric_logger = None
        profile_step_timing = bool(self.hypers.get("profile_step_timing", False))
        profile_step_timing_sync_cuda = bool(
            self.hypers.get("profile_step_timing_sync_cuda", False)
        )
        use_sync_cuda_timing = profile_step_timing_sync_cuda and device.type == "cuda"
        if profile_step_timing:
            logging.info(
                "Step timing profiler enabled "
                "(training-loop stage breakdown per epoch)."
            )
            if profile_step_timing_sync_cuda and device.type != "cuda":
                logging.warning(
                    "profile_step_timing_sync_cuda is enabled, but device is not CUDA. "
                    "Falling back to non-synchronized timing."
                )
            elif use_sync_cuda_timing:
                logging.info(
                    "Using CUDA-synchronized timing for profiling (higher overhead)."
                )

        def sync_timing_cuda() -> None:
            if use_sync_cuda_timing:
                torch.cuda.synchronize(device)

        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            if is_distributed:
                for train_sampler in train_samplers:
                    train_sampler.set_epoch(epoch)
            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator(
                    self.hypers["log_separate_blocks"]
                )
            timing_sums: Dict[str, float] = {
                "dataloader_wait": 0.0,
                "train_step_total": 0.0,
                "unpack_batch": 0.0,
                "batch_to": 0.0,
                "forward_loss": 0.0,
                "backward_opt": 0.0,
                "metrics_logging": 0.0,
                "validation_total": 0.0,
            }
            timed_steps = 0
            batches_with_energy_pos_grad = 0

            train_loss = 0.0
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True)
            last_step_end = time.perf_counter()
            for batch in pbar:
                sync_timing_cuda()
                batch_ready = time.perf_counter()
                timing_sums["dataloader_wait"] += batch_ready - last_step_end

                # Skip None batches (those outside batch_atom_bounds)
                if should_skip_batch(batch, is_distributed, device):
                    sync_timing_cuda()
                    last_step_end = time.perf_counter()
                    continue

                sync_timing_cuda()
                step_start = time.perf_counter()
                for opt in optimizers:
                    opt.zero_grad()

                sync_timing_cuda()
                t0 = time.perf_counter()
                systems, targets, extra_data = unpack_batch(batch)
                sync_timing_cuda()
                timing_sums["unpack_batch"] += time.perf_counter() - t0

                sync_timing_cuda()
                t0 = time.perf_counter()
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=batch_dtype, device=device
                )
                sync_timing_cuda()
                timing_sums["batch_to"] += time.perf_counter() - t0
                if any(
                    train_targets[key].quantity == "energy"
                    and "positions" in train_targets[key].gradients
                    for key in targets.keys()
                ):
                    batches_with_energy_pos_grad += 1
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16_autocast
                    else nullcontext()
                )
                sync_timing_cuda()
                t0 = time.perf_counter()
                with autocast_ctx:
                    predictions = evaluate_model(
                        model,
                        systems,
                        {key: train_targets[key] for key in targets.keys()},
                        is_training=True,
                    )

                    # average by the number of atoms
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )
                    train_loss_batch = loss_fn(predictions, targets, extra_data)
                sync_timing_cuda()
                timing_sums["forward_loss"] += time.perf_counter() - t0

                if is_distributed:
                    # make sure all parameters contribute to the gradient calculation
                    # to make torch DDP happy
                    for param in model.parameters():
                        train_loss_batch += 0.0 * param.sum()

                sync_timing_cuda()
                t0 = time.perf_counter()
                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                for opt in optimizers:
                    opt.step()
                for sched in lr_schedulers:
                    sched.step()
                sync_timing_cuda()
                timing_sums["backward_opt"] += time.perf_counter() - t0

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()

                sync_timing_cuda()
                t0 = time.perf_counter()
                scaled_predictions = (model.module if is_distributed else model).scaler(
                    systems, predictions
                )
                scaled_targets = (model.module if is_distributed else model).scaler(
                    systems, targets
                )
                train_rmse_calculator.update(
                    scaled_predictions, scaled_targets, extra_data
                )
                if self.hypers["log_mae"]:
                    train_mae_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )

                # Update tqdm progress bar with per-batch errors
                postfix = {"loss": f"{train_loss_batch.item():.4e}"}
                for key in scaled_predictions:
                    pred = scaled_predictions[key].block().values
                    tgt = scaled_targets[key].block().values
                    rmse = torch.sqrt(torch.mean((pred - tgt) ** 2)).item()
                    postfix[key] = f"{rmse:.4e}"
                    if scaled_predictions[key].block().has_gradient("positions"):
                        pg = (
                            scaled_predictions[key].block().gradient("positions").values
                        )
                        tg = scaled_targets[key].block().gradient("positions").values
                        postfix["forces"] = (
                            f"{torch.sqrt(torch.mean((pg - tg) ** 2)).item():.4e}"
                        )
                pbar.set_postfix(postfix)
                sync_timing_cuda()
                timing_sums["metrics_logging"] += time.perf_counter() - t0

                global_step += 1
                timed_steps += 1
                sync_timing_cuda()
                timing_sums["train_step_total"] += time.perf_counter() - step_start
                last_step_end = time.perf_counter()

                # Step-level wandb logging of training metrics
                if (
                    global_step % log_every_n_steps == 0
                    and global_step % val_every_n_steps != 0
                ):
                    for handler in ROOT_LOGGER.handlers:
                        if isinstance(handler, WandbHandler):
                            wandb_data = {
                                "step/loss": train_loss_batch.item(),
                                "step/learning_rate": optimizers[0].param_groups[0][
                                    "lr"
                                ],
                            }
                            for key in scaled_predictions:
                                p = scaled_predictions[key].block().values
                                t = scaled_targets[key].block().values
                                wandb_data[f"step/{key}_rmse"] = torch.sqrt(
                                    torch.mean((p - t) ** 2)
                                ).item()
                                if (
                                    scaled_predictions[key]
                                    .block()
                                    .has_gradient("positions")
                                ):
                                    pg = (
                                        scaled_predictions[key]
                                        .block()
                                        .gradient("positions")
                                        .values
                                    )
                                    tg = (
                                        scaled_targets[key]
                                        .block()
                                        .gradient("positions")
                                        .values
                                    )
                                    wandb_data["step/forces_rmse"] = torch.sqrt(
                                        torch.mean((pg - tg) ** 2)
                                    ).item()
                            handler.run.log(wandb_data, step=global_step)
                            break

                # Step-level validation
                if global_step % val_every_n_steps == 0:
                    sync_timing_cuda()
                    val_start = time.perf_counter()
                    val_rmse_calculator = RMSEAccumulator(
                        self.hypers["log_separate_blocks"]
                    )
                    if self.hypers["log_mae"]:
                        val_mae_calculator = MAEAccumulator(
                            self.hypers["log_separate_blocks"]
                        )
                    val_loss = 0.0
                    for val_batch in val_dataloader:
                        if should_skip_batch(val_batch, is_distributed, device):
                            continue
                        systems_v, targets_v, extra_data_v = unpack_batch(val_batch)
                        systems_v, targets_v, extra_data_v = batch_to(
                            systems_v,
                            targets_v,
                            extra_data_v,
                            dtype=batch_dtype,
                            device=device,
                        )
                        autocast_ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if use_bf16_autocast
                            else nullcontext()
                        )
                        with autocast_ctx:
                            predictions_v = evaluate_model(
                                model,
                                systems_v,
                                {key: train_targets[key] for key in targets_v},
                                is_training=False,
                            )
                            predictions_v = average_by_num_atoms(
                                predictions_v, systems_v, per_structure_targets
                            )
                            targets_v = average_by_num_atoms(
                                targets_v, systems_v, per_structure_targets
                            )
                            val_loss_batch = loss_fn(
                                predictions_v, targets_v, extra_data_v
                            )
                        if is_distributed:
                            torch.distributed.all_reduce(val_loss_batch)
                        val_loss += val_loss_batch.item()
                        scaled_preds_v = (
                            model.module if is_distributed else model
                        ).scaler(systems_v, predictions_v)
                        scaled_tgts_v = (
                            model.module if is_distributed else model
                        ).scaler(systems_v, targets_v)
                        val_rmse_calculator.update(
                            scaled_preds_v, scaled_tgts_v, extra_data_v
                        )
                        if self.hypers["log_mae"]:
                            val_mae_calculator.update(
                                scaled_preds_v, scaled_tgts_v, extra_data_v
                            )
                    sync_timing_cuda()
                    timing_sums["validation_total"] += time.perf_counter() - val_start

                    # Finalize validation metrics
                    finalized_val_info = val_rmse_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                    if self.hypers["log_mae"]:
                        finalized_val_info.update(
                            val_mae_calculator.finalize(
                                not_per_atom=["positions_gradients"]
                                + per_structure_targets,
                                is_distributed=is_distributed,
                                device=device,
                            )
                        )

                    # Finalize training metrics (window since last reset)
                    finalized_train_info = train_rmse_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                    if self.hypers["log_mae"]:
                        finalized_train_info.update(
                            train_mae_calculator.finalize(
                                not_per_atom=["positions_gradients"]
                                + per_structure_targets,
                                is_distributed=is_distributed,
                                device=device,
                            )
                        )

                    finalized_train_info = {
                        "loss": train_loss,
                        **finalized_train_info,
                    }
                    finalized_val_info = {
                        "loss": val_loss,
                        **finalized_val_info,
                    }

                    # Initialize or log via MetricLogger
                    if metric_logger is None:
                        metric_logger = MetricLogger(
                            log_obj=ROOT_LOGGER,
                            dataset_info=(
                                model.module if is_distributed else model
                            ).dataset_info,
                            initial_metrics=[finalized_train_info, finalized_val_info],
                            names=["training", "validation"],
                        )
                    metric_logger.log(
                        metrics=[finalized_train_info, finalized_val_info],
                        epoch=global_step,
                        rank=rank,
                        learning_rate=optimizers[0].param_groups[0]["lr"],
                    )

                    # Best model tracking
                    val_metric = get_selected_metric(
                        finalized_val_info,
                        self.hypers["best_model_metric"],
                    )
                    if val_metric < self.best_metric:
                        self.best_metric = val_metric
                        self.best_model_state_dict = copy.deepcopy(
                            (model.module if is_distributed else model).state_dict()
                        )
                        self.best_epoch = epoch
                        self.best_optimizer_state_dict = copy.deepcopy(
                            [opt.state_dict() for opt in optimizers]
                        )

                    # Reset training accumulators for next window
                    train_rmse_calculator = RMSEAccumulator(
                        self.hypers["log_separate_blocks"]
                    )
                    if self.hypers["log_mae"]:
                        train_mae_calculator = MAEAccumulator(
                            self.hypers["log_separate_blocks"]
                        )
                    train_loss = 0.0

            if profile_step_timing and timed_steps > 0 and rank == 0:
                loop_total = (
                    timing_sums["train_step_total"] + timing_sums["dataloader_wait"]
                )
                avg_step_ms = 1000.0 * timing_sums["train_step_total"] / timed_steps
                avg_wait_ms = 1000.0 * timing_sums["dataloader_wait"] / timed_steps
                avg_loop_ms = 1000.0 * loop_total / timed_steps
                avg_unpack_ms = 1000.0 * timing_sums["unpack_batch"] / timed_steps
                avg_batch_to_ms = 1000.0 * timing_sums["batch_to"] / timed_steps
                avg_forward_ms = 1000.0 * timing_sums["forward_loss"] / timed_steps
                avg_backward_ms = 1000.0 * timing_sums["backward_opt"] / timed_steps
                avg_metrics_ms = 1000.0 * timing_sums["metrics_logging"] / timed_steps
                denom = max(loop_total, 1e-12)
                logging.info(
                    "Epoch %d timing (avg/step ms): loop_total=%.2f "
                    "| dataloader_wait=%.2f "
                    "(%.1f%%) | train_step=%.2f (%.1f%%) | unpack=%.2f (%.1f%%) | "
                    "batch_to=%.2f (%.1f%%) | forward+loss=%.2f (%.1f%%) | "
                    "backward+opt=%.2f (%.1f%%) | metrics/log=%.2f (%.1f%%) | "
                    "validation_total=%.2fs | energy_posgrad_batches=%d/%d",
                    epoch,
                    avg_loop_ms,
                    avg_wait_ms,
                    100.0 * timing_sums["dataloader_wait"] / denom,
                    avg_step_ms,
                    100.0 * timing_sums["train_step_total"] / denom,
                    avg_unpack_ms,
                    100.0 * timing_sums["unpack_batch"] / denom,
                    avg_batch_to_ms,
                    100.0 * timing_sums["batch_to"] / denom,
                    avg_forward_ms,
                    100.0 * timing_sums["forward_loss"] / denom,
                    avg_backward_ms,
                    100.0 * timing_sums["backward_opt"] / denom,
                    avg_metrics_ms,
                    100.0 * timing_sums["metrics_logging"] / denom,
                    timing_sums["validation_total"],
                    batches_with_energy_pos_grad,
                    timed_steps,
                )

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = [o.state_dict() for o in optimizers]
                self.scheduler_state_dict = [s.state_dict() for s in lr_schedulers]
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = [o.state_dict() for o in optimizers]
        self.scheduler_state_dict = [s.state_dict() for s in lr_schedulers]

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
        if self.best_model_state_dict is not None:
            self.best_model_state_dict["finetune_config"] = model.finetune_config
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
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

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
        trainer.epoch = checkpoint["epoch"]
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        return trainer

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["trainer_ckpt_version"] == v:
                update = getattr(checkpoints, f"trainer_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["trainer_ckpt_version"] = v + 1

        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using "
                f"trainer version {checkpoint['trainer_ckpt_version']}, while the "
                f"current trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
