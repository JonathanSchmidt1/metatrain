"""
ELECTRAFY Trainer — adapted from the PET trainer for charge density prediction.

Key differences from PET trainer:
- No additive (composition) models or scalers — density has no simple baseline.
- No energy/force/stress wrapping — output is a per-structure density grid.
- Uses NMAE loss by default (paper convention).
- No torch.compile path (density pipeline not yet compile-friendly).
- No fine-tuning support (experimental).
"""

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
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


def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """Cosine-annealing LR scheduler with linear warmup."""
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

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    ds, num_replicas=world_size, rank=rank,
                    shuffle=True, drop_last=True,
                )
                for ds in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    ds, num_replicas=world_size, rank=rank,
                    shuffle=False, drop_last=False,
                )
                for ds in val_datasets
            ]
        else:
            train_samplers = [None] * len(train_datasets)
            val_samplers = [None] * len(val_datasets)

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
        for ds, sampler in zip(train_datasets, train_samplers, strict=True):
            if len(ds) < batch_size:
                raise ValueError(
                    f"Training dataset has fewer samples ({len(ds)}) "
                    f"than batch size ({batch_size}). Reduce batch_size."
                )
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

        # Loss function
        loss_hypers = dict(self.hypers.get("loss", {"type": "nmae", "weight": 1.0}))
        loss_fn = LossAggregator(
            targets=train_targets,
            config={k: LossSpecification(**v) if isinstance(v, dict) else v
                    for k, v in loss_hypers.items()} if isinstance(loss_hypers, dict)
            else loss_hypers,
        )

        # Optimizer
        if self.hypers.get("weight_decay") is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["learning_rate"],
                weight_decay=self.hypers["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["learning_rate"]
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
            if is_distributed:
                for sampler in train_samplers:
                    sampler.set_epoch(epoch)

            train_rmse_calculator = RMSEAccumulator(False)
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

                # Forward pass: model produces density TensorMaps
                predictions = evaluate_model(
                    model,
                    systems,
                    {key: train_targets[key] for key in targets.keys()},
                    is_training=True,
                )

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
                    val_rmse_calculator = RMSEAccumulator(False)
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
                            predictions_v = evaluate_model(
                                model, systems_v,
                                {key: train_targets[key] for key in targets_v},
                                is_training=False,
                            )
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
                    train_rmse_calculator = RMSEAccumulator(False)
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
