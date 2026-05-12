"""
End-to-end smoke test for ``Trainer.train()``.

Wires up:

- CHGCAR → Systems + per-sample charge_density TensorMaps
- ``Dataset.from_dict`` → ``DataLoader`` via the trainer's own collate setup
- Tiny ELECTRAFY model (d_pet=4, d_node=8, 4³ grid) on CPU, float64, 1 epoch

Verifies that training runs end-to-end, a checkpoint is written, and at
least one parameter has moved. Does NOT assert on loss values — this is
a plumbing test, not a convergence test.
"""

from pathlib import Path

import pytest
import torch
from metatensor.learn.data import Dataset

from metatrain.experimental.electrafy.model import DENSITY_KEY, ELECTRAFY
from metatrain.experimental.electrafy.modules.chgcar import (
    chgcar_to_system_and_density,
    density_to_single_sample_tmap,
)
from metatrain.experimental.electrafy.trainer import Trainer
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo

from metatensor.torch import Labels, TensorBlock, TensorMap


REPO_ROOT = Path(__file__).resolve().parents[6]
CHGCAR_PATH = REPO_ROOT / "CHGCAR"


def _make_density_layout(grid_shape):
    n_grid = grid_shape[0] * grid_shape[1] * grid_shape[2]
    block = TensorBlock(
        values=torch.empty((0, n_grid), dtype=torch.float64),
        samples=Labels(names=["system"], values=torch.zeros((0, 1), dtype=torch.int32)),
        components=[],
        properties=Labels(
            names=["grid_point"],
            values=torch.arange(n_grid, dtype=torch.int32).unsqueeze(1),
        ),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def _make_dataset_info(atomic_types, grid_shape):
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={
            DENSITY_KEY: TargetInfo(
                layout=_make_density_layout(grid_shape),
                quantity="",
                unit="",
            ),
        },
    )


def _tiny_model_hypers(grid_shape):
    return {
        "cutoff": 4.0,
        "cutoff_function": "Cosine",
        "cutoff_width": 0.5,
        "d_pet": 4,
        "d_node": 8,
        "d_feedforward": 8,
        "num_heads": 2,
        "num_attention_layers": 1,
        "num_gnn_layers": 1,
        "normalization": "RMSNorm",
        "activation": "SiLU",
        "transformer_type": "PreLN",
        "attention_temperature": 1.0,
        "gaussians_per_electron": 1,
        "gamma": 0.1,
        "grid_shape": list(grid_shape),
        "fourier_chunk_size": 256,
    }


def _trainer_hypers():
    return {
        "distributed": False,
        "distributed_port": 39591,
        "batch_size": 1,
        "num_epochs": 1,
        "warmup_fraction": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": None,
        "log_interval": 1.0,
        "validation_interval": 1.0,
        "checkpoint_interval": 1,
        "loss": {
            DENSITY_KEY: {
                "type": "nmae",
                "weight": 1.0,
                "reduction": "mean",
                "gradients": {},
            }
        },
        "best_model_metric": "loss",
        "grad_clip_norm": 10.0,
        "num_workers": 0,
    }


@pytest.mark.skipif(not CHGCAR_PATH.is_file(), reason="CHGCAR fixture missing")
def test_trainer_train_end_to_end(tmp_path):
    grid_shape = (4, 4, 4)
    dtype = torch.float64
    device = torch.device("cpu")

    # Build two training samples and one val sample from the same CHGCAR.
    # (CollateFn drops the last partial batch on unsampled training, so
    # len(train_ds) must be >= batch_size — 2 samples is safe at batch_size=1.)
    systems_train = []
    targets_train = []
    for i in range(2):
        sys_, flat = chgcar_to_system_and_density(
            CHGCAR_PATH, grid_shape, dtype=dtype
        )
        systems_train.append(sys_)
        targets_train.append(density_to_single_sample_tmap(flat, sample_index=i))

    sys_val, flat_val = chgcar_to_system_and_density(
        CHGCAR_PATH, grid_shape, dtype=dtype
    )
    systems_val = [sys_val]
    targets_val = [density_to_single_sample_tmap(flat_val, sample_index=0)]

    train_ds = Dataset.from_dict({"system": systems_train, DENSITY_KEY: targets_train})
    val_ds = Dataset.from_dict({"system": systems_val, DENSITY_KEY: targets_val})

    # Atomic types present: Mn(25), Zn(30), N(7)
    dataset_info = _make_dataset_info([7, 25, 30], grid_shape)
    model = ELECTRAFY(_tiny_model_hypers(grid_shape), dataset_info).to(dtype)

    params_before = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }

    trainer = Trainer(_trainer_hypers())
    trainer.train(
        model=model,
        dtype=dtype,
        devices=[device],
        train_datasets=[train_ds],
        val_datasets=[val_ds],
        checkpoint_dir=str(tmp_path),
    )

    # Training advanced some parameters.
    moved = [
        name
        for name, before in params_before.items()
        if not torch.allclose(before, dict(model.named_parameters())[name])
    ]
    assert moved, "No model parameter changed after training"

    # Checkpoint was written (checkpoint_interval=1 → epoch 0 saves model_0.ckpt).
    ckpts = list(tmp_path.glob("model_*.ckpt"))
    assert ckpts, f"No checkpoint written under {tmp_path}"

    # Trainer state consistent with one-epoch run.
    assert trainer.epoch == 0
    assert trainer.optimizer_state_dict is not None
    assert trainer.best_metric is not None and trainer.best_metric < float("inf")
