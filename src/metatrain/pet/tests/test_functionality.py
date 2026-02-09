import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.pet.modules.transformer import AttentionBlock
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_pet_padding():
    """Tests that the model predicts the same energy independently of the
    padding size."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    lone_output = model([system], outputs)

    system_2 = System(
        types=torch.tensor([6, 6, 6, 6, 6, 6, 6]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 6.0],
            ]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_2 = get_system_with_neighbor_lists(
        system_2, model.requested_neighbor_lists()
    )
    padded_output = model([system, system_2], outputs)

    lone_energy = lone_output["energy"].block().values.squeeze(-1)[0]
    padded_energy = padded_output["energy"].block().values.squeeze(-1)[0]

    assert torch.allclose(lone_energy, padded_energy, atol=1e-6, rtol=1e-6)


def test_consistency():
    """Tests that the two implementations of attention are consistent."""

    num_centers = 100
    num_neighbors_per_center = 50
    hidden_size = 128
    num_heads = 4
    temperature = 2.0

    attention = AttentionBlock(hidden_size, num_heads, temperature)

    inputs = torch.randn(num_centers, num_neighbors_per_center, hidden_size)
    radial_mask = torch.rand(
        num_centers, num_neighbors_per_center, num_neighbors_per_center
    )

    attention_output_torch = attention(inputs, radial_mask, use_manual_attention=False)
    attention_output_manual = attention(inputs, radial_mask, use_manual_attention=True)

    assert torch.allclose(attention_output_torch, attention_output_manual, atol=1e-6)


@pytest.mark.parametrize("per_atom", [True, False])
def test_nc_stress(per_atom):
    """Tests that the model can predict a symmetric rank-2 tensor as the NC stress."""
    # (note that no composition energies are supplied or calculated here)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "non_conservative_stress": get_generic_target_info(
                "non_conservative_stress",
                {
                    "quantity": "stress",
                    "unit": "",
                    "type": {"cartesian": {"rank": 2}},
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                },
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 1.0]]),
        cell=torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"non_conservative_stress": ModelOutput(per_atom=per_atom)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(1, 2))


def test_muon_optimizer(tmp_path):
    """Test that Muon optimizer trains without error and splits params correctly."""
    torch.manual_seed(0)

    systems = read_systems(DATASET_WITH_FORCES_PATH)
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_WITH_FORCES_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["use_muon"] = True
    hypers["training"]["weight_decay"] = 0.01
    hypers["training"]["atomic_baseline"] = {"energy": 0.0}
    loss_conf = {"energy": init_with_defaults(LossSpecification)}
    loss_conf["energy"]["gradients"] = {
        "positions": init_with_defaults(LossSpecification)
    }
    loss_conf = OmegaConf.create(loss_conf)
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )
    model = PET(MODEL_HYPERS, dataset_info)

    # Verify parameter split: check that Muon would get 2D+ non-embed/norm params
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
            muon_params.append(name)
        else:
            adamw_params.append(name)

    assert len(muon_params) > 0, "Muon should get at least some parameters"
    assert len(adamw_params) > 0, "AdamW should get at least some parameters"
    # All bias params should be in adamw
    for name in adamw_params:
        if "bias" in name:
            assert name not in muon_params
    # All embedding params should be in adamw
    for name in adamw_params:
        if "embed" in name:
            assert name not in muon_params

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=str(tmp_path),
    )
