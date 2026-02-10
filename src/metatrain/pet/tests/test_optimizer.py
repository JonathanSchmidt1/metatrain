"""Tests for the Muon optimizer integration."""

import copy

import torch

from metatrain.pet import PET
from metatrain.pet.modules.optimizer import MuonWithAuxAdamW, get_optimizer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info


def _make_model():
    """Create a small PET model for optimizer tests."""
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers["d_pet"] = 8
    hypers["d_head"] = 8
    hypers["d_node"] = 8
    hypers["d_feedforward"] = 8
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    return PET(hypers, dataset_info)


def _make_hypers(**overrides):
    """Create training hypers with optional overrides."""
    hypers = copy.deepcopy(get_default_hypers("pet")["training"])
    hypers.update(overrides)
    return hypers


def test_get_optimizer_adam():
    """get_optimizer with Adam returns torch.optim.Adam."""
    model = _make_model()
    hypers = _make_hypers(optimizer="Adam")
    opt = get_optimizer(model, hypers)
    assert isinstance(opt, torch.optim.Adam)
    assert not isinstance(opt, torch.optim.AdamW)


def test_get_optimizer_adamw():
    """get_optimizer with AdamW returns torch.optim.AdamW."""
    model = _make_model()
    hypers = _make_hypers(optimizer="AdamW", weight_decay=0.01)
    opt = get_optimizer(model, hypers)
    assert isinstance(opt, torch.optim.AdamW)


def test_get_optimizer_muon():
    """get_optimizer with Muon returns MuonWithAuxAdamW."""
    model = _make_model()
    hypers = _make_hypers(optimizer="Muon")
    opt = get_optimizer(model, hypers)
    assert isinstance(opt, MuonWithAuxAdamW)


def test_muon_param_split():
    """Muon gets GNN/MLP 2D weights; AdamW gets biases, embeddings, heads."""
    model = _make_model()
    hypers = _make_hypers(optimizer="Muon")
    opt = get_optimizer(model, hypers)

    muon_params = set(id(p) for p in opt.muon_optimizer.param_groups[0]["params"])
    adamw_params = set(id(p) for p in opt.adamw_optimizer.param_groups[0]["params"])

    for name, param in model.named_parameters():
        pid = id(param)
        if param.ndim >= 2 and (
            ("gnn_layers" in name and "neighbor_embedder" not in name)
            or "combination_mlps" in name
        ):
            assert pid in muon_params, f"{name} should be in Muon"
            assert pid not in adamw_params, f"{name} should not be in AdamW"
        else:
            assert pid in adamw_params, f"{name} should be in AdamW"
            assert pid not in muon_params, f"{name} should not be in Muon"


def test_muon_adjust_lr_fn():
    """Muon inner optimizer uses match_rms_adamw LR scaling."""
    model = _make_model()
    hypers = _make_hypers(optimizer="Muon")
    opt = get_optimizer(model, hypers)

    # Check that adjust_lr_fn was set on the Muon optimizer's param groups
    for group in opt.muon_optimizer.param_groups:
        assert group.get("adjust_lr_fn") == "match_rms_adamw", (
            "Muon should use match_rms_adamw LR scaling"
        )


def test_muon_state_dict_roundtrip():
    """MuonWithAuxAdamW state_dict can be saved and loaded back."""
    model = _make_model()
    hypers = _make_hypers(optimizer="Muon", learning_rate=1e-3)
    opt = get_optimizer(model, hypers)

    # Do a dummy step to populate optimizer state
    loss = sum(p.sum() for p in model.parameters())
    loss.backward()
    opt.step()
    opt.zero_grad()

    # Save state
    state = opt.state_dict()
    assert "muon_optimizer" in state
    assert "adamw_optimizer" in state

    # Create a fresh optimizer and load state
    model2 = _make_model()
    # Copy model weights so parameter shapes match
    model2.load_state_dict(model.state_dict())
    opt2 = get_optimizer(model2, hypers)
    opt2.load_state_dict(state)

    # Verify loaded state matches
    state2 = opt2.state_dict()
    for key in ["muon_optimizer", "adamw_optimizer"]:
        assert state[key].keys() == state2[key].keys()


def test_muon_step_updates_params():
    """A Muon optimizer step actually changes both Muon and AdamW parameters."""
    model = _make_model()
    hypers = _make_hypers(optimizer="Muon", learning_rate=1e-2)
    opt = get_optimizer(model, hypers)

    # Snapshot params before step
    before = {n: p.clone() for n, p in model.named_parameters()}

    # Forward + backward + step
    loss = sum((p**2).sum() for p in model.parameters())
    loss.backward()
    opt.step()

    # Check that at least some Muon and AdamW params changed
    muon_changed = False
    adamw_changed = False
    for name, param in model.named_parameters():
        if not torch.equal(param, before[name]):
            if param.ndim >= 2 and (
                ("gnn_layers" in name and "neighbor_embedder" not in name)
                or "combination_mlps" in name
            ):
                muon_changed = True
            else:
                adamw_changed = True

    assert muon_changed, "Muon parameters should have been updated"
    assert adamw_changed, "AdamW parameters should have been updated"
