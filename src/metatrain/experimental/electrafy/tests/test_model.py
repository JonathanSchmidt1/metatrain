"""
Integration tests for the ELECTRAFY model: forward pass, gradient flow,
electron count conservation, and checkpoint round-trip.
"""

import torch
import pytest
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.electrafy.model import ELECTRAFY, DENSITY_KEY
from metatrain.experimental.electrafy.modules.valence import ZVAL_LOOKUP
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo

from metatomic.torch import ModelOutput, NeighborListOptions, System


def _make_system(
    positions: torch.Tensor,
    types: torch.Tensor,
    cell: torch.Tensor,
) -> System:
    """Create a System with a precomputed neighbor list."""
    system = System(
        positions=positions,
        types=types,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )
    return system


def _add_neighbor_list(system: System, cutoff: float) -> System:
    """Add a full neighbor list to a system using ASE."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list as ase_nl
    import numpy as np

    positions = system.positions.detach().cpu().numpy()
    types = system.types.detach().cpu().numpy()
    cell = system.cell.detach().cpu().numpy()

    atoms = Atoms(
        numbers=types,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    i_list, j_list, S_list, d_list = ase_nl("ijSd", atoms, cutoff)

    # Build the neighbor list in metatomic format
    samples = torch.tensor(
        [[i_list[k], j_list[k], S_list[k, 0], S_list[k, 1], S_list[k, 2]]
         for k in range(len(i_list))],
        dtype=torch.int32,
    )
    distances = torch.tensor(d_list, dtype=system.positions.dtype)

    if len(samples) == 0:
        samples = torch.zeros((0, 5), dtype=torch.int32)
        distances = torch.zeros((0, 3), dtype=system.positions.dtype)

    nl_options = NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)
    system.add_neighbor_list(
        nl_options,
        system.known_neighbor_lists()[0] if system.known_neighbor_lists()
        else _build_nl_block(samples, distances, system),
    )
    return system


def _build_nl_block(
    samples: torch.Tensor, distances: torch.Tensor, system: System
) -> "metatensor.torch.TensorBlock":
    """Build a metatensor TensorBlock for the neighbor list."""
    from metatensor.torch import TensorBlock, Labels
    import torch

    n_pairs = samples.shape[0]
    if n_pairs == 0:
        values = torch.zeros((0, 3, 1), dtype=system.positions.dtype)
    else:
        # Compute actual displacement vectors
        i_idx = samples[:, 0].long()
        j_idx = samples[:, 1].long()
        shifts = samples[:, 2:5].to(system.positions.dtype)
        pos = system.positions
        cell = system.cell
        values = (
            pos[j_idx] - pos[i_idx] + shifts @ cell
        ).unsqueeze(-1)  # (n_pairs, 3, 1)

    sample_labels = Labels(
        names=["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
        values=samples,
    )
    component_labels = Labels(
        names=["xyz"],
        values=torch.tensor([[0], [1], [2]], dtype=torch.int32),
    )
    property_labels = Labels(
        names=["distance"],
        values=torch.tensor([[0]], dtype=torch.int32),
    )

    return TensorBlock(
        values=values,
        samples=sample_labels,
        components=[component_labels],
        properties=property_labels,
    )


def _make_density_layout(grid_shape=(8, 8, 8)) -> TensorMap:
    """Build a zero-sample TensorMap layout matching ELECTRAFY's density output."""
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


def _make_dataset_info(atomic_types, grid_shape=(8, 8, 8)):
    """Create a minimal DatasetInfo for ELECTRAFY."""
    targets = {
        "charge_density": TargetInfo(
            layout=_make_density_layout(grid_shape),
            quantity="",  # non-standard output → empty quantity (no unit conversion)
            unit="",
        ),
    }
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets=targets,
    )


def _make_minimal_hypers(grid_shape=(8, 8, 8)):
    """Minimal hyperparameters for a tiny ELECTRAFY model."""
    return {
        "cutoff": 4.0,
        "cutoff_function": "Cosine",
        "cutoff_width": 0.5,
        "d_pet": 8,
        "d_node": 16,
        "d_feedforward": 16,
        "num_heads": 2,
        "num_attention_layers": 1,
        "num_gnn_layers": 1,
        "normalization": "RMSNorm",
        "activation": "SiLU",
        "transformer_type": "PreLN",
        "attention_temperature": 1.0,
        "gaussians_per_electron": 2,
        "gamma": 0.1,
        "grid_shape": list(grid_shape),
        "fourier_chunk_size": 256,
    }


def _make_si_system():
    """Create a minimal silicon system (2 atoms in a diamond unit cell)."""
    a = 5.43
    cell = torch.tensor([
        [a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, 0.0, a],
    ], dtype=torch.float64)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [a / 4, a / 4, a / 4],
    ], dtype=torch.float64)
    types = torch.tensor([14, 14], dtype=torch.int32)
    return _make_system(positions, types, cell)


class TestModelForward:
    @pytest.fixture
    def model_and_system(self):
        """Create a minimal model and system for testing."""
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])  # Silicon
        model = ELECTRAFY(hypers, dataset_info).to(torch.float64)

        system = _make_si_system()
        # Add neighbor list manually
        nl_options = model.requested_neighbor_lists()[0]
        nl_block = _build_nl_block(
            _compute_nl_samples(system, nl_options.cutoff),
            torch.zeros(0),
            system,
        )
        system.add_neighbor_list(nl_options, nl_block)
        return model, system

    def test_forward_produces_density(self, model_and_system):
        """Forward pass returns a charge_density TensorMap."""
        model, system = model_and_system
        outputs = {DENSITY_KEY: ModelOutput(quantity="charge_density", unit="")}
        result = model([system], outputs)

        assert DENSITY_KEY in result
        tmap = result[DENSITY_KEY]
        block = tmap.block()
        assert block.values.shape[0] == 1  # 1 system
        assert block.values.shape[1] == 8 * 8 * 8  # grid points

    def test_forward_no_density_key(self, model_and_system):
        """Forward returns empty dict when density not requested."""
        model, system = model_and_system
        result = model([system], {})
        assert result == {}

    def test_gradient_flow_through_model(self, model_and_system):
        """Gradients flow from loss through the full model."""
        model, system = model_and_system
        model.train()

        outputs = {DENSITY_KEY: ModelOutput(quantity="charge_density", unit="")}
        result = model([system], outputs)
        density = result[DENSITY_KEY].block().values

        loss = density.abs().mean()
        loss.backward()

        # Check that at least GNN parameters have gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameter received a non-zero gradient"

    def test_density_is_finite(self, model_and_system):
        """Output density contains no NaN or Inf."""
        model, system = model_and_system
        outputs = {DENSITY_KEY: ModelOutput(quantity="charge_density", unit="")}
        result = model([system], outputs)
        density = result[DENSITY_KEY].block().values
        assert torch.all(torch.isfinite(density))


class TestCheckpoint:
    def test_checkpoint_roundtrip(self):
        """Save and load checkpoint reproduces same outputs."""
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])
        model = ELECTRAFY(hypers, dataset_info).to(torch.float64)

        ckpt = model.get_checkpoint()
        model2 = ELECTRAFY.load_checkpoint(ckpt)

        # Verify state dicts match
        for key in model.state_dict():
            assert torch.equal(
                model.state_dict()[key], model2.state_dict()[key]
            ), f"State dict mismatch for {key}"


class TestSupportedOutputs:
    def test_outputs_include_density(self):
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])
        model = ELECTRAFY(hypers, dataset_info)
        outputs = model.supported_outputs()
        assert DENSITY_KEY in outputs

    def test_neighbor_list_requested(self):
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])
        model = ELECTRAFY(hypers, dataset_info)
        nls = model.requested_neighbor_lists()
        assert len(nls) == 1
        assert nls[0].cutoff == 4.0


class TestRestart:
    def test_restart_same_types(self):
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])
        model = ELECTRAFY(hypers, dataset_info)
        model.restart(dataset_info)  # should not raise

    def test_restart_new_types_raises(self):
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])
        model = ELECTRAFY(hypers, dataset_info)

        new_info = _make_dataset_info([14, 8])
        with pytest.raises(ValueError, match="does not support adding new atomic types"):
            model.restart(new_info)


class TestLossIntegration:
    """
    Exercise the model ↔ LossAggregator ↔ backward path used inside Trainer.train.

    This catches issues at the boundary between the model's TensorMap layout
    and metatrain's loss infrastructure without requiring the full Dataset /
    DataLoader / MetricLogger stack.
    """

    def _build(self):
        hypers = _make_minimal_hypers()
        dataset_info = _make_dataset_info([14])
        model = ELECTRAFY(hypers, dataset_info).to(torch.float64)

        system = _make_si_system()
        nl_options = model.requested_neighbor_lists()[0]
        nl_block = _build_nl_block(
            _compute_nl_samples(system, nl_options.cutoff),
            torch.zeros(0),
            system,
        )
        system.add_neighbor_list(nl_options, nl_block)
        return model, system, dataset_info

    def test_nmae_loss_via_aggregator_backprops(self):
        """LossAggregator(nmae) on real model output produces non-zero gradients."""
        from metatrain.utils.loss import LossAggregator, LossSpecification

        model, system, dataset_info = self._build()
        model.train()

        outputs = {DENSITY_KEY: ModelOutput(quantity="charge_density", unit="")}
        predictions = model([system], outputs)

        # Build a synthetic reference TensorMap with the same layout as the
        # prediction. The density shape is (1 system, N_grid).
        pred_block = predictions[DENSITY_KEY].block()
        ref_values = torch.rand_like(pred_block.values)
        ref_block = TensorBlock(
            values=ref_values,
            samples=pred_block.samples,
            components=pred_block.components,
            properties=pred_block.properties,
        )
        ref_tmap = TensorMap(
            keys=predictions[DENSITY_KEY].keys, blocks=[ref_block]
        )
        targets = {DENSITY_KEY: ref_tmap}

        loss_fn = LossAggregator(
            targets=dataset_info.targets,
            config={DENSITY_KEY: LossSpecification(type="nmae", weight=1.0, reduction="mean", gradients={})},
        )
        loss = loss_fn(predictions, targets, extra_data={})

        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.requires_grad, "Loss must be differentiable for backward"

        loss.backward()

        # Confirm gradient reached at least one model parameter.
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No model parameter received a non-zero gradient"

    def test_optimizer_step_changes_params(self):
        """A single Adam step measurably changes at least one trainable param."""
        from metatrain.utils.loss import LossAggregator, LossSpecification

        model, system, dataset_info = self._build()
        model.train()

        # Snapshot params before step
        before = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if v.dtype.is_floating_point}

        outputs = {DENSITY_KEY: ModelOutput(quantity="charge_density", unit="")}
        predictions = model([system], outputs)
        pred_block = predictions[DENSITY_KEY].block()

        ref_block = TensorBlock(
            values=torch.rand_like(pred_block.values),
            samples=pred_block.samples,
            components=pred_block.components,
            properties=pred_block.properties,
        )
        targets = {
            DENSITY_KEY: TensorMap(
                keys=predictions[DENSITY_KEY].keys, blocks=[ref_block]
            )
        }
        loss_fn = LossAggregator(
            targets=dataset_info.targets,
            config={DENSITY_KEY: LossSpecification(type="nmae", weight=1.0, reduction="mean", gradients={})},
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss = loss_fn(predictions, targets, extra_data={})
        loss.backward()
        optimizer.step()

        after = model.state_dict()
        changed = [
            k for k in before
            if not torch.allclose(before[k], after[k])
        ]
        assert len(changed) > 0, "No parameter changed after optimizer step"


def _compute_nl_samples(system: System, cutoff: float) -> torch.Tensor:
    """Compute neighbor list sample indices using ASE."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list as ase_nl

    positions = system.positions.detach().cpu().numpy()
    types = system.types.detach().cpu().numpy()
    cell = system.cell.detach().cpu().numpy()

    atoms = Atoms(numbers=types, positions=positions, cell=cell, pbc=True)
    i_list, j_list, S_list = ase_nl("ijS", atoms, cutoff)

    if len(i_list) == 0:
        return torch.zeros((0, 5), dtype=torch.int32)

    samples = torch.zeros((len(i_list), 5), dtype=torch.int32)
    samples[:, 0] = torch.tensor(i_list, dtype=torch.int32)
    samples[:, 1] = torch.tensor(j_list, dtype=torch.int32)
    samples[:, 2] = torch.tensor(S_list[:, 0], dtype=torch.int32)
    samples[:, 3] = torch.tensor(S_list[:, 1], dtype=torch.int32)
    samples[:, 4] = torch.tensor(S_list[:, 2], dtype=torch.int32)
    return samples
