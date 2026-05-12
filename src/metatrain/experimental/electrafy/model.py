"""
ELECTRAFY with PET backbone — metatrain ModelInterface implementation.

Periodic charge density prediction via Gaussian splatting + analytic Fourier
transform, using PET's CartesianTransformer as the GNN backbone.

Pipeline
--------
1. PET feature extraction (residual mode):
   systems_to_batch -> NEF tensors -> num_gnn_layers x CartesianTransformer
   -> node_features_list, edge_features_list

2. Dyadic neighbor aggregation (ELECTRAFY Appendix B, per GNN layer):
   S^[l] (N, C), V^[l] (N, C, 3), T^[l] (N, C, 3, 3)
   + cross-layer softmax aggregation -> S, V, T

3. Gaussian density head:
   (S, V, T) + atom positions + valence -> weights, centers, covs
   n_gauss(a) = M * ZVAL(a)

4. Fourier density:
   rho_hat(G) = sum_j w_j exp(-1/2 G^T Sigma_j G) exp(-i G . mu_j)
   IFFT + volume prefactor + electron-count normalization -> rho(r)

Output
------
``"charge_density"``: per-structure TensorMap.
  - Samples: system indices.
  - Properties: flattened grid points (0 … N1*N2*N3-1).
  - Values: ``(sum_i N1_i*N2_i*N3_i,)`` — the concatenated densities of all
    systems in the batch. When per-system grids vary, the model also writes
    ``self._last_density_offsets`` (prefix sum, length ``n_systems + 1``) so
    callers can slice each structure's density back out.
  - Grid shape: see :ref:`grid-shape-resolution`.

.. _grid-shape-resolution:

Grid shape resolution
---------------------
Two channels feed the Fourier head's output grid:

1. **Per-batch override (canonical training path).** Call
   :py:meth:`ELECTRAFY.set_override_grid_shapes` with one
   ``(N1, N2, N3)`` per system before each forward; the model uses those
   exact shapes for the IFFT and emits a variable-size flat output.

   The metatrain trainer's ``_apply_grid_shapes`` helper does this for
   you: it reads ``extra_data["grid_shape"]`` (a TensorMap with shape
   ``(n_systems, 3)`` emitted by
   :class:`~metatrain.experimental.electrafy.modules.cache_dataset.CachedChgcarDataset`)
   and installs the override automatically before forward, then clears it
   after.

2. **Fallback hyper.** When no override is set,
   ``ModelHypers.grid_shape`` is broadcast to every system in the batch.
   This is the inference / smoke-test path and is the only knob without
   which the model is not a valid module.

There is no other channel: ``metatomic.torch.System`` does not carry a
grid field, and ``model.forward(systems, outputs)`` has no per-system
grid-shape parameter, so the stash-on-self override is the only way to
hand per-system native grids to the Fourier head.

References
----------
- ELECTRAFY: "Global Plane Waves From Local Gaussians: Periodic Charge
  Densities in a Blink", Elsborg et al. (2026), arXiv:2501.09146
- PET: "Equivariant Point-Edge Transformers", Bochkarev et al. (2024),
  arXiv:2305.19302
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo

from metatrain.pet.modules.structures import systems_to_batch
from metatrain.pet.modules.transformer import CartesianTransformer

from .documentation import ModelHypers
from .modules.dyadic_aggregation import DyadicAggregation
from .modules.fourier_density import periodic_density_from_gaussians
from .modules.gaussian_density import GaussianDensityHead
from .modules.valence import ZVAL_LOOKUP, MAX_ZVAL, build_zv_index_lookup

#: Name of the charge-density output as registered in metatrain.
DENSITY_KEY = "charge_density"


class ELECTRAFY(ModelInterface[ModelHypers]):
    """
    ELECTRAFY periodic charge density model with PET backbone.

    :param hypers: Model hyperparameters; see :class:`ModelHypers`.
    :param dataset_info: Dataset metadata (atomic types, targets).
    """

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "https://arxiv.org/abs/2501.09146",  # ELECTRAFY
                "https://arxiv.org/abs/2305.19302",  # PET
            ]
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        # -- Cached hypers --
        self.cutoff = float(hypers["cutoff"])
        self.cutoff_function = hypers["cutoff_function"]
        self.cutoff_width = float(hypers["cutoff_width"])
        self.d_pet = hypers["d_pet"]
        self.d_node = hypers["d_node"]
        self.num_gnn_layers = hypers["num_gnn_layers"]
        self.gaussians_per_electron = hypers["gaussians_per_electron"]
        self.fourier_chunk_size = hypers["fourier_chunk_size"]
        self.use_triton_fourier = bool(hypers.get("use_triton_fourier", False))
        self.grid_shape: Tuple[int, int, int] = tuple(hypers["grid_shape"])  # type: ignore[assignment]

        self.atomic_types = dataset_info.atomic_types
        num_species = len(self.atomic_types)

        # C = M * max_ZVAL (total feature channels per atom)
        self.n_channels = self.gaussians_per_electron * MAX_ZVAL

        # -- (Z, v_a) admissible-set embeddings (Appendix B.3) --
        # Each element-valence pair gets its own embedding row, so the model
        # can distinguish (e.g.) Mn vs Mn_pv vs Mn_sv. For dataset_info that
        # doesn't carry per-atom valence annotations, we look up the canonical
        # default at forward time (matches what VASP/MP wrote into the CHGCAR).
        self._zv_index = build_zv_index_lookup()
        n_zv_embeddings = self._zv_index.num_embeddings

        # -- PET GNN layers (residual featurization) --
        self.node_embedders = torch.nn.ModuleList(
            [
                torch.nn.Embedding(n_zv_embeddings, self.d_node)
                for _ in range(self.num_gnn_layers)
            ]
        )
        self.edge_embedder = torch.nn.Embedding(n_zv_embeddings, self.d_pet)

        # Note: CartesianTransformer's internal `neighbor_embedder` is also
        # indexed by the (Z, v) embedding index (same as our top-level
        # node/edge embedders), so it must be sized to `n_zv_embeddings`.
        self.gnn_layers = torch.nn.ModuleList(
            [
                CartesianTransformer(
                    cutoff=self.cutoff,
                    cutoff_width=self.cutoff_width,
                    d_model=self.d_pet,
                    n_head=hypers["num_heads"],
                    dim_node_features=self.d_node,
                    dim_feedforward=hypers["d_feedforward"],
                    n_layers=hypers["num_attention_layers"],
                    norm=hypers["normalization"],
                    activation=hypers["activation"],
                    attention_temperature=hypers["attention_temperature"],
                    transformer_type=hypers["transformer_type"],
                    n_atomic_species=n_zv_embeddings,
                    is_first=(i == 0),
                )
                for i in range(self.num_gnn_layers)
            ]
        )

        # -- Dyadic aggregation --
        self.dyadic_aggregation = DyadicAggregation(
            n_gnn_layers=self.num_gnn_layers,
            d_node=self.d_node,
            d_edge=self.d_pet,
            n_channels=self.n_channels,
            mlp_hidden=hypers.get("dyadic_mlp_hidden", 0),
        )

        # -- Gaussian density head --
        self.gaussian_head = GaussianDensityHead(
            n_channels=self.n_channels,
            gamma=hypers["gamma"],
            mlp_hidden=hypers.get("head_mlp_hidden", 64),
        )

        # -- Species lookup: atomic_number -> (Z, v) embedding index --
        # Required by `systems_to_batch` which expects a buffer mapping Z to
        # an integer index used by both node_embedder and edge_embedder.
        # Without per-atom valence annotations we route every Z through its
        # canonical (Z, v_canonical) embedding row.
        max_z = max(self.atomic_types) + 1
        self.register_buffer(
            "species_to_species_index",
            torch.full((max_z,), -1, dtype=torch.long),
        )
        for z in self.atomic_types:
            try:
                self.species_to_species_index[z] = self._zv_index.canonical_index_for(z)
            except KeyError:
                raise ValueError(
                    f"no admissible (Z, v) entry for atomic number {z}; "
                    f"add to ADMISSIBLE_ZVALS in modules/valence.py"
                )

        # -- Valence lookup: atomic_number -> ZVAL --
        self.register_buffer("zval_lookup", ZVAL_LOOKUP.clone())

        # -- Optional surgical compile of the GNN backbone --
        # Each ``CartesianTransformer`` is pure-tensor and shape-stable except
        # for the (n_atoms, n_edges) dynamic dimensions — `dynamic=True` keeps
        # those fluid and emits one shared graph per layer. Together with the
        # `_fourier_chunk_body` compile in modules/fourier_density.py this
        # covers the two heaviest blocks of the forward without exposing the
        # Python-int specializations elsewhere.
        #
        # Set ELECTRAFI_COMPILE_GNN=0 to disable (debugging / eager comparison).
        # Set ELECTRAFI_REPLACE_SILU=1 to also apply PET's
        # `replace_silu_modules` helper (some versions of inductor compile
        # PET's SiLU more cleanly after this swap).
        if os.environ.get("ELECTRAFI_REPLACE_SILU", "0") != "0":
            try:
                from metatrain.pet.modules.utilities import replace_silu_modules
                for layer in self.gnn_layers:
                    replace_silu_modules(layer)
            except (ImportError, AttributeError):
                pass  # PET helper not available; skip silently
        if os.environ.get("ELECTRAFI_COMPILE_GNN", "0") != "0":
            for i in range(len(self.gnn_layers)):
                self.gnn_layers[i] = torch.compile(  # type: ignore[assignment]
                    self.gnn_layers[i], dynamic=True
                )

        # -- Neighbor list request --
        self.requested_nl = NeighborListOptions(
            cutoff=self.cutoff,
            full_list=True,
            strict=True,
        )

        # -- Metatrain output registry --
        n_grid = int(self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2])
        self.outputs: Dict[str, ModelOutput] = {
            DENSITY_KEY: ModelOutput(
                quantity="charge_density",
                unit="electrons/angstrom^3",
                per_atom=False,
            )
        }

        # Pre-build labels that are reused across forward calls
        self._grid_property_labels: Optional[Labels] = None
        self._n_grid = n_grid

        # Per-batch grid override -- the canonical channel for per-system
        # native NGXF/NGYF/NGZF grids during training.
        #
        # ``self.grid_shape`` (the ModelHypers value) is only a fallback used
        # when this attribute is None at forward time. The metatrain trainer's
        # ``_apply_grid_shapes`` helper sets this from the batch's
        # ``extra_data["grid_shape"]`` (emitted by CachedChgcarDataset) before
        # every forward and clears it after, so the hyper is never read on the
        # training path -- only by ad-hoc inference paths that don't supply
        # per-batch grids.
        #
        # Set via :meth:`set_override_grid_shapes`, cleared via
        # :meth:`clear_override_grid_shapes`. Length must equal ``len(systems)``.
        self._override_grid_shapes: Optional[List[Tuple[int, int, int]]] = None

        # Per-batch electron-count override. When set, ``forward`` uses these
        # values (one per system, in batch order) for the Eq-11 electron-count
        # normalization instead of the canonical ``ZVAL_LOOKUP`` sum. This is
        # the right thing to use for CHGCARs computed with non-canonical
        # pseudopotentials (e.g. Fe_pv vs Fe), and is also more numerically
        # robust because the CHGCAR-integrated electron count is exact
        # rather than reconstructed from an element->valence table.
        self._override_n_electrons: Optional[List[float]] = None

        # Populated at the end of each forward; (n_systems+1,) prefix-sum offsets
        # into the flat density tensor so micro-batched callers can slice the
        # per-system density when grids vary.
        self._last_density_offsets: Optional[torch.Tensor] = None

    def set_override_grid_shapes(self, shapes: List[Tuple[int, int, int]]) -> None:
        """Install per-system output grid shapes for the next ``forward()`` call.

        This is the **canonical mechanism** for handing per-system native
        NGXF/NGYF/NGZF grids to the Fourier head. ``metatomic.torch.System``
        has no grid field and ``forward(systems, outputs)`` has no
        per-system grid parameter, so the only way to vary the output grid
        across structures in a batch is via this stash-on-self override.

        The override **takes precedence over** ``ModelHypers.grid_shape``
        (the fallback). When set, the model uses ``shapes[i]`` for system
        ``i`` and emits a concatenated flat density of total length
        ``sum_i N1_i*N2_i*N3_i``; ``self._last_density_offsets`` is the
        prefix sum that lets callers slice per-system densities back out.

        The metatrain trainer calls this for you (see
        ``_apply_grid_shapes`` in the trainer module) from each batch's
        ``extra_data["grid_shape"]``. You only need to call it directly for
        ad-hoc inference / debugging when the batch is constructed by hand.

        Always pair with :py:meth:`clear_override_grid_shapes` in a
        ``finally:`` to avoid carrying stale per-batch state into the next
        unrelated forward.

        :param shapes: One ``(N1, N2, N3)`` tuple per system in the batch
            (must equal ``len(systems)`` at forward time).
        """
        self._override_grid_shapes = [tuple(s) for s in shapes]  # type: ignore[assignment]

    def clear_override_grid_shapes(self) -> None:
        """Drop the per-batch grid override; subsequent forwards will use
        the ``ModelHypers.grid_shape`` fallback."""
        self._override_grid_shapes = None

    def set_override_n_electrons(self, n_electrons: List[float]) -> None:
        """Set per-system electron counts for the next forward call.

        :param n_electrons: One float per system in the batch — the integrated
            valence electron count for that structure (typically
            ``∫ρ_ref dV``). Used for Eq-11 normalization.
        """
        self._override_n_electrons = [float(n) for n in n_electrons]

    def clear_override_n_electrons(self) -> None:
        self._override_n_electrons = None

    # ------------------------------------------------------------------
    # ModelInterface API
    # ------------------------------------------------------------------

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def restart(self, dataset_info: DatasetInfo) -> "ELECTRAFY":
        # ELECTRAFY does not support new atomic types or targets at restart.
        new_types = [
            t for t in dataset_info.atomic_types if t not in self.atomic_types
        ]
        if new_types:
            raise ValueError(
                f"ELECTRAFY does not support adding new atomic types: {new_types}"
            )
        self.dataset_info = dataset_info
        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Predict periodic charge density for each system.

        :param systems: List of metatomic System objects with precomputed
            neighbor lists at ``self.cutoff``.
        :param outputs: Requested output quantities.  If ``"charge_density"``
            is not present the function returns an empty dict.
        :param selected_atoms: Unused (density is always per-structure).
        :return: ``{"charge_density": TensorMap}`` with shape
            ``(n_systems, N1*N2*N3)``.
        """
        if DENSITY_KEY not in outputs:
            return {}

        device = systems[0].device
        n_systems = len(systems)

        # ---- Stage 0: batch prep ----
        (
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
            system_indices,
            _neighbor_atom_indices,
            sample_labels,
            species,
        ) = systems_to_batch(
            systems,
            self.requested_nl,
            self.atomic_types,
            self.species_to_species_index,
            self.cutoff_function,
            self.cutoff_width,
        )

        # ---- Stage 1: PET feature extraction ----
        node_features_list, edge_features_list = self._extract_features(
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
        )

        # ---- Stage 2: Dyadic aggregation ----
        S, V, T = self.dyadic_aggregation(
            node_features_list, edge_features_list, edge_vectors, padding_mask
        )

        # ---- Stages 3-4: Per-system density ----
        positions_all = torch.cat([sys.positions for sys in systems], dim=0)
        # ``species`` may arrive as float (recent metatrain casts System.types
        # to the model dtype for downstream attention plumbing); cast back to
        # long for the integer-index lookup.
        n_gaussians_per_atom = self.zval_lookup[species.long()] * self.gaussians_per_electron

        # Resolve per-system output grid shapes (paper convention: native
        # NGXF/NGYF/NGZF). When ``set_override_grid_shapes`` has been called,
        # those shapes are used; otherwise fall back to ``self.grid_shape``.
        if self._override_grid_shapes is not None:
            if len(self._override_grid_shapes) != n_systems:
                raise ValueError(
                    f"override grid shapes ({len(self._override_grid_shapes)}) "
                    f"!= n_systems ({n_systems})"
                )
            per_system_shapes = self._override_grid_shapes
        else:
            per_system_shapes = [self.grid_shape] * n_systems

        if self._override_n_electrons is not None and len(
            self._override_n_electrons
        ) != n_systems:
            raise ValueError(
                f"override n_electrons ({len(self._override_n_electrons)}) "
                f"!= n_systems ({n_systems})"
            )

        density_rows: List[torch.Tensor] = []
        atom_offset = 0
        for sys_idx, system in enumerate(systems):
            n_atoms = len(system)
            atom_slice = slice(atom_offset, atom_offset + n_atoms)

            weights, centers, covs, _ = self.gaussian_head(
                S[atom_slice], V[atom_slice], T[atom_slice],
                positions_all[atom_slice],
                n_gaussians_per_atom[atom_slice],
            )

            if self._override_n_electrons is not None:
                n_elec = self._override_n_electrons[sys_idx]
            else:
                n_elec = float(
                    (n_gaussians_per_atom[atom_slice].sum()
                     / self.gaussians_per_electron).item()
                )

            rho = periodic_density_from_gaussians(
                weights=weights,
                centers=centers,
                covs=covs,
                cell=system.cell,
                grid_shape=per_system_shapes[sys_idx],
                n_electrons=n_elec,
                chunk_size=self.fourier_chunk_size,
                use_triton=self.use_triton_fourier,
            )
            density_rows.append(rho.reshape(-1))

        # Variable-grid handling: when all systems in this batch share the
        # same grid we keep the legacy (n_systems, grid_size) stacked layout
        # — back-compat with batch=1 callers that read
        # `out.block().values.reshape(-1)`. When grids differ, we concatenate
        # into a flat (sum_grid_sizes,) tensor and stash per-system offsets
        # on `self._last_density_offsets` so micro-batched callers can slice
        # densities per system. The TensorMap surface stays a single block.
        sizes = [row.numel() for row in density_rows]
        unique_sizes = set(sizes)
        if len(unique_sizes) == 1:
            density_values = torch.stack(density_rows, dim=0)
            self._last_density_offsets = torch.arange(
                0, n_systems * sizes[0] + 1, sizes[0],
                device=device, dtype=torch.long,
            )
        else:
            flat = torch.cat(density_rows, dim=0)
            density_values = flat.unsqueeze(0)  # (1, sum_grid_sizes)
            offsets = torch.zeros(n_systems + 1, device=device, dtype=torch.long)
            for i, s in enumerate(sizes):
                offsets[i + 1] = offsets[i] + s
            self._last_density_offsets = offsets

        # ---- Wrap in TensorMap ----
        tmap = self._wrap_density_tmap(density_values, density_values.shape[0], device)
        return {DENSITY_KEY: tmap}

    # ------------------------------------------------------------------
    # Checkpoint helpers (required by ModelInterface)
    # ------------------------------------------------------------------

    def export(self, metadata: Optional[ModelMetadata] = None) -> Any:
        raise NotImplementedError(
            "ELECTRAFY does not support metatomic export yet. "
            "Use get_checkpoint() / load_checkpoint() for saving and loading."
        )

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.state_dict(),
            "hypers": self.hypers,
            "dataset_info": self.dataset_info,
        }

    @classmethod
    def load_checkpoint(
        cls, checkpoint: Dict[str, Any], context: str = "restart"
    ) -> "ELECTRAFY":
        model = cls(
            hypers=checkpoint["hypers"],
            dataset_info=checkpoint["dataset_info"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        # v1 is the only version; nothing to upgrade yet.
        return checkpoint

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        element_indices_nodes: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_distances: torch.Tensor,
        padding_mask: torch.Tensor,
        reverse_neighbor_index: torch.Tensor,
        cutoff_factors: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run PET layers in residual featurization mode (per PET's implementation)."""
        use_manual_attention = edge_vectors.requires_grad and self.training
        node_features_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []

        input_edge_embeddings = self.edge_embedder(element_indices_neighbors)
        for node_embedder, gnn_layer in zip(self.node_embedders, self.gnn_layers):
            input_node_embeddings = node_embedder(element_indices_nodes)
            out_node, out_edge = gnn_layer(
                input_node_embeddings,
                input_edge_embeddings,
                element_indices_neighbors,
                edge_vectors,
                padding_mask,
                edge_distances,
                cutoff_factors,
                use_manual_attention,
            )
            node_features_list.append(out_node)
            edge_features_list.append(out_edge)

            new_msgs = out_edge.reshape(
                out_edge.shape[0] * out_edge.shape[1], out_edge.shape[2]
            )[reverse_neighbor_index].reshape(out_edge.shape)
            input_edge_embeddings = 0.5 * (input_edge_embeddings + new_msgs)

        return node_features_list, edge_features_list

    def _wrap_density_tmap(
        self,
        density_values: torch.Tensor,
        n_systems: int,
        device: torch.device,
    ) -> TensorMap:
        """
        Pack a (n_systems, N_grid) density tensor into a TensorMap.

        Layout:
          - keys: ``Labels.single()``
          - samples: [("system", 0), ("system", 1), ...]
          - components: []
          - properties: [("grid_point", 0), ..., ("grid_point", N_grid-1)]
          - values: (n_systems, N_grid)
        """
        n_grid = int(density_values.shape[1])
        property_labels = Labels(
            names=["grid_point"],
            values=torch.arange(n_grid, device=device).unsqueeze(1),
        )

        sample_vals = torch.arange(n_systems, device=device, dtype=torch.int32).unsqueeze(1)
        sample_labels = Labels(names=["system"], values=sample_vals)

        block = TensorBlock(
            values=density_values,
            samples=sample_labels,
            components=[],
            properties=property_labels,
        )
        return TensorMap(keys=Labels.single().to(device), blocks=[block])
