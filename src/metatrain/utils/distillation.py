"""
Multi-teacher knowledge distillation utilities.

This module provides tools for distilling knowledge from multiple pre-trained
teacher models into a student model's embedding space.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from .io import load_model


class TeacherModelManager:
    """
    Manages multiple pre-trained teacher models for knowledge distillation.

    Handles loading teacher models, extracting embeddings, and optional
    projection alignment when teacher embedding dimensions differ.

    :param teacher_configs: List of teacher configurations. Each config dict contains:
        - ``path``: Path to teacher checkpoint or exported model.
        - ``embedding_key``: Which embedding to extract (default: "features").
        - ``weight``: Relative weight of this teacher (default: 1.0).
        - ``projection_dim``: If not None, project embeddings to this dimension.
    :param device: Device to load teachers on.
    :param dtype: Data type for teacher models.
    """

    def __init__(
        self,
        teacher_configs: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if device is None:
            device = torch.device("cpu")
        self.teachers: List[torch.nn.Module] = []
        self.embedding_keys: List[str] = []
        self.weights: List[float] = []
        self.projections: List[Optional[torch.nn.Module]] = []
        self._projection_dims: List[Optional[int]] = []

        self.device = device
        self.dtype = dtype

        for config in teacher_configs:
            # Load teacher model
            path = config["path"]
            if isinstance(path, str):
                path = Path(path)

            teacher = load_model(path)
            teacher = teacher.eval()
            teacher = teacher.to(device=device, dtype=dtype)

            # Freeze teacher parameters
            for param in teacher.parameters():
                param.requires_grad = False

            self.teachers.append(teacher)
            self.embedding_keys.append(config.get("embedding_key", "features"))
            self.weights.append(config.get("weight", 1.0))
            self._projection_dims.append(config.get("projection_dim"))

            # Projection will be built lazily when dimensions are known
            self.projections.append(None)

    @property
    def num_teachers(self) -> int:
        """Return the number of teacher models."""
        return len(self.teachers)

    @torch.no_grad()
    def get_teacher_embeddings(
        self,
        systems: List[System],
    ) -> List[torch.Tensor]:
        """
        Extract embeddings from all teacher models.

        :param systems: Input systems to get embeddings for.
        :return: List of embedding tensors, one per teacher.
        """
        embeddings = []

        for teacher, emb_key in zip(self.teachers, self.embedding_keys, strict=True):
            # Request the embedding output from the teacher
            outputs = {emb_key: ModelOutput(per_atom=True)}
            predictions = teacher(systems, outputs)

            if emb_key not in predictions:
                raise ValueError(
                    f"Teacher model does not output '{emb_key}'. "
                    f"Available outputs: {list(predictions.keys())}"
                )

            emb = predictions[emb_key].block().values
            embeddings.append(emb)

        return embeddings

    def build_projections(
        self,
        teacher_dims: List[int],
        target_dim: int,
    ) -> None:
        """
        Build projection layers to align teacher embeddings to a target space.

        :param teacher_dims: Embedding dimensions of each teacher.
        :param target_dim: Target dimension to project to.
        """
        for i, (t_dim, proj_dim) in enumerate(
            zip(teacher_dims, self._projection_dims, strict=True)
        ):
            if proj_dim is not None:
                self.projections[i] = torch.nn.Linear(t_dim, proj_dim)
            elif t_dim != target_dim:
                # Auto-project if dimensions don't match
                self.projections[i] = torch.nn.Linear(t_dim, target_dim)
            else:
                self.projections[i] = torch.nn.Identity()

            # Move to device
            proj = self.projections[i]
            if proj is not None:
                self.projections[i] = proj.to(device=self.device, dtype=self.dtype)

    def project_embeddings(
        self,
        embeddings: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply projections to teacher embeddings.

        :param embeddings: List of teacher embeddings.
        :return: List of projected embeddings.
        """
        projected = []
        for emb, proj in zip(embeddings, self.projections, strict=True):
            if proj is not None:
                projected.append(proj(emb))
            else:
                projected.append(emb)
        return projected


class TeacherEmbeddingCallable:
    """
    CollateFn callable that computes teacher embeddings and adds them to extra_data.

    This runs during data collation to precompute teacher embeddings, which can
    then be used by distillation losses.

    :param teacher_manager: TeacherModelManager with loaded teacher models.
    :param project: Whether to apply projections to embeddings.
    """

    def __init__(
        self,
        teacher_manager: TeacherModelManager,
        project: bool = False,
    ):
        self.teacher_manager = teacher_manager
        self.project = project

    def __call__(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Compute teacher embeddings and add to extra_data.

        Adds ``teacher_embeddings_0``, ``teacher_embeddings_1``, etc. to extra_data.

        :param systems: List of systems in the batch.
        :param targets: Dictionary of target TensorMaps.
        :param extra_data: Dictionary for extra data (will be modified).
        :return: Tuple of (systems, targets, extra_data) with teacher embeddings.
        """
        if extra_data is None:
            extra_data = {}

        # Get teacher embeddings
        teacher_embeddings = self.teacher_manager.get_teacher_embeddings(systems)

        # Optionally project
        if self.project:
            teacher_embeddings = self.teacher_manager.project_embeddings(
                teacher_embeddings
            )

        # Add to extra_data as TensorMaps
        for i, emb in enumerate(teacher_embeddings):
            extra_data[f"teacher_embeddings_{i}"] = self._to_tensormap(emb, systems)

        return systems, targets, extra_data

    def _to_tensormap(
        self,
        embeddings: torch.Tensor,
        systems: List[System],
    ) -> TensorMap:
        """
        Convert embedding tensor to TensorMap.

        :param embeddings: Embedding tensor of shape [n_atoms, n_features].
        :param systems: List of systems (for metadata).
        :return: TensorMap containing the embeddings.
        """
        n_atoms = embeddings.shape[0]
        n_features = embeddings.shape[-1]

        return TensorMap(
            keys=mts.Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=embeddings,
                    samples=mts.Labels(
                        names=["atom"],
                        values=torch.arange(n_atoms).unsqueeze(-1),
                    ),
                    components=[],
                    properties=mts.Labels(
                        names=["feature"],
                        values=torch.arange(n_features).unsqueeze(-1),
                    ),
                )
            ],
        )


def create_teacher_manager(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> TeacherModelManager:
    """
    Create a TeacherModelManager from a configuration dictionary.

    :param config: Configuration with "teachers" key containing list of teacher configs.
    :param device: Device to load teachers on.
    :param dtype: Data type for teacher models.
    :return: Configured TeacherModelManager.
    """
    if "teachers" not in config:
        raise ValueError("Configuration must contain 'teachers' key")

    return TeacherModelManager(
        teacher_configs=config["teachers"],
        device=device,
        dtype=dtype,
    )
