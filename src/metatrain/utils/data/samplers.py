"""Batch samplers for atom-count-aware batching."""

import logging
import math
from typing import Iterator, List

import numpy as np
import torch
import torch.utils.data


logger = logging.getLogger(__name__)


def _get_num_atoms(dataset: torch.utils.data.Dataset, i: int) -> int:
    """Return atom count for sample ``i``, resolving ``Subset`` wrappers."""
    if isinstance(dataset, torch.utils.data.Subset):
        return _get_num_atoms(dataset.dataset, dataset.indices[i])
    if hasattr(dataset, "get_num_atoms"):
        return dataset.get_num_atoms(i)
    raise TypeError(
        f"Dataset of type {type(dataset).__name__} does not support "
        "get_num_atoms(). Only MemmapDataset (and Subsets thereof) is "
        "currently supported with max_atoms_per_batch."
    )


def _greedy_pack(
    indices: List[int],
    atom_counts: List[int],
    max_atoms: int,
) -> List[List[int]]:
    """Greedily pack ``indices`` into batches where total atoms <= ``max_atoms``.

    Single structures that alone exceed ``max_atoms`` are skipped with a warning.
    """
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_atoms = 0

    for idx, n in zip(indices, atom_counts):
        if n > max_atoms:
            logger.warning(
                f"Structure {idx} has {n} atoms which exceeds max_atoms_per_batch "
                f"({max_atoms}). Skipping this structure."
            )
            continue
        if current_atoms + n > max_atoms and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_atoms = 0
        current_batch.append(idx)
        current_atoms += n

    if current_batch:
        batches.append(current_batch)

    return batches


class MaxAtomDistributedBatchSampler(torch.utils.data.Sampler):
    """Distributed batch sampler that packs structures greedily up to ``max_atoms``.

    Structure-to-batch packing is performed **once at construction** using ``seed``
    (stable across epochs). Each epoch, the *order* in which batches are presented
    to each rank is reshuffled using ``seed + epoch``, mirroring the fairchem
    ``MaxAtomDistributedBatchSampler`` design.

    Batch data is stored in CSR (compressed-sparse-row) numpy arrays rather than
    Python lists to avoid fork-induced copy-on-write pressure from Python's
    garbage collector traversing reference-counted objects in worker processes.

    :param dataset: The dataset to sample from. Must support ``get_num_atoms(i)``
        (currently only ``MemmapDataset`` and ``Subset`` wrappers thereof).
    :param max_atoms: Maximum total number of atoms across all structures in a batch.
    :param num_replicas: Number of distributed processes (world size).
    :param rank: Rank of the current process.
    :param shuffle: Whether to shuffle batch presentation order each epoch.
    :param seed: Base random seed. Packing uses ``seed``; per-epoch order uses
        ``seed + epoch``.
    :param drop_last: If ``True``, drop tail batches so the count is evenly divisible
        by ``num_replicas`` (no padding/repetition). If ``False``, repeat batches from
        the front to pad.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_atoms: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.start_iter = 0

        n = len(dataset)
        # Fast path: avoid a Python loop over millions of structures.
        # MemmapDataset exposes get_all_atom_counts() which returns np.diff(na)
        # in one vectorised operation.
        inner = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        if hasattr(inner, "get_all_atom_counts"):
            all_counts = inner.get_all_atom_counts()
            if isinstance(dataset, torch.utils.data.Subset):
                atom_counts = all_counts[np.array(dataset.indices, dtype=np.int64)]
            else:
                atom_counts = all_counts
        else:
            atom_counts = np.array(
                [_get_num_atoms(dataset, i) for i in range(n)], dtype=np.int64
            )

        # Pack once at init; only batch *order* changes each epoch.
        # Store as CSR numpy arrays (flat indices + offsets) to avoid Python
        # list reference-counting in forked DataLoader workers, which would
        # COW-copy gigabytes of Python objects and exhaust /dev/shm.
        self._batch_indices, self._batch_offsets = self._build_batches_csr(atom_counts)
        # atom_counts is no longer needed after packing
        del atom_counts

        num_batches = len(self._batch_offsets) - 1
        assert num_batches >= self.num_replicas, (
            f"Only {num_batches} batches were packed but "
            f"num_replicas={self.num_replicas}. Increase the dataset size or "
            "reduce max_atoms."
        )

        if self.drop_last and num_batches % self.num_replicas != 0:
            self.num_samples = math.floor(num_batches / self.num_replicas)
        else:
            self.num_samples = math.ceil(num_batches / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling. Call before each epoch."""
        self.epoch = epoch
        self.start_iter = 0

    def set_epoch_and_start_iteration(self, epoch: int, start_iter: int) -> None:
        """Set epoch and starting batch index for mid-epoch checkpoint resumption."""
        self.epoch = epoch
        self.start_iter = start_iter

    def _build_batches_csr(
        self, atom_counts: np.ndarray
    ) -> tuple:
        """Pack structures into batches and return as CSR numpy arrays.

        Returns ``(flat_indices, offsets)`` where batch ``i`` contains
        ``flat_indices[offsets[i]:offsets[i+1]]``.  Using numpy arrays avoids
        Python list reference-counting in forked DataLoader workers.
        """
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        counts = atom_counts[indices]
        batches = _greedy_pack(indices.tolist(), counts.tolist(), self.max_atoms)
        # Convert to CSR
        flat = np.concatenate([np.asarray(b, dtype=np.int64) for b in batches])
        lengths = np.array([len(b) for b in batches], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return flat, offsets

    def _get_batch(self, i: int) -> List[int]:
        """Return batch ``i`` as a Python list of indices."""
        return self._batch_indices[
            self._batch_offsets[i] : self._batch_offsets[i + 1]
        ].tolist()

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self._batch_offsets) - 1
        # Shuffle batch presentation order per epoch.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(num_batches, generator=g).tolist()
        else:
            batch_indices = list(range(num_batches))

        if not self.drop_last:
            # Pad to total_size, wrapping multiple times if needed.
            padding_size = self.total_size - len(batch_indices)
            if padding_size <= len(batch_indices):
                batch_indices += batch_indices[:padding_size]
            else:
                batch_indices += (
                    batch_indices * math.ceil(padding_size / len(batch_indices))
                )[:padding_size]
        else:
            batch_indices = batch_indices[: self.total_size]

        assert len(batch_indices) == self.total_size

        # Assign to this rank via interleaved striding.
        batch_indices = batch_indices[self.rank : self.total_size : self.num_replicas]
        assert len(batch_indices) == self.num_samples

        return (
            self._get_batch(i)
            for i in batch_indices[self.start_iter :]
        )

    def __len__(self) -> int:
        return self.num_samples


class MaxAtomBatchSampler(MaxAtomDistributedBatchSampler):
    """Single-process version of :class:`MaxAtomDistributedBatchSampler`.

    Convenience wrapper that fixes ``num_replicas=1`` and ``rank=0``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_atoms: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            max_atoms=max_atoms,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
