"""Size-aware samplers for ELECTRAFY's variable-grid training.

ELECTRAFY trains on charge densities with native FFT grids that range from
~200k to ~10M voxels across the MP-CHGCAR corpus. With a standard random
``DistributedSampler``, each DDP step's wall time is bounded by the slowest
rank, which (for variable system sizes) is the rank that happened to draw
the largest structure. Bench job 3346830/3346831 showed this load imbalance
costs ~50% of step time at 8x H100 (mean 0.94 s vs max 1.47 s/step).

This module provides two distributed samplers that group similar-sized
systems into the same DDP step:

* :class:`SortedBucketSampler` -- 1.64x e2e speedup on 2x4 H100 (bench
  3348578). Each DDP step's grid-size spread is bounded by ``bucket_tol``
  (default 10%). Pure random data order is sacrificed.
* :class:`GridBudgetBatchSampler` -- variable batch size, packed up to a
  grid-point budget. Matches bucketed-sampler perf in bench 3356591/3356703
  (no further speedup), but yields more uniform per-step memory use.

Both samplers expose ``set_epoch(epoch)`` so the metatrain trainer's
existing per-epoch reseed loop works without modification.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from torch.utils.data import Sampler


__all__ = ["SortedBucketSampler", "GridBudgetBatchSampler"]


class SortedBucketSampler(Sampler[int]):
    """Distributed index sampler that groups similar-sized systems into the
    same DDP step, eliminating max-rank gating from variable system sizes.

    Two modes via ``tol`` (max within-bucket ``max_grid / min_grid`` ratio):

    Strict (``tol=0``)
        Each bucket is exactly ``world_size`` consecutive sorted indices.
        Maximum size matching and within-epoch gradient correlation.

    Relaxed (``tol>0``)
        Greedy-build POOLS where ``pool_max / pool_min <= 1 + tol``. Each
        epoch the pool is reshuffled, then chopped into ``world_size``-chunks
        (each chunk = one DDP step). Admits up to ``(1+tol)x`` grid variance
        within each DDP step in exchange for breaking the strict per-epoch
        gradient correlation.

    Per-epoch behaviour (both modes):

    * Bucket order across the epoch is permuted (same permutation on all
      ranks).
    * Within-bucket column assignment is permuted each epoch.

    :param grid_sizes: per-sample grid point counts (``N1 * N2 * N3``) in
        dataset index order. ``len(grid_sizes) == len(dataset)``.
    :param world_size: number of distributed ranks (== ``WORLD_SIZE``).
    :param rank: this process's rank in ``[0, world_size)``.
    :param seed: base seed for per-epoch shuffling (synced across ranks).
    :param tol: maximum within-bucket grid-size spread; ``0.0`` = strict.
    """

    def __init__(
        self,
        grid_sizes: List[int],
        world_size: int,
        rank: int,
        seed: int = 0,
        tol: float = 0.0,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(
                f"rank must be in [0, {world_size}), got {rank}"
            )
        if len(grid_sizes) < world_size:
            raise ValueError(
                f"need >= world_size={world_size} samples, "
                f"got {len(grid_sizes)}"
            )

        self.world_size = world_size
        self.rank = rank
        self.seed = int(seed)
        self.epoch = 0
        self.tol = max(0.0, float(tol))

        order = sorted(range(len(grid_sizes)), key=lambda i: grid_sizes[i])
        gs_sorted = [grid_sizes[i] for i in order]

        if self.tol == 0.0:
            n_full = (len(order) // world_size) * world_size
            self._pools: List[List[int]] = [
                order[b : b + world_size]
                for b in range(0, n_full, world_size)
            ]
            self._reshuffle_within_pool = False
        else:
            threshold = 1.0 + self.tol
            pools_raw: List[List[int]] = []
            cur_pool: List[int] = []
            cur_min: float = 0.0
            for idx, gsz in zip(order, gs_sorted):
                if not cur_pool:
                    cur_pool = [idx]
                    cur_min = float(gsz)
                elif gsz <= cur_min * threshold:
                    cur_pool.append(idx)
                else:
                    pools_raw.append(cur_pool)
                    cur_pool = [idx]
                    cur_min = float(gsz)
            if cur_pool:
                pools_raw.append(cur_pool)
            self._pools = []
            for pool in pools_raw:
                n_full = (len(pool) // world_size) * world_size
                if n_full > 0:
                    self._pools.append(pool[:n_full])
            self._reshuffle_within_pool = True

        self._n_buckets = sum(len(p) // self.world_size for p in self._pools)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed * 7919 + self.epoch)
        all_buckets: List[List[int]] = []
        for pool in self._pools:
            if self._reshuffle_within_pool:
                local = pool.copy()
                rng.shuffle(local)
            else:
                local = pool
            for j in range(0, len(local), self.world_size):
                all_buckets.append(local[j : j + self.world_size])
        bucket_order = list(range(len(all_buckets)))
        rng.shuffle(bucket_order)
        within = list(range(self.world_size))
        rng.shuffle(within)
        col = within[self.rank]
        for bi in bucket_order:
            yield int(all_buckets[bi][col])

    def __len__(self) -> int:
        return self._n_buckets


class GridBudgetBatchSampler(Sampler[List[int]]):
    """Distributed batch sampler that greedy-packs systems up to a
    grid-point budget, then assigns batches to ranks by stride.

    All ranks deterministically build the same global batch list (sort by
    grid size, greedy-pack while ``sum_grid <= max_grid_per_batch``). Each
    epoch the global batch order is permuted with a synced RNG. Rank ``r``
    yields the batches at positions ``r, r + world_size, ...`` of the
    permuted order.

    Because batches have variable cardinality (more small systems per batch,
    fewer large), wire this through
    ``DataLoader(batch_sampler=GridBudgetBatchSampler(...))`` rather than the
    ``sampler=...`` + ``batch_size=...`` pair.

    Oversized systems (``grid > max_grid_per_batch``) are emitted as solo
    batches so they aren't silently dropped.

    :param grid_sizes: per-sample grid point counts in dataset index order.
    :param world_size: number of distributed ranks.
    :param rank: this process's rank.
    :param max_grid_per_batch: grid-point budget per batch.
    :param seed: base seed for per-epoch shuffling.
    """

    def __init__(
        self,
        grid_sizes: List[int],
        world_size: int,
        rank: int,
        max_grid_per_batch: int,
        seed: int = 0,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(
                f"rank must be in [0, {world_size}), got {rank}"
            )
        if max_grid_per_batch <= 0:
            raise ValueError(
                f"max_grid_per_batch must be > 0, got {max_grid_per_batch}"
            )
        if len(grid_sizes) == 0:
            raise ValueError("need >= 1 sample")

        self.world_size = world_size
        self.rank = rank
        self.seed = int(seed)
        self.epoch = 0
        self.max_grid_per_batch = int(max_grid_per_batch)

        order = sorted(range(len(grid_sizes)), key=lambda i: grid_sizes[i])

        batches: List[List[int]] = []
        cur: List[int] = []
        cur_sum = 0
        for idx in order:
            g = int(grid_sizes[idx])
            if g > self.max_grid_per_batch:
                if cur:
                    batches.append(cur)
                    cur = []
                    cur_sum = 0
                batches.append([idx])
                continue
            if cur and cur_sum + g > self.max_grid_per_batch:
                batches.append(cur)
                cur = [idx]
                cur_sum = g
            else:
                cur.append(idx)
                cur_sum += g
        if cur:
            batches.append(cur)

        n_full = (len(batches) // world_size) * world_size
        self._batches: List[List[int]] = batches[:n_full]
        self._batches_per_rank = n_full // world_size

        sizes_per_batch = [
            sum(int(grid_sizes[i]) for i in b) for b in self._batches
        ]
        items_per_batch = [len(b) for b in self._batches]
        self.stats: dict = {
            "n_batches_total": len(self._batches),
            "n_batches_per_rank": self._batches_per_rank,
            "items_min": min(items_per_batch) if items_per_batch else 0,
            "items_mean": (
                sum(items_per_batch) / len(items_per_batch)
                if items_per_batch else 0.0
            ),
            "items_max": max(items_per_batch) if items_per_batch else 0,
            "fill_mean": (
                sum(sizes_per_batch) / len(sizes_per_batch)
                if sizes_per_batch else 0.0
            ),
            "fill_max": max(sizes_per_batch) if sizes_per_batch else 0,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed * 7919 + self.epoch + 1)
        step_ids = list(range(self._batches_per_rank))
        rng.shuffle(step_ids)
        col_perm = list(range(self.world_size))
        rng.shuffle(col_perm)
        col = col_perm[self.rank]
        for sid in step_ids:
            yield list(self._batches[sid * self.world_size + col])

    def __len__(self) -> int:
        return self._batches_per_rank
