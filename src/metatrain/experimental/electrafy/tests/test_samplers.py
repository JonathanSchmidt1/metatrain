"""Unit tests for ELECTRAFY's size-aware samplers."""

from __future__ import annotations

from typing import List

import pytest

from metatrain.experimental.electrafy.modules.samplers import (
    GridBudgetBatchSampler,
    SortedBucketSampler,
)


def _gather_across_ranks(
    sampler_cls,
    grid_sizes: List[int],
    world_size: int,
    epoch: int = 0,
    **kwargs,
):
    out = []
    for r in range(world_size):
        s = sampler_cls(
            grid_sizes=grid_sizes,
            world_size=world_size,
            rank=r,
            **kwargs,
        )
        s.set_epoch(epoch)
        out.append(list(iter(s)))
    return out


class TestSortedBucketSamplerStrict:
    def test_per_rank_disjoint(self):
        grid_sizes = list(range(40))
        per_rank = _gather_across_ranks(
            SortedBucketSampler, grid_sizes, world_size=4, tol=0.0
        )
        union = set()
        for r in per_rank:
            assert len(set(r)) == len(r), "duplicates within a rank"
            assert union.isdisjoint(set(r)), "indices visited by >1 rank"
            union.update(r)

    def test_equal_steps_per_rank(self):
        grid_sizes = list(range(40))
        per_rank = _gather_across_ranks(
            SortedBucketSampler, grid_sizes, world_size=4, tol=0.0
        )
        lens = {len(r) for r in per_rank}
        assert len(lens) == 1, f"unequal steps across ranks: {lens}"
        assert next(iter(lens)) == 40 // 4

    def test_step_size_correlation(self):
        # In strict mode, each DDP step's grid sizes are consecutive in the
        # sorted order, so within-step max/min must be tight.
        grid_sizes = list(range(40))
        s0 = SortedBucketSampler(grid_sizes, world_size=4, rank=0, tol=0.0)
        s1 = SortedBucketSampler(grid_sizes, world_size=4, rank=1, tol=0.0)
        s2 = SortedBucketSampler(grid_sizes, world_size=4, rank=2, tol=0.0)
        s3 = SortedBucketSampler(grid_sizes, world_size=4, rank=3, tol=0.0)
        for sampler in (s0, s1, s2, s3):
            sampler.set_epoch(0)
        it0, it1, it2, it3 = (
            iter(s0), iter(s1), iter(s2), iter(s3)
        )
        for _ in range(10):
            step = sorted([
                grid_sizes[next(it0)],
                grid_sizes[next(it1)],
                grid_sizes[next(it2)],
                grid_sizes[next(it3)],
            ])
            assert step[-1] - step[0] <= 3, (
                f"strict step spread > world_size-1: {step}"
            )

    def test_determinism_per_epoch(self):
        grid_sizes = list(range(40))
        s = SortedBucketSampler(grid_sizes, 4, 0, seed=42, tol=0.0)
        s.set_epoch(7)
        run1 = list(iter(s))
        s.set_epoch(7)
        run2 = list(iter(s))
        assert run1 == run2

    def test_distinct_epochs_differ(self):
        grid_sizes = list(range(40))
        s = SortedBucketSampler(grid_sizes, 4, 0, seed=42, tol=0.0)
        s.set_epoch(0)
        ep0 = list(iter(s))
        s.set_epoch(1)
        ep1 = list(iter(s))
        assert ep0 != ep1


class TestSortedBucketSamplerRelaxed:
    def test_within_step_tol_respected(self):
        # Mixed distribution of grid sizes
        grid_sizes = (
            [100] * 10
            + [105] * 10
            + [110] * 10
            + [200] * 10
            + [400] * 10
        )
        ws = 4
        s = [
            SortedBucketSampler(grid_sizes, ws, r, tol=0.10) for r in range(ws)
        ]
        for sampler in s:
            sampler.set_epoch(0)
        iters = [iter(x) for x in s]
        n_steps = len(s[0])
        for _ in range(n_steps):
            step_sizes = [grid_sizes[next(it)] for it in iters]
            ratio = max(step_sizes) / min(step_sizes)
            # In relaxed mode each step is drawn from one pool whose
            # max/min <= 1+tol, so the step's max/min is also <= 1+tol.
            assert ratio <= 1.10 + 1e-6, (
                f"step spread {ratio:.3f} > 1+tol=1.10  {step_sizes}"
            )

    def test_per_rank_disjoint_relaxed(self):
        grid_sizes = (
            [100, 105, 110, 200, 210, 220, 400, 410, 420, 430] * 4
        )
        per_rank = _gather_across_ranks(
            SortedBucketSampler, grid_sizes, world_size=4, tol=0.20
        )
        union = set()
        for r in per_rank:
            assert union.isdisjoint(set(r))
            union.update(r)

    def test_pool_reshuffle_changes_within_pool_order(self):
        # 10 same-size structures in one pool; bucket order may vary across
        # epochs (pool reshuffle is on).
        grid_sizes = [100] * 10 + [101] * 10  # one pool of 20 indices at 1%
        ws = 4
        s = SortedBucketSampler(grid_sizes, ws, rank=0, seed=1, tol=0.05)
        s.set_epoch(0)
        ep0 = list(iter(s))
        s.set_epoch(1)
        ep1 = list(iter(s))
        assert ep0 != ep1


class TestSortedBucketSamplerEdges:
    def test_insufficient_samples_errors(self):
        with pytest.raises(ValueError, match="need >= world_size"):
            SortedBucketSampler([1, 2, 3], world_size=4, rank=0)

    def test_invalid_rank(self):
        with pytest.raises(ValueError, match="rank must be in"):
            SortedBucketSampler([1, 2, 3, 4], world_size=4, rank=4)

    def test_invalid_world_size(self):
        with pytest.raises(ValueError, match="world_size must be"):
            SortedBucketSampler([1, 2, 3, 4], world_size=0, rank=0)


class TestGridBudgetBatchSampler:
    def test_batch_size_respects_budget(self):
        grid_sizes = [100] * 50
        s = GridBudgetBatchSampler(
            grid_sizes, world_size=2, rank=0, max_grid_per_batch=500, seed=0
        )
        s.set_epoch(0)
        for batch in s:
            assert sum(grid_sizes[i] for i in batch) <= 500

    def test_per_rank_disjoint_batches(self):
        grid_sizes = [100] * 40
        ws = 4
        s = [
            GridBudgetBatchSampler(
                grid_sizes, ws, r, max_grid_per_batch=300, seed=0
            )
            for r in range(ws)
        ]
        for sampler in s:
            sampler.set_epoch(0)
        seen = set()
        for sampler in s:
            for batch in sampler:
                for i in batch:
                    assert i not in seen, "duplicate across ranks"
                    seen.add(i)

    def test_oversize_emits_solo(self):
        grid_sizes = [100, 100, 5000, 100, 100, 100, 100]
        ws = 1
        s = GridBudgetBatchSampler(
            grid_sizes, ws, rank=0, max_grid_per_batch=300
        )
        s.set_epoch(0)
        batches = list(s)
        # The 5000-grid system must appear in some batch (alone).
        flat = [i for b in batches for i in b]
        assert 2 in flat
        for b in batches:
            if 2 in b:
                assert len(b) == 1

    def test_equal_batches_per_rank(self):
        grid_sizes = [100] * 40
        ws = 4
        per_rank_batches = []
        for r in range(ws):
            s = GridBudgetBatchSampler(
                grid_sizes, ws, r, max_grid_per_batch=300, seed=0
            )
            s.set_epoch(0)
            per_rank_batches.append(list(s))
        lens = {len(b) for b in per_rank_batches}
        assert len(lens) == 1

    def test_invalid_budget(self):
        with pytest.raises(ValueError, match="max_grid_per_batch"):
            GridBudgetBatchSampler(
                [100, 100], world_size=1, rank=0, max_grid_per_batch=0
            )

    def test_empty_grid_sizes(self):
        with pytest.raises(ValueError, match="need >= 1 sample"):
            GridBudgetBatchSampler(
                [], world_size=1, rank=0, max_grid_per_batch=10
            )
