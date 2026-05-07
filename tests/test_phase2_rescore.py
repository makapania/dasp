"""Tests for phase2_rescore — the shared multi-seed rescore helper.

Spec: docs/plans/2026-05-06-exhaustive-multiseed-autoscale-tpe.md, section
"Shared phase-2 helper". Tests cover both ADAPTIVE mode (exhaustive Phase 2
use) and DEGENERATE mode (TPE Phase 4 multi-start union rescore).
"""

from __future__ import annotations

import math

import pytest

from spectral_predict.phase2_rescore import (
    DEFAULT_SEEDS,
    _aggregate_scores,
    _compute_jaccard,
    _rank_key,
    _select_diverse,
    phase2_adaptive_rescore,
)


# =============================================================================
# Helper-level tests
# =============================================================================


class TestAggregateScores:
    def test_finite_scores_aggregate_correctly(self):
        mean, std, n = _aggregate_scores([1.0, 2.0, 3.0])
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(0.8164965, abs=1e-5)
        assert n == 3

    def test_infinite_scores_dropped(self):
        mean, std, n = _aggregate_scores([1.0, float("-inf"), 3.0, float("nan")])
        assert mean == pytest.approx(2.0)
        assert n == 2

    def test_all_invalid_returns_sentinel(self):
        mean, std, n = _aggregate_scores([float("-inf"), float("nan")])
        assert mean == float("-inf")
        assert std == float("inf")
        assert n == 0

    def test_single_valid_score_zero_std(self):
        mean, std, n = _aggregate_scores([2.5])
        assert mean == pytest.approx(2.5)
        assert std == 0.0
        assert n == 1


class TestRankKey:
    def test_maximize_inverts_mean(self):
        # Higher mean should sort earlier (lower sort key)
        a = _rank_key(0.9, 0.01, ("a",), "maximize")
        b = _rank_key(0.5, 0.01, ("b",), "maximize")
        assert a < b

    def test_minimize_keeps_mean_ascending(self):
        # Lower mean should sort earlier (lower sort key)
        a = _rank_key(0.5, 0.01, ("a",), "minimize")
        b = _rank_key(0.9, 0.01, ("b",), "minimize")
        assert a < b

    def test_std_breaks_ties_lower_first(self):
        # Same mean: lower std wins
        low_std = _rank_key(0.7, 0.01, ("a",), "maximize")
        high_std = _rank_key(0.7, 0.10, ("b",), "maximize")
        assert low_std < high_std

    def test_key_breaks_remaining_ties(self):
        # Same mean and std: key_fn output decides
        a = _rank_key(0.7, 0.01, ("a",), "maximize")
        b = _rank_key(0.7, 0.01, ("b",), "maximize")
        assert a < b


class TestComputeJaccard:
    def test_identical_lists_one(self):
        a = [{"x": 1}, {"x": 2}]
        b = [{"x": 1}, {"x": 2}]
        assert _compute_jaccard(a, b, lambda c: (c["x"],)) == 1.0

    def test_disjoint_lists_zero(self):
        a = [{"x": 1}]
        b = [{"x": 2}]
        assert _compute_jaccard(a, b, lambda c: (c["x"],)) == 0.0

    def test_partial_overlap(self):
        a = [{"x": 1}, {"x": 2}]
        b = [{"x": 2}, {"x": 3}]
        # |a ∩ b| = 1, |a ∪ b| = 3
        assert _compute_jaccard(a, b, lambda c: (c["x"],)) == pytest.approx(1 / 3)

    def test_empty_lists_one(self):
        assert _compute_jaccard([], [], lambda c: (c,)) == 1.0


class TestSelectDiverse:
    def test_no_diversity_returns_top_n(self):
        rescored = [
            ("a", 0.9, 0.01, ("a",)),
            ("b", 0.8, 0.01, ("b",)),
            ("c", 0.7, 0.01, ("c",)),
        ]
        result = _select_diverse(rescored, None, top_n=2)
        assert result == ["a", "b"]

    def test_diversity_picks_one_per_key(self):
        # Two configs with same diversity key, plus a third with a different key.
        rescored = [
            ({"family": "A", "id": 1}, 0.9, 0.01, ("A1",)),
            ({"family": "A", "id": 2}, 0.85, 0.01, ("A2",)),
            ({"family": "B", "id": 1}, 0.7, 0.01, ("B1",)),
        ]
        result = _select_diverse(
            rescored, lambda c: (c["family"],), top_n=2
        )
        # Should pick A's best (id=1) and B's only entry, NOT both A's
        assert result == [
            {"family": "A", "id": 1},
            {"family": "B", "id": 1},
        ]

    def test_diversity_falls_back_to_best_overall(self):
        # Only one diversity key but top_n=2: second slot fills with best-overall
        rescored = [
            ({"family": "A", "id": 1}, 0.9, 0.01, ("A1",)),
            ({"family": "A", "id": 2}, 0.85, 0.01, ("A2",)),
        ]
        result = _select_diverse(
            rescored, lambda c: (c["family"],), top_n=2
        )
        assert len(result) == 2
        assert result[0] == {"family": "A", "id": 1}
        assert result[1] == {"family": "A", "id": 2}


# =============================================================================
# Main function tests
# =============================================================================


def _build_static_eval_fn(scores_by_id: dict[int, float]):
    """Eval fn that returns a fixed score per candidate (no seed dependence).

    Useful for tests where we want deterministic ordering without simulating
    fold randomness.
    """

    def eval_fn(cand: dict, seed: int) -> float:
        return scores_by_id[cand["id"]]

    return eval_fn


def _build_seed_dependent_eval_fn(scores_by_id_and_seed: dict[tuple[int, int], float]):
    """Eval fn whose output varies with (cand, seed)."""

    def eval_fn(cand: dict, seed: int) -> float:
        return scores_by_id_and_seed[(cand["id"], seed)]

    return eval_fn


def _key(cand: dict) -> tuple:
    return (cand["id"],)


class TestReturnsTopNWithCorrectMetadata:
    def test_returns_top_n_and_well_formed_metadata(self):
        candidates = [{"id": i} for i in range(20)]
        scores = {i: 1.0 - i * 0.01 for i in range(20)}  # higher id → lower score
        eval_fn = _build_static_eval_fn(scores)

        result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=15,
            pool_size_progression=[15],
            top_n=5,
            n_seeds=3,
        )

        assert len(result) == 5
        # Top-5 should be ids 0..4 since they have highest scores
        assert {c["id"] for c in result} == {0, 1, 2, 3, 4}
        assert meta["halt_reason"] == "single_iteration"
        assert meta["final_pool_size"] == 15
        assert meta["candidates_evaluated"] == 15
        assert meta["expansions"] == 0


class TestHaltOnConverged:
    def test_stable_rankings_halt_via_jaccard(self):
        # Build candidates where top-5 is the same regardless of pool size
        candidates = [{"id": i} for i in range(30)]
        scores = {i: 1.0 - i * 0.01 for i in range(30)}
        eval_fn = _build_static_eval_fn(scores)

        _result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=15,
            pool_size_progression=[15, 25, 40],
            top_n=5,
            n_seeds=3,
        )

        assert meta["halt_reason"] == "converged"

    def test_overlap_threshold_path_halts(self):
        # top-5 same except for one position swap: overlap = 4 (= top_n - 1)
        # Stable rankings → halts on either jaccard or overlap; verify halts.
        candidates = [{"id": i} for i in range(30)]
        scores = {i: 1.0 - i * 0.01 for i in range(30)}
        eval_fn = _build_static_eval_fn(scores)

        _result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=15,
            pool_size_progression=[15, 25, 40],
            top_n=5,
            n_seeds=3,
            halt_jaccard_threshold=2.0,  # impossible by jaccard
            halt_overlap_threshold=4,    # same as top_n - 1
        )

        assert meta["halt_reason"] == "converged"


class TestHaltOnCap:
    def test_unstable_rankings_hit_cap(self):
        # Eval is seed-dependent so each iteration's "top" can differ when the
        # pool grows: build scores so adding new candidates flips the top-5.
        candidates = [{"id": i} for i in range(50)]

        # For each candidate, the score depends on whether the seed forces a
        # large reshuffle. Make the scores depend chaotically on seed * id.
        def eval_fn(cand: dict, seed: int) -> float:
            return ((cand["id"] + seed * 7) % 23) * 0.01

        _result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=5,
            pool_size_progression=[5, 10, 20, 40],
            max_pool_multiplier=8,  # cap = 8 * 5 = 40
            top_n=5,
            n_seeds=5,
        )

        # Either converged via stability (chaotic but possible) or hit cap;
        # we want to assert at least that cap is reachable. Construct a stricter
        # case to guarantee cap.
        assert meta["halt_reason"] in ("cap", "converged")
        # The final pool size cannot exceed cap
        assert meta["final_pool_size"] <= 40

    def test_cap_with_jaccard_threshold_one_forces_cap(self):
        # Force halt_reason="cap" by making jaccard threshold unreachable
        candidates = [{"id": i} for i in range(50)]
        scores = {i: 1.0 - i * 0.01 for i in range(50)}
        eval_fn = _build_static_eval_fn(scores)

        _result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=5,
            pool_size_progression=[5, 10, 20, 40],
            max_pool_multiplier=8,
            top_n=5,
            n_seeds=3,
            halt_jaccard_threshold=2.0,   # unreachable
            halt_overlap_threshold=10,    # > top_n=5, unreachable
        )

        assert meta["halt_reason"] == "cap"
        assert meta["final_pool_size"] == 40


class TestJaccardThresholdRespected:
    def test_jaccard_path_takes_precedence_when_overlap_unreachable(self):
        candidates = [{"id": i} for i in range(30)]
        scores = {i: 1.0 - i * 0.01 for i in range(30)}
        eval_fn = _build_static_eval_fn(scores)

        # overlap_threshold=10 is > top_n; only jaccard path can trigger halt
        _result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=15,
            pool_size_progression=[15, 25, 40],
            top_n=5,
            n_seeds=3,
            halt_jaccard_threshold=0.5,  # easy
            halt_overlap_threshold=10,   # unreachable (top_n=5)
        )

        # Stable scores → jaccard=1.0 between iterations → halts converged
        assert meta["halt_reason"] == "converged"


class TestTieBreakingByStdThenKey:
    def test_lower_std_wins_on_equal_mean(self):
        # Two candidates tied on mean across seeds, one with lower std.
        # Use integer-valued scores so means are bit-exact equal under
        # floating-point summation — float scores like 0.7 are inexact and
        # let mean(0.5, 0.7, 0.9) drift to 0.7000000000000001, which would
        # make this test pass for the wrong reason (mean comparison, not std).
        candidates = [{"id": 1}, {"id": 2}]
        # id=1: scores [3, 3, 3] → mean=3.0 exact, std=0
        # id=2: scores [1, 3, 5] → mean=3.0 exact, std≈1.633
        scores_by_pair = {
            (1, 42): 3.0, (1, 0): 3.0, (1, 7): 3.0,
            (2, 42): 1.0, (2, 0): 3.0, (2, 7): 5.0,
        }
        eval_fn = _build_seed_dependent_eval_fn(scores_by_pair)

        result, _meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=2,
            pool_size_progression=[2],
            top_n=2,
            n_seeds=3,
        )

        # id=1 (lower std) should be ranked first
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_key_breaks_remaining_ties(self):
        # Two candidates with identical mean AND identical std across seeds.
        # key_fn determines order. We sort tuples lexicographically, so the
        # smaller tuple should appear first.
        candidates = [{"id": 2}, {"id": 1}]  # provide in non-key-sorted order
        scores_by_pair = {
            (1, 42): 0.7, (1, 0): 0.7, (1, 7): 0.7,
            (2, 42): 0.7, (2, 0): 0.7, (2, 7): 0.7,
        }
        eval_fn = _build_seed_dependent_eval_fn(scores_by_pair)

        result, _meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=2,
            pool_size_progression=[2],
            top_n=2,
            n_seeds=3,
        )

        # key_fn returns (id,); tuple (1,) < (2,) → id=1 first deterministically
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2


class TestDiversityAppliedAtEnd:
    def test_diversity_reduces_homogeneous_pool(self):
        # 5 candidates, all family "A" but with different ids and varying scores;
        # plus 2 family-"B" candidates with lower scores.
        candidates = [
            {"id": 1, "family": "A"},
            {"id": 2, "family": "A"},
            {"id": 3, "family": "A"},
            {"id": 4, "family": "B"},
            {"id": 5, "family": "B"},
        ]
        scores = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.70, 5: 0.65}
        eval_fn = _build_static_eval_fn(scores)

        # Without diversity: top-3 = [1, 2, 3] (all family A)
        # With diversity_key_fn=family: top-3 picks one A then one B,
        #   then fills the third slot with the best remaining (id=2)
        result, _meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=5,
            pool_size_progression=[5],
            top_n=3,
            n_seeds=3,
            diversity_key_fn=lambda c: (c["family"],),
        )

        # Expect: id=1 (best of A), id=4 (best of B), id=2 (next best A)
        assert [c["id"] for c in result] == [1, 4, 2]


# =============================================================================
# DEGENERATE-mode (TPE) specific
# =============================================================================


class TestDegenerateMode:
    def test_single_iteration_halt_reason(self):
        candidates = [{"id": i} for i in range(10)]
        scores = {i: 1.0 - i * 0.01 for i in range(10)}
        eval_fn = _build_static_eval_fn(scores)

        result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=10,
            pool_size_progression=[10],
            top_n=3,
            n_seeds=3,
            max_pool_multiplier=999,  # TPE-style: disable adaptive cap
        )

        assert meta["halt_reason"] == "single_iteration"
        assert meta["expansions"] == 0
        assert meta["final_pool_size"] == 10
        assert len(result) == 3

    def test_degenerate_no_jaccard_recorded(self):
        candidates = [{"id": i} for i in range(10)]
        scores = {i: 1.0 - i * 0.01 for i in range(10)}
        eval_fn = _build_static_eval_fn(scores)

        _result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=10,
            pool_size_progression=[10],
            top_n=3,
            n_seeds=3,
        )

        # No prior iteration to compare against → empty list
        assert meta["per_iteration_jaccards"] == []


# =============================================================================
# Score direction
# =============================================================================


class TestScoreDirectionMinimize:
    def test_minimize_picks_lowest(self):
        candidates = [{"id": 1}, {"id": 2}, {"id": 3}]
        # For minimize (e.g., RMSE): id=1 best (lowest score)
        scores = {1: 0.5, 2: 1.0, 3: 1.5}
        eval_fn = _build_static_eval_fn(scores)

        result, _meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="minimize",
            initial_pool_size=3,
            pool_size_progression=[3],
            top_n=2,
            n_seeds=3,
        )

        assert [c["id"] for c in result] == [1, 2]


# =============================================================================
# Robustness / edge cases
# =============================================================================


class TestEdgeCases:
    def test_empty_candidates(self):
        result, meta = phase2_adaptive_rescore(
            candidates=[],
            eval_fn=lambda c, s: 0.0,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=10,
            pool_size_progression=[10],
            top_n=3,
            n_seeds=3,
        )

        assert result == []
        assert meta["halt_reason"] == "empty_input"
        assert meta["candidates_evaluated"] == 0

    def test_eval_fn_exception_treats_as_invalid(self):
        candidates = [{"id": 1}, {"id": 2}, {"id": 3}]

        def eval_fn(cand, seed):
            if cand["id"] == 2:
                raise RuntimeError("simulated CV failure")
            return 1.0 - cand["id"] * 0.1  # id=1 → 0.9, id=3 → 0.7

        result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=3,
            pool_size_progression=[3],
            top_n=2,
            n_seeds=3,
        )

        # id=2 should sort to the bottom (all -inf)
        assert {c["id"] for c in result} == {1, 3}
        assert meta["candidates_evaluated"] == 3

    def test_pool_larger_than_candidates_clamps(self):
        candidates = [{"id": i} for i in range(5)]
        scores = {i: 1.0 - i * 0.01 for i in range(5)}
        eval_fn = _build_static_eval_fn(scores)

        result, meta = phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=100,           # larger than 5 available
            pool_size_progression=[100],
            top_n=3,
            n_seeds=3,
            max_pool_multiplier=1000,        # don't let cap interfere
        )

        assert len(result) == 3
        assert meta["final_pool_size"] == 5

    def test_invalid_score_direction_raises(self):
        with pytest.raises(ValueError, match="score_direction"):
            phase2_adaptive_rescore(
                candidates=[{"id": 1}],
                eval_fn=lambda c, s: 0.0,
                key_fn=_key,
                score_direction="bogus",  # type: ignore[arg-type]
                initial_pool_size=1,
                pool_size_progression=[1],
                top_n=1,
                n_seeds=1,
            )

    def test_n_seeds_exceeds_default_seeds_requires_explicit(self):
        with pytest.raises(ValueError, match="DEFAULT_SEEDS"):
            phase2_adaptive_rescore(
                candidates=[{"id": 1}],
                eval_fn=lambda c, s: 0.0,
                key_fn=_key,
                score_direction="maximize",
                initial_pool_size=1,
                pool_size_progression=[1],
                top_n=1,
                n_seeds=99,  # way more than DEFAULT_SEEDS has
            )

    def test_explicit_seeds_used(self):
        seeds_seen: list[int] = []

        def eval_fn(cand, seed):
            seeds_seen.append(seed)
            return 0.5

        phase2_adaptive_rescore(
            candidates=[{"id": 1}],
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=1,
            pool_size_progression=[1],
            top_n=1,
            n_seeds=3,
            seeds=[111, 222, 333],
        )

        assert seeds_seen == [111, 222, 333]

    def test_progress_callback_invoked(self):
        progress: list[tuple[int, int, str]] = []

        candidates = [{"id": i} for i in range(3)]
        scores = {i: 0.5 for i in range(3)}
        eval_fn = _build_static_eval_fn(scores)

        phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=3,
            pool_size_progression=[3],
            top_n=2,
            n_seeds=2,
            progress_callback=lambda c, t, m: progress.append((c, t, m)),
        )

        assert len(progress) == 3
        # Each call should report cumulative count and total
        assert progress[-1][0] == 3
        assert progress[-1][1] == 3


# =============================================================================
# Caching behavior
# =============================================================================


class TestCaching:
    def test_same_candidate_not_rescored_across_iterations(self):
        # Test that across pool expansions, candidates evaluated in earlier
        # iterations are NOT re-evaluated (cache hit).
        eval_calls: list[tuple[int, int]] = []  # (id, seed)

        def eval_fn(cand, seed):
            eval_calls.append((cand["id"], seed))
            return 1.0 - cand["id"] * 0.01

        candidates = [{"id": i} for i in range(40)]

        phase2_adaptive_rescore(
            candidates=candidates,
            eval_fn=eval_fn,
            key_fn=_key,
            score_direction="maximize",
            initial_pool_size=10,
            pool_size_progression=[10, 20, 40],
            top_n=5,
            n_seeds=3,
            max_pool_multiplier=8,  # cap = 40
            halt_jaccard_threshold=2.0,    # force full progression
            halt_overlap_threshold=10,
        )

        # ID 0 should be evaluated only n_seeds=3 times (first iteration only)
        # not 9 times (3 iterations × 3 seeds)
        id0_evals = sum(1 for (i, _s) in eval_calls if i == 0)
        assert id0_evals == 3
