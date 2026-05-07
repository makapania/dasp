"""Phase-2 multi-seed rescore helper.

Shared between exhaustive (Phase 2) and TPE (Phase 4) preprocessing-discovery
paths to denoise top-K rankings against CV-fold-randomness lottery.

Two operating modes, controlled by ``pool_size_progression``:

1. **ADAPTIVE MODE** (exhaustive Phase 2 use): pool grows through the
   progression list until top-N stabilizes between iterations or the
   cap is hit. Used when the candidate list is large (e.g., 238+
   single-seed-ranked configs) and we want to denoise just the top.

2. **DEGENERATE MODE** (TPE Phase 4 use): when ``pool_size_progression`` is
   a single element, the adaptive loop runs exactly one iteration —
   useful for multi-seed rescore of a pre-explored candidate pool
   (e.g., TPE multi-start union, where cross-study exploration is the
   responsibility of the caller, not the helper).

Spec: ``docs/plans/2026-05-06-exhaustive-multiseed-autoscale-tpe.md``,
section "Shared phase-2 helper".

Caller contract
---------------
* ``candidates``: pre-sorted by single-seed score (descending for
  ``score_direction="maximize"``, ascending for ``"minimize"``). The helper
  takes ``candidates[:K]`` per iteration; it does not re-rank by single-seed.
* ``eval_fn(candidate, random_state) -> float``: must return a finite score
  for valid candidates and ``-inf`` (maximize) / ``+inf`` (minimize) or raise
  for invalid ones. The helper handles exceptions internally.
* ``key_fn(candidate) -> tuple``: hashable identity tuple. Two distinct
  candidates must produce distinct keys.
* Aggregation: rank by mean across seeds, tie-break by std (lower = more
  stable wins on ties), then by ``key_fn`` for determinism.
* Halt criterion (adaptive mode): two consecutive iterations where
  ``Jaccard(prev_topN, current_topN) >= halt_jaccard_threshold`` OR
  ``len(prev ∩ current) >= halt_overlap_threshold``. Otherwise continues
  until the pool reaches ``max_pool_multiplier * top_n`` or
  ``len(candidates)``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal

import numpy as np

DEFAULT_SEEDS: list[int] = [42, 0, 7, 100, 31, 17, 88]


def _aggregate_scores(scores: list[float]) -> tuple[float, float, int]:
    """Aggregate per-seed scores. Returns ``(mean, std, n_valid)``.

    Drops non-finite scores before aggregating. If no valid scores remain,
    returns ``(-inf, +inf, 0)`` so the candidate sorts to the bottom under
    either score direction.
    """
    valid = [s for s in scores if math.isfinite(s)]
    if not valid:
        return (float("-inf"), float("inf"), 0)
    mean = float(np.mean(valid))
    std = float(np.std(valid)) if len(valid) > 1 else 0.0
    return (mean, std, len(valid))


def _rank_key(
    mean: float, std: float, key: tuple, score_direction: str
) -> tuple[float, float, tuple]:
    """Sort key for a candidate's aggregated score.

    Lower sort_key = better rank. For ``score_direction="maximize"``, the
    primary term is ``-mean`` so higher mean ranks earlier. ``std`` is always
    ascending (lower variance preferred on ties). ``key`` is the final
    tie-breaker for determinism.

    When ``mean`` is non-finite (no valid scores survived aggregation), the
    candidate is sent to the bottom of the ranking regardless of
    ``score_direction``. This prevents the ``_aggregate_scores`` "all-invalid"
    sentinel ``(-inf, +inf, 0)`` from accidentally ranking first under
    ``score_direction="minimize"``, where raw ``-inf`` would otherwise sort
    earliest.
    """
    if not math.isfinite(mean):
        return (float("inf"), float("inf"), key)
    primary = -mean if score_direction == "maximize" else mean
    return (primary, std, key)


def _select_diverse(
    rescored: list[tuple[Any, float, float, tuple]],
    diversity_key_fn: Callable[[Any], tuple] | None,
    top_n: int,
) -> list[Any]:
    """Diversity selection on rescored sorted output.

    ``rescored`` is already sorted by rank (best first). If
    ``diversity_key_fn`` is None, return the first ``top_n`` candidates.
    Otherwise: walk the sorted list selecting one candidate per diversity
    key, then fill remaining slots with best-overall candidates not yet
    selected.
    """
    if diversity_key_fn is None:
        return [item[0] for item in rescored[:top_n]]

    selected: list[Any] = []
    selected_dkeys: set = set()
    selected_keys: set = set()  # uniqueness-by-key_fn for second pass

    # First pass: one per diversity key
    for cand, _mean, _std, ckey in rescored:
        if len(selected) >= top_n:
            break
        dkey = diversity_key_fn(cand)
        if dkey not in selected_dkeys:
            selected.append(cand)
            selected_dkeys.add(dkey)
            selected_keys.add(ckey)

    # Second pass: fill remaining slots with best-overall not already selected
    if len(selected) < top_n:
        for cand, _mean, _std, ckey in rescored:
            if len(selected) >= top_n:
                break
            if ckey not in selected_keys:
                selected.append(cand)
                selected_keys.add(ckey)

    return selected


def _compute_jaccard(
    a: list[Any], b: list[Any], key_fn: Callable[[Any], tuple]
) -> float:
    """Jaccard similarity between two candidate lists, by ``key_fn``."""
    set_a = {key_fn(c) for c in a}
    set_b = {key_fn(c) for c in b}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def phase2_adaptive_rescore(
    candidates: list[Any],
    eval_fn: Callable[[Any, int], float],
    key_fn: Callable[[Any], tuple],
    score_direction: Literal["maximize", "minimize"],
    *,
    initial_pool_size: int,
    pool_size_progression: list[int],
    max_pool_multiplier: int = 8,
    top_n: int,
    n_seeds: int = 5,
    seeds: list[int] | None = None,
    halt_jaccard_threshold: float = 0.8,
    halt_overlap_threshold: int = -1,
    diversity_key_fn: Callable[[Any], tuple] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Any], dict]:
    """Adaptive-K multi-seed rescore with diversity selection.

    See module docstring for full spec. Returns
    ``(top_n_winners, halt_metadata)`` where ``halt_metadata`` is a dict with
    keys ``halt_reason``, ``final_pool_size``, ``expansions``,
    ``per_iteration_jaccards``, ``candidates_evaluated``.

    ``halt_reason`` is one of:
      * ``"converged"`` — top-N stabilized between iterations (adaptive mode).
      * ``"cap"`` — pool reached ``max_pool_multiplier * top_n`` or
        ``len(candidates)`` without converging (adaptive mode).
      * ``"single_iteration"`` — degenerate mode ran its one iteration.
      * ``"empty_input"`` — ``candidates`` was empty.
    """
    if score_direction not in ("maximize", "minimize"):
        raise ValueError(
            f"score_direction must be 'maximize' or 'minimize', got {score_direction!r}"
        )
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    if n_seeds <= 0:
        raise ValueError(f"n_seeds must be positive, got {n_seeds}")
    if not pool_size_progression:
        raise ValueError("pool_size_progression must be non-empty")

    if not candidates:
        return (
            [],
            {
                "halt_reason": "empty_input",
                "final_pool_size": 0,
                "expansions": 0,
                "per_iteration_jaccards": [],
                "candidates_evaluated": 0,
            },
        )

    if seeds is None:
        if n_seeds > len(DEFAULT_SEEDS):
            raise ValueError(
                f"n_seeds={n_seeds} exceeds DEFAULT_SEEDS length {len(DEFAULT_SEEDS)}; "
                "pass an explicit `seeds` list"
            )
        seeds = DEFAULT_SEEDS[:n_seeds]
    if len(seeds) < n_seeds:
        raise ValueError(
            f"seeds list has {len(seeds)} entries but n_seeds={n_seeds}"
        )
    seeds = list(seeds[:n_seeds])

    overlap_threshold = top_n - 1 if halt_overlap_threshold < 0 else halt_overlap_threshold
    cap = min(max_pool_multiplier * top_n, len(candidates))
    is_degenerate = len(pool_size_progression) == 1

    rescored_cache: dict[tuple, tuple[Any, float, float]] = {}
    prev_topN: list[Any] = []
    consecutive_stable = 0
    per_iteration_jaccards: list[float] = []
    iterations_done = 0
    final_pool_size = 0
    halt_reason: str | None = None

    for raw_K in pool_size_progression:
        # In degenerate mode we honor the literal pool size; in adaptive mode
        # we additionally bound by the cap so the helper can't run wider than
        # max_pool_multiplier * top_n even if the progression list specifies a
        # larger value.
        if is_degenerate:
            K = min(raw_K, len(candidates))
        else:
            K = min(raw_K, len(candidates), cap)
        if K <= 0:
            halt_reason = "empty_input"
            break

        # Rescore any candidate in pool[:K] not yet cached
        pool = candidates[:K]
        for cand in pool:
            ckey = key_fn(cand)
            if ckey in rescored_cache:
                continue
            scores: list[float] = []
            for seed in seeds:
                try:
                    s = float(eval_fn(cand, seed))
                except Exception:
                    s = float("-inf") if score_direction == "maximize" else float("inf")
                scores.append(s)
            mean, std, _ = _aggregate_scores(scores)
            rescored_cache[ckey] = (cand, mean, std)
            if progress_callback is not None:
                progress_callback(
                    len(rescored_cache), K, f"rescored candidate key={ckey}"
                )

        final_pool_size = K
        iterations_done += 1

        # Build sorted rescored from cache for the current pool only
        pool_keys = [key_fn(c) for c in pool]
        rescored = [
            (rescored_cache[k][0], rescored_cache[k][1], rescored_cache[k][2], k)
            for k in pool_keys
        ]
        rescored.sort(key=lambda x: _rank_key(x[1], x[2], x[3], score_direction))

        # Diversity selection for top-N
        current_topN = _select_diverse(rescored, diversity_key_fn, top_n)

        if is_degenerate:
            halt_reason = "single_iteration"
            prev_topN = current_topN
            break

        # Stability check on adaptive iterations (after first)
        if prev_topN:
            jaccard = _compute_jaccard(prev_topN, current_topN, key_fn)
            overlap = len(
                {key_fn(c) for c in prev_topN} & {key_fn(c) for c in current_topN}
            )
            per_iteration_jaccards.append(jaccard)

            stable = (jaccard >= halt_jaccard_threshold) or (
                overlap >= overlap_threshold
            )
            if stable:
                consecutive_stable += 1
                if consecutive_stable >= 2:
                    halt_reason = "converged"
                    prev_topN = current_topN
                    break
            else:
                consecutive_stable = 0

        prev_topN = current_topN

        # Cap reached — exit after this iteration's work is recorded
        if K >= cap or K >= len(candidates):
            halt_reason = "cap"
            break

    if halt_reason is None:
        # Loop completed without converging or hitting cap explicitly — that
        # means we exhausted pool_size_progression. Treat as cap.
        halt_reason = "cap"

    return (
        prev_topN,
        {
            "halt_reason": halt_reason,
            "final_pool_size": final_pool_size,
            "expansions": max(0, iterations_done - 1),
            "per_iteration_jaccards": per_iteration_jaccards,
            "candidates_evaluated": len(rescored_cache),
        },
    )
