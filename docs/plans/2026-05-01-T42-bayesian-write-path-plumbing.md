# T-42: Bayesian Write-Path Plumbing — Reduce Per-Trial Overhead Before `auto` Persistence Becomes Default

> **Status:** ROUGH_PLAN — investigation needed before specifying tasks. T-41 ships first with `default='never'`; T-42 is the prerequisite for raising the default to `'auto'`.
>
> **Dependency chain (read this before doing anything):**
> 1. **Pre-T-41 (current main):** All Bayesian SQLite uses Optuna's default journal mode (no WAL). PLS pays 28× slowdown, LightGBM ~1.9×, XGBoost ~1.36×.
> 2. **Post-T-41 (currently being implemented on `fix/T41-bayesian-sqlite-auto-calculator`):** WAL pragmas applied at migration time. PLS would pay 8× *if* the auto-calc routed it to SQLite, but it doesn't (median fit < 1s → in-memory). XGBoost on auto-calc-selected SQLite pays 1.36× (WAL ratio).
> 3. **Default-off in T-41 means WAL is essentially never exercised in practice** unless the user explicitly picks `'always'`. So the WAL pragmas are correct future-proofing but not a user-felt speedup at default settings.
> 4. **T-42 is the next step ONCE T-41 has merged**: drive the per-trial finalize overhead lower so that even XGBoost on `'auto'`-selected SQLite is fast enough that flipping the default from `'never'` to `'auto'` is a no-regret change.

**Goal:** Reduce the per-trial Optuna SQLite write overhead from ~200ms (with WAL) down to a level where `'auto'`-selected SQLite mode is acceptable as the default — i.e., heavy models pay ≤ 1.1× and the auto-decision threshold can be pushed below 1.0s median fit time without taxing the user.

**Source:** Conversation 2026-05-01 (post-T-41 implementation kickoff). User accepted T-41 with `default='never'` because `'auto'` mode still pays ~200ms/trial even after the auto-calc gates SQLite to slow models. Their framing: "before it could be used robustly there were also some plumbing changes we needed to make to make it faster." T-42 is the placeholder for that plumbing work.

---

## Why T-41 alone isn't enough

Empirical benchmark from T-41 (`tests/_bench_bayesian_per_model.py`):

| Model | In-mem fit | SQLite WAL | Slowdown |
|---|---|---|---|
| XGBoost | 534 ms | 728 ms | **1.36×** |
| LightGBM | 276 ms | 520 ms | **1.89×** |

T-41's auto-calc correctly migrates these to SQLite when median fit > 1.0s, but the user pays the slowdown anyway because the ~200ms overhead is roughly constant. For `'auto'` to be a sensible default, this constant has to come down.

---

## Where the 200ms goes (T-41 plan investigation, line 47-49)

> Each Bayesian trial calls `set_user_attr` ~25 times for metadata persistence (preprocessing fields, model params, n_vars, subset_tag, all_wavelengths, etc.). Per-call overhead ~5-10ms with default SQLite, ~2ms with WAL. Plus the trial-state COMMIT. Total per-trial finalize cost ~200ms with WAL.

So the cost is split roughly:
- **~50ms**: 25 `set_user_attr` calls (each is a separate UPDATE statement under WAL)
- **~150ms**: trial-state finalize COMMIT (Optuna's frame_state UPDATE + COMMIT)

The first chunk is the obvious target — 25 individual round-trips for what could be a single batched write.

---

## Candidate approaches (need to pick one before implementing)

### Approach A: Batch `set_user_attr` into a single JSON blob

Replace the 25 `trial.set_user_attr(key, value)` calls with one `trial.set_user_attr('dasp_metadata', json.dumps(...))`. Optuna user_attrs accept arbitrary serializable values, so storing a dict is supported.

**Pro:** Cuts ~25 round-trips to 1. Estimated ~50ms savings.
**Con:** Read path needs to know to deserialize the blob. Existing `convert_study_to_dataframe` and resume-on-startup code parse user_attrs by key — they'd need updating to read from the blob. Touches several call sites.
**Risk:** Low — the metadata is dasp-internal; Optuna doesn't care.

### Approach B: Move metadata out of `user_attrs` entirely

Use a parallel dasp-managed SQLite (or JSON sidecar) for the per-trial metadata; let Optuna's SQLite hold only what it needs (params, value, state, datetime). T-11 already has run_state machinery — this would extend it.

**Pro:** Removes ALL user_attr writes from the hot path; commit becomes pure Optuna state.
**Con:** Dual-write architecture. Resume-on-startup has to reconcile the two stores. Crash semantics get harder (what if Optuna commits but dasp's store doesn't?).
**Risk:** Moderate. The complexity might offset the savings.

### Approach C: Move heavy metadata to study-level `study.set_user_attr`

Some of the 25 keys are constant across trials (e.g., `subset_tag`, `all_wavelengths` — set once per Bayesian run, not per trial). Hoist those to `study.set_user_attr` at study-creation time so they aren't re-written on every trial finalize.

**Pro:** Cheapest implementation — purely a refactor of WHERE the writes happen, no schema change. Could save ~10-15 of the 25 calls.
**Con:** Doesn't address the fundamental per-trial COMMIT cost. Maybe gets us to ~150ms/trial instead of 200ms — meaningful but not transformative.
**Risk:** Very low. Likely the right first step regardless of which longer-term approach we pick.

### Approach D: Periodic-snapshot architecture (originally floated in T-41 out-of-scope)

Run fully in-memory always; have a background thread snapshot the in-memory study to SQLite every N seconds. Crash window = N seconds of work. The user's hot path never blocks on disk.

**Pro:** In-memory speed at all times. Scales to any model.
**Con:** Architectural change. Background thread + Optuna study state requires careful concurrency handling — Optuna studies aren't thread-safe by default. The snapshot has to copy under a lock.
**Risk:** Higher. Worth it only if A+C aren't enough.

---

## Suggested investigation order

1. **Measure** — instrument `unified_bayesian.py` to count and time the per-trial write breakdown empirically. Confirm the 25-call estimate and the 50/150 split. Drop a counter into `set_user_attr` and a timer around the COMMIT. Update the T-41 plan's Background section with the new numbers.

2. **Approach C first** (low-risk hoist) — list every key currently written via `trial.set_user_attr`, classify as per-trial vs per-run, move the per-run ones to `study.set_user_attr`. Re-bench. If this gets the slowdown ratio under 1.15× for XGBoost, T-42 may be done here.

3. **Approach A second** (batched blob) if C alone isn't enough. Update `convert_study_to_dataframe` and the resume-on-startup parser to read the blob.

4. **Approach D last resort** if A+C still leave heavy models above 1.15×.

---

## Out of scope for T-42

- The T-41 auto-calculator itself (already shipping with `default='never'`).
- Switching the default to `'auto'` — that's the *result* of T-42 succeeding, but it's a separate one-line ticket once the bench numbers justify it.
- Cross-platform tuning (Linux/macOS).

---

## Definition of done

- Bench `tests/_bench_bayesian_per_model.py` shows XGBoost SQLite WAL ratio ≤ 1.15× (down from 1.36×) and LightGBM ≤ 1.30× (down from 1.89×). Then we file the trivial follow-up ticket to flip T-41's default from `'never'` to `'auto'`.

## Effort estimate

Investigation + Approach C (hoist constants): ~1 day, low risk.
Plus Approach A if needed: another ~1 day, low-moderate risk.
Plus Approach D if needed: ~3-5 days, higher risk — but probably not needed.
