# T-42 Bayesian Write-Path Plumbing — Approach C Validation

**Branch:** `fix/T42-write-path-plumbing-approach-c`
**Plan:** `docs/plans/2026-05-01-T42-bayesian-write-path-plumbing.md`
**Status:** IMPLEMENTED.

## Surprise finding (changes T-42's framing)

Before implementing Approach C, the baseline benchmark
(`tests/_bench_bayesian_per_model.py`, n_trials=10, synthetic
n=100, n_features=200) on top of `fix/T43-resume-auto-restore-settings`
already meets the plan's "definition of done" target without any
T-42 work:

| Model | In-memory | SQLite WAL | Ratio | Plan target |
|---|---|---|---|---|
| PLS | 0.28 s | 0.20 s | **0.69×** | (n/a — auto-calc keeps PLS in-memory) |
| Ridge | 0.18 s | 0.19 s | **1.06×** | (n/a) |
| RandomForest | 14.58 s | 3.57 s | **0.25×** | (caching noise) |
| LightGBM | 2.35 s | 2.19 s | **0.93×** | ≤ 1.30× |
| XGBoost | 4.06 s | 4.09 s | **1.01×** | ≤ 1.15× |

The plan quoted XGBoost 1.36× and LightGBM 1.89× from pre-T-41
measurements. T-41's WAL pragmas + 30 s SQLite `busy_timeout`
collapsed the per-trial finalize cost from the plan-quoted
~200 ms to near-noise. **Flipping T-41's default from `'never'`
to `'auto'` is justified by the bench data alone — no longer
gated on T-42.** That is now a trivial follow-up ticket.

## What Approach C does anyway

The plan asked for the implementation. It is correct cleanup
regardless of the perf data:

- `cv_strategy` and `cv_n_repeats` were written on every trial
  but read by nobody. `convert_study_to_dataframe` takes them as
  function parameters, never inspects `trial.user_attrs`. Pure
  dead writes.
- `early_stopping_rounds` is constant per study (depends only on
  the model name and the call's `early_stopping_rounds` argument)
  but was being re-emitted on every trial. Hoist to
  `study.user_attrs` and read it once outside the trial loop.

### Implementation

`src/spectral_predict/unified_bayesian.py`:

1. Removed two dead `trial.set_user_attr` calls in the one-class
   path (`cv_strategy`, `cv_n_repeats`).
2. Removed three calls in the regression / classification path
   (`cv_strategy`, `cv_n_repeats`, `early_stopping_rounds`).
3. Added three `study.set_user_attr` calls right after the
   `optuna.create_study` call site so the values land on the
   in-memory study (also persists across `optuna.copy_study`
   migration to SQLite per Optuna 4.8 docs).
4. `convert_study_to_dataframe` now reads `early_stopping_rounds`
   once via `study.user_attrs.get(...)` outside the trial loop;
   trial-level fallback covers studies finalized before T-42
   (resume path picks up legacy SQLite files unchanged).

## Investigation: actual `set_user_attr` counts per trial

`tests/_bench_t42_set_user_attr_count.py` monkey-patches
`Trial.set_user_attr` to count calls per trial.

**Before Approach C:**

| Model | Task | Calls / trial | Unique keys |
|---|---|---|---|
| PLS | regression | 30 | 29 |
| Ridge | regression | 30 | 29 |
| PLS-DA | classification | 42 | 41 |

The plan's pre-implementation estimate was 25; the truth is
30 for regression and 42 for classification.

**After Approach C:**

| Model | Task | Calls / trial | Δ |
|---|---|---|---|
| PLS | regression | 27 | −3 |
| Ridge | regression | 27 | −3 |
| PLS-DA | classification | 39 | −3 |

The 3 dropped writes per trial match the 3 hoisted keys exactly.

## Post-fix benchmark

Re-run of `tests/_bench_bayesian_per_model.py` after Approach C
(n_trials=10, same synthetic data):

| Model | In-memory | SQLite WAL | Ratio | Δ vs pre-C |
|---|---|---|---|---|
| PLS | 0.29 s | 0.20 s | 0.70× | +0.01× (noise) |
| Ridge | 0.19 s | 0.20 s | 1.01× | −0.05× (improved) |
| RandomForest | 14.83 s | 3.65 s | 0.25× | (cached) |
| LightGBM | 2.08 s | 2.10 s | 1.01× | +0.08× (noise) |
| XGBoost | 3.85 s | 3.93 s | 1.02× | +0.01× (noise) |

The 3-write savings is dominated by run-to-run noise at this
sample size. The cleanup matters because:

- Each removed write is an SQLite UPDATE under WAL — at scale
  (300 trials × 5 models × 3 writes = 4 500 writes / run avoided)
  it is non-zero work.
- Removing dead writes makes future profiling and audits easier.

## Test coverage

`tests/test_t42_write_path_plumbing.py` — 6 cases, all green:

- `test_constants_hoisted_to_study_user_attrs`
- `test_early_stopping_hoisted_for_xgboost` (also asserts absence on
  trial.user_attrs — DeepSeek review LOW #3)
- `test_study_user_attrs_survive_copy_study_migration` (DeepSeek review
  MEDIUM #2 — verifies the T-41 auto-migration preserves the hoisted
  keys; same trap class as the
  `create_study(load_if_exists=True, sampler=...)` silent-ignore)
- `test_no_per_trial_writes_for_hoisted_keys`
- `test_convert_study_reads_hoisted_early_stopping_rounds`
- `test_legacy_study_fallback_to_trial_user_attrs`

`tests/test_cv_strategy.py::test_one_class_bayesian_writes_cv_strategy_to_trial`
updated to read from `study.user_attrs` instead of `best.user_attrs`
(DeepSeek review HIGH #1 — pre-existing test that still asserted the
old per-trial contract; the original audit doc's "132/132 green" claim
was wrong because that test file was excluded from the initial sweep).

Post-fix regression sweep — **225/225** across `test_t42`,
`test_cv_strategy`, `test_t41`, `test_run_state`,
`test_t43_resume_auto_restore`, `test_unified_bayesian_baseline`,
`test_autoscale_bayesian`, `test_cv_pls_clamp` — all green.

## Out of scope (deferred follow-ups)

- **Flipping T-41 default from `'never'` to `'auto'`.** Justified by the
  baseline numbers above; trivial one-liner. File as a separate ticket.
- **Approach A (batched JSON blob for the remaining ~27 writes/trial).**
  Bench shows post-T-41 + Approach C is already well below the plan's
  perf target. No need to add the read-path complexity Approach A
  would require.
- **Approach D (periodic-snapshot architecture).** Same — not needed.
