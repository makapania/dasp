# Implementation Plan — Option A: Pre-Fit Fingerprint Cache + `optuna.TrialPruned`

> **STATUS: SUPERSEDED 2026-05-06.** The TrialPruned design described
> below was implemented (commits `b73f22c`..`8778c9a`) and benchmarked,
> then reverted at `ed809f3` because PRUNED trials enter Optuna 4.8
> TPE's KDE with split-score `(1, 0.0)` — different from a duplicate
> COMPLETE-with-real-value's update. Multi-seed bench at 300 trials
> showed only 22-35% fingerprint overlap pre/post; pre's best
> fingerprint was NOT reached by post in any seed. The shipped design
> uses the same fingerprint hash but **value-cache-and-replay** instead
> of TrialPruned: duplicate trials return the prior trial's cached
> metric value, so TPE sees identical (params, value) pairs to a
> pre-dedup re-fit. KDE history is bit-identical → same parameter
> space, same models, just no redundant fits. See
> `src/spectral_predict/unified_bayesian.py:262-313` for the helpers
> and the 2026-05-06 entry in `docs/SESSION_LOG.md` for the full writeup.
> This document is preserved as the historical record of why the
> straightforward TrialPruned approach was abandoned.

**Branch context:** `docs/2026-05-07-final-wrapup-and-continuation`, builds on commits `9b86bc9 / a64004f / 727077f` (LVs reporting fix).

**User-locked decisions (2026-05-09):**
- **Mechanism (b)** — narrow `terminal_states` to `(COMPLETE,)` so 300 set = 300 unique fits. Wall-clock approximately equal to pre-fix; the "savings" appear as ~33% more useful search coverage per wall-clock-minute (75 fit-slots that today go to redundancy go to fresh fingerprints).
- **Higher fingerprint bar kept** — defensive redundancy is ~5 LOC + ~25 test LOC, sub-microsecond per trial; cheap insurance.
- **Missing one duplicate is acceptable** — fingerprint completeness risk drops from HIGH to MEDIUM. Defensive runaway cap is "log only" (F9 retained as warning).

---

## 1. Executive summary

Option A replaces the silent compute waste of duplicate fits (Problem 3a clamp collapses + Problem 3b TPE re-suggests) by hashing every trial's *fully-resolved, post-clamp* configuration before the fit and raising `optuna.TrialPruned` on a hit. The fingerprint must use clamped values (the entire purpose), live in the closure of `create_unified_objective` so it shares lifecycle with the existing preprocessing/region/importance caches, and be reconstructed on resume from `study.trials` user_attrs to avoid double-counting across sessions. To preserve the user-visible contract that "300 trials" means 300 unique fits, we narrow `terminal_states` at `unified_bayesian.py:~2518` and `~2557` to `(COMPLETE,)` so PRUNED trials don't burn the budget — Mechanism (b). The broad `except Exception` at `:~1689` must be preceded by an explicit `except optuna.TrialPruned: raise` so the prune isn't downgraded to a 1e10 penalty (which would re-introduce duplicates as failed-but-completed rows). Sister sites: `search.py` legacy bayesian path needs the same treatment; `bayesian_utils.create_objective_function` is structurally similar; `nsga2_search.py` is structurally immune (pymoo, not Optuna).

---

## 2. Mechanism choice — **(b)** with one explicit refinement

**Pick (b): narrow `terminal_states` to `(TrialState.COMPLETE,)` so the resume-budget clamp counts only successful unique fits against the user's `n_trials`.**

Reasoning:
- Optuna's `study.optimize(n_trials=N)` runs N callable invocations regardless of state. So the dedup loop **gets free re-draws inside one `optimize` call**: every prune triggers another sample. Mechanism (b) only changes the **resume math** (the `already_finished = sum(... t.state in terminal_states)` clamp at `:~2519` and `:~2555`).
- This means the Mechanism-(a) outer "increment `n_trials` in a callback" approach is unnecessary inside a fresh run.
- Mechanism (b) cleanly preserves the audit trail: Optuna stores PRUNED rows in the DB so the dedup rate is greppable from the study after the fact.

**Refinement:** Mechanism (b) alone is *not* sufficient on resume into a new `n_trials` target. If a study has 50 COMPLETE + 30 PRUNED and the user re-launches with `n_trials=100`, the original code computed `already_finished = 80`, `remaining = 20`. Post-fix it must compute `already_finished = 50` (COMPLETE only), `remaining = 50` — to give the user the 100 unique fits they asked for. That's exactly what narrowing `terminal_states` accomplishes. Confirm both sites — line ~2518 and line ~2557 (the post-migration restart) — get the same narrowing.

**Codex-anticipated objections, with rebuttals:**

| Objection | Response |
|---|---|
| "Mechanism (b) lets a pathological dataset inflate fits indefinitely — what if every trial prunes?" | The duplicate rate is bounded by the Cartesian product of the clamped search space. For PLS on n_features=10 (worst case in the diagnostic CSV), n_components ∈ [2,9] gives 8 distinct values × ~150 preprocessing/subset combinations × 5 cv folds-fixed × random_state-fixed ≈ ~6000 unique fingerprints. We need n_trials < ceiling. Add a defensive cap: if `study.trials` length exceeds `2 * n_trials_target`, log a warning and let the user decide. Implement as a safety valve, not a primary mechanism. |
| "What if `TrialPruned` propagates to the GUI as an error popup?" | The progress_callback wrapper at `:~2322` only acts on `cb_study`/`trial`; PRUNED trials still trigger the callback (Optuna calls it for every trial regardless of state). The wrapper must check `trial.state` before reading `.value`/`.user_attrs`. Existing code at `:~2500` already filters `if trial.value is not None and trial.value < 1e9` — that path is safe for None values. |
| "Why not just use `study.tell(trial, state=TrialPruned)` via the manual API?" | Inline `raise TrialPruned()` is the documented Optuna idiom; `tell()` is for ask-and-tell pattern which we don't use. Using both would split the seam. |

**Third option considered and rejected:** A "post-suggest, before-fit" `study.add_trial(FrozenTrial(state=PRUNED))` shim. This would let us mark the trial PRUNED without raising, but it requires manipulating the trial's lifecycle from inside the objective, which Optuna's contract forbids. Stick with `raise TrialPruned()`.

---

## 3. Fingerprint specification — exhaustive

The fingerprint must uniquely identify the *fitted* model. Anything that differs across two trials and changes the fit must be in the tuple; anything constant per study (cv_strategy, hoisted to `study.user_attrs`) **need not** be in the per-trial fingerprint but MUST be in the closure-level salt to handle resume into a different config (covered by the existing `study_name` config_hash at `:~2209` — so we don't need to re-encode it in the fingerprint).

### Universal dimensions (all model families)

```python
fingerprint = (
    # Preprocessing block — mirrors the cache_key at :~908
    preprocess_config['name'],            # 'raw' | 'snv' | 'snv_deriv1' | ...
    preprocess_config.get('deriv', 0),
    preprocess_config.get('window', 0),
    preprocess_config.get('polyorder', 0),
    preprocess_config.get('apply_baseline', False),
    preprocess_config.get('apply_smoothing', False),
    preprocess_config.get('apply_autoscale', False),  # T-36

    # Variable selection block
    subset_type,                          # 'importance' | 'cars' | 'region' | 'uve'
    subset_size,                          # 'full' | 10 | 20 | 50 | ...
    subset_tag,                           # final post-subsetting tag (e.g. 'top10_importance')
    n_vars,                               # actual feature count post-subset
    tuple(np.asarray(top_indices).tolist()) if top_indices is not None else None,
        # Selected wavelength indices — STORE THE TUPLE, not hash().
        # Two different importance orderings producing the same set in the same order are
        # actually the same fit; Python's tuple-of-ints is hashable and safe across runs.

    # Model block
    model_name,                           # post-normalization (uppercase 'PLS' etc.)
    frozenset(model_params.items()),      # POST-CLAMP params from suggest_model_params

    # Imbalance / weighting (defensive — encoded in model_name + imbalance_method already,
    # but cheap belt-and-suspenders since "missing one dup is acceptable" but extra coverage
    # is sub-microsecond)
    imbalance_method or None,
    repr(sorted((imbalance_params or {}).items())),
    use_sample_weight_for_classification,
)
```

**`cv_folds`, `cv_strategy`, `cv_n_repeats`, `random_state`, `early_stopping_rounds`** are CONSTANT per study (hoisted to `study.user_attrs` at `:~2300`, baked into `study_name` at `:~2209`). Not needed in per-trial fingerprint.

### Per-model-family clamped-params validation

| Model | Params from `suggest_model_params` (post-clamp) | Clamp risk? | Fingerprint covers? |
|---|---|---|---|
| **PLS / PLS-DA** | `n_components` (post-clamp at :462), `max_iter` (fixed 500), `tol` (fixed 1e-6) | YES — main case for 3a | YES — `frozenset(model_params.items())` includes clamped n_components |
| **Ridge / Lasso / ElasticNet** | `alpha` (suggest_float log), optional `l1_ratio` | NO clamp | YES |
| **RandomForest** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features='sqrt'` | NO clamp | YES |
| **LightGBM** | `n_estimators`, `learning_rate`, `num_leaves` (suggest-bound clamp at :510-514), `max_depth`, fixed extras | suggest-bound (TPE sees the bound) | YES — clamped value is the suggested value |
| **XGBoost** | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, fixed extras | NO clamp | YES |
| **CatBoost** | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg` | NO clamp | YES |
| **SVM / SVR** | `C`, `kernel`, `epsilon` (regression only), `gamma` (rbf only) | conditional params (kernel-dependent) | YES — frozenset captures present keys |
| **MLP** | `hidden_layer_sizes` (derived tuple), `alpha`, `learning_rate_init`, fixed `max_iter` | derived (n_layers + hidden_size → tuple) | YES — derived tuple is hashable |
| **PLS-DA tail LR** | `lr_C`, `lr_solver`, `lr_max_iter` (when present in model_params dict) | NO clamp | YES if model_params includes them |

**One-class models** (`PCA-SIMCA`, `OneClassSVM`, `IsolationForest`, `EllipticEnvelope`, `LOF`):
- Suggest path: `suggest_one_class_params` at :608-636. Returns dict; assigns to `oc_params`.
- Trial body for OC at :946-1121 has its **own** subsetting + `oc_params` flow; does not call `suggest_model_params`. Fingerprint must be inserted in the OC branch with `oc_params` in place of `model_params`.

### Edge cases for fingerprint completeness

1. **Float dimensions:** `frozenset({'alpha': 0.123456789})` is hashable as long as the float is exactly equal. Optuna's TPE+seed determinism does sometimes produce identical floats at boundaries — that's exactly what 3b is.
2. **`top_indices` order vs. set:** `np.argsort(importances, kind='stable')[-n_vars:]` is deterministic for a given importances array (which itself is cached per `(cache_key, subset_type, model_name, task_type)`). Use ordered tuple.
3. **`use_sample_weight_for_classification`:** derived at :1270-1288 from `imbalance_method`, `model_name`, and `model.fit` signature inspection. Defensive redundancy — kept per user direction (sub-microsecond cost, cheap insurance).
4. **PLS-DA sklearn pipeline keys (`pls__n_components` vs. `model__n_components` vs. bare):** Use `model_params` (bare keys) for fingerprinting, not `captured_params` (post-fit, prefix variability).
5. **`hash(tuple(...))` is process-local — DO NOT use it.** Use the actual tuple/frozenset as the dict key. For persistence to user_attrs (resume), serialize via `repr(tuple)` (round-trips through `ast.literal_eval`).
6. **Resume reconstruction:** on study load, before calling `study.optimize`, walk `study.trials` (only `state == COMPLETE`) and re-build the `seen_fingerprints` dict by reading each trial's stored `fingerprint` user_attr. PRUNED trials don't need their fingerprints rehydrated.
7. **The `n_components > n_features_final` early-return at :1224-1229:** Convert :1229 to `raise optuna.TrialPruned()` for consistency. Methodology-neutral simplification.

---

## 4. Code-site map

Authoritative line numbers should be re-grepped against `unified_bayesian.py` at implementation time.

| Site | File:Line | Current | Change |
|---|---|---|---|
| **F1. Fingerprint state in objective closure** | `unified_bayesian.py:~891` (existing cache block: `region_cache`, `importance_cache`, `preprocessing_cache`) | three caches | Add `seen_fingerprints: Dict[Tuple, int] = {}` alongside |
| **F2. Resume rehydration** | `unified_bayesian.py:~2270` (after study creation, before optimize) | none | Iterate `study.trials`, `if t.state == COMPLETE and 'fingerprint' in t.user_attrs: seen_fingerprints[ast.literal_eval(t.user_attrs['fingerprint'])] = t.number`. Wire `seen_fingerprints` into the closure via a parameter on `create_unified_objective` (don't rely on closure capture alone — explicit is better). |
| **F3. Fingerprint construction site** | `unified_bayesian.py:~1217` (right after `model_params = suggest_model_params(...)` at :1215, after the PLS clamp validation at :1219-1229) | none | Build the `fingerprint` tuple as specified in §3; check membership; if hit → `raise optuna.TrialPruned()`; else `seen_fingerprints[fingerprint] = trial.number; trial.set_user_attr('fingerprint', repr(fingerprint))` |
| **F4. One-class fingerprint site** | `unified_bayesian.py:~1054` (right after `oc_params = suggest_one_class_params(trial, model_name)`) | none | Same shape but using `oc_params` and OC subset state |
| **F5. PLS too-many-components branch** | `unified_bayesian.py:~1229` | `return 1e10` | `raise optuna.TrialPruned()` (consistency; methodology-neutral — neither path stores model_params anyway) |
| **F6. Trial-body broad except** | `unified_bayesian.py:~1689` | `except Exception as e:` | Add **before** it: `except optuna.TrialPruned: raise` |
| **F7. Resume budget — narrow terminal_states** | `unified_bayesian.py:~2518` | `terminal_states = (TrialState.COMPLETE, TrialState.PRUNED)` | `terminal_states = (TrialState.COMPLETE,)` |
| **F8. Auto-migration restart — narrow** | `unified_bayesian.py:~2557` | `t.state in (_TS_post.COMPLETE, _TS_post.PRUNED)` | `t.state in (_TS_post.COMPLETE,)` |
| **F9. Defensive runaway cap (log-only)** | `unified_bayesian.py:~2534` (just before `study.optimize(n_trials=remaining_trials, ...)`) | none | Optional safety: if `len(study.trials) > 2 * n_trials_target`, `logger.warning("dedup is consuming the search space; %d trials drawn for %d unique fits", ...)`. No abort, just visibility. |
| **F10. Inline check vs. callback decision** | `unified_bayesian.py:~1217` (the F3 site) | inline raise | Use **inline `raise optuna.TrialPruned()`** — direct, no callback registration, no double-trip through Optuna's pruning hook system. |

### Sister site `search.py` legacy Bayesian path

| Site | File:Line | Change |
|---|---|---|
| **S1. Trial body** | `bayesian_utils.py:~191-690` (`create_objective_function` body) | Add fingerprint dict at outer scope, fingerprint construction inside the trial body using the same dimension list as F3 (adapted to this seam's `model_params` shape) |
| **S2. Broad except** | grep `bayesian_utils.py` for `except Exception` in the trial body and add `except optuna.TrialPruned: raise` before each |
| **S3. Optimize call** | `search.py:~3994-3999` (`study.optimize(objective_fn, n_trials=n_trials, ...)`) | Pre-call: count `study.trials` filtered to `state == COMPLETE`; pass `n_trials = max(0, n_trials_target - already_complete)` (mirrors `unified_bayesian.py:~2527`). Today's `search.py` doesn't have a resume clamp — the optimize call uses raw `n_trials`. **This is a pre-existing gap**, not introduced by Option A; fix it here for completeness. |
| **S4. Convert path** | `bayesian_utils.py:~770-773` | Already filters `state != COMPLETE` — no change needed |

### Sister site `nsga2_search.py`

**Verdict: structurally immune.** Uses pymoo (NSGA-II algorithm) not Optuna. Different fitness-evaluation model. Genetic crossover *can* produce duplicate genomes, but pymoo handles that internally via `eliminate_duplicates=True` (default). No `TrialPruned` mechanism. No change.

### GUI

`spectral_predict_gui_optimized.py` exposes `n_unified_trials` as a target. Post-fix, the user's "300" still produces 300 unique fits (because Mechanism (b) makes optimize run until 300 COMPLETE trials accumulate). **No GUI change needed for n_trials semantics.** Optional follow-up: surface dedup rate in the progress wrapper ("Drew N candidates → 300 unique fits"). Defer.

---

## 5. Resume semantics — four scenarios

| Scenario | Pre-fix behavior | Post-fix behavior |
|---|---|---|
| **(a) Fresh run, n_trials=300** | 300 trials run; some are duplicates of each other (4-70 dup rate observed) | 300 *unique* fits run; ~330-450 callable invocations (Optuna re-draws on prune); duplicate trials end as PRUNED rows in the study DB. |
| **(b) Resume mid-run (e.g. 150 COMPLETE + 30 PRUNED of original target 300, user re-launches with n_trials=300)** | `terminal_states` includes PRUNED → already_finished=180, remaining=120 → final total: 300 trials, but only ~225 unique fits | already_finished=150 (COMPLETE only), remaining=150 → final total: 300 unique fits guaranteed. PRUNED trials from the prior session do NOT count against remaining. **Critically:** rehydrate `seen_fingerprints` from the 150 COMPLETE trials so the next 150 don't re-collide with prior fits. |
| **(c) Resume after completion (300 COMPLETE + 87 PRUNED, re-launch with n_trials=300)** | `already_finished=387`, `remaining=0` → "All trials already complete; skipping" | `already_finished=300`, `remaining=0` → "All trials already complete; skipping." Same behavior, but math is cleaner. |
| **(d) Resume into a different n_trials (e.g. originally 100, completed 100 unique + 23 PRUNED, re-launch with n_trials=200)** | `already_finished=123`, `remaining=77` → user gets only 77 fresh fits where they expected 100 fresh (200 total) | `already_finished=100`, `remaining=100` → user gets 100 fresh unique fits, total 200 unique fits as requested. **`seen_fingerprints` rehydrated from all 100 COMPLETE trials** so the 100 new ones avoid them. |

**Resume guard:** the `study_name` includes a `config_hash` of cv_strategy, random_state, imbalance, baseline, etc. (`:~2209`). Loading a study with mismatched config hash would create a NEW study, not load incompatible trials. So the fingerprint dimension list correctly excludes those (already gated upstream).

---

## 6. Test plan

### 6.1 Unit tests — new `tests/test_bayesian_dedup.py`

Following the existing `tests/test_cv_pls_clamp.py` pattern (synthetic data, small n_trials, black-box assertions on the result DataFrame and the study object).

```
tests/test_bayesian_dedup.py:
  TestFingerprintConstruction
    - test_pls_post_clamp_n_components_in_fingerprint
        synthetic n_features=10, run 25 trials, assert no two COMPLETE trials
        in study have identical (preprocess, subset_tag, n_vars, frozenset(model_params)) tuple
    - test_lightgbm_suggest_bound_clamp_in_fingerprint
        same assertion for LightGBM
    - test_one_class_pca_simca_fingerprint
        OC path with PCA-SIMCA, assert no duplicate (preprocess, oc_params) pairs
  TestPrunedExceptionPropagation
    - test_pruned_not_swallowed_by_broad_except
        monkeypatch to force a duplicate fingerprint on the 5th call;
        assert trial 5 ends in TrialState.PRUNED, not COMPLETE with value=1e10
  TestBudgetSemantics
    - test_n_trials_300_yields_300_unique_complete
        run with n_trials=20 (scaled-down for CI), assert
        len([t for t in study.trials if t.state == COMPLETE]) == 20
    - test_resume_does_not_double_count_pruned
        run 1: n_trials=10. Verify k COMPLETE + m PRUNED.
        run 2 (load_if_exists, same config): n_trials=15.
        Assert COMPLETE total == 15 (NOT 15 - m).
  TestResumeRehydration
    - test_resume_rehydrates_seen_fingerprints
        run 1: 10 trials, capture set of fingerprints from user_attrs.
        run 2: monkeypatch suggest_model_params to force the SAME 10 params again.
        Assert all 10 immediately PRUNED (none re-fitted).
```

### 6.2 Integration tests — extend `tests/test_cv_pls_clamp.py`

Add to `TestRunBayesianSearchPLSGridClamping`:
```
test_n10_loo_bayesian_no_duplicate_fits
    Same setup as existing test_n10_loo_bayesian_caps_at_9.
    Additional assertion: df.duplicated(subset=DEDUP_COLS).sum() == 0
    where DEDUP_COLS == DeepSeek's recommended subset from continuation prompt:
    ['PreprocessBase','Deriv','Window','Poly','Autoscale','baseline_method',
     'smoothing','n_vars','SubsetTag','Model','Params','imbalance_method',
     'top_vars','all_vars']
```

### 6.3 A/B harness — `tools/ab_dedup_compare.py` (new)

User-mandated: A/B test on `example/` folder data BEFORE merging.

Models to cover:
1. **PLS regression** — example folder dataset with regression target
2. **LightGBM regression** — same dataset, LightGBM target
3. **LightGBM classification** — example folder dataset with class target

For each:
- Run with `--tag old` on git ref before Option A merge (`727077f`)
- Run with `--tag new` on Option A branch
- `--diff` validates:
  - For each NEW row, find a matching OLD row with byte-identical model-fit columns (`Params`, `Preprocess`, `PreprocessBase`, `Deriv`, `Window`, `Poly`, `Autoscale`, `n_vars`, `SubsetTag`, `RMSE`, `R2`, `RMSEcv`, `R2cv`, `MAEcv` for regression; analogous for classification).
  - Assertion: top-N rows of NEW (sorted by primary metric) all appear in OLD's top-(N + dup_count).
  - **Comparable budget:** call OLD with `n_trials=300` (raw); call NEW with `n_trials=N_unique_old` where `N_unique_old = len(OLD.drop_duplicates(...))`. This makes the post-fix run target the same number of unique fits the pre-fix run actually produced.
- Tolerance: model-fit columns must be byte-identical (string compare) with seeds fixed to `random_state=42`. RMSEcv / Accuracycv compared at full float64 precision (the harness's existing 8-decimal pin from the continuation prompt).

### 6.4 Pytest invocations

```
# Unit + integration
pytest tests/test_bayesian_dedup.py -v
pytest tests/test_cv_pls_clamp.py::TestRunBayesianSearchPLSGridClamping -v

# Run the existing LV-reporting regressions to confirm no regression
pytest tests/test_cv_pls_clamp.py::TestLVsReportingMatchesFittedValue -v
pytest tests/test_t41_bayesian_sqlite_auto_calculator.py -v  # resume-path coverage

# A/B harness (manual, before/after merge)
git checkout 727077f && python tools/ab_dedup_compare.py --tag old --model PLS --task regression
git checkout option-a-branch && python tools/ab_dedup_compare.py --tag new --model PLS --task regression
python tools/ab_dedup_compare.py --diff --model PLS --task regression

# Repeat for LightGBM regression and LightGBM classification
```

---

## 7. Sister-site verdict

| Site | Verdict | Reason |
|---|---|---|
| `unified_bayesian.py` (primary) | **Needs change** (F1-F9) | Where the bug originates; main beneficiary |
| `unified_bayesian.py` one-class branch | **Needs change** (F4) | Shares the trial body, has its own suggest path; same TPE re-suggest issue |
| `search.py` legacy `run_bayesian_search` | **Needs change** (S1-S3) | Same Optuna + TPE shape; carries the same 3a/3b risks. Code path is "test-only" per comment at search.py:3872 (no GUI caller), but `tests/test_cv_pls_clamp.py::TestRunBayesianSearchPLSGridClamping` does exercise it — keep it correct. |
| `bayesian_utils.create_objective_function` | **Needs change** | The objective-function factory the legacy path uses; mirror F3+F6 there |
| `bayesian_utils.convert_optuna_result_to_dasp_format` | **No change needed** | Already filters `state != COMPLETE` at line ~772 — PRUNED trials drop out |
| `nsga2_search.py` | **Structurally immune** | pymoo (genetic) not Optuna; pymoo's `eliminate_duplicates=True` default already handles genome dups; no TPE re-suggest concept |
| `spectral_predict_gui_optimized.py` | **No change needed** | `n_unified_trials` semantics are preserved by Mechanism (b) |
| `convert_study_to_dataframe` (`unified_bayesian.py:~2630`) | **No change needed** | Already filters PRUNED via `state != COMPLETE` at :~2686 |

---

## 8. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Fingerprint missing a dimension** → real model silently skipped | MEDIUM (lowered from HIGH per user direction: "missing one dup is acceptable") | §3 enumerates per-model-family. Add a unit test that constructs two trials with the SAME `model_params` post-clamp but DIFFERENT raw `trial.params` and verifies the second prunes. Add inverse: two trials with intentionally different model_params don't collide. |
| **`TrialPruned` swallowed** by broad except | HIGH | F6 explicit re-raise. Add unit test that monkeypatches the duplicate path and asserts state == PRUNED, not COMPLETE-with-1e10. |
| **`terminal_states` regression in T-41 auto-migration restart** | MEDIUM | Both sites (F7 + F8) must be narrowed in lockstep; include a test that triggers the auto-migration path (T-41 test file `tests/test_t41_bayesian_sqlite_auto_calculator.py` already exercises this — extend rather than fork). |
| **Resume rehydration loses fingerprints** if user_attr write fails (e.g. SQLite locked, partial commit) | LOW | Storage failure of one user_attr is rare; the worst case is one duplicate gets re-fitted, which is acceptable per user direction. |
| **Study DB schema drift** — old DBs (pre-Option A) have no `fingerprint` user_attr | LOW | Resume code checks `if 'fingerprint' in t.user_attrs` before reading. Trials without it are NOT rehydrated → first new trial may collide with an old one and prune. Acceptable defensive behavior. |
| **Float dimensions clamp/round in future change** | LOW | Pin in unit test: `frozenset({'alpha': 0.123})` ≠ `frozenset({'alpha': 0.124})`; same for derived MLP `hidden_layer_sizes` tuple. |
| **Memory growth** of `seen_fingerprints` for very large `n_trials` (10k+) | LOW | Tuples + frozensets are tiny; 10k entries ≈ a few MB. Not a concern at user-facing budgets. |
| **Pathological dataset where most trials prune** | MEDIUM | Defensive cap at F9: log warning if `len(study.trials) > 2 * n_trials_target`. |
| **Pre-existing `search.py` resume gap (S3)** | LOW | Already broken; Option A makes the in-flight optimize loop correct but doesn't introduce the gap. Fix included per "fix here for completeness" choice. |
| **A/B test methodology — "comparable budget" choice** | MEDIUM | §6.3 picks "post-fix targets unique-fit count of pre-fix" as the comparable. Document in the PR. |
| **Wall-clock expectation** | LOW | User-aware: Mechanism (b) is wall-clock-neutral; the "savings" appear as 33% more useful coverage at the same time cost, not less time. Documented in PR description so users don't expect a faster run. |

---

## 9. Decided / open questions

| # | Question | Resolution |
|---|---|---|
| 1 | Should the `n_components > n_features_final` branch (F5) become `raise TrialPruned` or stay `return 1e10`? | **Resolved:** TrialPruned for consistency. Methodology-neutral. |
| 2 | Defensive runaway cap (F9): log-and-continue, or hard abort? | **Resolved:** log-and-continue per user direction. |
| 3 | Search.py legacy `run_bayesian_search` resume gap (S3): fix here or separate ticket? | **Resolved:** fix here for completeness. |
| 4 | A/B harness budget choice: target same n_unique_fits in both runs, or target same `n_trials=300`? | **Resolved:** target same n_unique_fits (per §6.3). |
| 5 | GUI surface for "drew N drew K unique": needed now or follow-up? | **Resolved:** follow-up. |
| 6 | Should the fingerprint be persisted to `study.user_attrs['fingerprint']` or only `trial.user_attrs['fingerprint']`? | **Resolved:** per-trial user_attr (different per trial). |

---

## 10. Files to touch

- `src/spectral_predict/unified_bayesian.py` (primary)
- `src/spectral_predict/bayesian_utils.py` (sister site)
- `src/spectral_predict/search.py` (legacy bayesian path resume gap)
- `tests/test_bayesian_dedup.py` (new)
- `tests/test_cv_pls_clamp.py` (extend)
- `tools/ab_dedup_compare.py` (new, sister-tool to `tools/ab_lv_compare.py`)

**Estimated effort:** ~2 hours implementation + ~1 hour tests + A/B harness run (manual, ~30 min) + cross-family review.
