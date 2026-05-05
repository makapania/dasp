# Continuation prompt — 2026-05-09 — Bayesian search dedup follow-ups

## Where we left off (2026-05-08)

Shipped on `docs/2026-05-07-final-wrapup-and-continuation`:

| Commit | Tip | Change |
|---|---|---|
| `9b86bc9` | LVs reporting fix | `unified_bayesian.convert_study_to_dataframe` now reads `LVs` from `n_components_actual` user_attr (post-clamp) instead of `trial.params` (pre-clamp). Sister site fixed in `bayesian_utils.convert_optuna_result_to_dasp_format`. GUI Model Dev rebuild prefers `Params` over `LVs`. New `_extract_fitted_n_components` helper handles bare/`model__`/`pls__` keys. Three regression tests + A/B harness at `tools/ab_lv_compare.py`. |
| `a64004f` | NSGA-II sister site | Codex pre-merge review caught `nsga2_search.py:3979` calling `.get()` on `str(dict)`. Routed through the same helper. Test duplicate collapsed. |

A/B harness verification: 19 fit columns byte-identical pre/post fix; only `LVs` column changed on rows where the clamp had fired (18/22 rows in the test scenario).

Cross-family pre-merge review: GLM 5.1 → READY_TO_MERGE; Codex → NEEDS_FIX (closed by `a64004f`).

## What was empirically discovered but DEFERRED

While diagnosing the LVs reporting bug, three additional issues surfaced in `outputs/results_N_20260505_124946.csv` that the user prioritized below the LVs fix:

### Problem 3a — Clamp-induced duplicate fits (small, 4/300 rows)

When `_suggest_hyperparams` clamps Optuna's suggestion (e.g., 18 → 9 because n_features=10), different raw suggestions can collapse to the same fitted model. Two cases in the diagnostic CSV:

- `top10_importance` subset, n_vars=10 → fitted_nc=9: ranks 288 (LVs label 14) and 289 (LVs label 18), identical RMSEcv=0.87212573
- `top20_uve` subset, n_vars=20 → fitted_nc=19: ranks 264 (LVs 19) and 265 (LVs 20), identical RMSEcv=0.76209147

After 9b86bc9, both rows now correctly *display* LVs=9 and LVs=19 respectively, making the duplicates obvious in the CSV.

### Problem 3b — TPE re-suggesting the same point (large, 70/300 rows)

Optuna's TPE re-suggests the same parameter vector multiple times, especially at boundaries. Most striking: ranks 1–5 in the diagnostic CSV are all the same model fit 5 times (snv_deriv, deriv=1, window=33, poly=2, full features, n_components=20) — trials 67, 171, 173, 180, 181, RMSEcv=0.26600518 to 8 decimals.

This is independent of the clamp. TPE doesn't track "have I sampled this exact point before."

### Problem 3c — Fixed `n_components` suggest range with hidden clamp (methodology)

`unified_bayesian.py:457` suggests `n_components ∈ [2, 20]` regardless of feature count. The clamp at `:462` rescues the fit but Optuna's TPE believes it's exploring a wider space than it actually is. Comment at `:455` says fixed range was used "to avoid Optuna's dynamic space error" — DeepSeek confirmed Optuna 4.8.0 (the pinned version, see pyproject.toml) supports dynamic ranges, but `multivariate=True` TPE's joint KDE depends on stable support.

## Which option does what (fast scan)

| Option | Saves compute? | Fixes 3a (clamp dupes) | Fixes 3b (TPE re-suggest dupes) | Slow-fit model benefit (CatBoost / XGBoost / LightGBM) |
|---|---|---|---|---|
| **A** pre-fit fingerprint cache + `TrialPruned` | **Yes — proportional to dup rate** | Yes (all models) | Yes (all models) | **Large** — these models cost 10–60s per fit; 25% duplicates = 10–60 min saved per 300-trial run |
| **B** post-hoc dedup | **No** — duplicates still run | Yes (all models) | Yes (all models) | None — same wall-clock as today, just cleaner leaderboard |
| **C** data-aware `n_components` suggest range | Tiny — only the few clamp-impossible PLS trials | Yes (PLS only) | **No** — TPE still re-samples valid points | **None** — only PLS has the post-suggest n_components clamp; CatBoost/XGBoost/LightGBM clamp on suggest bounds, never had this bug |

**Headline:** Option A is the only one that saves wall-clock on CatBoost/XGBoost/LightGBM. Option B is the conservative "ship now" leaderboard cleanup. Option C is a PLS-specific methodology improvement; do not pick it expecting non-PLS speedups.

## Three candidate fixes (per cross-family review 2026-05-08)

### Option B — Post-hoc dedup *(lowest risk, recommended first)*

Add `df.drop_duplicates(subset=[...])` at `unified_bayesian.py:~2861` before sort/rank. Mirror at `search.py:~4051` for the legacy bayesian path.

DeepSeek's recommended subset:
```python
['PreprocessBase', 'Deriv', 'Window', 'Poly',
 'Autoscale', 'baseline_method', 'smoothing',
 'n_vars', 'SubsetTag', 'Model', 'Params',
 'imbalance_method', 'top_vars', 'all_vars']
```

Pros: ~8 LOC, tautologically safe (drops only byte-identical rows), fixes both 3a and 3b.
Cons: Doesn't save compute (duplicates still run). Worth ~4–10 min wall-clock per 300-trial run.

**Verdict from cross-family review:** Codex + DeepSeek picked this. GLM picked Option A. Per user's safety+simplicity criteria, B wins on review.

### Option A — Pre-fit fingerprint cache + `optuna.TrialPruned` *(higher risk, larger savings)*

In the trial objective (~`unified_bayesian.py:1230`), after the clamp at line 462 but before the fit at ~1232:

```python
fingerprint = (
    preprocess_name, deriv, window, poly,
    baseline_method, baseline_params, smoothing,
    smoothing_window, smoothing_polyorder, autoscale,
    subset_tag, n_vars, hash(tuple(selected_wavelengths)),
    model_name, frozenset(clamped_model_params.items()),
    cv_folds, random_state,
)
if fingerprint in seen_fingerprints:
    raise optuna.TrialPruned()
seen_fingerprints[fingerprint] = trial.number
```

Caveat from Codex review: `terminal_states = (TrialState.COMPLETE, TrialState.PRUNED)` at `unified_bayesian.py:~2518` makes `PRUNED` count as a terminal state for resume. A 300-trial budget could shrink to ~225 unique fits without telling the user. Bumping `n_trials` to compensate is the workaround, but the bump amount depends on duplicate rate which depends on data.

Caveat 2: existing `try/except Exception` wrapper at `unified_bayesian.py:~1689` could swallow `TrialPruned` — must explicitly re-raise.

Pros: **the only option that saves wall-clock on CatBoost/XGBoost/LightGBM/CatBoost runs.** PLS fits in <1s, so on PLS the savings are real but modest (4-10 min on a 300-trial run). On boosters, each fit costs 10-60s; saving 25% of 300 trials = 10-60 minutes per run, scaling linearly with `n_trials`.
Cons: fingerprint completeness is load-bearing — miss a dimension and a valid model gets silently skipped.

### Option C — Data-aware `n_components` suggest range *(PLS-only methodology change)*

**Scope:** PLS regression and PLS-DA only. CatBoost / XGBoost / LightGBM / RandomForest / Ridge / Lasso / SVM / MLP are unaffected — those models clamp on `suggest_int`/`suggest_float` *bounds* (e.g., `trial.suggest_int('num_leaves', 15, max_valid)`), so TPE never sees an out-of-range value to silently rescue. The post-suggest clamp is unique to PLS `n_components` at `unified_bayesian.py:457-462`.

Replace `unified_bayesian.py:457` with:

```python
max_valid = max(2, min(20, n_features_final - 1))
n_components = trial.suggest_int('n_components', 2, max_valid)
# Clamp at line 462 becomes vestigial; remove it.
```

**Why this is methodology, not a bug fix:** TPE in `multivariate=True` mode (`unified_bayesian.py:~2127`) builds a joint KDE. Fixed range gives stable support; dynamic range makes `n_components` conditional on `subset_size`, changing the KDE shape and TPE's exploration pattern. Convergence rate and reproducibility against fixed-range baselines are not pinned.

Pros: eliminates 3a *for PLS only* at the source. With the LVs reporting fix already in, this is the conceptually-honest version of the PLS search space.
Cons: methodology change → needs user signoff. Possible convergence regression. Old study DB resume would mix fixed-range and dynamic-range histories in the same KDE. **Does nothing for non-PLS model speed** — pick A if compute on CatBoost/XGBoost is the goal.

Optuna 4.8.0 supports dynamic ranges syntactically (the `warn_independent_sampling=False` at `unified_bayesian.py:~2129` already suppresses the related warning).

## Recommended pickup order

1. **Option B first** — 8 LOC, tautological safety, ships under "bug fix" framing per `feedback_chemometrics_relevance_per_ticket.md`. Fixes leaderboard distortion for ALL model families (3b duplicates affect every model — TPE re-suggests across the board). Defer A and C.
2. **Option A second, only if user runs slow-fit models routinely.** Watch for: 300-trial CatBoost or XGBoost or LightGBM runs taking >30 min where the user has commented on slowness. The fingerprint-completeness work is worthwhile when compute savings are 10-60 min per run, not when they're 4-10 min. PLS-only users can skip A indefinitely; B is sufficient.
3. **Option C only as a methodology paper requirement** — if/when documenting the search procedure for publication, the implicit-clamp design becomes unsightly. Otherwise it's a marginal PLS-specific improvement at high risk. Do not pick C expecting compute savings on any non-PLS model.

## Toolkit-panel review findings deferred from this PR

Claude-family toolkit panel ran on PR #52 cumulative diff (code-reviewer + comment-analyzer + pr-test-analyzer + silent-failure-hunter):

### Deferred (medium priority, file as next-session work)

- **Helper return-value ambiguity** — `_extract_fitted_n_components` collapses three failure modes (parse failure, missing key, non-numeric value) into one `None`. Two of those are now logged at WARNING level (closed in this PR), but callers still cannot programmatically distinguish "non-PLS row, no n_components expected" from "Params corrupted, bug signal." Cleanest fix: small dataclass return type or raise on parse failure. Defer until a caller actually needs the distinction.
- **NSGA-II `include_best_from_all` LVs=None silent drop** — when the helper returns None for a PLS/PLS-DA row in `nsga2_search.py:3980-3983`, the `'LVs': None` flows into the V1 results DataFrame. Logging at the helper level (added in this PR) covers parse failures; missing-key on a PLS model would still be silent. Add a NSGA-specific log when `model in ('PLS', 'PLS-DA')` AND helper returns None.
- **`print()` → `logger` migration** — silent-failure-hunter flagged that the GUI rebuild block at `spectral_predict_gui_optimized.py:36710-36800` uses `print(f"DEBUG: ...")` and `print(f"WARNING: ...")` rather than `logger.{debug,warning}`. The new code in this PR matched the pre-existing convention in that file (consistent locally) but violates the project-wide `code-style.md` rule "Log errors with context using logging module." File a sweep ticket for the whole 36710-36800 block.
- **Comment rot risk** — comment-analyzer flagged two cross-file pins to `unified_bayesian.py:462`: `bayesian_utils.py:794` and `tests/test_cv_pls_clamp.py:322`. Currently accurate but rot-prone; replace with symbolic anchors (function name + clamp expression).
- **Pre-existing inaccurate line ref** at `bayesian_utils.py:835` cites `search.py:4532` which is unrelated polyorder code; the actual matching scheme is at `search.py:5080-5081`. Not introduced by this PR but worth a one-line cleanup next time `bayesian_utils.py` is touched.

### Test follow-ups (pr-test-analyzer)

- **Legacy `convert_optuna_result_to_dasp_format` end-to-end test** — helper is unit-tested but the full Params→user_attr→trial.params fallback chain isn't exercised end-to-end. ~30 LOC.
- **NSGA-II `include_best_from_all` LVs test** — pin that the helper migration in `a64004f` works on the actual stringified-dict shape `decode_solution()` produces. ~20 LOC.
- **Tab7-mocked Model Dev rebuild test** — covers the priority inversion at `spectral_predict_gui_optimized.py:36710-36748` end-to-end. Needs Tk dialog mocking. ~50 LOC.
- **PLS-DA end-to-end variant** — copy of `test_unified_bayesian_lvs_matches_fitted_n_components` with `model_name='PLS-DA'` to validate `pls__n_components` shape end-to-end. ~30 LOC.

## Other adjacent items observed but not in scope

From Codex review of 9b86bc9:

- **Pre-existing edge case** in `search.py:338`: `n_lvs = row.get('LVs', None)` reads from CSV — a pre-fix CSV with inflated LVs would feed the wrong value, but the next-line `set_params(**model_kwargs)` parsing of `Params` overrides it. Less robust than the GUI's new priority inversion. Not introduced here.
- **Helper duplication risk** between GUI `spectral_predict_gui_optimized.py:~36733` and `bayesian_utils._extract_fitted_n_components`: GUI does its own ast.literal_eval inline (key order: `model__`, `pls__`, bare). Same logic but separate call site. Could canonicalize via the helper if the duplication ever drifts.

From GLM 5.1 review:

- **Test coverage gap**: no test exercises the `bayesian_utils.convert_optuna_result_to_dasp_format` legacy fallback chain end-to-end (Params → user_attr → trial.params). Only the helper is unit-tested. Add if the legacy path stays load-bearing.
- **A/B harness scope**: covers PLS regression only. PLS-DA goes through the same `_suggest_hyperparams` clamp but uses `pls__n_components` Pipeline key. Helper test covers the key shape; A/B doesn't run PLS-DA.

## Acceptance criteria for Option B

- A 300-trial PLS run on the user's data produces no row groups where `(PreprocessBase, Deriv, Window, Poly, n_vars, SubsetTag, Params, imbalance_method, top_vars, all_vars)` is identical and `RMSEcv` matches to 8 decimals.
- Top-5 leaderboard contains 5 *distinct* model configurations (no duplicate fits).
- Existing tests pass; new test pinned: synthetic study with 5 trials including 2 duplicate fits → `convert_study_to_dataframe` returns 4 rows.
- A/B harness rerun: model-fit columns byte-identical; row count drops by exactly the duplicate count; LVs unchanged on surviving rows.

## Files most likely to need touching

- `src/spectral_predict/unified_bayesian.py` (~line 2861, before `df.sort_values`)
- `src/spectral_predict/search.py` (~line 4051, after `df_results` build, before scoring)
- `tests/test_cv_pls_clamp.py` or new `tests/test_bayesian_dedup.py`

Estimated effort for Option B: ~30 minutes implementation + ~15 minutes test + cross-family review. Same review pattern as 9b86bc9.
