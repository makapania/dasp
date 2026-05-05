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

Caveat from Codex review: `terminal_states = (TrialState.COMPLETE, TrialState.PRUNED)` at `unified_bayesian.py:2510` makes `PRUNED` count as a terminal state for resume. A 300-trial budget could shrink to ~225 unique fits without telling the user. Bumping `n_trials` to compensate is the workaround, but the bump amount depends on duplicate rate which depends on data.

Caveat 2: existing `try/except Exception` wrapper at `unified_bayesian.py:1681` could swallow `TrialPruned` — must explicitly re-raise.

Pros: saves ~25% compute on this run. Compute savings would be much larger for slow-fit models (LightGBM, XGBoost, CatBoost) because per-trial cost dominates.
Cons: fingerprint completeness is load-bearing — miss a dimension and a valid model gets silently skipped.

### Option C — Data-aware `n_components` suggest range *(methodology change)*

Replace `unified_bayesian.py:457` with:

```python
max_valid = max(2, min(20, n_features_final - 1))
n_components = trial.suggest_int('n_components', 2, max_valid)
# Clamp at line 462 becomes vestigial; remove it.
```

**Why this is methodology, not a bug fix:** TPE in `multivariate=True` mode (`:2119`) builds a joint KDE. Fixed range gives stable support; dynamic range makes `n_components` conditional on `subset_size`, changing the KDE shape and TPE's exploration pattern. Convergence rate and reproducibility against fixed-range baselines are not pinned.

Pros: eliminates 3a at the source. With the LVs reporting fix already in, this is the conceptually-honest version of the search space.
Cons: methodology change → needs user signoff. Possible convergence regression. Old study DB resume would mix fixed-range and dynamic-range histories in the same KDE.

Optuna 4.8.0 supports dynamic ranges syntactically (the `warn_independent_sampling=False` at `:2121` already suppresses the related warning).

## Recommended pickup order

1. **Option B first** — 8 LOC, tautological safety, ships under "bug fix" framing per `feedback_chemometrics_relevance_per_ticket.md`. Defer A and C.
2. After running B for a few sessions on real data, evaluate whether the compute waste is genuinely painful for LightGBM/XGBoost runs. If yes, Option A becomes worth the fingerprint-completeness work. If no, leave as B.
3. **Option C only as a methodology paper requirement** — if/when documenting the search procedure for publication, the implicit-clamp design becomes unsightly. Otherwise it's a marginal improvement at high risk.

## Other adjacent items observed but not in scope

From Codex review of 9b86bc9:

- **Pre-existing edge case** in `search.py:338`: `n_lvs = row.get('LVs', None)` reads from CSV — a pre-fix CSV with inflated LVs would feed the wrong value, but the next-line `set_params(**model_kwargs)` parsing of `Params` overrides it. Less robust than the GUI's new priority inversion. Not introduced here.
- **Helper duplication risk** between GUI `spectral_predict_gui_optimized.py:36732` and `bayesian_utils._extract_fitted_n_components`: GUI does its own ast.literal_eval inline (key order: `model__`, `pls__`, bare). Same logic but separate call site. Could canonicalize via the helper if the duplication ever drifts.

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
