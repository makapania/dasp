# T-37 — TPE Preprocessing Discovery

**Branch:** `feature/T37-tpe-preprocessing-discovery`
**Plan:** `docs/plans/2026-05-01-T37-tpe-quick-preprocessing-discovery.md`
**Filed:** 2026-05-01

---

## Summary

Adds a TPE (Tree-structured Parzen Estimator) preprocessing discovery mode that replaces the exhaustive basic-discovery and GA paths with a smarter 5-D search space (preproc x window x autoscale x baseline x smoothing). Uses Optuna TPESampler with `multivariate=True` to exploit the ordered-window dimension. Returns top-N diverse configs tested against ALL enabled models — preserves model diversity.

## Architecture choice

**Architecture A** (model-agnostic surrogate). TPE evaluates preprocessing quality using a LightGBM proxy model (same surrogate as current `discover_preprocessing`), returns top-N configs, then the search loop fits ALL enabled models against each. Output contract identical to basic discovery — the search loop doesn't know whether configs came from exhaustive or TPE search.

Architecture B (per-model parallel mini-studies) was rejected because the user explicitly preferred A and it's simpler.

## Search space

| Dimension | Type | Levels | Notes |
|---|---|---|---|
| preproc | categorical | 14 | Same as PREPROCESSING_CANDIDATES |
| window | integer | derivative-aware | Ported from ga_preprocessing.py DERIVATIVE_WINDOW_RANGES |
| autoscale | boolean (cat) | 2 | T-36 dependency |
| baseline_method | categorical | 5 | None, als, polynomial, rubber_band, airpls |
| smoothing | boolean (cat) | 2 | SG smoothing window=17, polyorder=2 |

**Trial budget:** 75 trials default (50/75/100/150 in GUI dropdown). 20 random startup trials before TPE kicks in.

**multivariate=True:** The TPESampler uses multivariate=True to exploit the ordered-window dimension. This justifies TPE over exhaustive on a discrete categorical space — the window dimension has neighborhood structure that TPE can learn even when individual axes are categorical-only.

## Files changed

| File | Changes |
|---|---|
| `src/spectral_predict/tpe_preprocessing_discovery.py` | NEW — core TPE module (~440 LOC) |
| `src/spectral_predict/search.py` | +170 LOC — TPE blocks in run_search + run_one_class_search |
| `spectral_predict_gui_optimized.py` | +74 LOC — GUI checkbox + n_trials dropdown |
| `tests/test_tpe_preprocessing_discovery.py` | NEW — 21 tests |

## Consumer-surface audit

| # | Surface | Status | Notes |
|---|---|---|---|
| 1 | Search loop | PASS | TPE configs converted to standard preprocess_configs dict |
| 2 | Result row | PASS | name field now uses display_name with all prefixes |
| 3 | Validation rebuild | PASS | Reads Autoscale column; parses Preprocess for baseline/smoothing |
| 4 | Model Dev refinement | PASS | Uses build_preprocessing_pipeline with standard keys |
| 5 | Saved-model metadata | PASS | Standard preprocess_configs keys preserved in training_config |
| 6 | Exported scripts | PASS | code_generator reads from model_config; standard keys |

## Self-review findings

### Finding #1 — `name` field used pipeline name instead of display name (FIXED)

The TPE config conversion set `preprocess_cfg["name"]` to the clean pipeline name (e.g., "deriv") instead of the display name (e.g., "als+deriv_snv_w23+autoscale"). The result rows and validation rebuild parse the `Preprocess` column for baseline, smoothing, and autoscale prefixes — missing prefixes would silently lose those downstream settings.

**Fix:** Set `"name": display_name` and `"base_name": pipeline_name`.

### Finding #2 — Doubling blocks re-doubled TPE-discovered per-config settings (FIXED)

The baseline, smoothing, and autoscale doubling blocks (search.py ~1893-1888 and ~5521-5535) run unconditionally after the config-building phase. TPE configs already have per-config baseline/smoothing/autoscale values; the global doubling would clobber them by adding extra config variants with overridden values.

**Fix:** Set `baseline_method = None`, `autoscale = False`, `smoothing = False` inside the TPE success block so doubling blocks are no-ops.

## Out-of-scope

| Item | Why |
|---|---|
| Bayesian path integration | Bayesian search uses its own per-trial preprocessing Optuna; TPE preprocessing discovery is a separate pre-step for grid search |
| NSGA-II support | Separate code path (`nsga2_search.py`); not modified |
| `run_bayesian_search` (test-only) | No GUI caller; comment added in T-36 |
| Programmatic baseline params (lam, p) tuning | Fixed to defaults for speed; revisit if results look noisy |
| n_trials scaling with dataset size | Fixed dropdown (50/75/100/150); conservative |

## Test sweep

21 new tests in `tests/test_tpe_preprocessing_discovery.py` — all green:
- 3 module structure tests
- 5 end-to-end (regression/classification/one_class/diversity/dimensions)
- 4 dimension disablement (individual axes + all-disabled)
- 5 edge cases (tiny dataset, constant columns, n_top, windows, progress)
- 3 integration (run_search + run_one_class_search + mutual exclusion)
- 1 reproducibility (deterministic with seed)

## Commit history

| Commit | Phase | Description |
|---|---|---|
| `ef5f61e` | 1 | Core TPE module |
| `7ca673f` | 2 | Wire into run_search + run_one_class_search |
| `467b384` | 3 | GUI checkbox + dropdown |
| `bd63ad1` | 4 | 21 tests |
| `5ca7080` | 5 | Self-review fixes |
| `7d8f940` | 6 | Audit doc + PROJECT_STATUS + SESSION_LOG entry |
| `03d95cb` | 7 | GLM 5.1 external-review fixes (see Review trail) |
| `6a0eb28` | 8 | Post-merge cross-family review fixes (Codex + DeepSeek + Gemini + CodeRabbit) |

## Review trail

| Pass | Reviewer | Verdict | Findings |
|---|---|---|---|
| Self-review (Phase 5) | DeepSeek (implementer) | shipped 2 silent-mismatch fixes in commit `5ca7080` | (a) preprocess_cfg name field used pipeline name not display name with prefix cascades; (b) baseline/smoothing/autoscale doubling blocks re-doubled TPE-discovered per-config settings — guarded by nulling global flags inside TPE block. |
| External review of Phase 1 commit `ef5f61e` | **GLM 5.1** (separate dispatch 2026-05-01 — corrected attribution per user 2026-05-01 PM) | READY_WITH_REVISIONS — refined ratings: only 1 finding upgraded to MEDIUM-fix-now; the rest are MEDIUM/LOW and partly NOT-AN-ISSUE. **Critical correction from GLM:** the original "fix C1/H2 by varying search space per trial" recommendation **would break Optuna** — `trial.suggest_categorical` requires CONSTANT choices across trials per the ask/tell interface contract. The current "suggest from union, let invalid combos return -inf" approach is the correct workaround. | See "Open follow-ups" below — fix list re-prioritized per GLM's analysis. |

## Resolved findings (Phase 7 fix commit `03d95cb`)

### MEDIUM F2: `_tpe_baseline_params` never attached to output — RESOLVED

`tpe_preprocessing_discovery.py:397` writes `_tpe_baseline_method` but never `_tpe_baseline_params`. Both `run_search:1474` and `run_one_class_search:5479` read `cfg.get('_tpe_baseline_params')` — always returned `None`. `build_preprocessing_pipeline` then fell back to defaults via `baseline_params or {}`.

**Fix in `03d95cb`:** Added `'_tpe_baseline_params': _BASELINE_DEFAULT_PARAMS.get(baseline_method, {}) if baseline_method else None` to the output dict at `tpe_preprocessing_discovery.py:398`.

### MEDIUM F6: PLS fallback uses `neg_root_mean_squared_error` regardless of task_type — RESOLVED

`tpe_preprocessing_discovery.py:186-191` exception handler scored with RMSE even when `task_type='classification'` or `'one_class'`. If LightGBM was unavailable, classification TPE evaluations ranked preprocessing by RMSE on integer labels — wrong metric.

**Fix in `03d95cb`:** Exception handler now branches on `task_type`:
- `regression` → `neg_root_mean_squared_error` (unchanged)
- `classification` → `accuracy`
- `one_class` → `-np.inf` (failed trial, don't pollute TPE)

### MEDIUM F9: Mutual exclusion test provides false confidence — RESOLVED

`tests/test_tpe_preprocessing_discovery.py:326-349` passed both `tpe_preprocess=True` and `smart_preprocess=True` to `run_search` expecting TPE to win, but only checked `results is not None and len(results) > 0`. The test would pass identically if smart preprocessing ran instead of TPE.

**Fix in `03d95cb`:**
- `tpe_score` propagated into result rows in `search.py` (3 sites: regression/classification grid, one_class full-spectrum, one_class subset).
- Integration tests unpacked the `(df_ranked, label_encoder)` tuple and assert `tpe_score` column presence, non-null values, and `smart_score` absence.
- **Surprise:** `run_search` returns a tuple, not a plain list. Original iteration over the tuple hit `TypeError` when `label_encoder` was `None`.

### MEDIUM F12: Test coverage gaps — RESOLVED

Missing invariants now covered by 12 new tests:
1. `_apply_full_preprocessing` unit tests — `TestApplyFullPreprocessing` (5 tests: raw passthrough, SNV mean, baseline+deriv shape, smoothing+autoscale, full chain).
2. `_quick_evaluate` direct tests — `TestQuickEvaluateDirect` (4 tests: regression finite, classification finite, one_class with outliers finite, one_class no outliers returns 0.0).
3. Empty-TPE-result fallback — `TestEmptyTPEFallback::test_all_trials_fail_returns_empty_list` (monkeypatches `_quick_evaluate` to always raise, verifies `[]` return).
4. `build_preprocessing_pipeline` roundtrip — `TestPipelineRoundtrip` (2 tests: full chain baseline+deriv+autoscale, and SNV no-extras).
5. Items 4–7 from the original list (result-row prefix verification, code-export pipeline, saved-model metadata roundtrip) remain out of scope for Phase 7; they require GUI-level or export-level testing infrastructure not yet present.

### LOW F1 / F10: Window constraint + `_resolve_window_choices` dead code — RESOLVED

F1: The "union + return-inf-on-invalid" approach is the correct workaround per GLM's Optuna-semantics correction. F10: `_resolve_window_choices` was commented as RESERVED with an explanation that per-trial window filtering would break Optuna's ask/tell contract. Applied in `03d95cb`.

---

## Open / Deferred follow-ups

### LOW F7: `select_diverse_configs` mutates non-selected configs — DEFERRED

`preprocessing_discovery.py:783-816` adds `_combined_score` to ALL config dicts (line 784), only deletes from selected subset (lines 814-816). Non-selected configs retain the temporary key.

**Why LOW:** TPE module passes a fresh `unique_configs` list to `select_diverse_configs` so there's no downstream pollution. Pure hygiene. Deferred to a general preprocessing-discovery cleanup ticket.

### LOW F8: `print()` instead of logger — DEFERRED

Consistent with `preprocessing_discovery.py` conventions. Deferred to a logging cleanup ticket.

### NOT-AN-ISSUE (per GLM analysis)

- F4: polyorder for raw/snv configs — correctly stored as None; downstream `build_preprocessing_pipeline` doesn't create SavgolDerivative for those base names, so polyorder is ignored. No bug.
- F5: One-class baseline/smoothing reset — `run_one_class_search` has no baseline/smoothing doubling blocks (verified by explore agent), so the reset isn't needed.
- F11: Display name prefix order between `run_search` and `one_class` — verified identical (`baseline+sg0+base_w17+autoscale`). No mismatch.

### Original H3 reclassified

The `selected_wavelengths` key omission is real but doesn't matter — `search.py` consumer doesn't read this key from TPE-origin configs. Update the docstring claim at `tpe_preprocessing_discovery.py:230` from "identical" to "compatible superset" — that's the entire fix.
