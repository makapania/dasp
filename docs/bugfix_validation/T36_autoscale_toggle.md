# T-36 — Autoscale (UV scaling) Preprocessing Toggle

**Branch:** `feature/T36-autoscale-toggle`
**Plan:** `docs/plans/2026-05-01-T36-autoscale-toggle.md`
**Filed:** 2026-05-01

---

## Summary

Adds UV scaling (mean-center + unit variance per wavelength) as a user-selectable preprocessing option that doubles the config grid for grid search and becomes a per-trial Optuna categorical for Bayesian search. Bundles three pre-existing-bug fixes that were uncovered while planning the autoscale wiring.

## Field-alignment check

| Tool | Default | Knob name |
|---|---|---|
| SIMCA (Umetrics) | UV scaling **default** for PLS | "Scaling: UV" |
| PLS_Toolbox (Eigenvector) | optional | `autoscale` |
| Unscrambler (CAMO) | optional | "Center & Scale" |
| dasp (this PR) | optional, off by default | `autoscale` checkbox in Preprocessing tab |

Adding autoscale matches the chemometrics-canon expectation. Defaulting to off preserves the prior dasp behavior. No literature reference recommends *always* using autoscale — it depends on whether wavelength-band variance heterogeneity is informative or noise-driven, which varies by application. Letting the user opt in (or letting Optuna learn it per dataset in Bayesian search) is the right contract.

## Architecture summary

| Path | Mechanism |
|---|---|
| Grid search (regression / classification) | `preprocess_configs` doubled — every selected combo runs once with autoscale=False, once with True. Mirrors the existing baseline / smoothing toggle pattern at `search.py:1813-1849`. Display name suffix `+autoscale`; `PreprocessBase` stays clean. |
| Grid search (one-class) | Same doubling block, applied after smart/manual config builders. Uses `'method'` (not `'base_name'`) for the clean pipeline name, per existing one-class convention. |
| Bayesian search (regression / classification / one-class) | `apply_autoscale: bool` becomes a per-trial Optuna categorical in `suggest_preprocessing` when `enable_autoscale=True` is passed in. TPE then learns whether autoscale helps for the dataset. |
| Validation rebuild | `compute_validation_metrics_for_top_models` and `compute_validation_metrics_for_top_one_class_models` both read `Autoscale` column from result rows and pass it to `build_preprocessing_pipeline`. Cache keys include the flag so doubled rows don't collide. |
| Per-model `StandardScaler` | Skipped for `SCALE_SENSITIVE_MODELS` when autoscale is on, in both grid and Bayesian paths. PLS-DA's internal scaler (operating on PLS scores) is preserved. |

## Three pre-existing bugs fixed

### BUG #1 — `apply_preprocessing()` early-return prevented autoscale from ever firing

`unified_bayesian.py:apply_preprocessing` returned early in every named-preprocessing branch (`raw`, `snv`, `'deriv' in name`). A naive autoscale step appended at the end of the function would have been unreachable for every preprocessing name — the Bayesian path would have silently no-op'd autoscale.

**Fix:** Restructured each branch to assign `X = ...` rather than `return ...`, then a single trailing `if config.get('apply_autoscale'): X = StandardScaler().fit_transform(X)` followed by `return X`.

**Regression test:** `tests/test_autoscale_bayesian.py::test_no_autoscale_bit_identical_to_pre_T36` — 14 parametrized cases (every preprocessing name in `PREPROCESSING_OPTIONS`) verify bit-identical output to the pre-T-36 early-return form when `apply_autoscale=False`. Companion test `test_autoscale_fires_in_every_branch` verifies the trailing block fires correctly when `apply_autoscale=True` (column means ≈ 0, column stds ≈ 1).

### BUG #2 — Bayesian preprocessing cache key omitted `apply_autoscale`

`unified_bayesian.py:create_unified_objective` cached `X_prep` keyed by 6 preprocessing fields. Adding `apply_autoscale` as a per-trial Optuna toggle would cause two trials with the same other params to collide on the cache — the second trial would silently receive the first's `X_prep`, defeating the whole point of varying the toggle.

**Fix:** Added `preprocess_config.get('apply_autoscale', False)` as the 7th element of the cache_key tuple.

**Regression test:** `test_cache_key_includes_apply_autoscale` (source-inspection check on `create_unified_objective`).

### BUG #3 — Contamination validation cache key omitted autoscale

`contamination.py:compute_validation_metrics_for_top_one_class_models` cached preprocessed train/val splits keyed by 8 preprocessing fields. Same class of bug as #2 — two top-N rows differing only in `Autoscale` would collide on the cache and produce wrong validation metrics for the second.

**Fix:** Added `autoscale` as the 9th element of the cache_key tuple, with the same robust parser as `search.py` (handles bool, numpy.bool_, NaN-float, int 0/1, and string forms — bool("False") == True trap).

## Bundled adjacent fix — one-class metadata write

`run_one_class_search` result rows previously omitted `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder`. The validation rebuild path at `contamination.py:1098-1107` reads them with `row.get(...)` defaults, meaning saved one-class `.dasp` files silently rebuilt validation pipelines using default smoothing/baseline regardless of what the search actually ran. Same code surface as the autoscale row writes — fixed together.

**Regression test:** `tests/test_autoscale_one_class.py::test_metadata_columns_present`.

## Out-of-scope (filed as follow-ups if the user wants them)

| Item | Why |
|---|---|
| NSGA-II support | Separate code path (`nsga2_search.py`) with its own `build_preprocessing_pipeline` calls. |
| `run_bayesian_search` (test-only path; no GUI caller) | Only used by `tests/test_cv_pls_clamp.py` and `tests/test_golden_standard_performance.py`. Comment added in source. |
| Tree-model skip in doubling loop (Phase 1.5 refinement) | Tree models are scale-invariant so the autoscaled half is wasted compute, but skipping needs cross-cut verification that varsel + imbalance still flow correctly. Defer. |
| Smart-preprocessing-discovery integration | Subsumed by T-37 (TPE quick-discovery). Don't bolt onto basic discovery. |
| Smoothing column in `_varsel_cache_key` (sister bug to autoscale add) | Pre-existing latent bug — `name` discriminates today via the `sg0+` prefix from doubling, so this is currently safe but defensively-fragile. Out of T-36 scope. |
| Robust string-bool parsing for legacy `smoothing` column | Same family of fragility as the autoscale parser DeepSeek flagged in Phase 3. Existing behavior is fine for the normal `pd.to_csv`/`pd.read_csv` roundtrip; only hand-edited CSVs are affected. |

## Review trail

Per-phase DeepSeek V4 Pro Max reviews (direct DeepSeek API, not opencode-go) ran after each commit. Final pre-merge Codex review pending.

| Phase | Commit | DeepSeek verdict | Findings addressed |
|---|---|---|---|
| 2 — preprocess.py + tests | `3279836` → `fec44e5` | READY_TO_PROCEED | 3 LOW (test ordering coverage tightened) |
| 3 — search.py grid path | `30e1093` → `f52a727` | READY_TO_PROCEED | 1 MEDIUM (string-bool parse), 2 LOW (fallback comment, smoothing×autoscale test added) |
| 4 — one-class + contamination | `158536d` → `54711e4` | READY_TO_PROCEED | 1 MEDIUM (silent-skip assertion hardened) |
| 5 — Bayesian path + 3 bugs | `d486962` → `73047d2` | READY_TO_PROCEED | 1 MEDIUM (Bayesian display-name suffix), 1 LOW (deriv3/4 test coverage) |
| 6 — GUI checkbox | `cc89bd9` | READY_TO_PROCEED | 0 findings |
| 7 — full sweep + Codex | _pending_ | _pending_ | _pending_ |

## Test sweep

Full sweep on `feature/T36-autoscale-toggle` after Phase 6:
- **2181 passed**
- **2 skipped** (pre-existing)
- **10 failed**, all in I/O loaders (`tests/test_io_jcamp.py`, `tests/test_io_spc.py`, `tests/test_io_vendor_formats.py`) — pre-existing failures unrelated to T-36; these touch jcamp / SPC / OPUS / PerkinElmer reader modules that T-36 does not modify.
- **1 environmental flake** (`tests/test_export_code.py::test_python_script_execution`) — spawns subprocess `python` which uses system Python, not `.venv312`, so can't import sklearn. Not a regression; deselected.

T-36-specific tests:
- `tests/test_preprocess_extended.py::TestAutoscaleStep` — 9 passes
- `tests/test_autoscale_grid_doubling.py` — 5 passes
- `tests/test_autoscale_one_class.py` — 4 passes
- `tests/test_autoscale_bayesian.py` — 44 passes (regression for BUG #1, #2, display name)

Total new T-36 tests: **62 passing**.
