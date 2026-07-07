# Spectral Predict (dasp) — Production-Readiness Review

**Date:** 2026-07-07
**Reviewed version:** 0.5.0b2 (Development Status: Alpha), post-PR #64 on `main`
**Method:** Four parallel evidence-based reviews (architecture/fragility, testing/CI/release, UI/UX, competitive positioning). All line numbers verified against current `main`.

---

## TL;DR

dasp is **further along than a 0.5-alpha label suggests, and its core thesis is genuinely differentiated** — no commercial competitor at any price does automated preprocessing × variable-selection × model search with a ranked, overfit-filtered leaderboard. The chemometric diagnostics (Hotelling T², Q-residuals, Mahalanobis, applicability domain at predict time) are already near-commercial-grade, and the reproducible-code export (Python + R + notebooks) is something none of the paid tools match.

The gap to "sellable product" is **not features** — it's **engineering process and presentation polish**:

1. **CI has been red for 8 months and gates nothing** — and two of the residual failures are real, shipping, user-facing codegen bugs.
2. **A 60,000-line / 926-method single-class GUI** is the structural root of a recurring bug pattern and a merge-conflict magnet across the multi-machine workflow.
3. **Presentation layer reads as "freeware"** — no DPI awareness (blurry on the default 150%-scaled Windows laptop), no dark mode, no determinate progress bar for the main sweep, 735 `print()`s invisible in the installed build.
4. **No reproducible path from repo to the installer users receive**, and no validation-report generation — the thing labs actually paste into SOPs and papers.

Highest-leverage sequence: **get CI green (fix the 2 codegen bugs) → DPI + progress-bar + global exception-handler polish pass → validation-report generator + headless predictor → exit beta with a validation white-paper.** None of this requires chasing 21 CFR Part 11 or real-time PAT — those are competitors' moats, not your market.

---

## 1. Testing, CI & Release Engineering

**The headline problem is process, not test content.** The test *suite* is production-grade; the *discipline of acting on it* is broken.

### What's strong
- **2,961 tests / 149 files**, including real numerical regression testing: `tests/gold_standards/` holds committed `.npz` reference outputs consumed by `tests/numerical/test_preprocessing_correctness.py` (~25 tests, `assert_allclose` against frozen gold + against scipy directly). This is not a smoke-test suite.
- Template-execution tests actually run generated scripts end-to-end — which is how the codegen bugs below were caught.
- **Reproducibility is a design habit:** 352 `random_state`/`seed` occurrences across 26 source files; a `RANDOM_STATE` module constant; seeds pinned into generated user scripts.
- **Version discipline is good:** `0.5.0b2` consistent across `pyproject.toml`, `__init__.py`, the installer `.iss`, and `version_info.txt`; threaded into exports, model bundles, and CLI `--version`.

### What's broken (ranked)
1. **[CRITICAL] Red CI is normalized.** `.github/workflows/ci.yml` exists (2 OS × 3 Python, black/flake8, build+twine) but **every completed run on `main` since the workflow was added (~2025-10-27) is a failure — zero green in ~8 months.** Runs take **1h50m–3h46m** with no tiering/caching/fail-fast, so the signal is both red and unread; the team's own SESSION_LOG admits merging by manually diffing failure sets. A permanently red CI gates nothing.
2. **[HIGH] Two of the 5 residual failures are real user-facing product bugs**, marinating known-and-unfixed since ~2026-05-08 because red-on-red hides urgency:
   - `test_cv_strategy.py::...test_classification_metrics_template_has_no_nameerror` — **exported classification scripts raise `NameError: '_fit_fold' is not defined`** (a template refactor calls a helper it doesn't emit). Every user exporting a classification script from this path gets a crashing script.
   - `test_t19_class_weight_per_library.py::test_xgboost_threads_sample_weight_via_fit_kwargs` — **exported XGBoost scripts silently dropped `**fit_kwargs`** → sample weights lost → silently-wrong results (worse than a crash).
   - The 3rd (`...does_not_emit_fit_kwargs_plumbing`) is a likely stale over-broad negative assertion from the same refactor; the 2 `test_export_code.py` failures are test-infra (subprocess uses bare `python` not `sys.executable`, hitting a non-venv interpreter). **Fixing the two real bugs + the subprocess env is roughly an afternoon and takes CI to near-green.**
   - Note: **T-CI-4 (CLI `--help`/`--version`) already passes** — docs still list it as open.
3. **[HIGH] No reproducible installer build in the repo.** The Inno Setup `.iss` references `installer/build_installer_py312.py` which **does not exist** at that path. The `build` CI job makes an sdist/wheel but nothing publishes or attaches installers. **End-user delivery is manual and unreproducible from the repo.**
4. **[MEDIUM] CI wall-clock 2–4h, no tiering/cache;** `tests/gui` is `--ignore`d on Linux over an unresolved deadlock (T-CI-2); the diagnostic GUI job is `continue-on-error`.
5. **[MEDIUM] Zero coverage measurement** — no `pytest-cov`, no `[tool.coverage]`, no CI step. With 2,961 tests it's probably decent, but the largest file (the GUI) is CI-excluded on Linux and nobody knows the blind spots.
6. **[LOW]** `CHANGELOG.md` two months stale.

---

## 2. Architecture & Fragility

### The single root cause: the god-class GUI
- **`spectral_predict_gui_optimized.py` = 60,289 lines, one `SpectralPredictApp` class with 926 methods and ~12,657 `self.` references.** 13 top-level tabs, ~45 sub-tab builders, plus non-UI responsibilities crammed in (theming engine, sound threads, model reconstruction, wavelength matching, run-state/resume).
- **Merge-conflict surface is real, not hypothetical: 343 of 805 commits since 2026-01-01 (43%) touch this one file.** With multi-machine development, nearly every pair of concurrent branches collides here; semantic conflicts inside a 926-method shared-`self` class are invisible to git's merge driver.
- This structure is the documented cause of the "sister-site" bug pattern — a fix lands in one tab and the same bug survives in a sibling tab because no one can hold 45 builders in their head. The project's whole "sister-site sweep" review practice exists *because* of this file.

### Business logic trapped in the GUI
- **Six sklearn `BaseEstimator` wrapper classes are defined only in the GUI file** (`WavelengthSubsetWrapper`, `GAPreprocessWrapper`, `CombinedPreprocessWrapper` + 3 Classifier twins, ~lines 2168-2528). No copies exist in `src/spectral_predict/` — **anything pickled with them can't be unpickled headless**, and they're untestable without Tk.
- **Inline CV loops and copy-pasted modeling stacks live in the GUI**: `cross_val_predict` at :25000, manual `KFold` at :25103/:38054, verbatim-triplicated `LogisticRegression`/`StandardScaler` blocks at :39623/:39751/:39808, full pipeline reconstruction in `_reconstruct_models_from_results` (:24280).
- **Risk:** two modeling code paths (backend search-time vs GUI refine/rebuild-time) that drift — PROJECT_STATUS already records exactly this class of bug (Refine-tab silently re-coupling autoscale flags).
- **Highest-leverage single move:** extract those 6 wrappers + `_reconstruct_models_from_results` + inline CV blocks into `src/spectral_predict/`. They're already backend-shaped (no Tk dependency); this makes them testable/pickle-safe and takes the first real bite out of the monolith without a risky big-bang split.

### Error handling
- **355 broad/bare excepts in the GUI file alone** (15 bare `except:`), 39+ confirmed silent `except Exception: pass`. `_update_widget_colors` (:4988) swallows everything recursively across the whole widget tree.
- The **"crash destroyed a half-built window, user saw nothing" pattern is acknowledged in the code's own docstrings** and has recurred ≥3 times (PROJECT_STATUS:10). Being fixed case-by-case; **no global `report_callback_exception` handler exists.**
- **cp1252/Unicode `print` crash is a confirmed recurring bug class** (non-ASCII literal → `UnicodeEncodeError` on Windows console, inside a worker thread under a broad except → silently dead analysis). **735 `print()` sites** in the GUI; a regression AST-scan guard now exists but the root cause (raw `print` to platform-encoded stdout instead of a logging layer) is untouched.
- For contrast the **backend is far more disciplined** — worst module `search.py` is 49 excepts over 8,402 lines.

### State & threading
- All state is mutable attributes on the god object; any of 926 methods can mutate any of them. Guard flags (`_clearing_filters`, debounce timers) are tell-tale patches of trace-cascade fragility.
- 8 worker-thread launch sites, all `daemon=True` (process exit mid-analysis abandons work, mitigated only for Bayesian runs via resume sidecar). Marshaling back to Tk via `root.after` is correct-by-convention across 60K lines, not structural — nothing stops a new worker from touching a widget directly.

### Lower-severity
- **Dual-import trap (`src.spectral_predict` vs `spectral_predict`)** is mostly cleaned in the GUI but kept armed by a runtime `sys.path.insert` (:147) + `pyproject` packaging both `src` and `.`. Recommend killing the sys.path hack for an editable install and adding a test that greps for `from src.`.
- **Unpinned vendor deps** (`specdal`, `brukeropus`, `specio-py310`, `requests`) + **no lock file** → frozen-bundle reproducibility rests on whatever pip resolves that day. Otherwise pinning discipline is above-average (dated, justified floors).
- **Top-level `shap` + `matplotlib` imports** (:121-126, :252) add a fixed 2–5s startup tax; most sklearn imports are correctly deferred.

---

## 3. UI / UX

**Net:** the concurrency/progress architecture and export story are already commercial-grade; the gaps are presentation polish, error-surfacing consistency, and non-expert onboarding.

### Strong today
- **Long-run UX is the best area.** Sweeps run on a daemon worker (UI doesn't freeze), with full **Pause / Resume / Stop**, an honest "current trial may take 20+ min" acknowledgement, a live capped log, "Best Model So Far", time estimate, disk-mirrored logs surviving process death, and crash-resume on startup.
- **Results & export are a standout:** sortable/multi-sort leaderboard with genuinely good scientific header tooltips, quartile row coloring, double-click → refinement/decision view, and export to CSV / `.dasp` model / **Colab notebook + repro script** / 300-DPI plots.

### Top UI improvements (prioritized)
1. **Global exception handler** — install `root.report_callback_exception` + wrap Toplevel render paths to show the error *before* destroying the window. Kills the recurring "user saw nothing" class structurally instead of per-site.
2. **DPI awareness** — `SetProcessDpiAwareness` + `tk scaling` + derive figure DPI from the real screen. Blurry text at 150% scaling (the modern-laptop default) reads as "old freeware" instantly. Cheapest polish win.
3. **Route `print()` → the existing status/log layer** — 735 prints are invisible in the installed `pythonw` build; ~half the diagnostic surface is dark in production.
4. **Determinate progress bar for the main sweep** — the `current`/`total` plumbing already exists (`_progress_callback_impl` :29651); only the prediction tab got an actual `Progressbar`. Add bar + % + ETA to the Analysis Progress header.
5. **Progressive disclosure in Analysis Configuration** — **461 input widgets on one tab** (~852 app-wide). Default to a "Recommended" card (tier + target + CV) with advanced sections collapsed; make the Quick/Standard/Comprehensive tier preset the *whole* config, not just the model list.
6. **Move data loading off the main thread** — 29 `self.root.update()` calls (the freeze-prone anti-pattern); large-file load currently locks the window. Reuse the worker+`after` pattern the sweep already has.
7. **Tooltip coverage pass** — only ~13% of inputs (112 of ~852) have tooltips; extend the excellent leaderboard-tooltip standard to Variable Selection and Model Config first.
8. **Navigation cleanup + dark mode** — 13 tabs / ~35 panes with two overlapping import paths and duplicated icons (🔍 twice, 🎯 twice); no first-run wizard. Consider numbered workflow signposting in the existing sidebar. Six light themes exist but no dark mode.

---

## 4. Competitive Positioning

**dasp is not a weaker Unscrambler — it's a different product (automated model search) that happens to already carry surprisingly complete chemometric diagnostics.**

### Genuine, unmatched differentiation
- **The automation thesis is real:** the internal T-16 survey confirmed **none** of Unscrambler / SIMCA / PLS_Toolbox / OPUS expose even model-A-vs-model-B comparison, let alone ranking hundreds of preprocessing × varsel × model candidates. They are manual, one-model-at-a-time, PLS/PCA-centric workbenches.
- **Model breadth** (10+ regressors incl. LightGBM/XGBoost/CatBoost/MLP/SVM, 5 one-class models, multi-class SIMCA) exceeds every commercial package.
- **Reproducible-code export** (Python + R + notebooks) — Eigenvector's paid Model_Exporter exports predictors, not auditable pipelines that re-derive from raw data. Publishable-science angle none of the closed tools match.
- **Price:** free/MIT vs PLS_Toolbox $3,395/seat (+MATLAB), SIMCA/Unscrambler quote-only enterprise (~$5–15k historically), OPUS QUANT bundled with Bruker hardware. **The only zero-cost GUI in the category.**
- **File-format reach** (ASD/OPUS/SPC/JCAMP/PerkinElmer/Agilent/Omnic) rivals vendor software and beats all open-source alternatives.

### Table-stakes audit — the assumed gap list was mostly wrong; dasp already has:
- **Hotelling T² / Q-residuals / Mahalanobis / leverage** — dedicated "Data Quality Check" tab + `outlier_detection.py` + `diagnostics.py`.
- **Scores/loadings plots** (PCA Explore tab), **spectral region exclusion** (`WavelengthExcluder`/`RegionExcluder` + moisture-band defaults), **batch prediction** (Tab 7 loads `.dasp`, predicts new files with T²/Q applicability-domain checks), **calibration transfer** (DS/PDS — absent from base SIMCA/Unscrambler tiers).

### Genuine gaps
- **Report generation is weak** (markdown top-5 only; no PDF/HTML validation report).
- **No headless/embeddable prediction engine** like Eigenvector Solo_Predictor (watch-folder/TCP) — though `model_io` + `cli.py` are most of the plumbing.
- **No classic per-model regression-coefficient (b-vector) overlay plot** — chemometricians trust models by eyeballing b against known bands; VIP/importance plots only partially cover this.
- **No audit trail / 21 CFR Part 11**, **no real-time PAT** — but these are deliberately *out of scope* (SIMCA's moats, multi-year efforts, wrong market).

### Realistic target & close-the-gap list
**Target:** lab scientist / grad student / at-line analytical lab priced out of $3.4k–$15k seats, currently hand-rolling scikit-learn or nursing an old Unscrambler license. **Not** regulated pharma QA, **not** real-time PAT.

1. **Polished validation report (PDF/HTML)** — predicted-vs-observed, coefficients, diagnostics, method summary. This is what gets pasted into SOPs and papers.
2. **Headless prediction engine** — documented `dasp predict model.dasp newdata.csv` + watch-folder mode. Most plumbing exists.
3. **Per-model regression-vector / loadings plots** in Results / Model Development.
4. **Pairwise model-comparison stats (T-16 Tier 1–3: paired bootstrap CI + permutation + McNemar)** — verified no competitor has it; directly hardens the leaderboard (your whole pitch) against "you picked the noise." ~3–4 days.
5. **Trust/stability signaling** — exit beta, guarantee the model-file format across versions, publish a validation white-paper against reference datasets. **Alpha + 0.5.0b2 is the single biggest adoption blocker for a lab manager, independent of features.**

---

## Consolidated priority roadmap

**Phase 0 — Stop the bleeding (days):**
- Fix the two codegen bugs (`_fit_fold` emission, XGBoost `**fit_kwargs`) + the `sys.executable` subprocess fix → **get CI green.** A green CI is the precondition for every other claim of quality.
- Add CI tiering/caching so the green signal is fast enough to read; add `pytest-cov`.

**Phase 1 — Polish that reads as "product" (1–2 weeks):**
- Global `report_callback_exception` handler; route `print()` → logging layer.
- DPI awareness + determinate sweep progress bar + move data load off the main thread.
- Progressive disclosure on the 461-widget config tab; tooltip coverage pass; dark mode.

**Phase 2 — Structural de-risking (ongoing, incremental):**
- Extract the 6 BaseEstimator wrappers + `_reconstruct_models_from_results` + inline CV into `src/spectral_predict/` (first bite out of the monolith; makes them testable + pickle-safe).
- Kill the `sys.path` dual-import hack (editable install) + guard test.
- Pin vendor deps + add a lock file; commit the missing installer build script → reproducible releases.

**Phase 3 — Close the commercial gap (weeks):**
- Validation report generator (PDF/HTML); headless predictor + watch-folder; b-vector/loadings plots; T-16 pairwise comparison stats.
- Exit beta + validation white-paper.

**Deliberately NOT recommended:** 21 CFR Part 11 audit trails, real-time PAT integration — wrong market, competitors' moats, multi-year efforts.
