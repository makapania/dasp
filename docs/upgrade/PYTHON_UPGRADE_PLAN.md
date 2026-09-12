# Python 3.14 + dependency upgrade — execution plan and running record

**Started:** 2026-09-12
**Analysis input:** `docs/PYTHON_UPGRADE_DECISION.md` Part B. That document is
**read-only analysis** — its own Appendix B states nothing was ever installed, built
or run. This file records what was actually executed, and corrects the analysis
where it did not survive contact with the repo.

## Governing principle

**One variable at a time.** The interpreter and the numerical dependencies both move
numerical results, so they never move in the same phase. Otherwise a divergence is
undiagnosable.

## Decisions taken (user, 2026-09-12)

- Install CPython 3.14 and carry through to a validated frozen bundle.
- Dependencies in two tiers: Tier 1 (non-numerical) lands freely; Tier 2 (numerical)
  is gated behind a before/after results comparison.
- CI matrix becomes `['3.12','3.14']`, `requires-python` floor raised to match
  (drops 3.10/3.11 — a deliberate support-policy change).
- User-visible artifact names (`SpectralPredict-py312.exe`, install directory,
  installer filename) stay unchanged, so existing installs upgrade in place instead
  of appearing as a second application.
- Build files get the interpreter **parameterized into a single setting**, not
  renamed to `py314`. Rationale: if PyInstaller turns out to work better on 3.12 for
  this codebase, rollback is a one-line edit rather than a 25-site rename.

## The baseline harness

`docs/upgrade/baseline_harness.py`. Run it identically before and after each phase
and diff the outputs; because each phase moves one variable, a diff is attributable.

```
<python> docs/upgrade/baseline_harness.py <label> --outdir <dir>
```

It emits per-sample out-of-fold predictions (12 significant digits) as well as the
aggregate `run_search` ranked tables, because aggregate CV metrics can round two
different prediction vectors to the same RMSE and report "identical" while
individual predictions moved. It pins BLAS/OpenMP threads to 1 before importing
numpy, pins `PYTHONHASHSEED`, and builds models with `n_jobs=1`.

It covers PLS, Ridge, RandomForest, XGBoost, LightGBM, MLP (regression) and PLS-DA,
RandomForest, XGBoost, LightGBM (classification). The tree/NN models are included
deliberately: they are all seeded at 42 internally, so "stochastic" does not mean
untestable, and they carry threading risk the linear models never exercise.

## Status

| Phase | State | Result |
|---|---|---|
| 0 — Baseline on 3.12 | **done** | Verified self-reproducible: two consecutive runs byte-identical, per-sample predictions included. Full suite: **7 failed, 2971 passed, 33 skipped, 50m29s**. |
| 1 — Version-agnostic repo fixes | in progress | |
| 2 — Tier-1 dependency bumps | not started | |
| 3 — 3.14 interpreter, pins held | **verified early** | See below. |
| 4 — Tier-2 numerical bumps | not started | Ordering corrected, see C6. |
| 5 — Frozen bundle + validation gate | not started | |
| 6 — CI, docs | not started | |

### Phase 3 result, obtained early

Because a new virtualenv touches neither `.venv312` nor the repo, Phase 3 was
testable during Phase 0 and was run early.

- **All 114 pinned distributions install on CPython 3.14.7** (GIL-enabled, x64).
  Only `jcamp` required a source build, and it succeeded.
- The full third-party stack and all 12 `spectral_predict` backend modules import.
- All three saved `.dasp` models in `example/` load on 3.14.
- **Results are byte-identical between 3.12 and 3.14 with identical pins** — both
  the per-sample predictions and the ranked tables.

Wheel availability is not working software, and this is not yet the frozen bundle.
But the interpreter itself is numerically neutral for this workload.

### Baseline pytest failure set on 3.12 (the anchor)

```
tests/gui/test_comprehensive.py::TestAllModelsViaGUI::test_catboost_via_gui
tests/gui/test_multiclass_gui.py::test_tab9_rejects_multiclass_primary
tests/test_cv_strategy.py::TestPostMergeReviewFixes::test_classification_metrics_template_has_no_nameerror
tests/test_export_code.py::test_python_script_execution
tests/test_export_code.py::test_full_workflow_python
tests/test_t19_class_weight_per_library.py::test_xgboost_threads_sample_weight_via_fit_kwargs
tests/test_t19_class_weight_per_library.py::test_non_xgboost_classification_does_not_emit_fit_kwargs_plumbing
```

**`PROJECT_STATUS.md`'s known-red list is stale**: it names 5 (`test_export_code` 2,
`test_cv_strategy` 1, `test_t19_class_weight_per_library` 2). The two `tests/gui/`
failures are also pre-existing but undocumented.

## Corrections to `PYTHON_UPGRADE_DECISION.md`

Found by direct execution, and by an adversarial Codex review of both documents.

- **C1.** Its jcamp "live drift" claim (installed 1.2.1 vs lockfile 1.2.2) is wrong.
  It read the package's stale `jcamp.__version__` string; pip metadata reports 1.2.2,
  matching the lockfile. The API-rename finding is still correct and still required.
- **C2.** B.4 says the build path hardcodes 3.12 "in four places". It is ~25
  occurrences across the three build files, several user-visible.
- **C3.** The spec docstring defers to a *"production 3.11 spec
  (spectral_predict.spec)"* that no longer exists. The py312 path is the only build
  path. Additional B.7 item.
- **C4.** It assumed 3.14 was available to build on. It was not installed.
- **C5.** B.2 frames source-only `jcamp==1.2.2` as the one blocker "resolved in B.2".
  It is not a blocker — it source-builds cleanly on 3.14. The bump to 1.3.2 remains
  worth doing, but it is desirable, not required.
- **C6.** The 3.12 float `sum()` change (its §0.3) does **not** apply to this
  migration. That landed in 3.11→3.12; this project already starts at 3.12, so the
  new algorithm is already in effect. 3.14 adds specialized *complex* summation only.
  Any plan citing it as a Phase 3 risk is citing the wrong rationale.
- **C7.** B.3's two build-tool pins do not constitute a locked build toolchain:
  PyInstaller also pulls `altgraph` and, on Windows, `pefile`, none of which are in
  the lockfile. The declared dev extra `pytest-timeout` is likewise absent.
- **C8.** B.1 overstates reader import protection. The PerkinElmer import
  (`readers/perkinelmer_reader.py:47`) is lazy but **not** inside `try/except`, and
  the JCAMP parser *call* sits outside the block that guards its import
  (`io.py:3293` vs `:3304`). Lazy importing reduces startup exposure; it does not
  guarantee a clean per-format failure.
- **C9.** B.6's "CI installs from floors" is imprecise. CI resolves *whatever
  satisfies* the floors, so each job can test a different stack; it installs neither
  the floors nor the certified lock.

## Corrections to this plan itself

- **P1. Phase 4's internal ordering was impossible.** `numba 0.66.0` requires
  `numpy<2.5` *and* `llvmlite<0.49` (verified from installed metadata), so numpy
  2.5.3 cannot be installed while numba 0.66 is pinned. **numba/llvmlite must be
  upgraded first**, holding numpy at 2.4.4; then numpy/scipy/pandas, then sklearn.
- **P2.** The first baseline script sorted on a column named `Preprocessing`; the
  real column is `Preprocess`, so the sort key was a partial no-op. Fixed in
  `baseline_harness.py`, which now raises if the expected columns are absent rather
  than silently sorting on nothing.
- **P3.** The first baseline exported only aggregate metrics and excluded the
  tree/NN models. Both fixed.

## Pre-existing issues found en route (NOT caused by the upgrade)

These are separate tickets. They are recorded here because the upgrade makes several
of them more dangerous, not because this work should fix them.

- **Optuna resume can mix numerical environments.** Study identity
  (`unified_bayesian.py:2562`) includes the application version and analysis config
  but **not** the Python or dependency versions. A resumed study reloads completed
  trial fingerprints and returns cached scores without refitting — so after an
  upgrade, with the app version unchanged, one study can blend results computed
  under two different numerical stacks. Use fresh studies for all migration
  comparisons, and preserve existing databases for rollback.
- **Installer build failure is silent.** In `build_installer_py312.py:281–338`,
  missing ISCC / ISS files and compiler failures are non-fatal, `main()` ignores the
  return value, and prior installer output is not cleared — so a failed build can
  report success while leaving a **stale** installer at the expected path.
- **`MultiGroupEPO` seeds off `hash(label)`** (`contaminant_analysis.py:2323`), which
  is `PYTHONHASHSEED`-dependent, and those vectors determine a projection applied to
  spectra.
- **NSGA-II's seed does not control all its streams** (`nsga2_search.py:187`, `:387`):
  mutation and sampling construct unseeded `default_rng()`, and initialization uses
  global `np.random`.
- **`.gitignore` enumerated venvs individually**, so `.venv314/` was not ignored.
  Fixed in this work (globbed to `.venv*/`) because it would have committed a
  virtualenv.
