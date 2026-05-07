# Continuation prompt — CI hygiene (T-CI-1)

**Filed:** 2026-05-07 late evening, after PR #55 merge (`228bbc2`)
**Pickup:** next session

---

## Context

GitHub Actions CI on `main` has been **red since the workflow was added 2025-10-27** (commit `cc83e83`). Zero green runs in 6 months. `gh run list --branch main --status success` returns an empty list.

This was discovered while merging PR #55 — investigation showed the 91 failing tests on PR #55 were identical to the 91 failing tests on `origin/main`'s pre-merge state. The deletion didn't add or remove any. The user's `feedback_tests.md` discipline ("Don't run full test suite for small changes — use targeted tests instead") has kept local development clean while CI rotted in parallel.

**Net assessment:** CI rot is largely environmental (Linux headless + library drift), not code bugs. Fixing it is mostly workflow-config edits + a few dependency pins. **3–5 real codegen-template bugs are out of scope here** (filed separately — see bottom of this prompt) so the next session doesn't mix concerns.

The goal of this ticket is to get the CI badge green — or at least into a state where a single new failure means "investigate this specific real bug" rather than "ignore, pre-existing rot."

## Failure categories on `main` (PR #55 baseline)

Total: **91 failing tests**. From `gh run view 25471686544 --log-failed` (any post-PR-#55 run will show the same set minus the 3 SQLite-rehydration-related lines that are just PASS noise in failed-job output).

| # | Category | Count | Root cause | Fix |
|---|---|---|---|---|
| 1 | GUI / Tkinter | 72 | `tests/gui/test_*.py` collection-time fixture failure on headless Linux. Tkinter requires `$DISPLAY`. | Add `xvfb-run` wrapper to the Linux test job in `.github/workflows/ci.yml`, OR skip-mark the GUI test files when `os.environ.get('DISPLAY') is None and platform.system() == 'Linux'`. |
| 2 | matplotlib TkAgg | 1 (`test_run_logging.py::test_cli_main_calls_setup_app_logger`) | `ImportError: Cannot load backend 'TkAgg'` — matplotlib trying Tk in headless mode. | Set `MPLBACKEND=Agg` env var in the workflow (apply to all jobs), OR have the test force `matplotlib.use('Agg')` before importing the module under test. |
| 3 | jcamp library API drift | 5 (`test_io_jcamp.py`) | `AttributeError: module 'jcamp' has no attribute 'jcamp_write'. Did you mean: 'jcamp_parse'?`. Upstream `jcamp` package renamed/removed `jcamp_write`. | Either pin `jcamp<X.Y.Z` (whatever version still had `jcamp_write`) in `pyproject.toml`, OR update `src/spectral_predict/io.py` to use the new API name. Pin is cheaper; API update is more durable. |
| 4 | Test fixtures missing/bad | 4 | `tests/test_io_spc.py` × 2 (`Buffer size too small (102 instead of at least 512 bytes)` — bad fixture file); `tests/test_io_vendor_formats.py` × 2 (`File not found: dummy.0` and `dummy.sp`). | Replace the SPC fixture with a real ≥512-byte SPC binary; create the OPUS / PerkinElmer dummy files OR rewrite the tests to use `tmp_path`. |
| 5 | Numerical precision drift | 2 (`test_savgol_first_derivative_matches_scipy`, `test_savgol_second_derivative_matches_scipy`) | `Mismatched elements: 6 / 1000 (0.6%)`. Max absolute difference: **3.86e-12**. Test tolerance `rtol=atol=1e-12` is below the realistic precision floor for SG derivatives across scipy versions. | Relax to `1e-10` in both tests. Still proves correctness without false-flagging FP drift. |
| 6 | Optuna API behavior drift | 1 (`test_tpe_preprocessing_discovery.py::TestEdgeCases::test_progress_callback_fires`) | `assert 14 == 10` — Optuna 4.8 fires the progress callback more often than 4.7 did. | Change to `assert callback_count >= 10` (the contract is "fires at least once per trial," not "fires exactly N times"). |
| 7 | Stochastic ML test flake | 1 (`test_neural_boosted.py::TestClassifierIntegration::test_train_test_split_performance`) | `Large train-test gap suggests overfitting`. Stochastic; can flake on different RNG seeds across Python/sklearn versions. | Either pin `random_state` more aggressively + relax the gap threshold, OR `@pytest.mark.flaky(reruns=2)` if `pytest-rerunfailures` is acceptable. |

**73 of 91 failures resolve in category 1** alone (xvfb-run). **+1 in category 2.** **+5 in category 3.** **+4 in category 4.** **+2 in category 5.** **+1 in category 6.** **+1 in category 7.** Total: **87 of 91 resolved by this ticket's scope.** The remaining 4 are real code bugs filed separately (see "Out of scope" below).

---

## Plan

Read each task before acting; the order matters because category-1 fixes dominate the noise floor — get those in first so subsequent failures are visible as real signal.

### Task 1 — Workflow-level Linux environment fix (highest leverage)

**Files:** `.github/workflows/ci.yml`

Add to the Linux test job (currently runs `pytest -v --tb=short`):

```yaml
    - name: Install xvfb (Linux only)
      if: matrix.os == 'ubuntu-latest'
      run: sudo apt-get update && sudo apt-get install -y xvfb

    - name: Run tests
      env:
        MPLBACKEND: Agg
      run: |
        if [ "${{ matrix.os }}" = "ubuntu-latest" ]; then
          xvfb-run -a pytest -v --tb=short
        else
          pytest -v --tb=short
        fi
      shell: bash
```

`MPLBACKEND=Agg` is set for ALL jobs (not just Linux) — Windows currently passes the matplotlib test by accident (Tk works there) but defensive is cheaper than discovering it.

Also apply the same `env: MPLBACKEND` + `xvfb-run` wrapper to the `test-optional-deps` job (which currently fails the GUI tests for the same reason).

Verification: push to a throwaway branch, watch the GitHub Actions run, expect ~73 ERROR → PASS conversion (or at minimum the GUI tests should reach actual test bodies and either pass or report real assertion failures, not collection ERROR).

### Task 2 — jcamp library pin

**Files:** `pyproject.toml`

Find the `jcamp` entry (probably under `[project.optional-dependencies]` or similar). Check current PyPI history at `pip index versions jcamp` to find the last version with `jcamp_write`. Pin to `jcamp>=A.B,<C.D` (the upper bound being the version that removed it).

If pinning isn't viable (e.g. a security advisory on the older version), update `src/spectral_predict/io.py` to use whatever the current jcamp API is. Likely the writer was renamed or moved into a submodule.

Verification: `pip install -e .[dev]` then `pytest tests/test_io_jcamp.py -v` — expect 5 FAILED → PASS.

### Task 3 — Numerical precision relaxation

**Files:** `tests/numerical/test_preprocessing_correctness.py`

Find `TestSavgolDerivativeCorrectness::test_savgol_first_derivative_matches_scipy` and the `_second_` sibling. Locate the `np.testing.assert_allclose(..., rtol=1e-12, atol=1e-12, ...)` calls and change to `rtol=1e-10, atol=1e-10`.

The actual measured drift is 3.86e-12 — relaxing to 1e-10 keeps a 25× safety margin while resolving the false-positive.

Verification: `pytest tests/numerical/test_preprocessing_correctness.py::TestSavgolDerivativeCorrectness -v` — expect 2 FAILED → PASS.

### Task 4 — Optuna callback assertion relaxation

**Files:** `tests/test_tpe_preprocessing_discovery.py`

Find `TestEdgeCases::test_progress_callback_fires`. Change `assert callback_count == 10` (or whatever the exact assertion is) to `assert callback_count >= 10`. The contract being pinned is "TPE fires the callback for each trial" — exact count over-specifies what Optuna's behavior actually guarantees.

Verification: `pytest tests/test_tpe_preprocessing_discovery.py::TestEdgeCases::test_progress_callback_fires -v` — expect 1 FAILED → PASS.

### Task 5 — Test fixture cleanup

**Files:**
- `tests/fixtures/` (or wherever the SPC fixture lives — `find tests -name "*.spc"`)
- `tests/test_io_vendor_formats.py`

For SPC: replace the truncated 102-byte fixture with a real ≥512-byte SPC binary. If you don't have one, the test can be reframed to assert "function raises specific error on too-small input" rather than "function reads this specific file successfully."

For OPUS/PerkinElmer: the tests `test_opus_import_error` and `test_perkinelmer_import_error` are NEGATIVE tests asserting that the import-error path fires when the file isn't readable. The current code-under-test is raising `ValueError: File not found` BEFORE reaching the import-error branch (because the file doesn't exist). Either:
- Create empty `tests/fixtures/dummy.0` and `tests/fixtures/dummy.sp` (just `touch` them — the import-error branch fires when the file exists but the parser library isn't installed); or
- Use `tmp_path` and write a placeholder byte to a file there.

Verification: `pytest tests/test_io_spc.py tests/test_io_vendor_formats.py -v` — expect 4 FAILED → PASS.

### Task 6 — Stochastic test stabilization

**Files:** `tests/test_neural_boosted.py`

Find `TestClassifierIntegration::test_train_test_split_performance`. Two options:

A. **Tighten determinism:** thread `random_state=42` through every stochastic step (train/test split, model init, any resampling). Then re-run locally several times — if it's stable, leave the gap threshold as-is.

B. **Mark as flaky:** if `pytest-rerunfailures` is in `pyproject.toml`'s dev deps, add `@pytest.mark.flaky(reruns=2)`. If not, install it first or use option A.

Verification: run the test 10 times in a loop locally — should pass 10/10 after the fix.

### Task 7 — Verify CI is green

After Tasks 1-6 land, push to `main` and watch the next CI run. Expected: at most 4 remaining FAILED tests, all of which are the real codegen bugs filed in the out-of-scope section below.

If CI is fully green, mark T-CI-1 as CLOSED in `docs/PROJECT_STATUS.md`. If 4 codegen failures remain, that's expected — file follow-up tickets per the out-of-scope section.

---

## Out of scope — file as separate tickets

These are real code bugs (not CI environment problems) that have been masked by the broader rot. **Do not bundle them into T-CI-1.** Each deserves its own investigation:

### T-CI-2 (proposed): codegen template `_fit_fold` NameError

`tests/test_cv_strategy.py::TestPostMergeReviewFixes::test_classification_metrics_template_has_no_nameerror` — `NameError: name '_fit_fold' is not defined`. Some code-generation template references `_fit_fold` (the per-fold helper introduced for booster early-stopping) but doesn't emit/import it in the rendered output. Investigate `templates/validation.py` and `code_generator.py` for the metrics-block emission.

### T-CI-3 (proposed): T-19 codegen template regressions

`tests/test_t19_class_weight_per_library.py` × 2:

1. `test_xgboost_threads_sample_weight_via_fit_kwargs` — generated XGBoost script no longer contains `fold_model.fit(X_train_fold, y_train_fold, **fit_kwargs)`. The fit_kwargs splat is missing.
2. `test_non_xgboost_classification_does_not_emit_fit_kwargs_plumbing` — CatBoost generated script CONTAINS fit_kwargs plumbing in pure `class_weight` mode where it shouldn't.

Both look like regressions from PR #54's black-pass-and-comment-cleanup commit (`8ab4af1`?). Worth diffing `code_generator.py` and `templates/validation.py` against the last green version (impossible — there is no green version on `main`). Try `git log -p src/spectral_predict/code_generator.py src/spectral_predict/templates/validation.py` and look for changes near the fit-emission sites.

### T-CI-4 (proposed): CLI help/version exit code

`tests/test_cli_help.py::test_cli_help` and `test_cli_version` — both `assert 1 == 0`. The `python -m spectral_predict --help` (or equivalent) exits 1 instead of 0. Investigate `src/spectral_predict/cli.py` `main()` for the help/version handlers.

---

## Verification recipe (after T-CI-1 lands)

```bash
# Local-machine baseline
pytest tests/ --tb=short 2>&1 | tail -10
# Expect: same set of failures as CI (modulo Windows vs Linux differences)

# Verify nothing changed semantically — sanity check
pytest tests/test_legacy_deletion_snapshot.py 2>/dev/null || true  # file was deleted in PR #55, OK if missing
pytest tests/test_bayesian_dedup.py tests/test_t44_autoscale_wiring.py -v  # 28/28 expected
```

After PR for T-CI-1 merges:

```bash
gh run list --branch main --limit 3 --status success --json databaseId,conclusion
# Expect at least one entry — the first green main run on this repo
```

---

## Risk profile

**Low.** Every task in T-CI-1 modifies test infrastructure or test assertions, NOT production code. The worst case (e.g. tightening tolerance too much) shows up as a new failed test, not as a behavior regression in the GUI or in any model output.

The one task that could conceivably affect production behavior is Task 2 (jcamp pin or API update). If `src/spectral_predict/io.py` is updated to call the new jcamp API, a manual test of "load + read + write a JCAMP-DX file via the GUI" should be done before merge. If a version pin is used instead, no production-behavior risk.

---

## Sequencing note

Do Tasks 1 and 2 first — together they fix 78/91 failures and represent ~70% of the visible CI noise. Tasks 3-6 each fix 1-4 failures and can be done in any order or batched. Task 7 is the final close-out.

Estimated effort: 1-3 hours for someone familiar with the repo. Most time will be spent on Task 5 (finding/creating fixture files) if the binary fixtures aren't trivially recoverable.
