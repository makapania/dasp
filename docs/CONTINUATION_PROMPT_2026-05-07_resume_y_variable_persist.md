# Continuation prompt — T-resume-y-variable-persist

**Filed:** 2026-05-05 wrap-up, after PR #41 (validation-rebuild MEDIUM) merged + PR #42 (T-resume-y-variable framing correction) merged. GLM 5.1 reviewed the framing and surfaced implementation specifics.

**For:** the next agent picking up the resume-y-variable persistence ticket.

**Branch state at filing:** all PRs from 2026-05-05 (#32, #33, #35, #36, #38, #39, #40, #41, #42) merged. PR #34 closed (false-premise), PR #37 closed (redundant docs). `origin/main` at `2db6311`. Working tree clean.

---

## What this ticket is

A **state-restoration feature gap**, NOT a bug. When a user disrupts a run and clicks Resume, the y-variable selection defaults to the file's first column (correct load behavior) rather than recalling the variable the original run was using. If the original run was a classification analysis but the file's first column happens to be a regression variable, resume effectively can't restart against the original target.

The fix is **additive**, not corrective: persist the y-variable selection alongside the rest of the run state already round-tripped through the sidecar, then restore it on resume.

---

## Pre-verified implementation surface (GLM 5.1 review of PR #42)

**Don't re-investigate this — GLM already verified all of it against the current code.**

### What to add

1. **`src/spectral_predict/run_gui_settings.py`** — add `"target_column"` to the `CAPTURABLE_SETTINGS` whitelist (lines 45-211). The Tk variable to capture is `self.target_column` (a `StringVar` at `spectral_predict_gui_optimized.py:2837`).

### What does NOT need to change

- **`src/spectral_predict/run_state.py`** — already stores `gui_settings: dict[str, Any] | None` generically (line 148), populated by `capture_gui_settings`. The dict is a flat structure and `from_dict` already ignores unknown keys (line 183) — adding `target_column` is forward-compatible with existing sidecars.
- **`restore_gui_settings`** — works generically; calls `gui_obj.<key>.set(value)` (line 259), which works fine for `StringVar`.

### The edge case you MUST handle

**If the restored column name doesn't exist in the newly-loaded dataset** (renamed column, different file, or even the same file with column reordering), the Combobox will display a stale string that isn't in its option list. The user gets a "stuck" combobox they can't interact with cleanly until they manually pick a different option.

The existing readback check at `run_gui_settings.py:299-309` only verifies `.set()`/`.get()` round-trip — it does NOT validate against the current data's columns. So you need to add validation:

```python
# Validate restored target_column against the current dataset's columns.
# If the column name from the sidecar isn't in the loaded data (different
# file, renamed column, etc.), fall back to the default rather than leaving
# the combobox in an inconsistent state.
if hasattr(gui_obj, '_get_available_target_columns'):
    available = gui_obj._get_available_target_columns()
    if value not in available:
        # Fall back to first column (the existing on-load default).
        # Optionally log to GUI run-log so the user knows.
        return False  # or whatever signals "skip this restore"
```

`_get_available_target_columns()` is defined at `spectral_predict_gui_optimized.py:16849`. Verify the actual signature/behavior before wiring.

### Related state already in the sidecar

`dataset_fingerprint` (`run_state.py:138`) already detects when the user loaded different data on resume. A stale-column-name scenario would likely co-occur with the existing fingerprint-mismatch warning. **The validation-against-current-columns is still required** for cases where the fingerprint matches but the column ordering/naming changed (user added a column, deleted a column, etc.) — fingerprint is a coarse data-identity check, not a per-column schema check.

---

## Test plan

Mirror existing patterns in `tests/test_run_gui_settings.py` (or wherever `restore_gui_settings` is tested):

1. **Round-trip test** — capture `gui_settings` with a non-default `target_column`; restore into a fresh GUI mock; assert `gui_obj.target_column.get()` matches.
2. **Stale-column fallback test** — capture `gui_settings` with `target_column='ProteinPct'`; restore into a GUI mock whose `_get_available_target_columns()` doesn't include `'ProteinPct'`; assert the restore falls back gracefully (no exception, combobox shows a valid value).
3. **Backward-compat test** — load a sidecar from an older version that doesn't have `target_column` in `gui_settings`; assert `restore_gui_settings` doesn't crash and leaves `target_column` at its current default.
4. **Forward-compat test** — load a sidecar with extra unknown keys in `gui_settings`; assert no crash (the existing line 183 unknown-key behavior).

End-to-end test (optional — high-friction GUI setup): start classification analysis with non-default y → kill mid-run → restart GUI with same data → click Resume → assert the y-variable dropdown matches the original.

---

## Recommended pickup order

1. Read `docs/PROJECT_STATUS.md` top section — captures all 2026-05-05 PRs and the deferred items.
2. Read `docs/SESSION_LOG.md` top entry (2026-05-05 wrap-up) — captures the framing-correction lesson and the audit-pyramid lessons from PR #41.
3. Read `src/spectral_predict/run_gui_settings.py` — understand the `CAPTURABLE_SETTINGS` whitelist + `capture_gui_settings` + `restore_gui_settings` flow before editing.
4. Read `src/spectral_predict/run_state.py` lines 130-200 to see the `dataset_fingerprint` + sidecar persistence shape.
5. Read `spectral_predict_gui_optimized.py:16849+` for `_get_available_target_columns` to confirm the validation API.
6. Branch `fix/Tresume-y-variable-persist` from `origin/main` (currently at `2db6311`).
7. Implement the addition + edge-case validation. Should be ~10-30 LOC across 2 files.
8. Write 4 tests (round-trip, stale-column fallback, backward-compat, forward-compat). ~50-80 LOC.
9. Targeted pytest sweep + the consolidated regression battery for sidecar/resume work:
   ```
   test_run_state    test_run_gui_settings    test_t41_bayesian_sqlite_auto_calculator
   test_t42_write_path_plumbing    test_t43_resume_auto_restore
   ```
   Plus the pre-existing flake list — `test_export_code.py::test_python_script_execution` and `test_full_workflow_python` (deselect).
10. Commit, push, open PR. Dispatch DeepSeek + GLM + Codex for cross-family review (small change, single round should suffice unless something surfaces).

Estimated effort: **1-2 hours** (much smaller than the validation-rebuild MEDIUM because the implementation surface is well-spec'd and the test patterns already exist).

---

## Smaller deferred items still in the queue (fold in or separate)

If time permits after the resume-y-variable work:

### PR #33 deferred HIGHs (medium leverage, ~30 LOC)

- **pr-test-analyzer HIGH #1**: behavioral test for `n_jobs` survival through codegen. Add `assert "'n_jobs': 1" in script` (substring check on generated script content) for both XGBoost and RandomForest. The structural pin tests do NOT subsume because a refactor that moves the bug to a different mechanism would pass the structural pins but fail the behavioral test.
- **pr-test-analyzer HIGH #2**: architectural module-walk test that asserts no module-level `*PIPELINE_PARAMS*` set in the codebase contains `n_jobs`. Prevents a future third-PIPELINE_PARAMS sister site from being silently born.

### PR #32 deferred MEDIUM (~80 LOC)

- pr-test-analyzer flagged that PR #32 added the XGBoost auto-with-correction parity row but the LightGBM and CatBoost rows are still missing the symmetric coverage. Add `test_lightgbm_binary_classification_parity_auto_with_correction` and `test_catboost_binary_classification_parity_auto_with_correction` mirroring the XGBoost row pattern.

### Polish (low priority — surface only if relevant)

- `_apply_class_weight_discriminator_for_rebuilt_model` at `search.py:408+` — silent-failure-hunter MEDIUM noted that `inspect.signature(model.fit)` is not wrapped in `try/except`. Theoretical risk on C-extension fits without proper introspection support; all current sklearn estimators have inspectable signatures, so deferred. If a real failure surfaces, wrap line ~496 in `try/except (ValueError, TypeError): fit_sig = None` to fall through to the warn-and-return-{} path.
- The same helper's MLP branch is sklearn-version-dependent (sklearn 1.7+ adds `sample_weight` to `MLPClassifier.fit`, so MLP would route through the sample_weight fallback rather than the no-mechanism warning). Currently this is correct behavior but untested. Add a version-aware test if MLP-with-class_weight surfaces in real use.

---

## Process rules (mandatory)

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** Per-fix: matching test file + the sidecar/resume regression sweep above.
- **Pre-existing flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental.
- **Commit ASAP after implementation.** Long pytest runs / agent dispatch / `git push` can collide with parallel sessions. See memory `feedback_commit_before_yielding_control.md`.
- **Don't use `git add -A` for code commits.** Stage explicitly by file. The 2026-05-05 afternoon session had a `.review-tmp/` accident that polluted PR #38 with 39k+ LOC of scratch artifacts. The `.gitignore` now covers `.review-tmp/` but explicit staging is the belt-and-suspenders.
- **Don't use `pytest | tail`.** Use `2>&1 | grep -E "passed|failed"` for length-limited output.
- **Comments default to none.** Only when WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`) in production code.
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case.**
- **Don't add features beyond what the ticket requires.** Bug fixes / hygiene fine; methodology changes need user confirmation.
- **For comments referencing other code, use stable function-name identifiers, not line numbers.** Line numbers rot on every edit. Function names are refactor-stable and grep-queryable. (Lesson from PR #41 comment-analyzer.)
- **Structural pins should be one-per-consumer, not one-per-producer.** Asserting a contract exists at the producer (function signature) doesn't catch a consumer dropping the call. (Lesson from PR #41 Codex review.)

---

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-fix: tests passing count, review verdicts, deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on a BLOCKER, file `docs/CONTINUATION_PROMPT_<date>_<topic>.md` for the next agent.
