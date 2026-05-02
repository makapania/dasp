# T-44: `n_trials_var` typo silently zeros `n_trials_per_model` metadata

**Status:** PLANNED — surgical fix, ~5 LOC.
**Filed:** 2026-05-02.
**Source:** DeepSeek V4 Pro Max review of T-43 (LOW #7), surfaced again in pr-review-toolkit pass on the consolidated T-43 + T-42 + T-38 PR.
**Priority:** LOW — metadata-only; doesn't affect search behavior, only the value stored in `RunMetadata.n_trials_per_model` and the resume banner's "Trials per model (target)" line.

## Problem

`spectral_predict_gui_optimized.py:25123` (line numbers may drift):

```python
n_trials_per_model=int(self.n_trials_var.get())
if hasattr(self, "n_trials_var") else None,
```

The Tk var on the GUI is `self.n_unified_trials`, NOT `self.n_trials_var`. The `hasattr` guard short-circuits to `None` on every Bayesian `start_run` call. As a result:

- `RunMetadata.n_trials_per_model` is always `None` in every sidecar.
- The resume banner shows "Trials per model (target): ?" instead of the actual count.
- T-43 captures `n_unified_trials` correctly via the whitelist, so auto-restore on resume still works — but the run-state metadata field is wrong.

## Fix

Change the line to `int(self.n_unified_trials.get())` and update the `hasattr` guard accordingly. One-line fix.

```python
n_trials_per_model=int(self.n_unified_trials.get())
if hasattr(self, "n_unified_trials") else None,
```

## Verification

After the fix, confirm via:

1. Start a Bayesian run with `n_unified_trials=200`.
2. Inspect the sidecar JSON: `n_trials_per_model` should equal `200`, not `null`.
3. Kill the process; relaunch; confirm the resume banner shows "Trials per model (target): 200".

A regression test in `tests/test_run_state.py` could pin this by mocking the GUI attribute, but the test would mostly assert "we passed the correct attribute" rather than testing real behavior — judgment call whether worth adding.

## Out of scope

- Renaming `n_unified_trials` to `n_trials_var` (or vice versa) — bigger refactor.
- Other phantom `hasattr`-guarded reads in the GUI; this fix is for the one site only.
