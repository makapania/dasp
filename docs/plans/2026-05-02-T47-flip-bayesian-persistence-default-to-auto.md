# T-47: Flip T-41 default persistence mode from `'never'` to `'auto'`

**Status:** PLANNED — one-liner; gated on T-42's consolidated PR merging.
**Filed:** 2026-05-02.
**Source:** T-42 baseline benchmark surfaced that the post-T-41 SQLite WAL ratios already meet T-42's "definition of done" target. Filed as a separate trivial follow-up so the merge of T-42 is clean of behavior-changing UX defaults.
**Priority:** LOW — pure default-flip; users who want the prior behavior set the radio button to "Never on."

## Background

T-41 shipped with `bayesian_persistence_mode='never'` as default. The plan called for `'auto'` initially; the user requested the flip to `'never'` because pre-T-41 measurements showed PLS at 28× SQLite slowdown, XGBoost 1.36×, LightGBM 1.89×.

T-42's baseline benchmark (after T-41's WAL pragmas + 30s `busy_timeout` landed) showed:

| Model | SQLite WAL ratio |
|---|---|
| PLS | 0.69× |
| Ridge | 1.06× |
| RandomForest | 0.25× (caching) |
| LightGBM | 0.93× |
| XGBoost | 1.01× |

All well below the plan's 1.15× / 1.30× targets. T-42's Approach C added another small improvement (3 fewer trial-level writes per trial) — even tighter.

**T-41's 'never' default is now leaving crash-resume capability on the table for no measurable user-visible cost.** Flipping back to `'auto'` (which T-41's auto-calculator was designed for) is the right call.

## Fix

Two locations:

1. `src/spectral_predict/run_state.py:start_run`: change the `bayesian_persistence_mode: PersistenceMode = "never"` default to `"auto"`.
2. `spectral_predict_gui_optimized.py` Tk var: `self.bayesian_persistence_mode = tk.StringVar(value='auto')` is already `'auto'` post-T-41 (the user's flip earlier). Verify no GUI radio-button callback overrides at startup.
3. `src/spectral_predict/unified_bayesian.py:run_unified_bayesian` `enable_sqlite_persistence` parameter: confirm default also `'auto'`.

Update `docs/PROJECT_STATUS.md` Performance section to reflect the new default.

## Verification

1. Fresh launch: Bayesian persistence radio button reads "Auto" (already true).
2. `start_run()` called from a script with no `bayesian_persistence_mode` kwarg: confirm sidecar JSON has `"bayesian_persistence_mode": "auto"`.
3. Re-run `tests/_bench_bayesian_per_model.py` to confirm post-default-flip ratios still hold.

## Out of scope

- Removing the `'never'` mode entirely. Some users may want pre-T-11 in-memory speed for fast-iteration debugging where crash-resume isn't needed.
- Auto-detecting bad filesystems (OneDrive/Dropbox/network) and falling back to `'never'` defensively. That's a path-validation heuristic and orthogonal to this default flip.

## Dependencies

- Merge of `fix/bayesian-resume-and-cleanup` (T-42 + T-43 + T-38 consolidated PR) first, so the bench numbers in PROJECT_STATUS reflect the production code at the moment of the flip.
