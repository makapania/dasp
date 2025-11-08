# CRITICAL R² MISMATCH BUG – INVESTIGATION HANDOFF
**Date:** November 7, 2025  
**Status:** Partially fixed – instrumentation + headless regression added, GUI trace still missing  
**Severity:** CRITICAL – Tab 7 (Model Development) can still collapse R² (0.97 → 0.58) during real GUI flows

## Latest Update (Current Session)
- Added `Tab7ModelState`, `tab7_trace`, and `resolve_tab7_wavelengths()` to `spectral_predict_gui_optimized.py`. Results-tab loads now persist all wavelengths, var-selection indices, preprocessing settings, CV metadata, backend, and expected metrics. `_tab7_run_model_thread()` refuses to fall back to textbox parsing when the GUI is in “Loaded from Results” mode, and it logs each lifecycle event when `TAB7_TRACE=1`.
- Implemented `tests/test_tab7_state.py` (unit guard for the resolver) and `tests/test_tab7_headless_integration.py`, a widget-free harness that instantiates `SpectralPredictApp`, runs the real `_tab7_run_model_thread()`, and asserts: (1) preserved wavelengths survive intact, and (2) Tab 7 fails loudly if the state vanishes mid-flow. Both tests pass locally via `python3 -m pytest tests/test_tab7_state.py tests/test_tab7_headless_integration.py -q`.
- Tried to spin up Tk in this CLI sandbox (`python3 - <<'PY'\nimport tkinter as tk\ntk.Tk()\nPY`) but the process aborts (`Abort trap: 6`). We cannot gather `TAB7_TRACE` output nor drive notebook/tab events here; a workstation with a display server is still required for manual reproduction.

## Why R² Still Diverges in the Real GUI
- Historical tests manually injected `tab7_loaded_config` or bypassed the GUI entirely, so they never exercised the actual “Results tab → Load button → Tab 7 run” event chain.
- Real users click “Load to Model Development,” switch tabs, and run models; along that path, Tk events or textbox edits can clear the preserved state or re-parse wavelengths in sorted order, yielding catastrophic R² drops (0.985 → 0.58).
- Until we capture a `TAB7_TRACE` log from the GUI, we cannot see where the lifecycle diverges (state cleared? tab change resetting widgets? derivative pipelines re-slicing data?). The new tracing + headless test give us the tooling; we still need real-world evidence.

## Immediate Next Steps
1. **Collect TAB7_TRACE evidence on a GUI-capable machine**
   - Run `TAB7_TRACE=1 python spectral_predict_gui_optimized.py`.
   - Reproduce the failing workflow (load dataset, run search, pick the problematic derivative+subset model, click “Load to Model Development,” switch to Tab 7, run it).
   - Ensure console shows `load_to_tab7:state_saved`, `tab7_selected`, and `run_tab7:start` with `state_has_wavelengths=True`, then record the R² values from Results vs Tab 7. Paste the transcript + metrics into this document.
2. **Add a GUI-driven regression test**
   - The new headless test covers the execution engine, but we still need a Tk-enabled test (e.g., `pytest.mark.gui`) that invokes the real buttons/notebook events to catch regressions where the UI wipes out `tab7_model_state`.
3. **Broader regression sweep once trace looks good**
   - Re-run `test_tab7_automated_full.py`, `test_wavelength_fix_simple.py`, and other Tab 7 diagnostics to confirm no collateral regressions.
4. **Stakeholder validation**
   - Share the trace instructions and headless test logs with the original reporter; confirm the fix on their hardware (especially for Julia backend and manual wavelength edits).

## Reference Commands
- Headless regression: `python3 -m pytest tests/test_tab7_state.py tests/test_tab7_headless_integration.py -q`
- GUI trace (needs display/Tk): `TAB7_TRACE=1 python spectral_predict_gui_optimized.py`

## Key Files
- `spectral_predict_gui_optimized.py` – Tab 7 state dataclass, resolver, tracing, and execution-path updates.
- `tests/test_tab7_state.py` – Unit tests for state-aware wavelength resolution.
- `tests/test_tab7_headless_integration.py` – Headless end-to-end Tab 7 runner coverage.
- `CODEX_HANDOFF.md` – Progress log + outstanding work checklist.
