# Codex Handoff – Tab 7 R² Mismatch

## Status Snapshot
- Introduced `Tab7ModelState`, wavelength resolver, and `TAB7_TRACE` helper in `spectral_predict_gui_optimized.py` so Results-tab loads persist the exact wavelengths, preprocessing metadata, CV folds, backend, and performance expectations.
- `_load_model_to_NEW_tab7()` now builds the dataclass, preserves VarSelectionIndices, and logs `load_to_tab7:state_saved`; `_tab7_run_model_thread()` consumes the preserved state through `resolve_tab7_wavelengths()` and refuses to fall back to the textbox when the GUI is in “Loaded from Results” mode.
- Added `tests/test_tab7_state.py` (guards resolver semantics) and `tests/test_tab7_headless_integration.py` (drives `_tab7_run_model_thread()` without Tk widgets to ensure wavelength order stays intact and the runner fails loudly if state disappeared). Verified via `python3 -m pytest tests/test_tab7_state.py tests/test_tab7_headless_integration.py -q`.
- Rehydrated documentation: `HANDOFF_R2_MISMATCH_INVESTIGATION_2025_11_07.md` now includes a “Latest Update” section plus reminders about gathering a real GUI trace; `CODEX_HANDOFF.md` (this file) tracks status/outstanding work.

## Outstanding Work
1. **Manual GUI verification with tracing**  
   - Command: `TAB7_TRACE=1 python spectral_predict_gui_optimized.py` (requires a machine with Tk display; CLI sandbox throws `Abort trap: 6` when creating `tk.Tk()`).
   - Load the failing dataset, run search, pick the problematic derivative+subset model, click “Load to Model Development,” switch to Tab 7, run it.  
   - Capture stdout showing `load_to_tab7:state_saved`, `tab7_selected`, `run_tab7:start`, and `[OK] Order MATCHES …`, plus the R² values from Results vs Tab 7. Drop the transcript + numbers back into the handoff doc.

2. **GUI-driven regression**  
   - The new headless test exercises the runner, but we still need a Tk-enabled test that invokes the real notebook/tab widgets (button `.invoke()`, `notebook.select(6)`, etc.) so CI can catch regressions where events wipe out `tab7_model_state`.

3. **Broader regression sweep**  
   - Once a GUI trace looks good, rerun the heavier scripts/tests: `test_tab7_automated_full.py`, `test_wavelength_fix_simple.py`, `test_tab7_diagnostics.py`, plus any workflow-specific smoke tests the stakeholders rely on.

4. **Stakeholder validation**  
   - Share the trace instructions with the original reporter, capture their before/after R² numbers, and confirm the fix on their hardware (esp. when using Julia backend or editing the wavelength textbox after loading).

## Quick Reference
- Code touchpoints: `spectral_predict_gui_optimized.py` (`Tab7ModelState`, resolver, `_load_model_to_NEW_tab7`, `_tab7_run_model_thread`), `tests/test_tab7_state.py`, `tests/test_tab7_headless_integration.py`.
- Docs: `HANDOFF_R2_MISMATCH_INVESTIGATION_2025_11_07.md` (investigation narrative + tracing guide), `CODEX_HANDOFF.md` (this progress log).
- Validation: `python3 -m pytest tests/test_tab7_state.py tests/test_tab7_headless_integration.py -q` (fast); GUI trace still pending due to sandbox Tk limitations.
