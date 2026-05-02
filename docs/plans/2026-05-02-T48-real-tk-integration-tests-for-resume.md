# T-48: Real-Tk integration tests for the resume flow

**Status:** PLANNED — testing-infra ticket.
**Filed:** 2026-05-02.
**Source:** pr-review-toolkit pr-test-analyzer Improvements #2 and #3 during the consolidated T-43+T-42+T-38 review.
**Priority:** LOW — current `_FakeVar`/`_FakeGUI` shim coverage is sound at the unit level; this ticket pins the contract to real Tcl behavior so future Tk version bumps can't break the production code while tests still pass.

## Background

T-43 unit-tests the GUI-settings capture/restore via `_FakeVar` / `_FakeGUI` shims that mimic Tk's `.get()`/`.set()` surface. The shims do NOT reproduce Tcl's actual behavior — most importantly, the silent-string-into-IntVar corruption pattern that the production code's set/get verify branch exists to catch.

If a future Tk version changes `IntVar` to raise on `set("abc")` (which Python's typing semantics might suggest), the production verify branch goes dead, but the unit tests still pass because `_PoisonableIntVar` simulates the old behavior unconditionally.

Similar concern for the resume dialog flow — `_check_for_incomplete_run` orchestrates 6 imports, dialog, restore, override, and three `_log_progress` branches. The order-dependent contract (restore-then-override) is unit-tested at the helper level on a fake-GUI but never exercised end-to-end on the real method.

## Tests to add

Use the existing `tests/gui/harness.py` pattern (already in the codebase for other Tk tests).

### 1. Real `tk.IntVar` silent-corruption smoke

Pins the Tcl behavior the verify branch protects against:

```python
@pytest.mark.skipif(no_display_available, reason="needs Tk")
def test_real_tk_intvar_silently_accepts_string():
    root = tk.Tk(); root.withdraw()
    try:
        var = tk.IntVar(master=root, value=0)
        var.set("abc")           # must NOT raise per Tcl behavior
        with pytest.raises(_tkinter.TclError):
            var.get()            # the verify branch's exact target
    finally:
        root.destroy()
```

Lives at `tests/gui/test_t43_real_tk_int_var_corruption.py`.

### 2. Real-Tk smoke for `_check_for_incomplete_run` resume-yes path

Asserts the order-dependent contract on the actual method, not on a re-implementation:

```python
def test_resume_yes_path_restores_then_overrides_persistence(monkeypatch, tmp_path):
    # 1. drop a sidecar with gui_settings (incl. bayesian_persistence_mode='auto')
    # 2. instantiate SpectralPredictApp with --no-mainloop equivalent
    # 3. monkeypatch messagebox.askyesnocancel -> True
    # 4. call app._check_for_incomplete_run()
    # 5. assert app.use_snv.get() reflects sidecar value (restore happened)
    # 6. assert app.bayesian_persistence_mode.get() == 'always' (override AFTER restore)
```

Catches a refactor that moves the override above the restore call inside the real method — which the existing fake-GUI test only documents on a stand-in.

### 3. Real-Tk smoke for `_check_for_incomplete_run` resume-yes path with `resume_run() returns None`

Asserts the new (T-43 pr-review-toolkit fix-of-fixes) `else: messagebox.showwarning(...)` actually fires when `resume_run` returns None.

## Why deferred to its own ticket

Adding real-Tk tests to the CI matrix:

- Requires display infrastructure on Windows runners (not all do).
- Adds non-trivial test runtime (~5-10 sec per real-Tk test for window create/destroy).
- The shim coverage is sound at the unit level; this is belt-and-braces.

Should land alongside other GUI test infrastructure work, not piggyback on T-43.

## Out of scope

- Mocking the entire Tk event loop for headless CI. That's a bigger investment.
- Testing the actual visual rendering of the resume banner. Out of scope for unit/integration tests; manual smoke before each release covers this.
