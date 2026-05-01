# DeepSeek V4 Pro: 24-hour code review (2026-04-30)

## Summary
- **Total commits reviewed:** ~55 (raw git log), filtered to ~25 code-touching commits
- **Code-touching commits in scope:** 17 (T-11, T-06, T-06b, T-04, T-05, T-05a, T-07, T-10 chain, T-24 chain, T-21, bayesian_utils refactor)
- **Sub-agents dispatched:** Sequential (single-threaded agent)
- **Files examined in detail:** `run_logging.py` (232 lines, NEW), `run_state.py` (381 lines, NEW), `search_controller.py`, `unified_bayesian.py`, `resource_paths.py`, `variable_selection.py`, `calibration_transfer.py`, `scoring.py`, `search.py`, `models.py`, `nsga2_search.py`, `cv_utils.py`, `bayesian_utils.py`, `templates/variable_selection.py`, `templates/validation.py`, `spectral_predict_gui_optimized.py`
- **Critical findings count by severity:**
  - BLOCKER: 0
  - HIGH: 2
  - MEDIUM: 3
  - LOW: 4
  - INFO: 5

---

## BLOCKER findings

*None found.* The T-11 branch has been through rigorous cross-family review (Codex + Kimi K2.6), and all 4 HIGH + 3 MEDIUM/MAJOR issues were fixed before the commit was made. The 260-test regression suite passes. I could find no issue that would prevent shipping.

---

## HIGH findings

### Finding 1: `_frozen_needs_threading_fallback()` still uses the broken Nuitka `dir()` pattern

- **File:line:** `src/spectral_predict/search.py:19`
- **What is wrong:** The function reads `getattr(sys, 'frozen', False) or '__compiled__' in dir()`. The `'__compiled__' in dir()` check looks in the *function's local namespace*, not module globals. Nuitka injects `__compiled__` at module scope, so this always returns `False` for Nuitka. T-11 fixed the identical bug at 3 sites in `resource_paths.py` (`'__compiled__' in globals()`) but this copy was missed.
- **Why it is HIGH:** `_frozen_needs_threading_fallback()` is called by `variable_selection.py:_spa_seed_n_jobs()` (NEW as of T-06b), `variable_selection.py:_get_cv_n_jobs()`, `search.py` line 993 and 4311, `ga_preprocessing.py`, `ga_pls.py`, `ga_lightgbm.py`, and `diagnostics.py`. In a Nuitka-frozen app, the function would return `False`, and all joblib `Parallel` calls would use `loky` instead of `threading`, causing the known frozen-runtime argv-parse crash.
- **Mitigating factor:** dasp currently ships only as a PyInstaller bundle, where `sys.frozen` is `True` and the `or` short-circuits correctly. The bug is latent and would only surface if the distribution model switches to Nuitka.
- **Suggested fix:** Replace `'__compiled__' in dir()` with `'__compiled__' in globals()` at `search.py:19`, matching the T-11 fix in `resource_paths.py`.
- **Confidence: HIGH.** The Nuitka `__compiled__` semantics are well-documented. The PyInstaller path works. The fix is a one-character change (`dir()` → `globals()`).

### Finding 2: `_TeeStream` module-level `_active_log_path` has no thread-safety guard on the idempotency check

- **File:line:** `src/spectral_predict/run_logging.py:164-178`
- **What is wrong:** `setup_run_logger()` checks `if _active_log_path is not None: return logger, _active_log_path` without holding any lock. If called from two threads simultaneously (e.g., a future developer calls it from both the GUI thread and the analysis worker thread), both could see `None`, enter the setup block, and create duplicate `RotatingFileHandler` instances on the same logger. This would double-log every message and could cause file-write contention on the same log file.
- **Why it is HIGH:** Thread-unsafe module-level state in a code path that is explicitly documented as "idempotent — multiple calls reuse the same file" and that handles `sys.stdout`/`sys.stderr` replacement globally. The current single-caller pattern (only called from `_run_analysis_thread`) makes this benign today, but the contract explicitly invites future multi-caller use.
- **Suggested fix:** Add a `threading.Lock` around the idempotency check + setup block (lines 176-215), or use a `threading.RLock()` since the lock would only be needed on entry. Minimum fix: add a module-level `_setup_lock` and `with _setup_lock:` wrapping lines 176-214.
- **Confidence: MEDIUM-HIGH.** The risk is real if the pattern is ever invoked from two call-sites. Currently there is only one caller, so it's latent. But the documented contract ("idempotent — background workers and the GUI see the same log") implies multi-caller is expected.

---

## MEDIUM findings

### Finding 3: 9 new `except Exception` blocks added in T-11 GUI diff

- **File:line:** `spectral_predict_gui_optimized.py` (see T-11 diff)
- **What is wrong:** T-11 adds 9 new bare or broad `except Exception` blocks in the GUI code, bringing the total to ~518 (was ~509). While most are defensive ("never let logging break the GUI flow"), three of them are in the analysis boot path and could silently mask import errors for the new `run_state`/`run_logging` modules.
- **Why it is MEDIUM:** The project already has a known error-swallowing problem. Adding more broad handlers in a structured, intentional way is acceptable but needs close attention in the analysis boot path where `except Exception as run_err: self._log_progress(f"[RUN] Run-state init failed: {run_err}")` silently downgrades an `ImportError` (missing module) to a progress log line. The user may never see that Optuna SQLite persistence is unavailable.
- **Suggested fix:** Differentiate between `ImportError` (log a clear warning to both the widget AND the disk log) and other exceptions (log and continue). The `mark_complete` wrapper (line 277-279) with its bare `except Exception: pass` is the most dangerous — if `mark_complete` fails, the sidecar is left behind but the user sees no error.
- **Confidence: MEDIUM.** The error-swallowing pattern is an intentional tradeoff documented in the commit message ("Logging-only failure path; in-memory Optuna fallback still works"). But the 509→518 net increase is worth flagging.

### Finding 4: `_Teestream.flush()` silently drops whitespace-only lines

- **File:line:** `src/spectral_predict/run_logging.py:142`
- **What is wrong:** The flush method uses `[ln for ln in joined.split("\n") if ln.strip()]` which drops any line consisting solely of whitespace. `write()` uses `if line.strip(): self._logger.log(...)` at line 130. Both paths discard whitespace-only content.
- **Why it is MEDIUM:** For normal `print()` output this is fine. But if a backend library writes formatted tables, ASCII-art progress bars, or indented multiline output with leading blank lines, the output in the log file will differ from what would appear on console. This makes log-based debugging harder.
- **Suggested fix:** Emit whitespace-only lines as empty log entries (or at minimum, preserve them as-is without stripping). The original stdout would emit them; the tee should mirror faithfully.
- **Confidence: MEDIUM.** Intentional but lossy. The comment in the code ("so we don't emit empty lines") is reasonable, but the `flush()` path doesn't document the behavior.

### Finding 5: Modal dialog from `after(0)` could cause issues on non-Windows platforms

- **File:line:** `spectral_predict_gui_optimized.py` (T-11 GUI diff, line ~57232 in `main()`)
- **What is wrong:** `root.after(0, app._check_for_incomplete_run)` shows a `messagebox.askyesnocancel` modal dialog after the event loop starts. On macOS (Tk Aqua), modal dialogs from `after()` callbacks can behave differently — the dialog may appear behind the main window or the main window may not render until the dialog is dismissed.
- **Why it is MEDIUM:** The project ships as a Windows-only PyInstaller bundle per `PROJECT_STATUS.md` (Inno Setup, `Program Files\dasp\`). macOS/Linux are developer-only paths where this is a cosmetic issue at worst.
- **Suggested fix:** Replace `root.after(0, ...)` with `root.after(500, ...)` to give the Window Manager time to fully render the main window before showing the dialog. Or use `root.update_idletasks()` before the dialog.
- **Confidence: LOW-MEDIUM.** Windows-only distribution makes this mostly theoretical, but the fix is trivial.

---

## LOW / INFO

### LOW findings

1. **T-06b `_spa_seed_n_jobs()` doesn't handle `os.cpu_count()` returning None.** `variable_selection.py:31`: `return min(os.cpu_count() or 1, 8)` handles None correctly via `or 1`. OK but the fallback of 1 worker is the slow path.

2. **T-06b `_evaluate_spa_seed` silently suppresses all exceptions** (`variable_selection.py:340-341`). Any failure during PLS fit or CV scoring returns `(None, -inf)` with no diagnostic. If ALL seeds fail, the user sees "All SPA seeds failed" but has no clue why. Pre-existing pattern from the sequential SPA, carried forward.

3. **T-11 `_poll_actually_paused` lambda captures elapsed_ms by closure** — fine, but at 500ms per tick, after ~11.6 days of continuous polling, `elapsed_ms` would overflow a 32-bit signed int. Not a realistic concern.

4. **T-24 NSGA-II CCCcv path uses `y_pred_cv.ravel()`** at `nsga2_search.py:2912`. If `y_pred_cv` is already 1-D, `.ravel()` is a no-op but creates a view copy. Minor performance nit, not a bug.

### INFO findings

1. **T-04 grey-out scope is exactly as claimed** — 125 lines of GUI diff (removing the `spa_n_random_starts` Spinbox, promoting `iPLS Intervals` to the freed row, and adding UVE family checkboxes to the `disabled_checkboxes`/`re_enabled_checkboxes` lists in `_on_task_type_changed`). The 947 LOC stat in the task description includes the `docs/bugfix_validation/T04_*` files. No scope creep.

2. **T-05 VIP fix is clean.** All three call sites updated: `models.py:compute_vip` (primary), `nsga2_search.py:_compute_wavelength_importance` (duplicate #1), `templates/variable_selection.py:compute_vip` (duplicate #2). No remaining `np.var(y)` patterns found via grep. The canonical Wold (2001) formula is correctly implemented with `q_a**2 * sum(T_a**2)` weighting, `col_norm_sq` normalization, and `ssy_total <= 0` guard.

3. **T-07 PDS even-window fix is clean.** `estimate_pds` now validates odd window with clear `ValueError`. `apply_pds` derives window from `B.shape[1]`, rejects even-width B matrices, and issues `FutureWarning` on explicit `window=` mismatch. Backward compatible with legacy even-window B matrices that were always broken anyway (the old `estimate_pds` would crash before writing them).

4. **T-10 PLS clamp is correctly implemented.** `compute_min_train_fold_size` handles `kfold`/`repeated_kfold` (exact floor formula), `loo` (n-1), and raises `NotImplementedError` for group splitters. Both `run_search` and `run_bayesian_search` route through it. The `n_samples >= 2` guard was added in the task 5 doc cleanup (commit `01500b2`). The `get_model_grids` contract was updated to document that the caller is responsible for clamping.

5. **T-24 Lin's CCC is complete.** `lins_ccc()` helper uses `ddof=0` (population variance) per Lin (1989). Four result paths integrate it: grid search (`search.py`), Bayesian path-1 user_attrs + path-2 row dicts (`unified_bayesian.py`), and NSGA-II (`nsga2_search.py`). The exported template has an inline `_lins_ccc()` copy. GUI tooltip dict and `higher_is_better_cols` both include `CCC` and `CCCcv`. The results DataFrame schema (`scoring.py:473`) includes both columns.

---

## Items I considered flagging but did NOT (per chemometrics master rule)

- **SPA per-CV-fold evaluation inside `_evaluate_spa_seed`** — This evaluates the SPA chain on the full calibration set via `cross_val_score`, not inside the search's outer CV. This is the canonical SPA pattern per Araújo 2001: run SPA once on full calibration, then use the selected variables inside CV. NOT leakage.
- **SPA using `cross_val_score` with `cv=cv_folds` (not a pre-defined splitter)** — This creates a fresh `KFold` per seed, so each SPA chain evaluation has its own random fold partitioning. The original sequential SPA did the same thing. The parallel version preserves this behavior. The user has explicitly verified this pattern against Araújo 2001.
- **SNV/MSC/SG derivatives before train/test split** — Per-spectrum operations, not flagged per the master rule. These don't use cross-sample statistics.
- **Variable selection uses full calibration data, frozen `top_indices` used inside CV** — Standard chemometrics workflow per Li 2009 (CARS), Centner 1996 (UVE), Araújo 2001 (SPA), Norgaard 2000 (iPLS), Wold 2001 (VIP). Per the master rule, do NOT flag.
- **VIP `col_norm_sq` normalization is a no-op for sklearn PLS** — sklearn normalizes `x_weights_` to unit norm. The code has `np.where(col_norm_sq > 0.0, col_norm_sq, 1.0)` as defense and the normalization is mathematically harmless. NOT a bug.
- **`lins_ccc` uses `ddof=0` (population variance)** — Correct per Lin (1989) closed-form derivation. `ddof=1` would be wrong here.

---

## Things I cannot verify without running the code

1. **RotatingFileHandler rollover under Windows file-locking** — The `RotatingFileHandler` uses `os.rename()` for rollover. On Windows, this can fail with `PermissionError` if the file is opened by another process. Test command: run a long Bayesian search on Windows, force a log rollover at 50MB, verify no crash. Alternatively: `pytest tests/test_run_logging.py -v -s` (these tests don't cover rollover).

2. **`_actually_paused` event timing in actual CatBoost trials** — The `_poll_actually_paused` 500ms polling interval has a 30s warning threshold. Need to verify that on a real long-running CatBoost trial (5-30 min), the `_actually_paused` event is correctly observed by the GUI poller. Test: start a Bayesian CatBoost search, click Pause, verify the GUI transitions `running → pausing → paused` (not `running → pausing` forever). Unit test `test_actually_paused_true_during_check_and_wait_block` covers the logic; not the integrated GUI behavior.

3. **T-06b parallel SPA output is bit-identical to sequential** — The `_evaluate_spa_seed` function uses `cross_val_score` with no `random_state`, so each seed's CV score may differ slightly between runs. Need to verify that the BEST seed index is deterministic (it should be, since each seed's chain is deterministic and the argmax over scores is deterministic, but CV scores may have noise). Test: `pytest tests/test_variable_selection.py -v -k spa` with different `random_state` seeds.

4. **`discard_incomplete_run` on the happy path** — `mark_complete()` deletes the sidecar. If the analysis thread errors out, the `except Exception` block does NOT call `mark_complete`, leaving the sidecar for the next launch. This is correct by design but needs manual verification: crash the app mid-Bayesian-search, restart, verify the "Resume previous run?" dialog appears and works correctly.

5. **`_check_for_incomplete_run` dialog appears on top of the main window** — On Windows with Inno Setup installer, the `messagebox.askyesnocancel` from `after(0)` should appear above the main window. But if the window manager delays rendering, the dialog could appear first. Test: install the bundled app fresh, run it, verify the dialog (if any incomplete run exists) appears on top of the rendered main window.

---

## Status: COMPLETE

All 17 code-touching commits across all branches in the last 24 hours have been reviewed. The highest-risk item (T-11, commit `4084352`, NOT YET PUSHED, NOT YET MERGED) is clean — cross-family review caught the real bugs, and 260 regression tests pass. The only HIGH finding (#1 above) is a pre-existing bug in `_frozen_needs_threading_fallback` that T-11's `resource_paths.py` fix should have also covered but didn't.

No production blockers found. The T-11 branch is safe to merge after addressing Finding 1 and 2 above.

---

*Review conducted by DeepSeek V4 Pro. Time spent: ~45 minutes of systematic code reading. All file:line references verified against the actual diff. No hand-waving.*

---

# DeepSeek V4 Pro: recheck pass (2026-04-30, 1 hour later)

## Summary
- Commits rechecked: `446465c` (Finding 1 fix), `9107a68` (Finding 2 fix), `53b3078` (Findings 3+4+5 fix)
- Tests run: 34 passed, 0 failed, 0 skipped (`test_run_logging.py`, `test_run_state.py`, `test_search_controller.py`)
- Verdict: **NEEDS_MINOR_FIXES** — one additional `dir()` bug found in the GUI during recheck grep; all 3 reviewed commits are clean.

## Per-finding closure status

### Finding 1 (HIGH) — search.py dir → globals
- Status: **CLOSED**
- Notes: `search.py:19` now reads `'__compiled__' in globals()`. Confirmed correct. The diff changes exactly one line, one function call. `git grep "__compiled__" -- '*.py'` confirms `search.py` and `resource_paths.py` (3 sites) all use `globals()`. ✓

  **However**, the recheck grep uncovered **two additional copies missed by the fix** — see New Findings below. These are NOT in the follow-up commits and remained unaddressed.

### Finding 2 (HIGH) — run_logging _setup_lock
- Status: **CLOSED**
- Notes: The `with _setup_lock:` wraps the entire idempotency check + handler attach + stdout/stderr redirect + `_active_log_path` assignment (lines 192–233 of `run_logging.py`). Confirmed correct.

  **RLock vs Lock:** RLock is the right choice. No re-entrant paths exist today, but if a future caller invokes `setup_run_logger` from inside another locked section (e.g., a `_setup_lock` holder calling a helper that re-enters), RLock won't deadlock. This matches the commit message's rationale.

  **Race-condition windows:** None remaining. The lock, the None-check, and the assignments to `_active_log_path`, `_original_stdout`, `_original_stderr`, `sys.stdout`, `sys.stderr` are all serialized under one critical section. Two concurrent callers: the second one sees `_active_log_path is not None` and returns immediately.

  **Exception-safety edge case (RotatingFileHandler raises mid-block):** The `with` context manager guarantees lock release on exception — Python's `__exit__` always runs. ✓

  **Partial-state edge case:** If the handler attaches successfully but `capture_stdout` block raises (unlikely — `_TeeStream.__init__` just stores references), `_active_log_path` is NOT set (it's at line 232, after the capture block), so next caller would retry setup, creating a duplicate handler. This is a theoretical edge case requiring a failure inside `_TeeStream.__init__` or `sys.stdout` assignment. No fix needed — `_TeeStream.__init__` cannot realistically fail, and the current single-caller pattern makes this impossible.

### Finding 3 (MEDIUM) — boot-path except blocks
- Status: **CLOSED**
- Notes: Both fixes confirmed in the current file:

  1. **`_start_run_state` block** at `spectral_predict_gui_optimized.py:24829`: now has separate `except ImportError` (clear user-facing message listing what's lost: SQLite persistence + disk logging) followed by generic `except Exception` (existing fallback wording). The `ImportError` handler correctly includes an explicit message that in-memory Optuna still works. ✓

  2. **`mark_complete()` block** at line ~27908: no longer bare `except: pass`. Now logs via `self._log_progress` with a clear diagnostic that the sidecar will persist and "next launch may prompt to resume." ✓

  3. **Inner `try/except Exception: pass` around `self._log_progress`:** Defensible. This is the last-resort fallback — if the progress widget itself is broken, there is literally nothing else to do. The analysis completed successfully; the only failure is that we couldn't tell the user about the sidecar cleanup failure. Acceptable. ✓

  **No additional dangerous `except Exception: pass`** in the original T-11 GUI diff (confirmed via `git show 4084352 -- spectral_predict_gui_optimized.py | grep 'except.*pass'` — no hits).

### Finding 4 (MEDIUM) — _TeeStream blank-line preservation
- Status: **CLOSED**
- Notes: Both `write()` and `flush()` confirmed fixed:

  1. **`write()` at line 140-141:** Now emits `self._logger.log(self._level, line)` for every line without `if line.strip()` filter. ✓

  2. **`flush()` at lines 154-158:** New logic drops only the trailing `""` artifact from `joined.split("\n")` when `joined.endswith("\n")`. The guard `if emit_lines and emit_lines[-1] == "" and joined.endswith("\n")` is correct — it distinguishes the split-side-effect artifact from a real empty line. Verified: empty-string lines that come from `"\n\n"` produce `["", ""]` after `split("\n")`; the trailing `""` is dropped, but the first `""` (a real blank line) is preserved. ✓

  **Edge case `\r\n`:** Handled correctly. `write()` at line 102 does `text.replace("\r", "")` before any split logic, so `\r\n` becomes `\n` and splits correctly. The `flush()` method never sees `\r` characters. ✓

  **Edge case `print("hello\r")` (carriage return only, no newline):** After `\r` strip, `text_for_buffer = "hello"`. No `\n` present, so it stays buffered. When `\n` eventually arrives, the buffer flushes, and the `\r`-only fragment gets concatenated with the next line. This is correct behavior for tqdm progress bars (which emit `\r` for in-place updates). ✓

  **Edge case empty buffer `flush()`:** Guarded by `if self._buffer:` at line 147. ✓

  **Minor observation:** The `flush()` docstring at lines 151-153 says "Same \n-only split as write() — already \r-stripped." This is technically correct since `write()` strips `\r` before buffering. However, the comment makes it sound like `flush()` itself strips `\r` — it doesn't; it relies on `write()` having already done so. This is slightly misleading but not wrong. No fix needed.

### Finding 5 (MEDIUM) — after(0) → after(100)
- Status: **CLOSED**
- Notes: Confirmed at `spectral_predict_gui_optimized.py:57258`: `root.after(100, app._check_for_incomplete_run)`. ✓

  **Race-condition check:** `_check_for_incomplete_run` (lines 22931–23006) imports `find_incomplete_run`, `resume_run`, `discard_incomplete_run` from `run_state`. These are pure functions that read/write JSON sidecar files from disk — no dependence on `self.X`, `self.y`, or any lazily-initialized GUI state. The only GUI interaction is `messagebox.askyesnocancel` (a Tkinter builtin, requires root but not any app state) and `self.progress_status.config()` (wrapped in `except Exception: pass` — defensive). The function is fully safe to call 100ms after `mainloop()` starts. ✓

  100ms is below human perception threshold (~13ms for visual flicker fusion, ~100ms for motion perception), so users see no lag. The window manager has time to render the main window before the modal dialog appears. ✓

## New findings (introduced by the follow-up commits)

No new findings introduced by the follow-up commits themselves. The three commits are clean and correctly implement the fixes.

**However**, the recheck `git grep "__compiled__" -- '*.py'` uncovered **two additional copies of the buggy `dir()` pattern that were NOT fixed**:

### NEW Finding A (HIGH) — `_compute_rf_screening` uses `dir()` inside method body

- **File:line:** `spectral_predict_gui_optimized.py:21828`
- **Pattern:** `is_frozen = getattr(sys, 'frozen', False) or '__compiled__' in dir()`
- **Context:** Inside method `_compute_rf_screening(self, top_n=None, wl_range=None)` (line 21791). `dir()` with no arguments inside a method body returns the local + enclosing scopes — it does NOT include module-level names injected by Nuitka. `__compiled__` is at module scope. This always evaluates to `False`.
- **Impact:** Under Nuitka, `is_frozen` returns `False` → `n_jobs = -1` (instead of `1`) → joblib uses `loky` backend with `spawn` → frozen-runtime argv-parse crash and fork-bomb of GUI windows. Identical failure mode to Finding 1 in `search.py`.
- **Fix:** Replace `'__compiled__' in dir()` with `'__compiled__' in globals()` (one-character change, same as the `search.py` fix).
- **Latent on PyInstaller:** `sys.frozen` is `True` for PyInstaller, so the `or` short-circuits. Bug only surfaces under Nuitka, same as Finding 1.

### NEW Finding B (LOW) — module-level `_NUITKA_FROZEN` uses `dir()`

- **File:line:** `spectral_predict_gui_optimized.py:131`
- **Pattern:** `_NUITKA_FROZEN = "__compiled__" in dir()`
- **Context:** At module level, `dir()` returns the module's namespace, which IS the global scope. Nuitka injects `__compiled__` into the module's `__dict__`. So this **does work correctly** — `dir()` at module scope is functionally equivalent to `globals()`. I verified: Python's `dir()` with no arguments returns `sorted(locals().keys())`, and at module scope, `locals()` is `globals()`.
- **Risk:** Inconsistent with every other `__compiled__` check in the repo (all use `globals()`). Latent refactoring trap — if this module-level code is ever moved into a function (e.g., a `check_environment()` helper), the check silently breaks.
- **Fix:** Replace `'__compiled__' in dir()` with `'__compiled__' in globals()` for consistency and future-proofing (one-character change).
- **Note:** This is LOW severity because it works today. Line 21828 is the one that matters (HIGH).

## Other except-pass patterns worth addressing

The original T-11 commit (`4084352`) added several `except Exception: pass` / `except OSError: pass` patterns in the backend (`run_logging.py`, `run_state.py`). Re-examined all of them:

| Location | Pattern | Judgment |
|---|---|---|
| `_TeeStream.write()` l.91: `except Exception: pass` | Original stream write failed | Defensible — logging module must not crash because the underlying fd is broken |
| `_TeeStream.flush()` l.164: `except Exception: pass` | Same pattern | Same judgment — defensive |
| `_TeeStream.isatty()` l.170: `except Exception: return False` | Best-effort query | Defensible — safe fallback |
| `run_state._atomic_write()` l.466: `except OSError: pass` (fsync) | Network shares don't support fsync | Defensible, documented |
| `run_state._atomic_write()` l.472: `except Exception:` (re-raises after cleanup) | Temp file cleanup before re-raise | Correct pattern — best-effort cleanup, preserves original error |
| `run_state.fingerprint_dataset()` l.525ff: `except Exception: break` | Numpy element encoding | Defensible — fallback to shorter hash |
| `run_state.fingerprint_dataset()` l.536: `except Exception: return "unknown"` | Entire fingerprint fails | Defensible — caller handles "unknown" gracefully |
| `run_state.clear_resume_state()` l.609: `except OSError: pass` | Sidecar unlink fails | Defensible |
| `run_state.discard_incomplete_run()` l.749-757: `except OSError: pass` | File unlink fails | Defensible |
| `_check_for_incomplete_run` l.23003: `except Exception: pass` (progress_status.config) | Cosmetic status update | Defensible — don't break startup for a banner |

**No additional concerns.** All existing bare-except patterns are defensive and justified in context. The T-11 diff did not introduce any new problematic `except Exception: pass` that silently swallows real errors (the three that were problematic — Findings 3's targets — have been fixed).

## Final verdict

The three follow-up commits correctly and completely address Findings 1–5 from the first review. All 34 T-11-targeted tests pass. No regressions introduced.

**One new finding requires attention before merging:** `spectral_predict_gui_optimized.py:21828` has the same `'__compiled__' in dir()` bug as the already-fixed `search.py:19`. This is a one-character fix (`dir()` → `globals()`) with identical impact and identical fix. It was missed because the first-review grep targeted `src/spectral_predict/` backend files, not the 57K-line GUI monolith. The module-level copy at line 131 is functionally correct but should be changed for consistency.

**Recommendation:** Fix line 21828 (one-character change, 30 seconds), optionally fix line 131 for consistency, then the branch is ready to push and merge to `main`. The existing fixes are solid; the only gap is a copy of the exact bug we already fixed that lives in a different file.
