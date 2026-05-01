# T-40: Stop button vs natural completion + concurrent-instance footgun

> **Note 2026-05-01:** Originally drafted as T-40 during the T-11 PR
> review wrap-up; renumbered to T-40 to avoid collision with the
> in-flight T-40 (TPE quick preprocessing discovery) reservation
> drafted in parallel by another session. Content unchanged.

**Status:** PLANNED — deferred from T-11 PR #6 review.
**Filed:** 2026-04-30.
**Source:** code-reviewer specialist agent during T-11 PR #6 review (two related items combined).
**Related:** T-34 (per-model Bayesian failure handling) — overlapping lifecycle question.

## Problem 1: Stop button silently kills resume option

When the user clicks Stop mid-Bayesian-run:

1. `study.stop()` is called from the controller.
2. The optimize() loop exits cleanly (no exception).
3. The worker thread reaches the success-path `mark_complete()` call at
   `spectral_predict_gui_optimized.py:~27958`.
4. `mark_complete()` (now correctly checks `_active_run_id` matches —
   T-11 NEW BUG #1 fix) deletes the sidecar.
5. **The user has no resume option for the partial run they just
   stopped**, even though the SQLite store still has the trials.

Stopping is functionally indistinguishable from completing — the
existing lifecycle has only one terminal state ("complete" =
`mark_complete` deletes the sidecar). "I want to come back to this
later" has no representation.

**Fix shape:** Distinguish "stopped" from "completed" in the lifecycle.

Two viable approaches:

**Approach A — Don't call `mark_complete` on stop.**
The simplest fix: in the worker thread, check whether the stop was
user-initiated (controller flag) before calling `mark_complete()`.
If user-initiated, leave the sidecar in place so the next launch
offers the resume dialog.

```python
# In _run_analysis_thread, success path:
if not self.search_controller.user_stopped:
    try:
        from spectral_predict.run_state import mark_complete as _mark_complete
        _mark_complete()
    except Exception as _mc_err:
        # ... existing handler
```

**Approach B — Distinguish via dialog.**
On stop, show a dialog: "Stop search? Choose: 'Stop and save partial
run for later resume' vs 'Stop and discard'". The latter calls
`discard_incomplete_run`; the former leaves the sidecar in place.

**Recommendation:** Approach A for the V1 fix (one-line change, no UX
change), with Approach B as a follow-up if user feedback requests
explicit discard control on stop. Stop-and-resume-later is the more
common user intent; stop-and-discard is rare and already reachable
via the resume dialog at next launch.

**Preserving the surface change:** the Stop button's tooltip should
update to mention resume-availability:

> Stop the search. Partial trials are saved automatically; you'll be
> prompted to resume on next launch.

## Problem 2: Concurrent-instance footgun

If the user opens a second dasp window while the first is mid-run:

1. The second instance's startup `_check_for_incomplete_run` reads the
   *currently-running* sidecar.
2. Dialog appears: "Found unfinished run from <timestamp>. Resume?"
3. If the user clicks "No, discard," the second instance's
   `discard_incomplete_run` unlinks the SQLite that the first instance
   is actively writing to.
   - On POSIX: silent file deletion; first instance keeps writing to a
     deleted inode; data lost on process exit.
   - On Windows: `OSError` (file lock) → propagates as a `DiscardResult`
     error (post-T-11 fix), so the user sees a warning. But a flicker
     of contention can still corrupt the SQLite store.

**Fix shape:** Encode the writer PID in the sidecar at `start_run` time;
on `_check_for_incomplete_run`, skip the dialog (and the "Decide later"
state) if the encoded PID is still alive.

```python
# In start_run():
import os
meta = RunMetadata(
    ...,
    writer_pid=os.getpid(),
    writer_pid_started_iso=datetime.now().isoformat(),  # for stale-PID tolerance
)

# In _check_for_incomplete_run():
if meta.writer_pid is not None and _pid_is_alive(meta.writer_pid):
    # Another dasp instance owns this sidecar — skip the dialog.
    return
```

`_pid_is_alive` is platform-specific:
- POSIX: `os.kill(pid, 0)` raises `ProcessLookupError` if dead.
- Windows: `psutil.pid_exists(pid)` (psutil is already an indirect dep).
  Or `OpenProcess` / `GetExitCodeProcess` if we want to avoid psutil
  for this one check.

Stale-PID tolerance: store `writer_pid_started_iso` so that if the
recorded PID is alive but its boot time is more recent than the
sidecar's own `started_iso`, we treat it as stale (a different process
got assigned that PID after the original instance died).

## Files touched

- `src/spectral_predict/run_state.py` — add `writer_pid` and
  `writer_pid_started_iso` fields to `RunMetadata`; add `_pid_is_alive`
  helper.
- `spectral_predict_gui_optimized.py` — `_check_for_incomplete_run`
  consults `_pid_is_alive` before showing the resume dialog;
  `_run_analysis_thread` success path checks `user_stopped` flag
  before calling `mark_complete`.
- `src/spectral_predict/search_controller.py` — track `user_stopped`
  flag (or expose existing one if present).
- `tests/test_run_state.py` — regression: stopping leaves the sidecar
  intact; PID-aliveness check skips dialog when current process owns
  the sidecar.

## Open questions

1. **Should the second-instance dialog be silenced entirely, or just
   reframed?** Silencing means the user starting a second window
   doesn't see the resume prompt — fine, since the first window owns
   the run. But if the user *wanted* to attach to the running run
   from a second window, silencing prevents that. Counter: dasp's
   single-process Optuna SQLite + GUI design doesn't support
   "two-windows-one-run" anyway. Silencing is correct.
2. **Cross-machine** (NFS / Dropbox / OneDrive sidecar storage): PID
   encoding is meaningless if the second "instance" is on a different
   machine. The path-validation defense in T-11 (resume_run +
   discard_incomplete_run validate paths under `get_user_optuna_dir()`)
   already handles the most-common case (Dropbox sync conflict
   creating a duplicated sidecar). Cross-machine resume isn't
   supported and isn't planned.
3. **Stale PID tolerance threshold**: how long after `started_iso`
   should we trust `os.kill(pid, 0)` not having flipped due to PID
   reuse? Practical answer: if `started_iso` is older than 24 hours
   AND the PID's boot time is newer than `started_iso`, treat as
   stale. Don't over-engineer.

## Success criteria

- Stop button leaves the sidecar in place; next launch offers resume.
- Tooltip on Stop button mentions auto-save behavior.
- Opening a second dasp window during a run does NOT show the resume
  dialog for the active run.
- If the first instance crashed (PID gone), the second instance's
  resume dialog still works as today.
- Existing `mark_complete` natural-completion behavior unchanged.

## Estimated effort

~1 day. Most of the work is the PID-aliveness helper + tests; the
Stop-vs-Complete distinction is a one-line check.

## Coordination with T-34

T-34 (per-model Bayesian failure) and T-40 (Stop-vs-Complete) both
touch the sidecar lifecycle. Suggested order: land T-40 first (Stop
button distinction is a small, well-scoped change), then T-34's
larger per-model state machinery can build on the now-three-state
lifecycle (running / stopped / complete). Implementing T-34 first
would require a third pass when T-40 lands.
