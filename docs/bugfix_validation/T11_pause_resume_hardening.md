# T-11 Pause/resume hardening + Optuna SQLite + disk logging — MERGED

**Branch:** `fix/T11-pause-resume-hardening` (deleted post-merge)
**Status:** MERGED 2026-05-01 via PR #6 (rebase merge, linear history)
**Merge tip:** `50057af` (post-merge `main` HEAD prior to PROJECT_STATUS update)
**PR:** https://github.com/makapania/dasp/pull/6
**Date:** 2026-04-30 (initial APPROVED) → 2026-05-01 (MERGED after pass-3 verdict)
**Investigation:** [T11_findings.md](T11_findings.md)
**Full review trail:** [docs/reviews/deepseek_v4pro_24h_review_2026-04-30.md](../reviews/deepseek_v4pro_24h_review_2026-04-30.md)

## 1. Scope shipped

Three sub-units from the T-11 findings document, decomposed and shipped together:

- **Unit A — Disk-mirrored logging.** New `src/spectral_predict/run_logging.py`
  with `RotatingFileHandler` writing to `<user_data_dir>/dasp/logs/`. Tee
  proxy over `sys.stdout` / `sys.stderr` captures backend `print()` calls
  that would otherwise hit `/dev/null` in the bundled GUI.
- **Unit B — Thread-alive check + UI pause-state honesty.** New
  `_actually_paused` event in `SearchController` set inside `check_and_wait`
  while the worker is genuinely blocked. New 'pausing' UI state with the
  Pause button greyed and a "Pause requested — waiting for current trial
  to finish" status. `_resume_search` calls `analysis_thread.is_alive()`
  before signaling resume.
- **Unit D — Optuna SQLite persistence + resume dialog.** New
  `src/spectral_predict/run_state.py` managing per-run UUID + SQLite
  storage URL + sidecar JSON. `unified_bayesian.create_study` plumbs the
  storage URL when active. App-startup dialog offers Resume / Discard /
  Decide-later when an unfinished sidecar is found.

**Closed simultaneously:** T-12 (disk logging) — same work as Unit A.

**Deferred from original scope:** Unit C (trial-internal pause for boosting
models). Helps only Bayesian users with CatBoost/LightGBM/XGBoost; pause
latency for those models stays at "current trial duration" until a
boosting-callback hook is added. Filed as informal follow-up.

## 2. Cross-family review and fixes

Codex (US-trained) and Kimi K2.6 (Moonshot, Chinese-trained, via opencode-go
flat-rate subscription — explicitly NOT z.ai per recurring memory rule)
reviewed the diff in parallel.

### 2a. Codex HIGH (4) — all fixed before merge

1. **Trial-count overrun on resume** (`unified_bayesian.py:1904`). `study.optimize(n_trials=n_trials)` runs n_trials *more* trials, not until total reaches n_trials. Crashed 80/100 study would have resumed with 100 fresh trials → 180 total. Fixed with `remaining = max(0, target − already_finished)` using Optuna's terminal-state filter (`TrialState.COMPLETE | PRUNED`).

2. **Non-atomic sidecar writes** (`run_state.py:_atomic_write_json`). Plain `write_text()` could leave partial JSON if process dies mid-write. Fixed with `tempfile.mkstemp` + `fsync` + `os.replace` (atomic on POSIX + Windows since Python 3.3).

3. **Dataset fingerprint stored but never enforced.** Fingerprint was being saved at `start_run` but `verify_resume_fingerprint` had no callers. User could resume on different data → Optuna would silently pick up trials with stale objective values. Fixed: GUI calls `verify_resume_fingerprint` before `start_run` when `is_resuming()`. On mismatch, `discard_incomplete_run` (NOT `clear_resume_state` per Kimi MAJOR #3b — see below).

4. **Study names too coarse.** `unified_bayesian_{model_name}` reused across runs with different task types, sampler seeds, CV strategies, etc. `load_if_exists=True` would have silently mixed incompatible trials. Fixed: `study_name = f"unified_bayesian_{model_name}_{config_hash}"` where `config_hash` includes `version | task_type | cv_config | random_state | imbalance | baseline | smoothing | uve | inlier`. n_trials deliberately omitted per Kimi MINOR #6 (the trial-count clamp handles the user-changes-target case correctly without orphaning studies).

### 2b. Codex MEDIUM (3) — all fixed before merge

5. **`_TeeStream` not thread-safe** (`run_logging.py`). `setup_run_logger` replaces process-global `sys.stdout`/`sys.stderr`, so worker threads + library threads can write concurrently. Fixed with `threading.Lock` around the buffer + emit-outside-lock pattern + 64KB byte threshold to prevent newline-free OOM.

6. **No log rotation** — multi-GB log files possible on 12-hour runs. Switched `FileHandler` to `RotatingFileHandler` (50MB × 3 backups = 200MB ceiling).

7. **Run state started for grid + NSGA-II too.** A crashed grid run would leave a misleading sidecar that triggered "Resume?" dialog even though no Optuna study existed. Fixed: `start_run` only fires when `optimization_method == "unified"` (Bayesian).

### 2c. Kimi MAJOR (2) — both fixed

8. **`verify_resume_fingerprint` didn't check run_id.** A second app instance could overwrite the sidecar between `resume_run()` and now; the function would compare against the wrong run's fingerprint. Coincidental match would resume the wrong run. Fixed: assert `data["run_id"] == _active_run_id` before reading the fingerprint.

9. **GUI used `clear_resume_state` on fingerprint mismatch.** That only clears in-memory state — sidecar persists, so the user gets re-prompted to resume the same rejected run on every launch. Fixed: GUI captures `_active_run_id` first, then calls `discard_incomplete_run(rejected_id)` which removes both the sidecar and the SQLite file.

### 2d. Kimi MINOR (4) — all fixed

10. **Tee splits on `\r` (tqdm spam).** `splitlines()` separates on both `\r` and `\n`; tqdm's `\r<bar>\r<bar>...\n` would emit one log line per bar update. Fixed: strip `\r` before buffering, then split on `\n` only.

11. **Nuitka `is_frozen()` always returns False.** `"__compiled__" in dir()` checks the function's local namespace; Nuitka injects `__compiled__` at module scope. Fixed at all 3 sites in `resource_paths.py`: replaced with `"__compiled__" in globals()`. (Pre-existing bug surfaced by the T-11 review — included since it's a one-line change.)

12. **Orphaned Optuna studies accumulate.** With `n_trials` in `study_name`, changing the target spawned a new study and orphaned the old one. Fixed: removed `n_trials` from the hash; rely on the trial-count clamp instead.

13. **SQLite locking on Windows.** Default short busy-timeout could cause "database is locked" errors under concurrent access. Fixed: storage URL appended with `?check_same_thread=False&timeout=30`.

### 2e. Documented as known limitations (deferred)

14. **Cross-process sidecar locking.** Two app instances can race on `active_run.json` writes. The atomic-write pattern prevents partial JSON corruption (worst case is "whoever finishes second wins"), but a hard cross-process lock would require adding `filelock` as a dependency. dasp's GUI is single-instance in practice (most users won't run two copies); deferred until reported as an actual problem.

15. **Search-space hyperparameter bounds NOT in `config_hash`.** Kimi MAJOR #1 flagged that user-customized hyperparameter ranges (Tab 4C overrides for grid search) aren't captured by the current hash. The customizations don't currently flow into Bayesian search, so the bug is theoretical for now — but if Bayesian honoring custom ranges is added later, the config_hash will need a corresponding extension. The included `__version__` in the hash is a defensive proxy: any dasp upgrade forces a new study even if surface params look identical.

16. **GUI thread-integration test.** Tk threading tests are flaky and hard to maintain; the unit tests cover all logic-level invariants. Deferred per Codex MEDIUM #8.

17. **Legacy `search.py:3153 run_bayesian_search`** uses `create_optuna_study` with hardcoded `storage=None`. Not GUI-reachable; documented as legacy/non-resumable.

18. **Multi-worker barrier semantics for `_actually_paused`.** Current single-worker design is correct (Codex LOW #1, Kimi confirmed). If T-11's controller is ever reused by genuine multi-worker code, the single-Event design needs to become a paused-worker count plus expected-worker ownership.

## 3. Test coverage

- **`tests/test_run_state.py`** — 20 unit tests covering: idempotency, sidecar atomic-write recovery, run_id mismatch detection, fingerprint enforcement, corrupt-sidecar self-heal, dataset fingerprint determinism, Optuna integration smoke.
- **`tests/test_search_controller.py`** — 7 tests covering: pause acknowledgment via `_actually_paused`, resume clearing, end unblocking, reset.
- **`tests/test_run_logging.py`** — 7 tests covering: file creation, idempotency, label-in-filename, stdout tee, user-data-dir routing.

Full sweep: **260 passed in 5:01** on `.venv312`.

## 4. Verdict

**APPROVED for merge.**

Reasoning:
- T-11 is a productivity infrastructure ticket, not a chemometrics methodology question. Master rule applies in the lighter form: bundled-app distribution still matters (pause/resume buttons are GUI-visible; resume dialog is GUI-visible; logs are GUI-discoverable via the path printed at run start).
- Field-alignment frame: commercial chemometrics tools don't run multi-hour searches and have no analog. AutoML libraries (auto-sklearn, H2O AutoML, TPOT, FLAML) all ship checkpoint-with-resume; dasp catches up to that adjacent space.
- Both reviewers (Codex US-trained, Kimi K2.6 Moonshot Chinese-trained) cross-checked and surfaced 4 HIGH + 5 MEDIUM/MAJOR + 6 MINOR issues. All HIGH and MEDIUM/MAJOR fixed before merge. MINOR items either fixed or documented as known limitations.
- 260/260 regression tests pass.

## 5. Performance budget

Disk-mirrored logging: negligible overhead (~microseconds per emit; rotation triggered at 50MB).

Optuna SQLite persistence: per-trial write overhead is sub-second for SQLite + small object sizes. For a 30-minute CatBoost trial, persistence is microscopic. For a sub-second PLS sweep trial, persistence may add a few percent — acceptable.

Pause-acknowledgment polling: 500ms `after()` interval until the worker reaches `check_and_wait()`. For a Bayesian CatBoost trial this can be 20+ minutes; the GUI logs an extra "[PAUSE PENDING]" message at the 30-second mark so the user knows the trial is just slow rather than stuck.

## 6. Future re-evaluation criteria

Re-open T-11 (or open T-11b) only if:
- Users report cross-process sidecar corruption from running multiple dasp instances concurrently. → Add `filelock`.
- Bayesian search starts honoring user-customized hyperparameter ranges from Tab 4C. → Extend `config_hash` to include the customizations.
- A user running CatBoost-heavy Bayesian explicitly requests sub-second pause latency. → Wire trial-internal pause callbacks (Unit C).
- The legacy `search.py:3153 run_bayesian_search` becomes GUI-reachable. → Plumb storage URL through `create_optuna_study`.

## 7. Known false-alarm patterns NOT triggered

T-11 is the first ticket since T-26 to actually NEED most of its claimed fixes — no false alarm here. The original PROJECT_STATUS Follow-Ups note from 2026-04-23 was directionally correct on all 4 items + the bonus disk-logging gap, with one minor framing correction (gap #3's "errored worker → resume on dead thread" path is mostly closed by the existing `except Exception` handler; the realistic case is "worker hung but alive," same one-liner fix either way).

## 8. Post-APPROVED review trail (added 2026-05-01)

After Codex + Kimi cross-family review (sections 2a–2e above), four
additional review passes ran before merge. Each surfaced new findings the
prior reviewers had missed because of charter scope, search-window scope,
or the inherent limit of "diff-only" review.

### Pass 1 — DeepSeek V4 Pro 24h sweep (commits ahead of `main` over the prior day)

Found 0 BLOCKER + 2 HIGH + 3 MEDIUM + 4 LOW + 5 INFO:

- HIGH-1: `search.py:19` `'__compiled__' in dir()` was a missed copy of the
  Nuitka detection pattern (3 sites in `resource_paths.py` were fixed by
  T-11 itself; this 4th copy lived outside the diff and was missed).
  Closed in `446465c`.
- HIGH-2: `setup_run_logger` idempotency check + setup block had no
  `_setup_lock`, contradicting the docstring's "background workers and
  the GUI see the same log" multi-caller promise. Closed in `9107a68`.
- MEDIUM #3: `except Exception` in boot-path masked `ImportError` from
  `_start_run_state`; bare `except: pass` around `mark_complete()` hid
  failures. Closed in `53b3078`.
- MEDIUM #4: `_TeeStream.write/flush` stripped whitespace-only lines via
  `if line.strip()`. Lost log fidelity vs the actual stdout stream.
  Closed in `53b3078`.
- MEDIUM #5: `root.after(0, _check_for_incomplete_run)` could fire before
  the main window finished rendering on macOS Tk-Aqua / slow Windows
  machines, causing the modal to appear behind. Closed in `53b3078`.

### Pass 2 — DeepSeek V4 Pro recheck

Confirmed pass-1 findings closed. Surfaced 2 NEW HIGH (the same
`__compiled__ in dir()` pattern in two more sites — `gui:131` module-level
and `gui:21828` inside `_compute_rf_screening`). The prior pass had
scoped grep to `src/spectral_predict/`; the recheck broadened to the GUI
monolith. Both closed in `e62c15d`.

### Pass 3 — 5 specialist agents in parallel

After pass 2, ran code-reviewer / pr-test-analyzer / silent-failure-hunter
/ comment-analyzer / type-design-analyzer in parallel against the diff
through `e62c15d`. 13 findings:

- Cluster A (3 interlocking bugs in `run_state.py`): `mark_complete()`
  swallowed OSError on unlink and cleared in-memory state on failure;
  `find_incomplete_run()` missed `OSError`/`UnicodeDecodeError` and
  silently deleted corrupt sidecars; `discard_incomplete_run()` lied
  about success even when both unlinks failed.
- Cluster B (3 bugs in `run_logging.py`): `_TeeStream._logger.log()`
  unprotected → worker thread crash on handler failure;
  `RotatingFileHandler.handleError` feedback loop via the tee;
  `_BUFFER_BYTE_THRESHOLD` flush dead for stated purpose (no-newline
  buffer at threshold re-buffered verbatim).
- Cluster C: `start_run()` idempotent-return path returned a Frankenstein
  `RunMetadata` mixing the original on-disk `run_id`/`storage_url` with
  the second caller's `label`/`fingerprint`/`model_names`/`started_iso`.
  Found independently by type-design-analyzer + code-reviewer.
- Cluster D: 5 test-coverage gaps for headline T-11 contracts (concurrent
  `setup_run_logger`, concurrent `_TeeStream.write`, rotation, blank-line
  preservation, locked-file discard).
- Plus 8 "Important" + 4 "Suggestion" items spanning study-hash
  completeness, GUI integration of new contracts, defensive path
  validation, comment/docstring lag, and type-design suggestions.

All Critical + High closed across commits `37a71ea` / `5db0b40` /
`4dc7ec3` / `6fd8dab`. Test coverage closed via `29fe489` (13 new tests
covering each contract).

### Pass 4 — Codex meta-review (synthesis check)

Asked Codex to evaluate the 5-agent synthesis. Verdict: NEEDS_ADJUSTMENT
because the synthesis was directionally right but missed one big
integration bug (NEW BUG #1: `mark_complete()` deleted unrelated
sidecars unconditionally — GUI calls it after every successful analysis,
so a paused Bayesian sidecar was destroyed by a fresh grid run) and
flagged one architectural-cost deferral (NEW BUG #2: per-model Bayesian
failure swallowing → `mark_complete()` still runs, can erase resume
state for partially-completed multi-model runs). NEW BUG #1 closed by
the `mark_complete()` rewrite; NEW BUG #2 deferred (filed as T-34, see
`docs/plans/2026-04-30-T34-per-model-bayesian-failure-handling.md`).

Codex also corrected several failure-mode framings: "discard prompt
reappears every launch" was partially wrong (sidecar-failure → loud,
SQLite-failure → silent orphan); concurrent-logger tests are
regression-hardening, not merge-blocking.

### Pass 5 — DeepSeek V4 Pro pass-3 (high-effort, opencode `--variant high`)

Final gate-pass before merge. Verdict: **READY_TO_MERGE**.

- All 12 fix claims verified against diff hunks.
- NEW BUG #2 correctly classified as DEFERRED (architectural cost).
- 47/47 T-11 tests passing.
- 1 cosmetic observation (duplicate `_original` write on logger fallback)
  — DeepSeek explicitly noted "not worth fixing — the fallback's job is
  crash-prevention, not output-fidelity."
- 1 bundle-mode edge case (no error signal on rollover failure when
  `_original_stderr is None` because `console=False` PyInstaller bundle
  redirects fd 2 to /dev/null) — explicitly accepted as "the best
  achievable behavior in a no-console Windows bundle."

**Reviewer count for the final state**: 7 independent passes (Codex
initial + Kimi K2.6 initial + DeepSeek pass 1 + DeepSeek pass 2 + 5
specialists in parallel + Codex meta-review + DeepSeek pass 3 high-effort).

## 9. Deferrals filed as new tickets

After T-11 merge:

- **T-34** — per-model Bayesian failure handling (NEW BUG #2 deferral
  from this PR). `docs/plans/2026-04-30-T34-per-model-bayesian-failure-handling.md`.
- **T-35** — type-design follow-ups: `RunMetadata` `frozen=True, slots=True`
  + `__post_init__` validation, `_TeeStream` extract `_LineBuffer` helper,
  `_TeeStream` add missing file-protocol attrs (`closed`, `encoding`,
  `writable()`, `readable()`). `docs/plans/2026-04-30-T35-t11-type-design-followups.md`.
- **T-36** — `fingerprint_dataset` numpy 2.x repr stability. `str(X.flat[idx])`
  format differs across numpy versions; spurious "data has changed"
  rejections on resume after numpy upgrade.
  `docs/plans/2026-04-30-T36-fingerprint-numpy-stability.md`.
- **T-37** — Stop-vs-Complete distinction + concurrent-instance footgun.
  Clicking Stop mid-run silently kills resume option (success-path
  `mark_complete` runs); two app instances can race on the resume dialog.
  `docs/plans/2026-04-30-T37-stop-vs-complete.md`.
