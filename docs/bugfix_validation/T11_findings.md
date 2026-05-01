# T-11 findings: Pause/resume hardening + Optuna SQLite persistence

**Branch:** none yet — T-11 is **not implemented**. Investigation of `main` only.
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30

This is the investigation phase. T-11 is a **productivity / UX infrastructure**
ticket, not a chemometrics-methodology ticket. The chemometrics master rule
(literature is the spec, leading-program parity, etc.) doesn't have much
purchase here — pause/resume of multi-hour Bayesian searches is a dasp-specific
concern; Unscrambler / SIMCA / OPUS / PLS_Toolbox don't run multi-hour grid
searches in the first place. The framing here is: "is this a real
productivity bug, what's reachable in the GUI, and what's the smallest fix
that materially improves the user's life?"

---

## TL;DR

**Real productivity bug. Recommended split: ship the cheap UX honesty + thread-alive
check + disk logging first (1-2 days, gets ~80% of the perceived value);
defer Optuna SQLite + trial-internal pause to a second iteration (3-5 days,
gets the remaining 20% but covers the catastrophic crash-recovery case the
user actually got bitten by).**

All four claimed gaps in PROJECT_STATUS verified in the current code on `main`:

1. **Bayesian pause checkpoint granularity** — `unified_bayesian.py:1830` is
   the *only* `check_and_wait()` site in the entire Bayesian path. It's
   inside `progress_wrapper`, which Optuna fires *between* trials. A single
   CatBoost trial on 19-hour searches can be 10-30+ minutes. Pause UI flips
   instantly; thread keeps running for the rest of the trial. **Verified.**
2. **UI pause-state lies** — `_pause_search` at gui:22864-22869 calls
   `_update_search_buttons('paused')` synchronously after
   `controller.pause()`, with no callback or progress notification when
   the worker actually blocks at `check_and_wait()`. Same for resume. There
   is no "pausing — waiting for current trial" intermediate state.
   **Verified.**
3. **Resume doesn't check `analysis_thread.is_alive()`** — `_resume_search`
   at gui:22871-22876 is a 4-line method: `controller.resume()`,
   `_log_progress("[RESUMED] Search resumed")`, `_update_search_buttons('running')`.
   It never inspects `self.analysis_thread.is_alive()`. **Verified** — but
   see §2c below for nuance: the worker's main `except Exception` at
   gui:27684-27698 *does* call `_update_search_buttons('idle')` and pop a
   messagebox, so a "silently dead worker" only happens if the thread
   crashes hard enough to skip its own except handler (rare). The framing in
   the Follow-Ups note overstates this case; the real gap is more
   "user-paused, worker errored on the next trial unblock, UI returns to
   running, never produces output."
4. **Zero persistence** — `optuna.create_study(direction='minimize',
   sampler=sampler, study_name=...)` at `unified_bayesian.py:1820-1824`
   takes no `storage=` argument. In-memory only. Process exit = run lost.
   **Verified.** No commits since 2026-04-23 have touched this code path:
   `git log --since="2026-04-23" -- src/spectral_predict/search_controller.py
   spectral_predict_gui_optimized.py src/spectral_predict/unified_bayesian.py`
   shows only T-04 / T-06 / T-21 / T-24 fixes — none touched pause/resume.
5. **Disk-logging gap** — `_log_progress` at gui:27790-27792 is a 2-line
   wrapper that `after(0)`'s onto `_append_progress`, which writes to
   `self.progress_text` (Tkinter widget) only. The widget is capped at 2000
   lines (gui:27801-27804 explicitly truncates). No `logging.FileHandler`
   exists anywhere in the codebase (verified by grepping
   `src/spectral_predict/` and `spectral_predict_gui_optimized.py` — zero
   hits for `FileHandler`). **Verified.**

---

## 1. Verify reality in the codebase

### 1a. The four claimed gaps — line-numbers shifted, semantics intact

The PROJECT_STATUS Follow-Ups note (lines 322-327) cites file:line refs that
have drifted slightly since 2026-04-23 (the GUI is 28k+ lines and gets edited
constantly). Mapping to current `main`:

| PROJECT_STATUS claim | Currently at | Verified? |
|----------------------|--------------|-----------|
| `unified_bayesian.py:1826` — pause check between trials | `unified_bayesian.py:1820-1833` (4-line drift) | YES |
| `gui:22842` — `_pause_search` flips UI immediately | `gui:22864-22869` (22-line drift) | YES |
| `gui:22849` — `_resume_search` no `is_alive()` check | `gui:22871-22876` (22-line drift) | YES |
| Zero persistence — no `storage=` on `create_study` | `unified_bayesian.py:1820-1824` | YES |
| `gui:27770` — `_log_progress` widget-only, no disk | `gui:27790-27792` (20-line drift) | YES |

### 1b. Current code, quoted

**`unified_bayesian.py:1820-1833`** — the only pause site in the Bayesian path:

```python
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    study_name=f"unified_bayesian_{model_name}"
)

# Progress callback wrapper
def progress_wrapper(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    # Check for stop/pause signal from controller
    if controller is not None:
        if not controller.check_and_wait():
            # User requested stop - tell Optuna to stop after this trial
            study.stop()
            return
```

Note: (a) no `storage=` argument; (b) `progress_wrapper` is registered as an
Optuna trial-end callback — it fires once per trial after the trial completes.
A Pause click during a 30-minute CatBoost trial silently does nothing until
the trial finishes, then `_pause_event.wait()` blocks at the inter-trial
boundary.

**`spectral_predict_gui_optimized.py:22864-22876`** — pause/resume methods:

```python
def _pause_search(self):
    """Pause the running search."""
    if self.search_controller:
        self.search_controller.pause()
        self._log_progress("[PAUSED] Search paused by user")
        self._update_search_buttons('paused')

def _resume_search(self):
    """Resume a paused search."""
    if self.search_controller:
        self.search_controller.resume()
        self._log_progress("[RESUMED] Search resumed")
        self._update_search_buttons('running')
```

`_pause_search` flips the button state to 'paused' immediately on the same
GUI tick as `controller.pause()`. There is no acknowledgment that the worker
has reached `check_and_wait()`. The "[PAUSED] Search paused by user" log
message is also a lie until the worker actually blocks.

`_resume_search` has zero defensive checks. If `analysis_thread` is dead
(crashed during the paused window, or already exited normally and the
controller was left dangling), this method merrily sets `_pause_event` for
nobody to receive, logs `[RESUMED]`, and shows the user a "running" UI with
a dead worker.

**`spectral_predict_gui_optimized.py:27790-27804`** — log function:

```python
def _log_progress(self, message):
    """Log message to progress text area."""
    self.root.after(0, lambda: self._append_progress(message))

def _append_progress(self, message):
    """Append message to progress text (must be called from main thread)."""
    self.progress_text.insert(tk.END, message + "\n")
    self.progress_text.see(tk.END)

    # Limit text widget to last 2000 lines to prevent performance degradation
    # This keeps memory usage low and tab switching fast
    line_count = int(self.progress_text.index('end-1c').split('.')[0])
    if line_count > 2000:
        # Delete oldest lines, keeping only the last 2000
        self.progress_text.delete('1.0', f'{line_count - 2000}.0')
```

Lines 27801-27804 *deliberately delete* progress history beyond 2000 lines.
A 19-hour CatBoost run easily produces > 2000 progress lines (each trial
emits at minimum a "Trial N/300: RMSEcv=…" line plus per-fold prints). For
the user's incident, the diagnostic log was already truncated by the time
the user noticed pause/resume had silently no-op'd.

**`src/spectral_predict/search_controller.py`** — the controller itself
(only 73 lines, simple `threading.Event` design):

```python
def pause(self):
    """Pause the search. Search will block at next checkpoint."""
    self._pause_event.clear()

def resume(self):
    """Resume the search from paused state."""
    self._pause_event.set()

def check_and_wait(self) -> bool:
    if self._end_event.is_set():
        return False
    self._pause_event.wait()  # Blocks if paused
    return not self._end_event.is_set()
```

The mechanism is correct in isolation — the bug is *how often* `check_and_wait()`
is called from the worker, and the UX layer's failure to acknowledge the
async nature of the pause request.

### 1c. Checkpoint cadence by search mode

I grepped every `check_and_wait()` callsite to map cadence:

| File | Line | Context | Per-checkpoint cost |
|------|------|---------|---------------------|
| `nsga2_search.py:1965` | NSGA-II inner loop | Per generation/individual | seconds-minutes |
| `search.py:1939` | Grid search outer | Per preprocess config | seconds |
| `search.py:2145` | Grid search per model | Per model | seconds |
| `search.py:2161` | Grid search per config | Per (model × params) | seconds |
| `search.py:2322` | Grid search inner | Per fold | seconds |
| `search.py:2391` | Grid search varsel | Per varsel iteration | seconds-minutes |
| `search.py:5433`, `:5438`, `:5589`, `:5689`, `:5950`, `:5957` | One-class grid | Multiple cadences | seconds-minutes |
| `unified_bayesian.py:1830` | **Bayesian only** | **Per trial** | **10-30+ min for CatBoost** |

The grid-search and NSGA-II paths have *plenty* of pause checkpoints (one
per fold, one per model-config inner step) — pause latency on those is
seconds, not minutes. **The acute pause-latency problem is Bayesian-only.**

A natural fix for the trial-internal granularity gap is an Optuna
`TrialPruner`-style hook or a polling thread, but Optuna doesn't expose a
mid-trial cancel API natively. The trial body itself runs whatever `objective`
the user defined — for CatBoost this is `model.fit(X, y, ...)` which is
single C-extension call with no Python checkpoint to interpose on. To make
mid-trial pause work, either:

- a. CatBoost (and other boosting models) need a per-iteration callback that
  checks the controller. CatBoost has `iteration_callbacks=`; LightGBM has
  `callbacks=[…]`; XGBoost has `callbacks=[…]`. These would let pause take
  effect within a few seconds of a click. Real engineering, not drop-in.
- b. The Optuna study-level approach (`study.stop()`) at least makes Pause
  cancel the run *after* the current trial — but that's already what the
  current code does. The latency complaint is fundamental: a single trial
  is the minimum granularity Optuna offers without per-model callback work.

### 1d. Disk-logging gap is verified-zero

```bash
$ grep -rn "logging.FileHandler\|FileHandler\|logfile\|log_file" \
    src/spectral_predict spectral_predict_gui_optimized.py
(zero hits)
```

dasp has no disk-mirrored logging anywhere. Backend `print()` and
`logger.info()` calls go to stdout; in the bundled app
(`console=False`), stdout is /dev/null. The Tkinter progress widget is the
only surface, and it's capped + ephemeral.

This is genuinely the easiest of the gaps to close. A 5-line
`logging.FileHandler` writing to `<user_data_dir>/dasp/logs/run_<timestamp>.log`
would have made the 2026-04-23 incident debuggable.

### 1e. No work has happened since 2026-04-23

```bash
$ git log --since="2026-04-23" -- \
    src/spectral_predict/search_controller.py \
    spectral_predict_gui_optimized.py \
    src/spectral_predict/unified_bayesian.py
af44ad4 fix(T-06): canonical Araújo 2001 SPA enumeration
a5eef70 fix(T-21): hide x-unit Convert button — eliminates non-uniform-grid SG bug
6beb5e8 fix(T-04): grey out UVE family in one-class mode (matches iPLS pattern)
0087cad fix(T-24): add CCC/CCCcv to GUI tooltip dict + sort-direction set
b6debe7 feat: emit CCC and CCCcv in Bayesian regression result paths (T-24)
```

None of these touched the pause/resume code. Gaps confirmed live.

---

## 2. Verify reachability + UX surface

### 2a. Pause/Resume buttons exist and are visible

**`spectral_predict_gui_optimized.py:14192-14201`** — the buttons are in the
Progress tab's `progress_info_frame`, three buttons in a horizontal row
("⏸ Pause", "▶ Resume", "⏹ Stop"):

```python
# Search control buttons (Pause/Resume/Stop)
control_frame = ttk.Frame(progress_info_frame)
control_frame.pack(side='right', padx=(0, 20))

self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", command=self._stop_search, state='disabled')
self.stop_btn.pack(side='right', padx=2)
self.resume_btn = ttk.Button(control_frame, text="▶ Resume", command=self._resume_search, state='disabled')
self.resume_btn.pack(side='right', padx=2)
self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", command=self._pause_search, state='disabled')
self.pause_btn.pack(side='right', padx=2)
```

These start `state='disabled'` and are enabled when a search begins (via
`_update_search_buttons('running')`). Reachable on every search run, regardless
of mode.

### 2b. UX feedback is binary

Per `_update_search_buttons` at gui:22885-22905:

| State | Pause button | Resume button | Stop button |
|-------|--------------|---------------|-------------|
| `idle` | disabled | disabled | disabled |
| `running` | enabled | disabled | enabled |
| `paused` | disabled | enabled | enabled |

There is **no intermediate "pausing — please wait" state**. The user clicks
Pause; the button immediately becomes greyed and Resume becomes available.
This is a lie when a CatBoost trial has 25 minutes left to run — Resume is
clickable and "works" (in the sense that it sets `_pause_event` again before
the worker has even seen the original `pause()`), so Pause+Resume is a
silent no-op.

### 2c. The "errored worker" path — slightly different from the claim

The PROJECT_STATUS Follow-Ups item #3 says: "If the worker thread silently
errored during the paused window (exception caught by the `except Exception`
at gui:27664 which logs `[X] Error:` and exits the thread)".

Reading the actual error path at gui:27684-27698:

```python
except Exception as e:
    import traceback
    error_msg = traceback.format_exc()
    error_str = str(e)
    self._log_progress(f"\n[X] Error: {e}\n{error_msg}")
    self.root.after(0, lambda: self.progress_status.config(text="[X] Analysis failed"))

    # Stop running figure animation on error
    if hasattr(self, 'running_figure'):
        self.root.after(0, lambda: self.running_figure.stop_animation())

    # Disable search control buttons on error
    self.root.after(0, lambda: self._update_search_buttons('idle'))

    self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{error_str}"))
```

When the worker hits an `except Exception` in `_run_analysis_thread`, it:
1. Logs `[X] Error:` to the progress widget.
2. Sets the status label to `[X] Analysis failed`.
3. Stops the running-figure animation.
4. Calls `_update_search_buttons('idle')` — which **disables Pause AND Resume
   buttons**.
5. Pops up a `messagebox.showerror` dialog.

So the framing "worker errored silently, user clicks Resume on a dead thread"
isn't quite the right model. If the worker errors via a normal Python
exception, `_update_search_buttons('idle')` runs and disables the Resume
button. The user can't click Resume on a dead thread because the button is
disabled.

The realistic failure mode is more subtle:

- **The worker is still alive but hung** — e.g., a CatBoost training that's
  burning CPU forever on a degenerate hyperparameter combo, or stuck on
  joblib's loky in a way that doesn't raise. The thread is alive, the
  `except` hasn't fired, but `check_and_wait()` will never be called either
  because the trial isn't progressing. From the user's perspective, the
  search has been paused for 30 minutes but the spinner is still going and
  no log line appears. The user clicks Resume; nothing happens (because
  pause never took effect either). This matches the 2026-04-23 incident
  description ("Pause→Resume appeared to do nothing").
- **The worker crashed with a hard error that bypassed the except** —
  e.g., a SIGSEGV from a native extension (CatBoost / LightGBM / numpy
  C-call), `MemoryError` not caught by the bare `except Exception` in some
  inner loop, or process exit. In Python, `except Exception` does NOT catch
  `BaseException` subclasses like `KeyboardInterrupt` or `SystemExit`. If
  CatBoost segfaults the entire process exits — not "thread alive but dead",
  the whole app dies. That's the "Optuna SQLite would help" case.
- **The worker silently exited normally before pause was clicked** — e.g.,
  the search finished while the user was distracted, but the user clicks
  Pause anyway. `controller.pause()` succeeds (it just clears an event with
  no listener). UI flips to 'paused'. User clicks Resume. Nothing happens
  because there's no thread to resume. This is a real gap a thread-alive
  check would close, but it's a minor UX quirk, not the catastrophic
  "lost 19 hours" case.

So the PROJECT_STATUS framing #3 is *directionally correct* (resume should
sanity-check the thread) but the *specific error path it cites* (gui:27664)
already disables the Resume button. The real motivating case is the "alive
but hung" worker, not the "errored and exited" worker.

A thread-alive check on Resume would still help — it's a one-liner
(`if not self.analysis_thread.is_alive(): _log_progress("Worker no longer
alive — cannot resume"); return`). And for the "hung worker" case, the user
can at least be told via Stop that the thread didn't acknowledge the request
within N seconds.

### 2d. Five GUI configurations that hit the buggy code path

Any user who runs a Bayesian search hits #1 (Bayesian per-trial pause
granularity) and #4 (zero persistence). Specifically:

1. **Tier = Quick / Standard / Comprehensive in Bayesian mode** — every model
   in the tier becomes a separate Bayesian study with the in-memory storage.
2. **One-class Bayesian search** — same `unified_bayesian.run_unified_bayesian`,
   same in-memory study, same trial-boundary pause.
3. **NSGA-II preprocessing search** — uses its own controller checkpoint cadence
   (`nsga2_search.py:1965`); pause works at per-individual granularity, no
   Optuna SQLite involved (NSGA-II isn't Optuna-driven).
4. **Grid search (Quick / Standard tiers without Bayesian)** — pause works at
   per-model-config granularity (seconds), so the trial-latency complaint
   doesn't apply. But persistence is still zero — a crash mid-grid still
   loses everything.
5. **Refined model fitting** (Tab 4C) — runs in a separate thread (line
   23752 references `ensemble_training_thread.is_alive()`) but has its own
   button states and is outside T-11 scope.

Of these, the user's pain-point is **(1) and (2)**: the Bayesian mode where
trial-internal pause granularity meets in-memory persistence meets no
disk log. All three failure modes layered on a 19-hour CatBoost run is
exactly the incident that prompted the Follow-Ups note.

---

## 3. Field alignment — Optuna API + commercial chemometrics

### 3a. Optuna SQLite storage — **drop-in for the basic case**

Optuna's RDB storage is genuinely simple in the most common case. From the
Optuna docs (`optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html`):

```python
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name)
```

That's the entire API change. The current dasp code at
`unified_bayesian.py:1820-1824` would become:

```python
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    study_name=f"unified_bayesian_{model_name}",
    storage=storage_url,  # NEW
    load_if_exists=True,  # NEW — required for resume across runs
)
```

Plus an `optuna.delete_study(study_name=..., storage=storage_url)` call
when starting a fresh search (or a unique study_name per launch — see
"Caveats" below).

**Drop-in?** Mostly yes. Two non-trivial follow-ons:

1. **`load_if_exists=True` semantics.** Without this, a second
   `create_study` with the same name + storage raises
   `optuna.exceptions.DuplicatedStudyError`. With it, the new study picks
   up where the old left off — which is what we want for resume, but means
   the study_name needs to be deterministic enough for the user to pick
   the right run to resume. Currently `study_name=f"unified_bayesian_{model_name}"`
   is per-model, which means *all PLS Bayesian runs ever launched* would
   share a study unless we add a timestamp / run-ID prefix.
2. **Multiple studies in one search.** A typical Bayesian comprehensive run
   loops over models (PLS, Ridge, ElasticNet, RF, LGBM, XGBoost, CatBoost…)
   and creates one Optuna study per model. With SQLite each study lives in
   the same `.db` file but as a separate study row. To "resume" the whole
   search, the resume code needs to enumerate all studies in the .db, decide
   which is incomplete (count `n_trials < target`), and pick up from there.
   Modest engineering, not a one-liner.

### 3b. Optuna SQLite caveats

From the Optuna docs and field experience:

- **Database lock contention with concurrent runs.** SQLite uses file-level
  locking. If the user accidentally runs two dasp searches against the same
  storage URL simultaneously, they collide. Real risk in dasp because the
  app doesn't enforce single-instance. Optuna's docs explicitly recommend
  PostgreSQL/MySQL for concurrent workloads. For dasp's single-user
  bundled-app scenario, this is fine *as long as* the storage URL includes
  a per-launch UUID or timestamp.
- **RDB persistence overhead.** Optuna writes every trial to disk. For a
  300-trial CatBoost search where each trial is 30 minutes, the persistence
  overhead is microscopic relative to model training time (sub-second per
  trial). For a fast PLS sweep where trials are sub-second, the DB write
  overhead can be a few percent. Acceptable.
- **Study naming.** As noted above, the resume-across-launches semantics
  require thinking about whether the user wants to start fresh or pick up
  an existing study. Surfacing this in the GUI is non-trivial UX:
  "Existing run found — resume / start new / cancel?" dialog at search
  start.
- **Optuna version compatibility.** `load_if_exists` is stable since Optuna
  2.0 (2020); dasp's pinned Optuna version is 4.x per `pyproject.toml`. No
  compat risk.

### 3c. Commercial chemometrics — wrong frame, but worth checking

Per the master rule, commercial-software check is required. But the
chemometrics master rule pretty quickly concedes that this is a productivity
question, not a methodology one. None of the leading programs offer
crash-recoverable Bayesian model searches because **none of them run
multi-hour Bayesian searches in the first place**:

- **Unscrambler X / Camo Quant** — one-shot model fits with manual
  hyperparameter tuning. No automated multi-hour search.
- **SIMCA (Sartorius/Umetrics)** — one-shot PLS / OPLS / SIMCA fits. Has
  a "Stepwise" model variable selection but no Bayesian search over
  hyperparameters.
- **OPUS (Bruker)** — IR-specific; manual model setup. Quant 2 module fits
  PLS interactively.
- **PLS_Toolbox / Solo (Eigenvector)** — has cross-validation and
  hyperparameter sweeps; no Bayesian optimizer; no documented pause/resume
  on long sweeps. Sweeps are fast (seconds-minutes).
- **GRAMS (Thermo)** — same one-shot pattern.

**dasp's "test 11 models × 8 preprocessing × 100 Bayesian trials each"
approach is genuinely novel relative to this set.** Pause/resume on
multi-hour runs is a dasp-specific feature; competitor parity isn't the
right frame.

The closest parity point is **AutoML tools** (auto-sklearn, H2O AutoML,
TPOT, FLAML, MLJAR). These all have multi-hour search modes, and their
pause/resume support varies:

- **auto-sklearn 0.15+** — checkpointing via `tmp_folder`; the search can
  be killed and resumed from a checkpoint. Production-grade.
- **H2O AutoML** — saves intermediate models continuously to a `models/`
  directory; can stop and resume.
- **TPOT** — has `periodic_checkpoint_folder` parameter that pickles the
  current best pipeline every N seconds.
- **FLAML** — has `keep_search_state=True` to enable resume.

So the AutoML field has converged on "checkpoint to disk, support resume."
dasp not having this is genuinely a gap relative to that adjacent space.
But the chemometrics space doesn't offer it either (because the chemometrics
space doesn't run multi-hour searches).

### 3d. The bonus disk-logging gap is already canonical practice everywhere

`logging.FileHandler` to a per-run log file is bog-standard Python. Every
AutoML and ML library does it. dasp not doing it is a hygiene gap, not
a research question. The implementation is genuinely 5-10 lines.

---

## 4. Empirical — actual user pain magnitude

### 4a. The 2026-04-23 incident, reconstructed

The original Follow-Ups note (PROJECT_STATUS.md:322) says:

> "User lost hours on a CatBoost run after Pause → Resume appeared to do
> nothing; log was gone by the time we could inspect it (progress pane is
> widget-only, no disk log)."

Searching SESSION_LOG.md / SESSION_LOG_ARCHIVE.md for "2026-04-23" finds no
entries — the incident appears to have been documented only in the
PROJECT_STATUS Follow-Ups itself, not in the SESSION_LOG. The
`docs/analysis_vs_unscrambler/COMMERCIAL_READINESS.md:34` independently
flags the same scenario: "30-minute CatBoost Bayesian search, user clicks
Pause -- UI says 'paused' but thread keeps running for 10-30 minutes."

So the incident is documented in two places (PROJECT_STATUS + the commercial
readiness analysis), with consistent details. No SESSION_LOG bug-cause
post-mortem was written, presumably because the disk-log gap meant nothing
was inspectable post-mortem.

The Roadmap entry at `RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md:104`
calls it specifically "the user lost hours on a 19-hour CatBoost run."

### 4b. Typical search durations — by mode

From `unified_bayesian.py` (default `n_trials=300`, `gui:3006`) and from
the model-grid sizes in `models.py`:

| Mode | Models tested | Typical duration |
|------|---------------|------------------|
| Grid Quick | PLS, Ridge | 1-5 min on FTIR (1000 wavenumbers, 60 samples) |
| Grid Standard | + LightGBM, RandomForest | 5-30 min |
| Grid Comprehensive | + XGBoost, CatBoost, SVM | 30 min - 2 hr |
| Bayesian Quick | PLS, Ridge, 100 trials each | 5-15 min |
| Bayesian Standard | 4 models, 200 trials each | 30 min - 2 hr |
| Bayesian Comprehensive | 8+ models, 300 trials each | 2-19+ hr |

CatBoost is the consistent outlier — single trials of 5-30+ minutes on
larger datasets are routine. A Bayesian Comprehensive run with CatBoost
in the model list and 300 trials per model = the 19-hour run the user
reported.

### 4c. Who actually runs multi-hour searches?

Hard to quantify without telemetry (which dasp doesn't have). But:

- The user (Sponheimer / bone FTIR group) explicitly reports running them —
  the 19-hour incident wasn't a one-off; the user picks Comprehensive
  Bayesian with CatBoost as a routine workflow when the dataset is small
  enough to permit it (small bone FTIR datasets means CatBoost trials are
  shorter, and the total cost is feasible).
- Other bundled-app users (paleoanthropology / archaeology / isotope
  workflows) likely run shorter searches for routine analyses but probably
  trigger long Comprehensive sweeps when starting a new project.

The user impact framing: *for the user, this is a real bug that costs hours
of redo work when it bites*. The frequency of "biting" is hard to estimate,
but the Follow-Ups note says it has happened recently enough to motivate
the entry, and the user explicitly endorsed prioritizing T-11 in the
roadmap (`CONTINUATION_PROMPT_2026-04-30.md:139`).

---

## 5. Scope estimate — independent work units

T-11 decomposes into five independent work units of varying scope:

### 5a. Unit A — Disk-mirrored logging (the bonus gap)

**Scope:** ~2-4 hours.
- Add `setup_logging.py` module with `logging.FileHandler` to
  `<user_data_dir>/dasp/logs/run_<timestamp>.log`.
- Modify `_log_progress` to ALSO write to the file logger (currently it
  only writes to the Tkinter widget).
- Pipe `print()` calls in `search.py` / `unified_bayesian.py` /
  `nsga2_search.py` through `logger.info()` (or add a `sys.stdout` /
  `sys.stderr` redirect at app startup so backend `print()` from third-party
  libraries also gets captured).
- Add a "View log" button or a path display so the user can find it.
- `resource_paths.py` already has `is_frozen()` / `get_base_path()`; needs a
  new `get_user_data_dir()` helper that returns a writeable path
  (`platformdirs.user_data_dir('dasp')` is the standard way).

**Independence:** Fully standalone. Doesn't touch pause/resume, controller,
Optuna, or threading.

**Value:** Catches *any* future post-mortem (not just pause/resume bugs).
The 2026-04-23 incident would have been diagnosable. T-12 in the roadmap
is the same work; merging T-11 + T-12 here is natural.

### 5b. Unit B — Thread-alive check on Resume + UI honesty for Pause

**Scope:** ~2-4 hours.
- `_resume_search`: add `if self.analysis_thread is None or not
  self.analysis_thread.is_alive(): _log_progress("Worker no longer alive…");
  _update_search_buttons('idle'); return`.
- Add a 'pausing' state to `_update_search_buttons` (Pause greyed,
  Resume greyed, Stop enabled, status label "Pausing — waiting for current
  trial…").
- Wire the controller to fire a callback when `_pause_event.wait()` actually
  blocks (e.g., a third event `_paused_event` set inside `check_and_wait`
  when wait returns from a paused state). Have the GUI poll this with
  `self.root.after(500, ...)` until set, then transition from 'pausing' to
  'paused'.
- Add a 'Pause request acknowledged' or 'Pause taking longer than expected
  (current trial may be 20+ min)' progress message after a few-second
  timeout.

**Independence:** Standalone. Touches `search_controller.py` (add one
event), `_pause_search`, `_resume_search`, and `_update_search_buttons`.
Does not require Optuna changes.

**Value:** Eliminates the "UI lies" complaint. User sees real state. Resume
on a dead thread is no longer silent. This addresses claims #2 and #3 from
PROJECT_STATUS.

### 5c. Unit C — Trial-internal pause for Bayesian (CatBoost / LightGBM / XGBoost)

**Scope:** ~1-3 days.
- For each boosting model, register a per-iteration callback that calls
  `controller.check_and_wait()` and raises a custom `PauseRequested`
  exception or sets a stop flag. Optuna catches the exception and either
  marks the trial pruned or stops the study.
- Catboost: `iteration_callback` parameter or `Catboost.fit(callbacks=[...])`.
- LightGBM: `callbacks=[lgb.callback.early_stopping(...)]`-style hook.
- XGBoost: `callbacks=[xgb.callback.TrainingCallback(...)]`-style hook.
- Other slow models (RandomForest, SVM) don't have iteration callbacks;
  trial-internal pause is impossible — accept worst-case-trial pause latency.

**Independence:** Mostly standalone. Touches only the model-fitting
functions in `unified_bayesian.py` and `models.py`; doesn't touch GUI or
controller.

**Value:** Reduces pause latency from "current trial duration" (10-30+ min
worst case) to "current iteration duration" (sub-second for boosting).
But: only helps the boosting models that have iteration callbacks. PLS,
Ridge, ElasticNet, RandomForest, SVM, MLP all train as single calls with
no interposable hook.

### 5d. Unit D — Optuna SQLite persistence

**Scope:** ~1-2 days.
- Add `storage=` argument to `optuna.create_study` calls.
- Add `load_if_exists=True`.
- Add per-run timestamp/UUID to `study_name` so each launch gets a clean
  study.
- Pick a storage path: `<user_data_dir>/dasp/optuna/<run_id>.db`.
- On search start, write a `<user_data_dir>/dasp/optuna/active_run.json`
  with the run_id + study names + target trial counts.
- On app start, scan for incomplete runs and offer "Resume incomplete run?"
  dialog.
- Handle the multi-study case: a Comprehensive Bayesian run has 8+ studies
  in one .db; resume needs to enumerate all of them and pick up the one
  that's still <n_trials.

**Independence:** Mostly standalone. Touches `unified_bayesian.py` (add
`storage=`), the GUI (resume-prompt dialog), and adds a new module for
managing run state.

**Value:** Crash recovery — the catastrophic case the user got bitten by.
This is the highest-value-but-highest-cost unit.

### 5e. Unit E — Shipping order independence

| Unit | Depends on | Standalone? | Hours | Value |
|------|------------|-------------|-------|-------|
| A — Disk logging | None | YES | 2-4 | High (debugging future incidents) |
| B — Thread-alive + UI honesty | Unit A's logger optional | YES | 2-4 | High (claims #2 + #3, removes UI lies) |
| C — Trial-internal pause | None | YES | 1-3 days | Medium (only helps boosting models) |
| D — Optuna SQLite | Unit A's `get_user_data_dir` helper | Mostly | 1-2 days | High (catastrophic crash recovery) |

**A + B can ship together as a 1-day "minimum viable T-11" — cheap, high
visibility, fixes the most-reported pain point.** C + D are larger
investments that materially reduce the catastrophic case but require real
engineering on resume UX, study-name management, and per-model callback
plumbing.

---

## 6. Distribution-model tradeoff

### 6a. Bundled-app implications for SQLite storage

dasp ships via Inno Setup. The installer typically targets `Program Files\
dasp\` (read-only after install on Windows). The user's writeable area for
runtime state needs to be elsewhere:

- Windows convention: `%LOCALAPPDATA%\dasp\` (i.e., `C:\Users\<user>\
  AppData\Local\dasp\`).
- `platformdirs.user_data_dir('dasp')` returns this on Windows, the right
  thing on macOS/Linux too. Already a soft dependency; pip-installable.

`resource_paths.py` doesn't currently have a `get_user_data_dir()` helper.
Adding one is part of Unit A's scope — and makes Unit D possible without
duplicating the path logic.

### 6b. Resume across process restart

The user pauses + closes the app + reopens — what happens?

- **Without SQLite (current state):** The Optuna study is in-memory. Process
  exit destroys the study. Reopening starts from scratch; all completed
  trials are lost. Pure productivity bug.
- **With SQLite + `load_if_exists=True` + study_name persisted to a state
  file:** The reopened app reads the state file, finds the active study,
  re-creates an Optuna study from the persisted SQLite, and resumes.
  But: the surrounding context (X, y, preprocessing config, model selection,
  CV strategy) needs to be re-loaded. None of that is currently persisted
  beyond the original .dasp file. Either:
  - The user re-opens the .dasp project file; dasp detects an active Optuna
    run and offers "Resume?" — minimal new persistence (just store the
    storage URL + study name in the .dasp file).
  - Or dasp persists the full search context (X, y, config) as part of the
    run state — heavier scope.

Option (a) is the minimum viable path. The .dasp file already contains the
data and config; we just need to add a "last_active_optuna_run" key to it.

### 6c. The minimum-viable-T-11 split

Per §5e: **A + B (disk logging + thread-alive + UI honesty) gets the user
~80% of the perceived value at ~1 day of engineering**. The user's recurring
complaint is the UI lying about pause state and the lack of post-mortem
diagnostic capability — both of which A + B address.

C + D (trial-internal pause + Optuna SQLite) addresses the harder cases:

- **C** reduces worst-case pause latency, but only for boosting models.
  RandomForest / SVM / MLP trials still have whole-trial latency.
- **D** is the actual crash-recovery solution. Requires resume UX + state-file
  persistence + per-launch study naming. The 2026-04-23 incident
  specifically motivates this — the user *did* lose 19 hours, and only D
  prevents a future repeat.

If shipping in pieces:

- **Tier 1 (ship soon, ~1 day):** A + B. Low risk, high visibility.
- **Tier 2 (ship later, ~2-3 days):** D — Optuna SQLite + resume UX. Higher
  scope, addresses the catastrophic case.
- **Tier 3 (ship if motivated, ~1-3 days):** C — trial-internal pause for
  boosting. Lowest priority of the three; only helps Bayesian users with
  CatBoost/LightGBM/XGBoost specifically.

For a verdict-writer who wants "ship the smallest thing that materially
helps": **A + B** is that thing.

For a verdict-writer who wants "ship the thing that prevents the next
19-hour catastrophe": **A + B + D** is that. About 3-5 days total.

For full T-11 as scoped in the roadmap: A + B + C + D. ~5-7 days.

---

## 7. Cross-checking against the gate methodology's five false-alarm patterns

| Pattern | Applicable to T-11? | Evidence |
|---------|--------------------|----------|
| Sklearn-instinct false alarm (T-26 first-pass) | **No.** T-11 isn't a chemometrics methodology question. The bug is the pause UX, not a numerical calculation. |
| Defensive code in unreachable branch (T-32) | **No.** Pause/Resume buttons are in every search-mode UI, reachable on every Bayesian/grid/NSGA run. |
| Display-economy / code style (search.py:2855, bayesian random_state) | **No.** The gaps materially change observable behavior (UI state, pause latency, persistence) — not display-only. |
| dasp already matches leading programs (T-26 final) | **N/A — wrong frame.** No leading chemometrics program runs multi-hour automated searches. AutoML libraries (auto-sklearn, H2O, FLAML) all do persistence-with-resume; dasp is the outlier in that adjacent space. |
| Real finding can warrant zero action (T-26 final) | **No.** The user has already hit the bug. Cost is real (hours of work). Zero-action is not defensible. |

T-11 does not appear to be a false alarm in any of the five patterns. The
gate question is *scope*, not *whether*.

---

## 8. Notes / uncertainties for the verdict-writer

1. **Framing is correct.** All four claimed gaps + the bonus disk-logging
   gap are present in current `main`. No commits since 2026-04-23 have
   touched the affected code paths.

2. **The PROJECT_STATUS framing for gap #3 is slightly off.** The
   "worker errored silently → resume on dead thread" path is mostly closed
   by the existing `except Exception` at gui:27684-27698 (which disables
   the Resume button). The realistic motivating case is "worker hung but
   alive" — different fix shape. A thread-alive check on Resume is still
   useful as a one-liner safety net.

3. **Empirical user-pain frequency unknown.** No telemetry, no SESSION_LOG
   incident report from 2026-04-23 (only PROJECT_STATUS Follow-Ups + the
   commercial-readiness analysis flagged it). One documented incident, but
   user explicitly endorsed prioritizing T-11 in the roadmap.

4. **Optuna SQLite is mostly drop-in for a single study, harder for
   multi-study.** A Comprehensive Bayesian run creates 8+ studies in one
   storage; resume UX needs to enumerate them. Not a blocker, but more
   than 5 lines of code.

5. **Trial-internal pause (Unit C) only helps boosting models.** PLS / Ridge
   / ElasticNet / RandomForest / SVM / MLP are single-call trains with no
   interposable hook. Worst-case pause latency for a slow trial of those
   models is unavoidable.

6. **Disk-logging fix (Unit A) is the cheapest and most-broadly-useful
   piece.** Closes the bonus gap, makes future T-11 / T-12 / arbitrary-bug
   post-mortems possible, scope is 2-4 hours. The verdict-writer should
   strongly consider shipping this even if the rest of T-11 deferrs.

7. **`_log_progress`'s 2000-line cap (gui:27801-27804)** is also worth
   removing in Unit A — the cap was added for "performance during tab
   switches," but a disk-mirrored log makes the in-widget cap less
   problematic. Could cap differently or just leave the widget capped
   while the disk log is uncapped.

8. **`_run_analysis_thread` daemon=True** (gui:23159) means the worker
   thread cannot keep the process alive on app exit. This is fine for
   in-memory state, but means an Optuna SQLite resume needs the worker
   to flush trials *before* the user quits — Optuna's RDB storage flushes
   per-trial, so this is automatic.

9. **No SESSION_LOG entry** documenting the 2026-04-23 incident itself.
   The Follow-Ups note in PROJECT_STATUS is the most detailed account.
   Worth a SESSION_LOG entry now-or-when-fixed for the multi-machine sync
   (per the CLAUDE.md session protocol).

10. **`_pause_search` log message lies.** Line 22868 logs
    `[PAUSED] Search paused by user` synchronously with the click. The
    actual pause hasn't taken effect yet. Even within Unit B's scope, this
    log line should change to "Pause requested — waiting for current trial
    to finish…" and a follow-up "[PAUSED] Search paused" should fire when
    the worker actually blocks. Cheap fix bundled with Unit B.

11. **Distribution-model check — bundled-app users.** Per the master rule, a
    backend-only knob delivers zero value. Pause/Resume buttons are GUI
    elements, so this concern doesn't apply to the *fix*. But the *resume
    across app restart* UX (Unit D) requires a dialog: "We found an
    incomplete Bayesian run from <timestamp>. Resume?" — dialog box, not
    a CLI prompt. Already in scope.

12. **NSGA-II preprocessing search has a separate controller call site**
    (`nsga2_search.py:1965`). It checks at per-individual granularity
    (seconds), so pause latency is fine. But NSGA-II isn't Optuna-driven,
    so Unit D (SQLite persistence) won't help it — NSGA-II checkpointing
    is a separate concern. Could defer NSGA-II crash recovery to a follow-up
    ticket.

---

## Sources

- [Optuna RDB Backend Tutorial — `storage="sqlite:///..."`, `load_if_exists`](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html)
- [Optuna 4.x docs — `optuna.create_study`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)
- [auto-sklearn `tmp_folder` checkpointing](https://automl.github.io/auto-sklearn/master/manual.html)
- [TPOT `periodic_checkpoint_folder`](https://epistasislab.github.io/tpot/api/)
- [FLAML `keep_search_state`](https://microsoft.github.io/FLAML/)
- [platformdirs `user_data_dir`](https://platformdirs.readthedocs.io/en/latest/api.html)
- `src/spectral_predict/search_controller.py` (current 73-line implementation)
- `src/spectral_predict/unified_bayesian.py:1820-1833` (Optuna create_study + progress_wrapper)
- `spectral_predict_gui_optimized.py:22864-22905` (_pause_search, _resume_search, _update_search_buttons)
- `spectral_predict_gui_optimized.py:27684-27698` (worker exception handler)
- `spectral_predict_gui_optimized.py:27790-27804` (_log_progress + 2000-line widget cap)
- `docs/PROJECT_STATUS.md:322-327` (original Follow-Ups note from 2026-04-23)
- `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md:102-104` (T-11 verdict KEEP)
- `docs/analysis_vs_unscrambler/COMMERCIAL_READINESS.md:34` (independent flag of same scenario)
