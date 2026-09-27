# T-51 PR D: GUI card for extra Optuna axes (plan, 2026-09-26, revision 2)

Supersedes §4 of `2026-09-13-T51-optuna-axes-implementation-plan.md`, whose GUI line numbers and
settings wiring predate PR #79 (crash-resume) and PR #80. Facts were mapped on `main` @ `4323076`.

**Revision 2** folds in the plan review:
- Codex (`gpt-6-astra`): REVISE, two HIGH findings.
- GLM 5.3: APPROVE-WITH-CHANGES, one HIGH finding.

Both HIGH findings concern legacy snapshots on resume (§3). The review log is at the end.

## What PR D does

It makes the 14 opt-in bundles (11 supervised from PR B, 3 one-class from PR C) and `n_startup_trials`
reachable from the GUI.
- **The backend is untouched.** `run_unified_bayesian` already validates, resolves, names studies and
  records attrs.
- PR D adds no bundles, so it does not hit the registry-wide assertions in
  `tests/test_t51_supervised_bundles.py`.
- With every bundle off and startup blank, the call is identical to today's: no `|space=` segment and
  startup 20. The default trajectories and the T3 fixture are therefore unchanged.

## Current state (verified by both reviewers)

- **Registry:** `search_spaces.BUNDLES` (:483). `BundleSpec` has `id, families, task_types, label, help,
  family_task_types`. `resolve_bundles(model, task, ids)` (:637) filters by model and task.
  - A bundle that doesn't apply to the model is dropped, and that model's study identity is unchanged
    (`canonical_space_identity` :984).
  - The "None of the enabled bundles apply …" progress message fires only when **no** enabled bundle
    applies to that model (`unified_bayesian.py:2676`). With a full tier that can be several log lines.
  - `svm_gamma` with SVM + regression raises `ExtraAxesConfigError`. The GUI can't produce that pair:
    `_on_task_type_changed` unticks and disables unsupported models (:17064-17077).
- **Study identity:** applicable bundles add a `|space=` segment to the study name. `n_startup_trials`
  does not; it is only recorded as a study attr. The backend reattaches a sampler with whatever startup
  count it is given (:3155) and **never checks it against the saved value**, so the GUI must.
- **GUI call sites:** both are in `_run_analysis_thread` (:27240): one-class at :29629, supervised at
  :30216.
  - `_setting(name)` (:27298) returns the snapshot value and raises `KeyError` if the key is missing.
    Without a snapshot (direct test calls) it falls back to live vars (:27301).
  - The task type is resolved from frozen data (`analysis_data` :27312, "auto" inference :27517-27536)
    before either branch, so filtering bundles by the resolved task from frozen data is safe.
- **Launch gate:** `_confirm_resume_before_launch` (:26620) requires every `BAYESIAN_REQUIRED_SETTINGS`
  key (:2638) to be *present* in the current `capture_gui_settings` snapshot.
  - It checks presence, not values. A blank `StringVar` is present; a blank `IntVar` fails to read and
    blocks the launch.
  - It checks the current capture, not old saved snapshots, so adding required keys does not break old
    snapshots.
- **Capture/diff/restore (`run_gui_settings.py`):**
  - `CAPTURABLE_SETTINGS` (:45) already has `bayes_enable_autoscale`.
  - `capture_gui_settings` skips attributes that don't exist (:295-297).
  - `diff_gui_settings` (:242) compares **only keys present in `saved`**. So a pre-PR-D snapshot
    compared with a GUI that has a bundle ticked reports no difference.
  - The settings-diff dialog (:26386-26412) passes **only the differing keys** to
    `restore_gui_settings` (a partial patch).
  - `restore(None/{})` is a documented no-op (`tests/test_t43_resume_auto_restore.py:187-203, :549-562`).
- **Pre-PR-D snapshots may exist in the field.** 0.5.0b2 was cut (`35c3fc9`, CHANGELOG 2026-05-03) with
  resumable runs, although there is no git tag or GitHub release. Treat old snapshots as real.
- **Per-model error handlers:** the existing `except Exception` handlers (:29679, :30275) increment
  `oc_model_errors` / `unified_model_errors`. That keeps a partly failed run paused and resumable
  (:27134-27139) instead of releasing its record.
- **Tab 4C:** `self.bayes_options_frame` (~:12630) is shown only for Bayesian runs
  (`_on_optimization_method_changed` :16951). Checkbutton pattern: UVE :12656, autoscale :12662.
- **Docs:** `AGENT_COMPOSITION.md` §7b and `tests/test_agent_composition_api.py:117-137` already cover
  the Python API. User guide anchors: Model Configuration §4.3 (`docs/UserGuide.md:1232`), Bayesian
  parameters (:1267), TPE reference (:6819).
- **`discover_suggested_names`:** measured at 0.01-0.5 ms per model. It is cheap enough for the GUI
  thread.

## Changes

### 1. The card (Tab 4C)
- Put a collapsible "Extra hyperparameter axes (advanced)" section inside `bayes_options_frame`, below
  the persistence radio. It must stay separate from the grid-only "Advanced Model Options".
- **Built from `BUNDLES`,** grouped by model family, in registry order: a checkbox with `label`, and a
  tooltip with `help`. PR C's `if_max_samples` limitation is in its `help` text; reuse it, don't reword
  it.
- **Vars are created eagerly** during UI construction. This applies to `self.bayes_axis_<bundle_id>`
  (`tk.BooleanVar(False)`) and `self.bayes_n_startup_trials` (`tk.StringVar("")`), even while the
  section is collapsed.
  - A lazily created var is missing from the capture, and the gate would then block every Bayesian
    launch.
  - The names come from one module-level function over the registry, `extra_axes_setting_names()`.
    Capture, required-settings and legacy defaults all use that function.
- **Grey out by task type** (decision 1): a helper, `_refresh_extra_axes_state()`, disables boxes whose
  bundle's `task_types` exclude the current task.
  - Call it on **both** paths of `_on_task_type_changed`, including the "auto with no data" early return
    at :17031-17048. Otherwise the boxes keep a stale state.
  - Disabled boxes keep their value but are not sent, because the collector filters by task.
- **`n_startup_trials`** (decision 4): an entry, where blank means `None`, i.e. the backend default of
  20.

### 2. Validation, collector and call sites
- **New gate validation** (new code; the gate checks only presence today): after the presence check,
  `bayes_n_startup_trials` must be blank or an integer >= 1.
  - Anything else raises the existing "Invalid settings" dialog and blocks the launch.
  - Grid and NSGA-II launches leave the gate before this point (:26279) and are unaffected. The value is
    never read on those paths.
- **Parsing happens in the collector, not the backend:**
  - `_collect_extra_axes(settings, task_type)` returns `(sorted tuple of ticked bundle ids whose
    task_types include task_type, n_startup_trials as int or None)`.
  - It reads the frozen snapshot through `_setting`, and uses the task **resolved in the worker**.
- Pass `enabled_extra_axes=` and `n_startup_trials=` at both call sites.
- **No new exception handler.** `ExtraAxesConfigError` is a `ValueError`, so the existing per-model
  handlers already catch it and increment the error counters. That keeps the run resumable.
  - Consequence to accept and document: a persistent config error re-offers a resume that fails the same
    way until the user deletes the saved run.
  - The existing message (:29670-29677) already points there. Include the bundle id in the log line.

### 3. Settings capture and resume (rewritten after review)
- Add every `bayes_axis_<id>` and `bayes_n_startup_trials` to **both** `CAPTURABLE_SETTINGS` and
  `BAYESIAN_REQUIRED_SETTINGS`.
- **`LEGACY_DEFAULTS`** in `run_gui_settings.py` maps each new key to its value before PR D: bundles
  `False`, startup `""`.
- **Normalise saved snapshots before comparing them.** A new helper,
  `normalize_saved_settings(saved)`, fills the new keys that are missing from a **non-empty** saved
  snapshot with their legacy defaults. `diff_gui_settings` compares the normalised snapshot. This closes
  the silent-fork path in two cases:
  - An old run resumed with a bundle now ticked shows up as a difference, so the existing
    restore/cancel/fresh dialog appears.
  - The same happens when startup has changed: `"40"` now against `""` saved.
- **Full versus partial restore:** only a **full** restore normalises. That is the startup auto-restore
  of a whole saved snapshot. The dialog's **partial patch** (the differing keys only) writes exactly the
  keys it is given and never fills defaults. This removes the restore loop Codex reproduced: restoring
  startup no longer resets bundles, and the other way round.
- **Empty or `None` snapshots keep today's no-op contract,** in both diff and restore: nothing was
  recorded, so there is nothing to compare. A pre-PR-D run always has a non-empty snapshot.
- **Restore report:** legacy-filled keys that actually change a widget are counted as restored, as
  usual. They also appear in the diff dialog, so the user sees them. There is no new report bucket
  (decision 2). The module docstring's drift note is updated.
- **Dialog text:** the settings-diff dialog (:26398) and the resume banner get "Changing extra axes or
  startup trials changes how a resumed search continues; extra axes start a new study."
- **Rollback:** do not roll back past PR D while a run started with bundles is pending. The older build
  ignores unknown saved keys and would resume it with the default search space. Complete or discard such
  runs first. This goes in the CHANGELOG entry.

### 4. Advisory (corrected after review)
- **Dimension per model** = the model's base suggests plus its applicable bundle axes, plus the shared
  axes.
  - Base suggests come from `discover_suggested_names` on the model sampler.
  - Shared axes are `preprocessing` plus any preprocessing sub-suggests enabled by the current Bayesian
    options, and `subset_type`, `n_vars` and `region_id`.
  - Shared axes are counted from the same recording mechanism, run on the preprocessing suggester with
    the GUI's current options. They are not hard-coded.
- Each model runs its own study, so show the **largest** dimension and name its model:
  "Up to D search dimensions (XGBoost); startup trials max(20, 3 x D) = S."
  - Suggesting a total trial count was dropped: the review found no defensible formula, and the
    downstream evidence says widened spaces often peak earlier.
  - Add the caveat "an upper-bound guide; opening axes improves the best candidates but lowers the
    average one. Validate externally."
- Show a runtime note when IsolationForest or a boosting bundle is ticked.
- Never write `n_unified_trials` or the startup entry.
- **Refresh** on changes to bundle ticks, model ticks, the task type and the dimension-changing Bayesian
  options (baseline, smoothing, autoscale, regions, UVE).
  - Cache per (model, task) plus an options key.
  - With no data loaded, use a nominal feature count, because names don't depend on it. When the task is
    "auto" with no data, show "load data to see dimensions".

### 5. Docs
- User guide §4.3 (`docs/UserGuide.md:1232`) and the Bayesian parameters section (:1267): the card,
  what a bundle is, the study-name consequence, the startup box, and the caveat.
- `CLAUDE.md:91`: grid hyperparameters are user-editable; Bayesian search opens extra axes only
  through opt-in bundles.
- A CHANGELOG entry under 0.5.0b3 "Added", including the rollback note.

## Tests
The GUI tests drive the real app (`tests/gui/conftest.py`). **Fixture teardown must reset every new
var** (bundles off, startup blank). The session app is shared, so a leftover `"abc"` would break
unrelated launch tests.

**Wiring**, modelled on `tests/gui/test_resume_round11.py`:
1. A ticked bundle reaches both call sites as a sorted tuple. Unticked means `()`.
2. A bundle for another task type is not sent. Also: "auto" resolved from the frozen data picks the
   bundles for the inferred task.
3. **Frozen at the click:** toggling a bundle or the startup value mid-run does not reach later models.
4. `n_startup_trials`: blank gives `None`, `"30"` gives `30`, and `"abc"`, `"0"` and `"-1"` block the
   launch with the dialog. Grid and NSGA-II launches ignore a bad startup value.
5. **One model succeeds, then a model raises `ExtraAxesConfigError`:** the error counter increments, and
   the run stays paused and resumable instead of being released.

**Resume** (the transitions the review found):

6. **Legacy launch:** an old snapshot with no new keys, plus a bundle ticked now, makes the gate show the
   restore dialog. Test it at gate level, not by calling `restore_gui_settings` directly.
7. **Changed startup on resume:** saved `"30"` against current `"40"` shows up in the diff.
8. **Partial-restore convergence:** saved bundle on and startup `"30"`, against current startup
   `"40"`. Applying the dialog's patch converges in one pass, with no alternation.
9. **Full restore of an old snapshot** onto ticked controls resets bundles to off and startup to blank.
   `restore(None/{})` stays a no-op.

**Registry and advisory:**

10. Every registry bundle has a var created at construction, and it is in both settings lists. Adding a
    bundle without GUI wiring must fail.
11. The advisory's dimension for a known model and options equals the real recorded suggest count,
    including the shared axes (no Tk).
12. Greying follows the task type, including the "auto with no data" early return.

**Study name:**

13. A real tiny run, not a monkeypatch: XGBoost, 2 trials, a temporary SQLite store, launched from the
    GUI worker with `xgb_sampling`. It gets the same study name as the direct Python call. This test
    cannot use round 11's monkeypatch model.

Run the full non-GUI suite **on this machine** (the T3 blessing machine), plus `tests/gui/test_resume_*.py`.

## Review
Codex + GLM 5.3 on the implementation, re-reviewing until clean.

## Decisions (approved 2026-09-26)
1. Grey out by task type only.
2. Legacy defaults for the new keys only, with no new report bucket. (Revised mechanism: normalise in
   diff and full restore, per §3.)
3. Compute the advisory dimension at runtime. (Revised: it now includes the shared axes, and the
   total-trials suggestion is dropped.)
4. Expose `n_startup_trials` in the GUI.

## Plan-review log (2026-09-26)
- **Codex REVISE:**
  - H1: legacy defaults inside restore loop forever on partial patches.
  - H2: diff ignores keys missing from the saved snapshot, so old runs silently accept new bundles or
    startup.
  - M: the advisory omitted the shared axes.
  - M: tests missed the resume transitions and fixture teardown.
  - M: a new handler would skip the error counters.
  - M: rollback with a pending run.
- **GLM 5.3 APPROVE-WITH-CHANGES:**
  - B1 (HIGH): the same H2, and "old snapshots are dev-only" is false because b2 shipped resumable runs.
  - M1: the startup validation is new code, and it should be parsed in the collector.
  - O1: the auto/no-data early return skips greying.
  - O2: vars must be created eagerly.
  - R1: a persistent config error keeps the run resumable.
  - R2: the study-name test needs a real run.
  - Low: the report docstring, the user-guide anchors, and the advisory with no data.
- Every finding is folded in above.
