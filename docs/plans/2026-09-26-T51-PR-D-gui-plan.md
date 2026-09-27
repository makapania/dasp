# T-51 PR D: GUI card for extra Optuna axes (plan, 2026-09-26)

Supersedes §4 of `2026-09-13-T51-optuna-axes-implementation-plan.md`, whose GUI line numbers and
settings wiring predate PR #79 (crash-resume) and PR #80. Facts below were mapped on `main` @ `4323076`.

## What PR D does

It makes the 14 opt-in bundles (11 supervised from PR B, 3 one-class from PR C) and `n_startup_trials`
reachable from the GUI. **The backend is untouched.** `run_unified_bayesian` already validates, resolves,
names studies and records attrs. PR D adds no bundles, so the registry-wide assertions in
`tests/test_t51_supervised_bundles.py` (the `SUPERVISED_BUNDLES` trap) are not hit.

## Current state (verified)

- **Registry:** `search_spaces.BUNDLES` (:483). `BundleSpec` has `id, families, task_types, label, help,
  family_task_types`. `resolve_bundles(model, task, ids)` (:637) filters by model and task.
  - Enabled bundles that don't apply to the model are skipped with a progress message.
  - `svm_gamma` with SVM + regression raises `ExtraAxesConfigError`. The GUI can't produce that pair:
    `_on_task_type_changed` disables models the task type doesn't support.
- **Study identity:** enabled bundles change the study name (`|space=` segment). `n_startup_trials`
  does not; it is only recorded as a study attr.
- **GUI call sites:** both are in `_run_analysis_thread` (:27240):
  - one-class `run_unified_bayesian(` at :29629;
  - supervised at :30216.
  - Both read settings through `_setting(name)` (:27298). That returns the snapshot value frozen at the
    click and raises `KeyError` if the key is missing. There is no fallback to live controls.
- **Launch gate:** `_confirm_resume_before_launch` (:26620) requires every `BAYESIAN_REQUIRED_SETTINGS`
  key (:2638) to be present in the `capture_gui_settings` snapshot.
  - "Present" means readable. A blank `StringVar` is present; a blank `IntVar` fails to read and blocks
    the launch.
- **Settings capture:** `run_gui_settings.CAPTURABLE_SETTINGS` (:45) already includes
  `bayes_enable_autoscale` (added in #79 round 9), so the plan's "autoscale gap" item is done.
  - Restore leaves keys that are missing from an old snapshot at their current widget value. There is
    no legacy-default mechanism.
- **Tab 4C:** the Bayesian options live in `self.bayes_options_frame` (a `ttk.LabelFrame`, ~:12628,
  shown only for Bayesian runs by `_on_optimization_method_changed` :16951). Checkbutton pattern:
  rows 2-5 (UVE :12657, autoscale :12662).
- **Docs:** `AGENT_COMPOSITION.md` §7b already documents the three kwargs, the bundle table and the
  precedence rule, and `tests/test_agent_composition_api.py` already pins them. The plan's docs items for
  the Python API are done.
- **`base_axes`:** the per-model count of default axes the advisory needs does not exist anywhere.

## Changes

### 1. The card (Tab 4C)
- Put a collapsible "Extra hyperparameter axes (advanced)" section inside `bayes_options_frame`, below
  the persistence radio, so it shows only for Bayesian runs. It must stay separate from "Advanced Model
  Options", which applies to grid search only.
- **Built from `BUNDLES`,** grouped by model family, in registry order. Each row has a checkbox with the
  bundle's `label` and a tooltip with its `help` text. PR C's `if_max_samples` limitation lives in its
  `help` text; reuse that, don't reword it.
- **Stable var names:** `self.bayes_axis_<bundle_id>` (`tk.BooleanVar(False)`). The list of names comes
  from a module-level function over the registry, so capture and required-settings stay in sync with it.
- **Grey out by task type:** in `_on_task_type_changed`, disable the checkboxes of bundles whose
  `task_types` exclude the current task. For example, one-class bundles are disabled in regression.
  - Disabled boxes keep their value but are **not sent**: the collector filters by task type at launch.
  - This avoids a user ticking a bundle that silently does nothing.
- **`n_startup_trials`:** a `tk.StringVar` entry, `self.bayes_n_startup_trials`. Blank means
  `None`, i.e. the backend default of 20.
  - The launch gate validates it: blank, or an integer >= 1.
  - Anything else gets the gate's existing "Invalid settings" dialog.

### 2. Collector and call sites
- `_collect_enabled_extra_axes(settings, task_type)` returns a sorted tuple of ticked bundle ids whose
  `task_types` include the task type.
  - It reads from the frozen snapshot, never live vars. #79's rule is that the worker reads what the gate
    froze.
  - The task type used is the resolved one (after "auto" inference in the worker).
- Pass `enabled_extra_axes=` and `n_startup_trials=` at both call sites. Do not omit them the way the
  supervised call already omits `early_stopping_rounds`.
- **Guard each model's call:** catch `ExtraAxesConfigError`, log it, and fail that model visibly. The
  run carries on with the next model.

### 3. Settings capture and resume
- Add every `bayes_axis_<id>` and `bayes_n_startup_trials` to **both** `CAPTURABLE_SETTINGS` and
  `BAYESIAN_REQUIRED_SETTINGS`.
  - Bundles change the study name. A value that isn't frozen means a changed checkbox silently starts a
    different study, and the finished run releases the saved one's record.
  - `n_startup_trials` changes the TPE trajectory. It is not in the study name, but a resume must use
    the same value.
- **Old snapshots:** add a small `LEGACY_DEFAULTS` for the new keys only (bundles off, startup blank).
  `restore_gui_settings` applies it for keys absent from the snapshot, before its empty-settings early
  return.
  - Since nothing is published, no "assumed" report category is needed. Old snapshots are dev-only.
- Settings-diff dialog (:26398) and resume banner: add "Changing extra axes starts a new study."

### 4. Advisory
- One caption line under the card:
  "Opens N extra axes for <ticked models>. Suggested: ~T trials; startup max(20, 3 x dim).
  Upper-bound guide: widened spaces often peak earlier. Validate externally."
- **Compute `dim` at runtime:** run `discover_suggested_names` on the base sampler for each ticked
  model, plus each applicable bundle's axes. This replaces a hand-maintained `base_axes` table, so there
  is no table to drift.
- Show a runtime note when IsolationForest or a boosting bundle is ticked.
- Never write `n_unified_trials`.
- Update it on checkbox, model and task changes. It must be cheap; cache per model.

### 5. Docs
- User guide (Tab 4C section): the card, what a bundle is, the study-name consequence, and the
  "improves the best, lowers the average" caveat.
- `CLAUDE.md:91`: soften "All hyperparameters are exposed and user-editable" to say that grid
  hyperparameters are user-editable and that Bayesian search opens extra axes only via opt-in bundles.
- CHANGELOG entry under 0.5.0b3 "Added".

## Tests
Model the GUI tests on `tests/gui/test_resume_round11.py`: monkeypatch `run_unified_bayesian`, click
`_run_analysis`, run the `_FakeThread` target, and inspect the kwargs.
1. A ticked bundle reaches both call sites as a sorted tuple. Unticked means `()`.
2. A bundle for a different task type is not sent (e.g. `lof_metric` ticked, then regression).
3. **Frozen at the click:** toggling a bundle mid-run does not reach later models (copy round 11).
4. `n_startup_trials`: blank gives `None`, `"30"` gives `30`, `"abc"` and `"0"` block the launch.
5. Capture/restore round-trip. An old snapshot without the new keys restores bundles to off even when
   the controls were ticked (the T11 case). Use `_FakeGUI` in `tests/test_t43_resume_auto_restore.py`.
6. Every registry bundle has a var, and it is in both settings lists. Adding a bundle without GUI
   wiring must fail a test.
7. Advisory: `dim` for a known model equals the number of base suggests plus the bundle axes (no Tk).
8. The greying follows the task type.
9. **Study-name check:** a GUI-launched run with `xgb_sampling` gets the same study name as the Python
   call with `enabled_extra_axes=("xgb_sampling",)`.

Run the full non-GUI suite **on this machine** (the T3 blessing machine), plus `tests/gui/test_resume_*.py`.
Nothing here should change default trajectories: all bundles are off by default.

## Review
Codex + GLM 5.3, re-reviewing until clean, as in earlier rounds.

## Decisions (recommendations; confirm or change)
1. **Grey out by task type only,** not by ticked models. Model-level greying needs traces on 13 model
   checkboxes; the advisory already says which ticked models a bundle applies to.
2. **Legacy defaults for the new keys only,** with no restore-report category, because nothing is
   published.
3. **Compute the advisory's `dim` at runtime** rather than keeping a `base_axes` table.
4. **Expose `n_startup_trials` in the GUI** (the original plan did). The alternative is to leave it
   Python-only and keep the card to checkboxes.
