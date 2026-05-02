# T-43: Resume Should Auto-Restore Settings, Not Just SQLite Store

> **Status:** ROUGH_PLAN — small UX-driven ticket. Filed 2026-05-02 from user feedback during T-41 verification: "it would be great if the user did not have to remember every setting and could just hit run, or at least be reminded of them."

**Goal:** When the user accepts the resume banner, auto-populate the GUI from the previous run's settings — preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, dataset path — so they can click Run Analysis without re-entering anything. At minimum, surface a read-only summary of what the run was using so they know what to recreate.

**Source:** Conversation 2026-05-02. After landing T-41's resume-override fix (`bayesian_persistence_mode='always'` after `resume_run()` succeeds), the user verified resume works end-to-end *but* reported the UX is rough: they had to manually re-create every preprocessing/model setting before the resumed SQLite would actually pick up the trial-level state. They commented: "i went and recreated things exactly and then it did pick up." T-43 closes that gap.

---

## Current state (post-T-41)

`run_state.RunMetadata` stores:
- `run_id`
- `label`
- `dataset_fingerprint`
- `model_names: list[str]`
- `n_trials_per_model: int | None`
- `started_iso: str`
- `bayesian_persistence_mode: str`

That's it. None of the preprocessing or hyperparameter-search settings are persisted — the user has to remember and reproduce them. If they remember wrong, the resumed study's trials are using a different objective function than the new fresh trials, and the results are subtly wrong (TPE samples don't transfer cleanly across objective changes).

The dataset fingerprint check at `run_state.py` already catches data mismatches, but settings mismatches are silent.

---

## Two scope levels

### Scope A (small): show the user what they need to reproduce

When `_offer_resume_if_pending` fires, expand the resume banner to list the previous run's settings (preprocessing config, models selected, n_trials, dataset_fingerprint match status). User reads it, recreates manually, clicks Run. Same workflow as today, just with explicit reminders instead of "load the same data + settings."

**Cost:** ~30 LOC. The data is already in the sidecar. The challenge: only what's already in `RunMetadata` is shown. If we want to show preprocessing toggles, we need Scope B's serialization.

### Scope B (medium): full auto-restore

Extend `RunMetadata` with a `gui_settings: dict` field that captures the full GUI state at `start_run` time — preprocessing toggles, autoscale, baseline_method + params, smoothing window/polyorder, variable selection settings, model-specific hyperparameter ranges, dataset path (so we can offer to re-load it).

On resume, the GUI reads `meta.gui_settings` and calls `_restore_gui_state(...)` which sets every Tk variable to the saved value. User just clicks Run.

**Cost:** ~100-150 LOC. Mostly serialization — there's already a save-as-dasp model save format in `model_io.py` that captures most of this; reuse the schema. The risk is that GUI settings drift over future releases — old sidecars won't have the new fields, so `_restore_gui_state` needs forgiving defaults.

---

## Recommended scope: B

The "remind the user" approach (A) doesn't really fix the user's pain — they still have to re-type things, and they can still get it wrong without warning. Auto-restore (B) is the right answer because:

1. Resume *only matters* for crash recovery, which is rare. Spending 100 LOC to make rare events smooth is good UX investment.
2. Settings drift across the time gap (close app at 5pm, reopen 9am next day) is the most common cause of "I thought I was resuming but the trials don't match" confusion.
3. The serialization machinery already exists in `model_io.py`'s `.dasp` save format — adapting it for sidecar storage is mostly reuse.

---

## Tasks (rough)

1. Define `RunGUISettings` dataclass in `run_state.py` capturing the full GUI state. Mirror the `.dasp` save schema where possible to keep the serialization layer single-sourced.
2. Extend `RunMetadata` with `gui_settings: RunGUISettings | None`. `from_dict` tolerates missing/legacy field.
3. `start_run()` accepts and stores `gui_settings`. GUI passes the current state when starting a Bayesian run.
4. New helper `_restore_gui_state(self, settings: RunGUISettings)` in the GUI: sets every Tk variable to the saved value. Robust to missing fields (forgiving defaults — log a warning if a field is missing).
5. `_offer_resume_if_pending` shows an extended banner with the captured settings summary, and offers an "Auto-restore + Run" button that calls `_restore_gui_state` then triggers Run Analysis.
6. Tests:
   - Round-trip: serialize GUI state → save sidecar → load sidecar → restore GUI state → assert all variables match.
   - Backward-compat: load a pre-T-43 sidecar (no `gui_settings` field) and confirm `_restore_gui_state` is a no-op without crashing.
   - End-to-end: simulate crash mid-run, restart with auto-restore, verify the resumed study and the new trials use the same preprocessing.

---

## Out of scope

- Auto-loading the dataset file path (the dataset can move/rename between sessions; safer to surface the previous path and let the user confirm).
- Auto-restoring across major version changes (settings schema evolves; cross-version restore is a separate engineering problem).

## Effort estimate

~150 LOC + tests. Single-day ticket.
