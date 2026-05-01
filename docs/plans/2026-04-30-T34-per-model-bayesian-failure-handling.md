# T-34: Per-model Bayesian failure handling — preserve resume state on partial runs

**Status:** PLANNED — deferred from T-11 (NEW BUG #2 in PR #6 review trail).
**Filed:** 2026-04-30 (during T-11 merge review).
**Source:** Codex meta-review of T-11 PR #6, surfaced as architectural cost; explicitly deferred and accepted by DeepSeek V4 Pro pass-3.
**Related:** T-11 (parent), `docs/bugfix_validation/T11_pause_resume_hardening.md` §8 + §9.

## Problem

`spectral_predict_gui_optimized.py:_run_analysis_thread` runs a per-model
loop for Bayesian search. Per-model exceptions are caught with `except
Exception` and the loop continues to the next model (line ~27234, the
`continue` path). At the end of the loop, the success path calls
`mark_complete()` (line ~27958), which now correctly only deletes the
sidecar matching the active run_id (T-11 NEW BUG #1 fix) — but it still
*deletes the active sidecar even if some models in the loop failed*.

**User-visible failure mode:**

1. User starts Bayesian search with 3 models: PLS, RandomForest, XGBoost.
2. PLS completes 80/100 trials, RandomForest fails on trial 42 (e.g. OOM
   on a wide spectral matrix), XGBoost completes 100/100.
3. The per-model `except` swallows the RandomForest failure; the loop
   continues to XGBoost.
4. End-of-loop, `mark_complete()` runs and unlinks the sidecar.
5. User now has no resume option for RandomForest's 42 partial trials,
   even though the on-disk SQLite store still has them.

The resume-from-crash workflow assumes "either everything completed or
the process died." A model-level partial-failure is a third state the
sidecar lifecycle doesn't model.

## Design (rough — refine at implementation)

Two viable approaches:

**Approach A: per-model sidecars.** One `active_run_<run_id>_<model>.json`
sidecar per model, written when that model starts and deleted only when
that model's trials reach `n_trials_per_model` (or when the user
explicitly discards). The active-run record becomes a directory of
per-model sidecars rather than a single JSON. `mark_complete()` only
cleans up sidecars whose model exited cleanly. The `_check_for_incomplete_run`
dialog enumerates per-model sidecars and offers "Resume model X?" /
"Discard model X?" / "Resume all" / "Discard all."

Pros: cleanly models the per-model state.
Cons: large change to the sidecar schema and the resume dialog UI.
Multi-model resume becomes a multi-step user workflow.

**Approach B: track failures in the existing single sidecar.** Add a
`models_completed: list[str]` and `models_failed: list[str]` field to
`RunMetadata`. The Bayesian per-model loop appends to those lists as it
runs (via a new `mark_model_complete(model_name)` /
`mark_model_failed(model_name, exception_type)` API). `mark_complete()`
only unlinks the sidecar if `set(models_completed) == set(meta.model_names)`;
otherwise it leaves the sidecar with the partial state for the next
launch's resume dialog to surface. The dialog grows a "X of N models
completed; resume the failed ones?" framing.

Pros: minimal sidecar-schema change, no new files.
Cons: requires an Optuna-storage-aware way to filter the resume to "only
the failed/incomplete models."

**Recommendation:** Approach B. The Optuna SQLite store already keeps
trials keyed by study_name (which includes model_name in T-11's
`unified_bayesian_{model_name}_{config_hash}` format), so filtering on
resume is "instantiate studies only for the models in `meta.models_failed
∪ (meta.model_names − meta.models_completed)`." That's a small filter,
not a schema rewrite.

## Files likely touched

- `src/spectral_predict/run_state.py` — add `models_completed` /
  `models_failed` fields to `RunMetadata`; add `mark_model_complete()`
  and `mark_model_failed()`; tweak `mark_complete()` to only finalize
  when all models accounted for.
- `spectral_predict_gui_optimized.py` — wire `mark_model_complete` /
  `mark_model_failed` into the per-model loop's success + except paths
  (line ~27234 area). Update resume dialog text to surface partial state.
- `tests/test_run_state.py` — regression coverage: partial-completion
  scenarios, mark_complete-as-no-op-when-incomplete, resume-skips-
  already-completed-models.

## Open questions

1. **Stop button vs partial failure**: when the user clicks Stop, is
   that "completed but I want to discard" or "completed but I want to
   keep partial trials for resume"? Currently Stop runs the success
   path including `mark_complete()`. Codex's "Stop vs Complete distinction"
   suggestion (filed as T-40) is closely related — may want to land
   T-40 first to disambiguate the lifecycle states before implementing
   T-34.
2. **`_active_metadata` cache invalidation**: T-11's `_active_metadata`
   cache exists for the idempotent-return contract. If `mark_model_*`
   mutates `RunMetadata` fields, the cache and the on-disk sidecar
   must stay in sync. Easiest approach: re-write the sidecar after
   each model boundary (idempotent + atomic via existing
   `_atomic_write_json`).
3. **Concurrent instance interaction**: if two dasp instances both
   write to the per-model sidecar mid-run, the existing "whoever
   finishes second wins" race could now lose model-completion records.
   Single-user single-instance is the assumed regime; document
   accordingly.
4. **Dialog UX**: the existing resume dialog is a 3-button modal
   (Yes / No / Decide-later). Per-model partial state may need a
   richer dialog (treeview of per-model status, with checkboxes).
   Consider a separate UX pass before implementing.

## Success criteria

- A Bayesian run with 3 models where model 2 fails leaves a sidecar
  describing models 1+3 as completed, model 2 as failed.
- Next launch's resume dialog offers "Resume the 1 failed model
  (RandomForest)?" with the partial 42 trials available.
- Discard removes only the model 2 sidecar/study; sidecars for 1+3
  are already gone (they completed).
- Existing single-model Bayesian regression behavior unchanged
  (single model = `mark_complete` runs as before since
  `models_completed == model_names`).
- Stop-button behavior preserved or improved per T-40.

## Estimated effort

~1-2 days for Approach B. Approach A would be ~3-5 days plus a UX pass.

## References

- T-11 PR #6 review trail (2026-04-30 → 2026-05-01)
- Codex meta-review (`docs/reviews/deepseek_v4pro_24h_review_2026-04-30.md`,
  the meta-review section)
- T-40 (Stop-vs-Complete) — companion ticket, possibly land first
