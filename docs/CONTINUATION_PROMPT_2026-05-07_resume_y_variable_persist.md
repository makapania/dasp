# Continuation prompt — T-resume-y-variable-persist (SUPERSEDED)

**Status:** Ticket closed via PR #45 (`fix/resume-banner-y-variable-instructions`, merged at `82d9363`). The original implementation strategy in this prompt was demonstrated wrong by Codex's cross-family review of PR #44 and is no longer load-bearing.

---

## Why this prompt was superseded

The original strategy: add `target_column` to `CAPTURABLE_SETTINGS` + add a validation hook in `restore_gui_settings` against `_get_available_target_columns()`. Implementation landed in PR #44 (`fix/Tresume-y-variable-persist`).

**Codex's review caught the load-bearing flaw:** `restore_gui_settings(self, resumed.gui_settings)` runs at GUI startup *before* the user reloads data (call site: `spectral_predict_gui_optimized.py:23219`). At that moment, `_get_available_target_columns()` returns `[]` because neither `combined_metadata_df` nor `ref` is populated. Every captured `target_column` therefore fails validation, routes through `RestoreReport.errors`, and surfaces as a banner error — the exact opposite of the auto-restore the PR was meant to deliver.

**GLM 5.1's pre-PR verification missed this** because it evaluated `restore_gui_settings` in isolation, without tracing the call site. **DeepSeek V4 Pro Max also approved at MEDIUM** by accepting "user picks manually post-load" as the success state — but that *is* the pre-PR behavior the PR was supposed to fix. Two of three reviewers approved a feature that does not work in production.

PR #44 closed without merge. PR #45 chose the simpler banner-only path per user directive: tell the user explicitly in the resume banner that they need to set Y manually.

## What actually shipped (PR #45)

- `spectral_predict_gui_optimized.py:23288-23303` — both banner branches updated to add the load-bearing instruction "set the Y variable manually (resume does not restore the Y selection)".
- `tests/test_t43_resume_auto_restore.py::test_resume_banner_instructs_user_to_set_y_variable` — source-text regression pin.

## If a future agent revisits auto-restore of Y variable

**Do not** re-attempt the persist-then-restore approach without re-reading why it failed. The validation-at-restore-time pattern is wrong because Y variable restoration depends on dataset state that isn't loaded until *after* `restore_gui_settings` has run.

The pattern that *would* work is the deferred-apply mirror of `_pending_validation_indices`:

1. Add a `target_column` field to `RunMetadata` (alongside `validation_indices`) — schema-additive, capture from start_run.
2. At resume site (`gui:23219` area), stash `resumed.target_column` into `self._pending_resumed_target_column`.
3. Add `_apply_pending_resumed_target_column` method that validates against `_get_available_target_columns()` and either overrides the auto-default or logs a warning and clears the pending slot.
4. Wire a call to that method into the **three** GUI data-load completion paths after the auto-default `self.target_column.set(...)` calls: `_browse_data_folder` (gui:16156), `_process_combined_file` (gui:16333), and `_auto_detect_columns` (gui:16379). Cast available column names to `str` for comparison so integer pandas headers don't trigger false-fail (DeepSeek MEDIUM).

That's a ~50-100 LOC change across 3 files plus test rework. The user's cost-benefit decision in 2026-05-07 was that banner-only is sufficient and the deferred-apply machinery isn't worth building unless the friction proves real.

## Lesson set

Captured in `docs/SESSION_LOG.md` 2026-05-07 entry. Seven generalisable takeaways; the load-bearing one for future continuation prompts: **continuation prompts that pre-spec implementation should go through a cross-family review pass at filing time, not single-reviewer pre-verification.** The PR #44 prompt was pre-verified by GLM 5.1 alone, and that reviewer's blind spot (function-in-isolation evaluation) propagated into the broken implementation strategy.
