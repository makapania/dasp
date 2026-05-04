# Continuation prompt — 2026-05-08 deferred follow-ups

**Filed:** 2026-05-07 final wrap-up. main at `a8eec70`. Working tree clean.

**For:** the next agent picking up the dasp queue. Could be future-me on the same machine, or fresh-me on a different machine — the project is worked across multiple computers per `CLAUDE.md`.

**Read first:** `docs/PROJECT_STATUS.md` top section (current state) + `docs/SESSION_LOG.md` 2026-05-07 entries (lessons from this batch). The lessons are non-obvious and load-bearing.

---

## Process rule from this session (DO NOT SKIP)

User flagged mid-session that PRs were being merged too freely. **From now on**: open PR → review → wait for explicit "merge it" greenlight from user → then merge. Even for tests-only and docs-only PRs. The cost of pausing for greenlight is low; the cost of merging something the user wanted to look at is real.

Continuation-prompt-specific addendum: any implementation strategy this prompt sketches should go through cross-family review at *pickup* time, not at filing time. The lesson from PR #44 (this session) was that single-reviewer pre-verification of a continuation prompt propagates that reviewer's blind spot into the next agent's implementation. Don't trust this prompt blindly; verify against the current code.

---

## Queue (priority order)

### 1. Kimi MEDIUM — `verbose` in `code_generator._PIPELINE_PARAMS` (methodology change)

**Status:** Deferred from PR #49 review. **Requires user confirmation before implementing.**

**The finding:** `code_generator._PIPELINE_PARAMS` is `{'memory', 'transform_input', 'verbose', 'steps'}`. The strip set is meant to drop sklearn Pipeline kwargs before the bare model gets constructed. But `verbose` is ALSO a model-constructor kwarg for RandomForest, SVM, LightGBM (and others). The codebase's own inline comment at `code_generator.py:1031-1039` acknowledges this — "the same architectural smell as `n_jobs` had". For LightGBM / RandomForest / SVM users who pass an explicit `verbose=N`, the strip drops it; for CatBoost specifically, a `verbose=0` injection at `code_generator.py:974-975` masks the issue.

**Why it's queued, not closed:** the cleanest fix is to remove `verbose` from `_PIPELINE_PARAMS` and handle it contextually like `n_jobs` is handled (PR #33). That's a methodology change to runtime codegen behavior. Per project rules ("Pipeline methodology change? STOP. Confirm with user."), it doesn't belong in a fix-forward PR.

**Three viable approaches** (pick one with user):

a. **Remove from strip set, mirror n_jobs pattern.** Take `verbose` out of `_PIPELINE_PARAMS`, audit downstream consumers for any reliance on the strip happening, add a contextual handling site if needed. Architectural pin can then be extended to ban `verbose` (currently can't because the existing site contains it).

b. **Pin against future sister sites only.** Keep `verbose` in the existing set but extend the architectural pin's failure message to call out: "any NEW `*PIPELINE_PARAMS*` set introduced in the codebase must not contain `verbose` either." Doesn't fix the existing leak but prevents propagation.

c. **Defer indefinitely** if user judges the leak isn't user-visible enough to warrant the methodology change. The CatBoost masking (`verbose=0` injection at line 974-975) plus the inline comment captures the engineering knowledge already.

**Pickup order if approach (a):**
1. Read `code_generator.py:1031-1039` (the existing inline comment) and `code_generator.py:974-975` (CatBoost verbose injection).
2. Grep the codegen for all uses of the verbose strip — confirm the strip is only doing meaningful work for sklearn Pipeline objects, not for bare model construction.
3. Remove `verbose` from `_PIPELINE_PARAMS`, add contextual handling at the bare-model-construction site if a Pipeline-vs-bare detection is needed.
4. Extend `test_no_codebase_pipeline_params_set_contains_n_jobs` (or rename to a more general pattern) to also ban `verbose`. Rename to something like `test_no_codebase_pipeline_params_set_contains_model_kwargs` if generalizing.
5. Add a behavioral pin for `verbose` survival, parametrized on RandomForest + LightGBM + SVM (the models that accept verbose AND were affected by the strip).
6. Cross-family review at pickup — DeepSeek max-thinking + Kimi for sister-site sweep + Codex for call-site verification.

### 2. Toolkit suggestions deferred from PR #51 (low-priority polish)

**Status:** Toolkit pr-test-analyzer flagged these on PR #51 review as rating 3-5. Not blocking; surface only if relevant.

a. **CatBoost `thread_count` survival test** (rating 4, ~10 LOC). Mirror of `test_user_set_n_jobs_survives_codegen` but for CatBoost's `thread_count` kwarg (CatBoost-specific, not n_jobs). Closes a coverage gap left by the parametrize set in PR #49 (XGBoost + RandomForest + LightGBM only — CatBoost uses `thread_count` and was explicitly skipped).

b. **Mock-Tk banner-render test for PR #45** (rating 3). The current source-text pin at `test_resume_banner_instructs_user_to_set_y_variable` asserts the load-bearing phrase exists in the GUI module source. The next-step coverage would be a Tk-mocked test that confirms the banner *actually surfaces* in the resume codepath, not just exists in source. Cost: needs `tkinter.messagebox` monkeypatch + a fixture invoking `_check_for_incomplete_run`. Skip unless the resume flow grows additional banner branches that could route around the existing surface.

c. **Negative-pin asserting runtime conditional source is in generated scripts** (rating 5). Complementary to PR #49 + #51's split-params shape. Asserts the literal `if IMBALANCE_METHOD == 'class_weight':` block is present in the rendered script content — a structural assertion on top of the existing prediction-parity assertion. Catches deletion of the conditional even if numerical parity passes for the wrong reason. ~3 lines per parity row.

### 3. Pre-existing inherited debt (NOT introduced this session)

**Line-number references at `tests/test_t20_saved_model_export_parity.py:1079, 1091`** cite `code_generator.py:988`. Currently accurate (line 988 is `params_full.setdefault('n_jobs', -1)`) but rot-prone per project rule. Defer to a dedicated cleanup PR if the rot surfaces (i.e., if a line-number drift causes the citation to point to the wrong code).

### 4. Polish (lowest priority)

**`inspect.signature(model.fit)` not wrapped in try/except** inside `_apply_class_weight_discriminator_for_rebuilt_model` at `search.py:408+`. Theoretical risk on C-extension fits without proper introspection support; all current sklearn estimators have inspectable signatures. Per project polish-defer rule, defer unless a real failure surfaces.

---

## Session-end state recap

- **main at `a8eec70`** (PR #51 squash merge). Working tree clean except the long-lived `C\357\200\272UsersmsponAppDataLocalTempopencodesearch_t30.py` tempfile leftover from session start (not from this work — leave alone).
- **Seven PRs merged this session:** #45 / #46 / #47 / #48 / #49 / #50 / #51. PR #44 closed without merge (timing bug caught by Codex).
- **Two-phase review pattern earned its slot.** Cross-family LLM panel (Codex + DeepSeek + GLM + Kimi) caught 2 findings; Claude-family toolkit (code-reviewer + pr-test-analyzer + comment-analyzer) caught 1 different finding. Each panel covered angles the other missed. Memory `feedback_review_method_signal.md` updated with calibration evidence.

## Process rules (mandatory, from this session's lessons)

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** Per-fix: matching test file + the relevant regression sweep.
- **Pre-existing flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental.
- **Commit ASAP after implementation.** Long pytest runs / agent dispatch / `git push` can collide with parallel sessions. See memory `feedback_commit_before_yielding_control.md`.
- **Don't use `git add -A` for code commits.** Stage explicitly by file.
- **Don't use `pytest | tail`.** Use `2>&1 | grep -E "passed|failed"` for length-limited output.
- **Comments default to none.** Only when WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`) in production code. Test docstrings can cite review trails (`pr-test-analyzer 2026-05-07 follow-on review`) — that's appropriate test-context history.
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case.**
- **Don't add features beyond what the ticket requires.** Bug fixes / hygiene fine; methodology changes need user confirmation.
- **For comments referencing other code, use stable function-name identifiers, not line numbers.** Line numbers rot.
- **Structural pins should be one-per-consumer, not one-per-producer.** Plus behavioral + architectural layers for fix-of-fixes.
- **Open PR → review → wait for explicit greenlight → merge.** Process correction from this session.
- **Cross-family review at PR-pickup time, not just at PR-filing time.** Single-reviewer pre-verification of a continuation prompt propagates that reviewer's blind spot.

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-fix: tests passing count, review verdicts, deviations from plan, user greenlights received.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on a BLOCKER, file `docs/CONTINUATION_PROMPT_<date>_<topic>.md` for the next agent.
