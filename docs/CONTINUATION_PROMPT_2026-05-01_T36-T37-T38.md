# Overnight Handoff — T-36 / T-37 / T-38 Preprocessing-Discovery Roadmap

**Filed:** 2026-05-01
**Branch:** `feature/T36-autoscale-toggle` (already created, rebased onto current main, NOT yet pushed)
**Estimated total work:** 5-9 days (T-36: 1-2 days, T-37: 3-5 days, T-38: 0.5-1 day)

---

## What this overnight session must accomplish

Implement T-36 end-to-end on its existing branch, ship a PR, and queue T-37 + T-38 for the next session. The user wants:

1. **DeepSeek V4 Pro Max double-checks work as part of the implementation loop** — invoke via `opencode-call` after each major task completes (not just at the end). DeepSeek catches bugs that Claude's self-review misses; the T-11 PR #6 review trail is the precedent.
2. **Once all changes from this conversation are done, Codex goes over the whole thing** — final cross-family review pass before merge. Use `codex-reviewer` agent.

This is the same multi-pass review pattern that successfully shipped T-11 (PR #6) after seven reviewer passes caught real bugs that would otherwise have shipped.

---

## Required reading order at session start

> **Read these IN THIS ORDER. Do not skip.**

1. **`CLAUDE.md`** at the repo root — non-negotiable session protocol. Especially the chemometrics-vs-ML literature rule and the cross-machine doc-update requirement.
2. **`docs/PROJECT_STATUS.md`** — current state of all tickets including the T-36/T-37/T-38 closeout block at the top.
3. **`docs/plans/2026-05-01-T36-autoscale-toggle.md`** — the implementation plan with 18 tasks, three pre-existing bug fixes, and an adjacent one-class metadata gap to bundle.
4. **`docs/plans/2026-05-01-T37-tpe-quick-preprocessing-discovery.md`** — sketch (not for this session, but read so you understand T-36's design choices in context).
5. **`docs/plans/2026-05-01-T38-dead-preprocessing-cleanup.md`** — sketch.

Then `git log --oneline -10` to confirm you're on `feature/T36-autoscale-toggle` and that main is at `da51f60` (other agent's renumbering commit) or later.

---

## Memory rules to observe (from user's auto-memory)

- **Chemometrics conventions ≠ ML conventions.** Don't flag SNV / SG-derivatives / baseline-on-full-data as leakage bugs. Per-spectrum operations are not leakage by chemometrics community convention. Validate against PLS_Toolbox / Unscrambler / SIMCA literature, not sklearn-pipeline-purity.
- **GLM 5.1 must NEVER be dispatched through opencode-go.** It bills against z.ai subscription. Use `llm-call` skill or direct opencode invocation.
- **GUI runs in `.venv312` (Python 3.12).** All install/test in `.venv312`. `.venv311` is removed.
- **Don't run full test suite for small changes.** Use `py_compile` or targeted tests.
- **Use Codex when asked.** Always use `codex-reviewer` skill when user says "codex" — never substitute.
- **CARS-Tree is dasp's invention.** Don't search Li 2009 for canonical "tree-mode CARS"; it doesn't exist.

---

## T-36 implementation strategy

### Phase 1: Read-and-confirm-context (30 min)

Verify the line numbers in the T-36 plan are still accurate. The plan was written 2026-05-01 against commit `da51f60`. If any of the touchpoints have moved, update the plan first, commit the plan update separately, then proceed.

Touchpoints to spot-check (full list in plan):

- `src/spectral_predict/preprocess.py:283` — `build_preprocessing_pipeline` signature
- `src/spectral_predict/search.py:914` — `run_search` signature, `smoothing_polyorder=2`
- `src/spectral_predict/search.py:1830-1849` — smoothing toggle doubling block (template)
- `src/spectral_predict/search.py:4272` — per-model `StandardScaler` for `SCALE_SENSITIVE_MODELS`
- `src/spectral_predict/search.py:4685` — result dict
- `src/spectral_predict/search.py:5040` — `run_one_class_search` signature
- `src/spectral_predict/contamination.py:1109-1112` — validation cache key (BUG #3)
- `src/spectral_predict/unified_bayesian.py:288-376` — `apply_preprocessing` (BUG #1)
- `src/spectral_predict/unified_bayesian.py:857-864` — preprocessing cache key (BUG #2)
- `spectral_predict_gui_optimized.py:11587` — preprocess checkbox area

### Phase 2: Tasks 1-2 (preprocess.py + tests) — ship and review

1. Implement T-36 plan **Tasks 1-2** (preprocess.py + unit tests for autoscale step presence, order, default).
2. Run targeted tests: `python -m pytest tests/test_preprocess_extended.py -v`.
3. Commit: `feat(T-36): add autoscale step to build_preprocessing_pipeline + unit tests`.
4. **DeepSeek review checkpoint** — dispatch `opencode-call` agent with this prompt:
   > Review the diff `git diff HEAD~1..HEAD` on branch `feature/T36-autoscale-toggle`. Focus on: (1) does the autoscale step go in the correct order — after SNV/derivatives, before imbalance? (2) is the parameter wiring through the function signature correct? (3) are tests covering the right invariants? Use DeepSeek V4 Pro Max via the DeepSeek API (not opencode-go). Report findings; no edits.

   Address findings before proceeding.

### Phase 3: Tasks 3-7 (search.py grid path + validation rebuild) — ship and review

5. Implement T-36 plan **Tasks 3-7** (run_search signature, doubling block, propagate to pipeline calls, skip per-model scaler, results column, varsel cache key, validation metrics path).
6. Run: `python -m pytest tests/test_autoscale_grid_doubling.py -v` (write the test as part of task 16).
7. Smoke-test: invoke `run_search` programmatically with autoscale=True on a tiny dataset (use `example/` data); confirm result rows count doubles and `Autoscale` column populates correctly.
8. Commit: `feat(T-36): grid-path autoscale doubling + per-model scaler skip + validation rebuild`.
9. **DeepSeek review checkpoint** with this prompt:
   > Review the diff for T-36 grid-path changes. Critical questions: (1) does the per-model `StandardScaler` skip correctly trigger only when `autoscale=True`? (2) does the doubling block correctly preserve `base_name` so `PreprocessBase` validation rebuild still works? (3) is the cache key in `compute_validation_metrics_for_top_models` updated to include `autoscale` (plan task 7)? Use DeepSeek V4 Pro Max via DeepSeek API.

### Phase 4: Tasks 8-10 (one-class + contamination) — ship and review

10. Implement **Tasks 8-10**. Bundle the adjacent one-class metadata gap (writing baseline_method/smoothing/etc. to one-class result rows — see plan Background).
11. Run targeted one-class tests.
12. Commit: `feat(T-36): one-class autoscale + bundle pre-existing metadata-write gap`.
13. **DeepSeek review checkpoint:**
    > Review T-36 one-class changes. Confirm BUG #3 fix (contamination.py:1109-1112 cache key includes autoscale) is correct and that the metadata-write fix doesn't change semantics for old `.dasp` files (column should be additive, defaults preserved on read).

### Phase 5: Tasks 11-14 (Bayesian path) — the trickiest part

14. Implement **Tasks 11-14**. **The three pre-existing bugs all surface here.** Read the plan's Background section carefully before touching `apply_preprocessing` — the early-return restructure must keep behavior identical for `apply_autoscale=False`.
15. Add a regression test: call `apply_preprocessing` with `apply_autoscale=False` and confirm output is bit-identical to pre-T-36 behavior on a fixture. (This test prevents the restructure from accidentally changing existing trial outputs.)
16. Run the existing Bayesian tests: `python -m pytest tests/test_unified_bayesian*.py tests/test_cv_pls_clamp.py -v`.
17. Commit: `feat(T-36): Bayesian-path autoscale + fix three pre-existing bugs (apply_preprocessing return, two cache keys)`.
18. **DeepSeek review checkpoint — focused on BUG #1 specifically:**
    > Review the `apply_preprocessing()` restructure in `unified_bayesian.py`. Verify: (1) every code path still produces the same output when `apply_autoscale=False` as it did before. (2) When `apply_autoscale=True`, autoscale fires for `raw`, `snv`, `deriv1-4`, `snv_deriv*`, `deriv*_snv` — i.e., it fires for EVERY preprocessing name, not just one. (3) The cache key fix at lines 857-864 includes `apply_autoscale`. Use DeepSeek V4 Pro Max.

### Phase 6: Task 15 (GUI) — ship and review

19. Implement **Task 15**. Manually launch the GUI (`python spectral_predict_gui_optimized.py` from `.venv312`) and verify the checkbox renders in the right place, tooltip shows, and a small grid run with autoscale=True actually doubles the result rows.
20. Commit: `feat(T-36): GUI autoscale checkbox + wire-through to all four search paths`.
21. **DeepSeek review checkpoint:**
    > Review the GUI wire-through for T-36 autoscale. Confirm: (1) `self.use_autoscale.get()` is passed to all four call sites: `run_search`, `run_one_class_search`, both `run_unified_bayesian` calls. (2) The state is preserved on `_collect_*` config dicts if such a save/restore mechanism exists. (3) The checkbox is in the right tab/frame.

### Phase 7: Tasks 16-18 (integration tests + validation note + PR)

22. Implement **Tasks 16-17**. Write the integration tests, write `docs/bugfix_validation/T36_autoscale_toggle.md`.
23. Run full sweep: `python -m pytest -x -q` (in `.venv312`). Address any failures before proceeding.
24. **Codex final review** — the user explicitly requested this:
    > Use the `codex-reviewer` agent (NOT opencode-call) to do a final cross-family review of the entire T-36 PR. Diff against `main`. Use this prompt:
    >
    > "Final pre-merge review of T-36 (autoscale toggle). Branch `feature/T36-autoscale-toggle`. Plan at `docs/plans/2026-05-01-T36-autoscale-toggle.md`. Diff against main. Focus on: (1) the three pre-existing bug fixes in `apply_preprocessing` and the two cache keys — verify each is correct and complete. (2) the per-model `StandardScaler` skip logic — is the gating correct? (3) are there any search-path call sites the plan missed? Especially: NSGA-II is documented as out-of-scope, but `run_bayesian_search` (search.py:3153) — is it really only test-only? (4) Cross-machine `.dasp` roundtrip — old files without `Autoscale` column must still load. (5) GUI wire-through completeness."
25. Address Codex findings.
26. Open PR with `gh pr create`. Title: `feat(T-36): autoscale (UV scaling) preprocessing toggle`. Body should reference the plan and the three pre-existing bug fixes explicitly.

### Phase 8: Update PROJECT_STATUS.md and SESSION_LOG.md

27. Append T-36 ship entry to `docs/SESSION_LOG.md`. Document the bug-fix-list (the three pre-existing bugs that would have shipped).
28. Update `docs/PROJECT_STATUS.md` top section: T-36 status changes from `PLAN_FILED` to `IN_PR` with PR number.
29. Commit + push the doc updates separately.

### Phase 9: Hand off to T-37 / T-38

30. Mark in PROJECT_STATUS.md that T-37 is now unblocked.
31. T-38 has three modules deletable immediately (zero callers) — note that this could be a small standalone PR done in parallel with T-37 work.

---

## Review-pass discipline

Echo the T-11 review pattern:

1. **Per-phase DeepSeek review** (4-5 dispatches over the implementation) — catches in-flight bugs cheaply.
2. **Final Codex review** — cross-family second-opinion before merge.
3. **Document review trail** in `docs/bugfix_validation/T36_autoscale_toggle.md` with each pass's findings.

If a DeepSeek pass finds something serious, stop, fix, re-review BEFORE moving to the next phase. Do not stack issues across phases.

---

## What success looks like at session end

- T-36 PR open, all reviewer-found issues addressed, full test suite green.
- `docs/PROJECT_STATUS.md` updated (T-36 → IN_PR).
- `docs/SESSION_LOG.md` updated with the bug-fix list.
- `docs/bugfix_validation/T36_autoscale_toggle.md` written with review trail.
- T-37 and T-38 left untouched (their plans already filed; this session is T-36 only).

## What failure modes to watch for

- **Don't push without explicit user permission.** Commit, but PR creation needs user approval per CLAUDE.md / project memory rules.
- **Don't merge.** PR opens, reviewer trail completes, but merge is the user's call.
- **Don't expand scope.** If a sixth bug surfaces during implementation, file it as a follow-up ticket and fix it in a separate branch. T-36 is autoscale + the three documented bugs + the one bundled metadata gap. Nothing else.
- **Don't optimize.** Tree-model skip in doubling loop is Phase 1.5 and explicitly out of scope.
- **Don't update CLAUDE.md or memory files** unless the user explicitly directs.

## Out-of-scope reminders

- NSGA-II support (separate ticket; nsga2_search.py has its own `build_preprocessing_pipeline` calls)
- `run_bayesian_search` (search.py:3153) — test-only, leave alone, document as out-of-scope in the plan if not already
- Tree-model skip in doubling loop (Phase 1.5)
- Smart-preprocessing-discovery integration (subsumed by T-37)
- Adding autoscale to NSGA-II decode/result-row paths
