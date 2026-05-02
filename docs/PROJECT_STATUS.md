# Project Status

> **Last updated:** 2026-05-02 (T-47 MERGED to main + T-46 IMPLEMENTED) —
>
> **T-47 MERGED via PR #11 at merge commit `a29ad2d`.** Default-flip + DeepSeek follow-ups all rebased onto main; both T-47 commits + branch deleted from origin and locally. The dataclass-vs-function default symmetry + the deliberate "never"-kept legacy-sidecar/corruption-coercion fallbacks are now load-bearing in 3 regression tests.
>
> **T-46 IMPLEMENTED on `fix/T46-wal-pragma-surface-return` (off post-T-47 main).** Surgical fix to surface `_apply_wal_pragmas` return value at both call sites — pre-T-46 silent fallback to DELETE journal mode now signals through the right channel for each site. Site 1 (`run_unified_bayesian:2117`, always-on/auto-migrated SQLite path): captured return + `progress_callback({"t41_decision": "wal_rejected", ...})` event so the GUI can display a user-visible warning. Site 2 (`_migrate_study_to_sqlite:1732`, auto-migration path): captured return + contextual `logger.warning("T-46: WAL rejected during auto-migration to ...")` — the inner helper already logs raw rejection, the wrapper log adds lifecycle-phase context so future log readers can pinpoint when in the run lifecycle the rejection occurred without correlating timestamps. 1 new regression test (`test_wal_rejection_surfaces_via_progress_callback`) monkeypatches `_apply_wal_pragmas` to return `False` and asserts a structured `t41_decision: "wal_rejected"` event lands in the captured callback. 262 + 1 skipped across the consolidated sweep. **OneDrive/Dropbox-synced data dirs no longer silently halve Bayesian throughput** (the failure mode the ticket was filed against).
>
> ---
>
> **Previously:** 2026-05-02 (PR #10 MERGED + T-47 IMPLEMENTED) —
>
> **PR #10 (consolidated T-43 + T-42 + T-38 + T-49) MERGED via merge commit `f063d9e`.** 17 commits flattened onto main. Final fix-of-fixes commit `30fa673` closed the Codex post-PR MEDIUM (partial-mutation in `_apply_pending_validation_indices` when X/y indices disagreed on a captured label — silent skip of validation metrics). Cross-family review verdicts on the PR: DeepSeek V4 Pro Max READY_TO_MERGE (5 MEDIUM, 4 LOW deferred); Codex CLI READY_WITH_REVISIONS (1 MEDIUM closed). Per-ticket branches deleted on origin + locally: `fix/T43-resume-auto-restore-settings`, `fix/T42-write-path-plumbing-approach-c`, `fix/T38-dead-preprocessing-cleanup`, `fix/bayesian-resume-and-cleanup`.
>
> **T-47 IMPLEMENTED on `fix/T47-bayesian-persistence-default-auto` (off post-PR-10 main).** One-line behavioral flip: `bayesian_persistence_mode` field default flips from `'never'` → `'auto'` at both surfaces — `RunMetadata` dataclass field default + `start_run()` function default — keeping the two consistent. Legacy-sidecar fallback in `from_dict` (line 162) and corruption-coercion path (line 168) deliberately stay at `'never'` — those are safety nets for malformed input, not "the field default for unspecified input". Justified by the post-T-42 baseline benchmark: PLS 0.69×, Ridge 1.06×, RandomForest 0.25× (caching), LightGBM 0.93×, XGBoost 1.01× — all well below the plan's 1.15×/1.30× targets. Crash-resume now on by default; users who want the prior in-memory-only behavior set the radio to "Never on". 3 new regression tests (`test_default_persistence_mode_is_auto`, `test_dataclass_default_persistence_mode_matches_start_run`, `test_legacy_sidecar_without_persistence_mode_defaults_to_never` — last one added post-DeepSeek MEDIUM, pinning the deliberate kept-fallback half of the two-pair contract); 261 + 1 skipped across the consolidated sweep. DeepSeek V4 Pro Max review (READY_WITH_REVISIONS): 1 MEDIUM (missing absent-key fallback test) closed; 2 LOWs (plan-doc drift at step 3, GUI hasattr safety-net comment) closed in the same commit.
>
> ---
>
> **Previously:** 2026-05-02 (T-43 + T-42 + T-38 + T-49 CONSOLIDATED — single PR shipped) —
>
> **All four tickets (T-43 + T-42 + T-38 + T-49) land on `fix/bayesian-resume-and-cleanup` (tip `6465306`, 14 commits off main).** User decision after the parallel pr-review-toolkit pass: ship as one PR rather than three. T-49 was originally filed as a follow-up but the user (correctly) flagged it as a correctness blocker for the resume flow and folded it into the same PR. The per-ticket branches (`fix/T43-resume-auto-restore-settings`, `fix/T42-write-path-plumbing-approach-c`, `fix/T38-dead-preprocessing-cleanup`) remain on origin as reference and can be deleted once the consolidated PR merges.
>
> **What's in the PR:**
> 1. **T-43 (auto-restore GUI settings on resume):** ~91 analysis-defining Tk vars captured at `start_run` time and restored when the user accepts the resume banner — preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, tier, CV strategy, imbalance method, one-class hyperparameters, the 6 `bayes_*` Bayesian-only controls, and the 5 external-validation-set Tk vars. New `src/spectral_predict/run_gui_settings.py` module + extended `RunMetadata.gui_settings: dict | None` with `dataclasses.fields()` filter + type-guard for malformed sidecars + `RestoreReport` with `restored / skipped_unknown / skipped_no_var / errors` buckets and `fully_succeeded` property. **Plus T-49 (folded in):** validation set indices persisted alongside `gui_settings` in the sidecar so non-deterministic algorithms (Random, Manual) and hand-picked partitions resume without silent leakage. New `_apply_pending_validation_indices` GUI helper re-slices `self.X` / `self.y` after the fingerprint check passes; respects the user's manual re-creation if they did one before clicking Run Analysis; defense-in-depth on missing labels.
> 2. **T-42 (Bayesian write-path Approach C):** hoisted three constant keys (`cv_strategy`, `cv_n_repeats`, `early_stopping_rounds`) from per-trial to per-study scope. Two were dead writes (read by nobody); the third is now read once outside the trial loop with a trial-level fallback for legacy/migrated studies. Hoist guarded against silent overwrite on resume — disagreeing values keep the stored value and log a warning. `_supports_early_stopping(model_name)` helper de-duplicates the trial-time and study-hoist gates. Per-trial set_user_attr count: PLS regression 30→27, Ridge regression 30→27, PLS-DA classification 42→39.
> 3. **T-38 (dead preprocessing cleanup):** deleted `learned_preprocessing.py` (775 LOC, zero callers) and `ensemble_preprocessing.py` (701 LOC, only referenced by a dead `HAS_ENSEMBLE_PREPROCESSING` flag). Did NOT delete `preprocessing_wrapper.py` (still imported by `ensemble.py:18`) or `tests/test_ensemble_preprocessing.py` (despite the matching name — actually tests `preprocessing_wrapper` and `ensemble`, both alive).
>
> **Surprise finding documented during T-42:** post-T-41 baseline benchmark already met T-42's "definition of done" target — PLS 0.69×, Ridge 1.06×, LightGBM 0.93×, XGBoost 1.01× SQLite-WAL ratios, all well below the plan's 1.15× / 1.30× targets. T-41's WAL pragmas + 30s `busy_timeout` collapsed per-trial overhead to near-noise; the plan's 1.36× / 1.89× ratios were PRE-T-41. Flipping T-41's default from `'never'` to `'auto'` is now justified by bench data alone — separate trivial follow-up ticket.
>
> **Review trail (7 review passes + 1 user catch, cross-family):**
> 1. **T-43** post-`f934c81` — DeepSeek V4 Pro Max: READY_WITH_REVISIONS, 2 HIGH + 1 MEDIUM closed in `3f071b3`.
> 2. **T-43** post-`487b1d9` — Codex CLI: BLOCKERS, 1 HIGH (`bayes_*` whitelist gap) + 1 MEDIUM (`skipped_no_var` not surfaced) + 1 LOW (docstring drift) closed in `2568be8`.
> 3. **T-42** post-`ab9d00f` — DeepSeek V4 Pro Max: READY_WITH_REVISIONS, 1 HIGH (stale `test_cv_strategy.py` assertion) + 1 MEDIUM (copy_study migration test gap) + 1 LOW closed in `e1daaa6`.
> 4. **T-42** post-`ab9d00f` — Codex CLI: READY_TO_MERGE; verified empirically that `optuna.copy_study` preserves user_attrs.
> 5. **T-38** post-`2eaef4f` — DeepSeek V4 Pro Max: READY_TO_MERGE, auto-applied 2 doc fixes in `c0b4c20`.
> 6. **T-38** post-`2eaef4f` — Codex CLI: READY_WITH_REVISIONS, 1 LOW (stale PROJECT_STATUS sentence) closed in `de55f32`.
> 7. **Stacked diff** post-`de55f32` — pr-review-toolkit parallel-5 (code-reviewer / silent-failure-hunter / type-design-analyzer / pr-test-analyzer / comment-analyzer): 3 HIGH + 3 MEDIUM closed in `fe5e25d`.
> 8. **User catch** during review wrap-up: validation partition not maintained on resume — silent leakage on Random/Manual algorithms, silent skip if user forgets the button on deterministic algorithms. Originally filed as T-49 follow-up; user (correctly) called it a correctness blocker. Folded in at `6465306` with 8 new tests.
>
> **Test coverage:** **257/257 + 1 skipped** across `test_t41_bayesian_sqlite_auto_calculator`, `test_t42_write_path_plumbing` (8 tests), `test_t43_resume_auto_restore` (33 tests including the 8 new T-49 cases), `test_run_state`, `test_cv_strategy`, `test_unified_bayesian_baseline`, `test_autoscale_bayesian`, `test_cv_pls_clamp`, `test_ensemble_preprocessing`, `test_ensemble_integration`. Sanity-imported every `spectral_predict.*` submodule via `pkgutil.walk_packages` post-T-38 deletion — clean.
>
> **Follow-up tickets filed (NOT in this PR):**
> - **T-44** (`docs/plans/2026-05-02-T44-n-trials-var-typo.md`): `n_trials_var` typo at `gui:25123`; hasattr-guarded so silent-no-op. T-43 captures `n_unified_trials` correctly so resume-restore is unaffected; only the sidecar's `n_trials_per_model` field is wrong. ~5 LOC.
> - **T-45** (`docs/plans/2026-05-02-T45-logger-warning-visibility-bundled-gui.md`): `logger.warning` calls invisible in the bundled GUI (no `logging.basicConfig` anywhere). Silent in production for sidecar corruption, WAL rejection, capture-time Tk failures, and the new T-42 hoist mismatch warning. File-handler at app startup is the cheap fix.
> - **T-46** (`docs/plans/2026-05-02-T46-wal-pragma-return-not-surfaced.md`): `_apply_wal_pragmas` return value discarded at both call sites despite "caller should surface that" docstring. OneDrive/Dropbox-synced data dirs silently fall back to DELETE journal mode.
> - **T-47** (`docs/plans/2026-05-02-T47-flip-bayesian-persistence-default-to-auto.md`): one-liner default flip from `'never'` to `'auto'`. Justified by the T-42 baseline benchmark numbers (XGBoost 1.01×, LightGBM 0.93× — well below 1.15× target). Gated on this PR merging first.
> - **T-48** (`docs/plans/2026-05-02-T48-real-tk-integration-tests-for-resume.md`): real-Tk integration tests for the resume flow. Pins Tcl set/get-verify behavior + order-dependent restore-then-override contract on the actual method instead of `_FakeGUI` shims. Belt-and-braces; current shim coverage is sound at the unit level.
>
> **Audit trail:**
> - T-42: `docs/bugfix_validation/T42_write_path_plumbing_approach_c.md`.
> - T-43 + T-38 + T-49: code + tests + this PROJECT_STATUS entry serve as audit trail. T-49 plan doc (`docs/plans/2026-05-02-T49-...md`) is marked IMPLEMENTED with implementation summary.
>
> ---
>
> **Previously: per-ticket framing (now consolidated):**
>
> **T-38 dead preprocessing module cleanup — IMPLEMENTED on `fix/T38-dead-preprocessing-cleanup` (1 commit, branched off T-42 tip).** Deleted `src/spectral_predict/learned_preprocessing.py` (775 LOC, zero callers — imports torch but never wired into the GUI) and `src/spectral_predict/ensemble_preprocessing.py` (701 LOC, only referenced by a dead `HAS_ENSEMBLE_PREPROCESSING` flag in the GUI try/except block at `gui:236-241`). Removed the dead flag too. **Did NOT delete `preprocessing_wrapper.py`** (still imported by `ensemble.py:18`) — plan was corrected on this point during T-37 review. Also did NOT delete `tests/test_ensemble_preprocessing.py` despite its matching name — that file actually tests `preprocessing_wrapper` and `ensemble`, both alive (the test name is misleading). Updated build-time torch-exclusion comments + `BUNDLED_APP_BUILD_GUIDE_PY312.md` Step 3 rationale. 244/244 across the regression sweep + clean `pkgutil.walk_packages` sanity-import of all `spectral_predict.*` submodules.
>
> ---
>
> **T-42 Bayesian write-path Approach C — IMPLEMENTED on `fix/T42-write-path-plumbing-approach-c` (tip `5670c26`, 3 commits, branched off T-43 tip).** Hoisted three constant `set_user_attr` keys (`cv_strategy`, `cv_n_repeats`, `early_stopping_rounds`) from per-trial to per-study scope in `unified_bayesian.py`. Two were dead writes (read by nobody — `convert_study_to_dataframe` takes them as function parameters); the third is now read once outside the trial loop with a trial-level fallback for legacy/migrated studies. Per-trial set_user_attr count: PLS regression 30→27, Ridge regression 30→27, PLS-DA classification 42→39 (each model saves 3 writes/trial).
>
> **Surprise empirical finding (changes T-42's framing):** the post-T-41 baseline benchmark (`tests/_bench_bayesian_per_model.py`, n_trials=10, synthetic n=100, n_features=200) **already met T-42's "definition of done" target**: PLS 0.69×, Ridge 1.06×, RandomForest 0.25× (caching), LightGBM 0.93×, XGBoost 1.01× — all well below the plan's 1.15× / 1.30× targets. T-41's WAL pragmas + 30s `busy_timeout` collapsed per-trial overhead to near-noise; the plan's 1.36× / 1.89× ratios were PRE-T-41 measurements. **Flipping T-41's default from `'never'` to `'auto'` is now justified by bench data alone — file as a separate trivial follow-up ticket.** Approaches A and D no longer needed.
>
> **T-42 review trail (2 passes, cross-family):**
> 1. DeepSeek V4 Pro Max post-`ab9d00f`: READY_WITH_REVISIONS — 1 HIGH (test_cv_strategy.py still asserted old per-trial contract; audit doc's "132/132 green" claim was wrong because that file was excluded from the curated sweep) + 1 MEDIUM (no test that user_attrs survive copy_study migration — same trap class as T-41's `load_if_exists+sampler` silent-ignore) + 1 LOW (XGBoost trial-absence assertion missing) all closed in `e1daaa6`. Wider sweep now 225/225 green.
> 2. Codex CLI post-`ab9d00f` (paired with T-43 review per overnight protocol): READY_TO_MERGE. Verified empirically against installed Optuna source that `copy_study()` does preserve user_attrs (study.py:1567-1568); confirmed ordering safety (study.set_user_attr lands before progress_wrapper closure runs); confirmed no other readers of the three keys remain in `trial.user_attrs`.
>
> **T-43 resume auto-restore GUI settings — IMPLEMENTED on `fix/T43-resume-auto-restore-settings` (tip `2568be8`, 3 commits).** Closes the user-reported gap from T-41 verification: previously the user had to manually recreate every preprocessing/model setting before the resumed SQLite study would actually pick up trial-level state. Now ~86 analysis-defining Tk vars (preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, tier, CV strategy, imbalance method, one-class hyperparameters, **plus the 6 `bayes_*` Bayesian-only baseline/smoothing/region/UVE controls Codex flagged as missing**) are auto-restored from the sidecar.
>
> **Architecture:** `RunMetadata.gui_settings: dict | None` (extended schema, backward-compat via `dataclasses.fields()` filter + type-guard). New `src/spectral_predict/run_gui_settings.py` module with `CAPTURABLE_SETTINGS` whitelist + `capture_gui_settings(gui)` + `restore_gui_settings(gui, dict) -> RestoreReport` + `summarize_gui_settings()`. GUI captures at `start_run` time and restores BEFORE the existing 'always' persistence override (order matters — comment + new test document this).
>
> **T-43 review trail (2 passes, cross-family):**
> 1. DeepSeek V4 Pro Max post-`f934c81`: READY_WITH_REVISIONS — 2 HIGH (banner-lies-on-errors + missing type-guard on gui_settings) + 1 MEDIUM (Tcl quirk: `IntVar.set("abc")` doesn't raise) all closed in `3f071b3`.
> 2. Codex CLI post-`487b1d9`: BLOCKERS — 1 HIGH (the 6 `bayes_*` Tk vars at gui:27554-27570 weren't in the whitelist; resumed Bayesian runs would silently use defaults despite the banner claiming "settings restored" — the exact silent-failure trap T-43 was meant to close) + 1 MEDIUM (skipped_no_var bucket not surfaced in resume log) + 1 LOW (docstring drift) all closed in `2568be8`.
>
> **Pre-existing bug surfaced for follow-up (NOT a T-43 regression):** `spectral_predict_gui_optimized.py:25123` reads `self.n_trials_var.get()` guarded by `hasattr`; the actual var is `self.n_unified_trials`, so `n_trials_per_model` is always None in `start_run` metadata. T-43 itself captures `n_unified_trials` correctly via the whitelist. To file as a separate ticket.
>
> **Test coverage:** 23/23 T-43 tests + 6/6 T-42 tests + the 225/225 wider regression sweep = comprehensive coverage of the resume-auto-restore + write-path-cleanup surface.
>
> **Audit trail:**
> - T-42: `docs/bugfix_validation/T42_write_path_plumbing_approach_c.md`.
> - T-43: code + tests speak for themselves; no separate audit doc.
>
> ---
>
> **Previously:** 2026-05-02 (T-41 MERGED to main after 5-pass review trail) —
>
> **T-43 resume auto-restore GUI settings — IMPLEMENTED on `fix/T43-resume-auto-restore-settings` (tip `3f071b3`, 2 commits).** Closes the user-reported gap from T-41 verification: previously the user had to manually recreate every preprocessing/model setting before the resumed SQLite study would actually pick up trial-level state. Now, when the user accepts the resume banner, ~80 analysis-defining Tk vars (preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, tier, CV strategy, imbalance method, one-class hyperparameters) are auto-restored from the sidecar — they just click Run Analysis.
>
> **Architecture:** `RunMetadata.gui_settings: dict | None` (extended schema, backward-compat via `dataclasses.fields()` filter). New `src/spectral_predict/run_gui_settings.py` module with `CAPTURABLE_SETTINGS` whitelist + `capture_gui_settings(gui)` + `restore_gui_settings(gui, dict) -> RestoreReport` + `summarize_gui_settings()` for the resume prompt. GUI captures at `start_run` time and restores BEFORE the existing 'always' persistence override (order matters — comment in code).
>
> **Review trail (1 pass):**
> 1. DeepSeek V4 Pro Max post-`f934c81`: READY_WITH_REVISIONS — 2 HIGH (banner-lies-on-errors + missing type-guard on `gui_settings`) + 1 MEDIUM (Tcl quirk: `IntVar.set("abc")` doesn't raise) all closed in `3f071b3`. 5 LOW deferred (whitelist debt for variable_penalty/gap_penalty, pre-existing `n_trials_var` typo at gui:25123, ga_preprocess settings on the T-38 deletion path).
>
> **Pre-existing bug surfaced for follow-up (NOT a T-43 regression):** `spectral_predict_gui_optimized.py:25123` reads `self.n_trials_var.get()` guarded by `hasattr`; the actual var is `self.n_unified_trials`, so `n_trials_per_model` is always None in `start_run` metadata. T-43 itself captures `n_unified_trials` correctly via the whitelist. To file as a separate ticket.
>
> **Test coverage:** 19/19 T-43 tests + 29 run_state + 23 T-41 = 71/71 passing (~18s on `.venv312`).
>
> **Audit trail:** code + tests speak for themselves; no `bugfix_validation/` doc filed for this small ticket.
>
> ---
>
> **Previously:** 2026-05-02 (T-41 MERGED to main after 5-pass review trail) —
>
> **T-41 Bayesian SQLite auto-calculator + WAL mode — MERGED via PR #9.** Five rebase commits on main: `cd406f0` (feat) → `e99d35a` (DeepSeek HIGH/MEDIUM + resume bug + default 'auto') → `2087134` (docs) → `081ad6a` (pr-review-toolkit findings: silent failures + Literal validation + cleanup gates) → `f745d37` (DeepSeek Finding 1: multi-model SQLite collateral deletion via `optuna.delete_study` instead of `Path.unlink`).
>
> **Default flipped from plan's `'auto'`→`'never'`→`'auto'` during review:** User originally requested `'never'` ("99.99% of the time this would not be used"), then flipped back to `'auto'` for resume-path troubleshooting. Resume reliability now confirmed end-to-end.
>
> **Review trail (8 passes total):**
> 1. DeepSeek V4 Pro Max pre-implementation: 6 plan findings, all applied
> 2. DeepSeek V4 Pro Max post-`cd406f0`: 2 HIGH (study_ref doesn't redirect Optuna writes; cb_study.stop crash) + 1 MEDIUM + 1 LOW, all closed
> 3. GLM 5.1 post-`e99d35a`: READY_TO_MERGE; 1 MEDIUM (integration test gap) deferred
> 4. Codex post-`e99d35a`: READY_WITH_REVISIONS; 1 MEDIUM (phantom resume prompt for in-memory crashes) folded into T-43, 1 LOW fixed
> 5. pr-review-toolkit (5 specialist agents in parallel): 1 type-design HIGH + 4 silent-failure HIGHs + 4 MEDIUMs + comment-quality polish — all closed in `081ad6a`
> 6. DeepSeek V4 Pro Max post-`081ad6a`: 1 MEDIUM (multi-model collateral deletion) + 2 LOWs, MEDIUM closed in `f745d37`
> 7. User runtime verification: resume banner → recreate settings → click Run Analysis → SQLite study picks up trial-level state ✓
>
> **Type design:** `PersistenceMode = Literal["auto","always","never"]` shared between `RunMetadata` and `run_unified_bayesian`; validated at all three call boundaries (`__post_init__`, `start_run`, `run_unified_bayesian` entry); `from_dict` coerces corrupted sidecar values to `'never'` with warning instead of crashing the resume flow.
>
> **Architecture (final):** auto-decision callback aborts in-memory `optimize()` via `cb_study.stop()`; outer scope detects `_auto_migrated` and restarts `optimize()` on the migrated SQLite-backed study so trials 11..N persist directly. Multi-model migration failures use `optuna.delete_study` (study-scoped) not `Path.unlink` (file-scoped) so prior models' trials survive a later model's failed migration on the shared SQLite.
>
> **Test coverage:** 127/127 passing across `test_t41_bayesian_sqlite_auto_calculator` (23) + `test_run_state` (29) + `test_unified_bayesian_baseline` (36) + `test_autoscale_bayesian` (14) + `test_cv_pls_clamp` (25). New test classes added in `081ad6a`: `TestPersistenceModeValidation`, `TestCleanupByTrialCount`, `TestWALPragmaReturnValue`, `TestMigrationOrphanCleanup`.
>
> **Performance:**
> - `'never'` mode: pre-T-11 in-memory speed (~1.0× ratio). Default for users who don't need crash-resume.
> - `'auto'` mode: PLS/Ridge stay in-memory (median fit < 1s); LightGBM/XGBoost migrate at trial 11. XGBoost runs at ~1.36× WAL overhead; that's the residual `set_user_attr` cost which T-42 (filed) targets to push toward ~1.15×.
> - `'always'` mode: SQLite+WAL from trial 0; PLS pays ~8× cost in exchange for universal crash-resume.
>
> **Audit trail:** `docs/bugfix_validation/T41_bayesian_sqlite_auto_calculator.md`.
>
> **Open follow-ups (filed):**
> - **T-42** (`docs/plans/2026-05-01-T42-bayesian-write-path-plumbing.md`): hoist constant `set_user_attr` keys to `study.set_user_attr` and batch per-trial keys; brings XGBoost overhead toward 1.15× as prerequisite for promoting `'auto'` from troubleshooting-default to no-regret-default.
> - **T-43** (`docs/plans/2026-05-02-T43-resume-auto-restore-settings.md`): auto-restore GUI settings on resume so users don't have to manually recreate preprocessing/model settings before clicking Run Analysis.
>
> **Previously:** 2026-05-01 (T-36 + T-37 BOTH MERGED to main after final cross-family adversarial sweep) —
>
> **PR #8 (T-36) MERGED** at `ff03193` (rebase merge: `bc989f2` + `ffd6451` + `ff03193`). Final adversarial sweep verdict: READY_TO_MERGE with zero new findings from both reviewers.
>
> **PR #7 (T-37) MERGED** at `9430d5c` (rebase merge — single squashed commit since the local merge-commit-of-T-36 had to be flattened after T-36 rebased onto main with new commit hashes).
>
> **Cross-family review trail:** Two reviewers via `opencode-call` with full repo access — DeepSeek V4 Pro Max (DeepSeek API direct, max thinking) + MiMo 2.5 Pro Max (opencode-go subscription) — dispatched in parallel. **Both converged independently on the same structural finding:** T-37 was branched off T-36 at `1f49d73` BEFORE T-36's two post-merge review fix commits landed, so T-37 inherited the silent-mismatch bugs T-36 already fixed (3 HIGH + 4 MEDIUM + 2 LOW per MiMo's tally; equivalent ratings from DeepSeek). Resolution applied via merge of T-36 fixes into T-37 before final ship. 123 T-37+autoscale tests + 27 code-export tests all green post-merge.
>
> **Lesson logged in SESSION_LOG.md:** in-band PR review tools (Codex, gemini-code-assist, CodeRabbit, pr-review-toolkit) operate on diff-against-main and are structurally blind to sister-branch inheritance gaps. A final cross-family sweep with full repo access is the right gate for this class of bug. Convergence between two family-orthogonal models is high-signal enough to act on without further verification beyond the initial sanity check.
>
> **Architectural correction logged this session:** I initially told the user (incorrectly) that the grid-search path applied autoscale per-fold via the sklearn Pipeline mechanic, while the Bayesian path applied it pre-CV global — claiming this was a real divergence. **That was wrong.** The grid path uses `skip_spectral_preprocessing=True` at every call site of `_run_single_config`, which means `pipe_steps = []` for the inner CV loop (search.py:4243-4259); spectral preprocessing including the autoscale `StandardScaler` is `fit_transform`'d once on full training data at search.py:2061, and CV runs on the already-preprocessed `X_for_models`. So **both paths apply autoscale pre-CV global, by design, matching PLS_Toolbox / Unscrambler chemometrics convention.** No grid/Bayesian divergence exists. The misleading line 4270 path (`else` branch with full Pipeline construction) is reachable in principle but not exercised by the live grid call sites. See SESSION_LOG.md for the trace details so this doesn't get re-discovered.
>
> **Deferred deliberately (chemometrics-convention calls, NOT bugs):**
> - **Gemini "Bayesian autoscale leakage" finding** — autoscale fits on full training data before CV in both paths. By ML orthodoxy this is mild leakage; by chemometrics convention (PLS_Toolbox / Unscrambler / SIMCA-P / Pirouette default behavior) it is correct. Project follows chemometrics convention. Bias is small for stable NIR data and dominated by sample-population shift risks that KS-style train/test splits address.
> - **Gemini "code_generator export uses fit_transform not transform" finding** — entangled with the Bayesian leakage architectural rewrite above; not applicable until/unless that decision changes.
>
> **Doc nits + plan corrections done this commit (T-37 branch):**
> - `docs/bugfix_validation/T37_tpe_preprocessing_discovery.md:97` — Phase 7 row was marked `_pending_` after commit `03d95cb` already landed; updated to landed-commit reference. Also added Phase 8 row for `6a0eb28`.
> - `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md:74-82, 176, 235, 250` — internal contradiction between architecture (10 trials, median fit time) and examples / tasks / acceptance criteria (5 trials, mean fit time). Architecture is authoritative per the DeepSeek review. All inconsistent references updated to 10 trials / median to match.
> - `docs/plans/2026-05-01-T36-autoscale-toggle.md:210` — fixed nonexistent `T34_autoscale_toggle.md` reference → `T36_autoscale_toggle.md`.
> - `docs/plans/2026-05-01-T38-dead-preprocessing-cleanup.md:27-32` — corrected the false claim that `preprocessing_wrapper.py` is doubly orphaned. It is actively imported by `ensemble.py:18` and instantiated at `ensemble.py:1309`, exercised by 2 test files. Plan changed to NOT delete it.
>
> **Open follow-ups (NOT done — not "definitely broken"):**
> - pr-review-toolkit T-37 M1: missing regression test for the new mutual-exclusion `ValueError`. Coverage gap, not a bug.
> - pr-review-toolkit T-37 M2: bare `except Exception` in `_quick_evaluate` could be narrowed to `(ImportError, ModuleNotFoundError)`. Defensive over-catch, not a bug.
> - pr-review-toolkit T-36 M2: `refined_config` doesn't carry an `'autoscale'` key, so the GUI fallback chain always falls through to `selected_model_config`. Works in practice today (selected_model_config always has 'Autoscale').
> - pr-review-toolkit T-36 L1: `contamination.py` cache key omits `baseline_params`. Latent — no current producer emits per-row baseline_params.
> - `tests/test_autoscale_bayesian.py:169` cache-key assertion uses bare substring `'apply_autoscale' in src` rather than asserting it appears specifically inside the `cache_key` tuple. Test passes today; would not catch a regression that drops `apply_autoscale` from the tuple but keeps it elsewhere in the function. Test-quality issue, not a bug.
>
> **Previously:** 2026-05-01 (T-37 Phase 7 fix commit `03d95cb` — F2, F6, F9, F10, F12 all closed) —
>
> **DeepSeek V4 Pro Max review trail (this session):**
>
> - **T-41 plan pre-implementation review** (read-only) — verdict: READY_WITH_PLAN_REVISIONS, 6 findings (2 HIGH, 2 MEDIUM, 4 LOW). All applied in commit `7861a81`. Critical empirical catch: DeepSeek tested `optuna.copy_study` + `optuna.load_study(sampler=TPESampler(...))` mid-run on Optuna 4.8 and confirmed TPE state is preserved. Key trap caught: `create_study(load_if_exists=True, sampler=...)` SILENTLY IGNORES the sampler kwarg on existing studies — the migration helper now uses `load_study(sampler=...)` explicitly. Plan also revised to use 10-trial median window (was 5-trial mean), corrected WAL durability claim, moved WAL pragma application from `start_run` to migration time (file doesn't exist until then), added stale-sidecar cleanup, added 4 missing tests.
>
> - **T-37 Phase 1 review of commit `ef5f61e`** by **GLM 5.1** (read-only, attribution corrected by user 2026-05-01 PM) — verdict: READY_WITH_REVISIONS. Re-prioritized findings against initial draft:
>   - **MEDIUM F2:** `_tpe_baseline_params` never attached to output config dicts. Currently safe by coincidence (TPE defaults match `preprocess.py` defaults), fragile if either changes.
>   - **MEDIUM F6:** `_quick_evaluate` PLS exception fallback uses `neg_root_mean_squared_error` regardless of task_type — wrong metric for classification/one_class when LightGBM is unavailable.
>   - **MEDIUM F9:** Mutual exclusion test only checks `results is not None`, doesn't verify TPE actually ran vs smart_preprocess fallback. False confidence.
>   - **MEDIUM F12:** Test coverage gaps — no `_apply_full_preprocessing` unit tests, no `_quick_evaluate` direct tests, no empty-TPE-result fallback test, no roundtrip-through-`build_preprocessing_pipeline` verification.
>   - **LOW F1/F10:** `_resolve_window_choices` is dead code AND **GLM corrected a critical earlier misdiagnosis:** the apparent "fix" (varying search space per trial) **would break Optuna** — `trial.suggest_categorical` requires CONSTANT choices across trials per the ask/tell contract. The current "union + return-inf-on-invalid" approach is the correct workaround. Status: leave as-is; remove or comment the unused helper.
>   - LOW F7, F8 (style/hygiene), several "NOT AN ISSUE" items (F4, F5, F11).
>
>   **Recommendation:** Phase 7 fix commit needed before T-37 ships, but smaller scope than original draft suggested. Fix F2 (~5 lines), F6 (~10 lines), F9 (test strengthening, ~15 lines), F12 (new tests, ~80 lines). Original "fix C1 by varying search space" approach explicitly rejected per GLM analysis.
>
> **T-41 plan ready for implementation** at `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md` (revised to 10-trial median window, explicit `load_study(sampler=...)` mechanism, WAL durability honest, stale sidecar cleanup, 4 added tests). User has the prompt for dispatching implementation when ready.
>
> **Previously:** 2026-05-01 (post-T-37 ship — TPE preprocessing discovery implemented on `feature/T37-tpe-preprocessing-discovery`) —
>
> **T-37 TPE preprocessing discovery IMPLEMENTED.** Branch `feature/T37-tpe-preprocessing-discovery` at `03d95cb` (7 commits past T-36 tip `1f49d73`).
>
> Key wins:
> - **New module `tpe_preprocessing_discovery.py`** — Optuna TPESampler(multivariate=True) over 5-D space (preproc x window x autoscale x baseline x smoothing), 75-trial default budget, model-agnostic LightGBM surrogate (Architecture A — preserves model diversity).
> - **Derivative-aware windows** ported from `ga_preprocessing.py` DERIVATIVE_WINDOW_RANGES.
> - **GUI checkbox + n_trials dropdown** in Preprocessing tab, mutual exclusion with smart/GA preprocessing.
> - **Self-review caught two silent-mismatch bugs** before they shipped: (a) preprocess_cfg name field used pipeline name instead of display name with prefix cascades, (b) baseline/smoothing/autoscale doubling blocks re-doubled TPE-discovered per-config settings — guarded by nulling global flags inside TPE block.
> - **Phase 7 fix commit (`03d95cb`)** closes GLM 5.1 review findings: F2 (_tpe_baseline_params now attached), F6 (_quick_evaluate branches on task_type), F9 (tpe_score propagated to result rows + test strengthening), F10 (_resolve_window_choices commented as RESERVED), F12 (12 new tests covering _apply_full_preprocessing, _quick_evaluate, empty-TPE fallback, pipeline roundtrip).
>
> Test coverage: **33 T-37 tests all green** (21 unit + 5 integration + 4 direct + 3 roundtrip/fallback). Module: 5 files changed, ~700 LOC new.
>
> Audit trail: `docs/bugfix_validation/T37_tpe_preprocessing_discovery.md`.
>
> **T-38 (dead preprocessing module cleanup) is now unblocked** — `ga_preprocessing.py` can be retired after T-37 merge.
>
> **Previously:** 2026-05-01 (post-T-36 ship — T-41 Bayesian SQLite auto-calculator plan filed in response to user-reported 5-28x Bayesian slowdown) —
>
> **T-41 Bayesian SQLite auto-calculator plan filed at `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md`.** User reported Bayesian 5x slower than pre-T-11; investigation traced to T-11's per-trial Optuna SQLite writes. Benchmarks (`tests/_bench_bayesian_per_model.py`) show overhead is ~200ms/trial constant — fast models pay 8-10x, heavy models 1.36-1.89x. Plan: per-model auto-calculator (first 5 trials in-memory + decide based on mean fit time + migrate via `optuna.copy_study` if heavy), 3-way GUI override (Auto/Always-on/Always-off), WAL mode bundled in. ~180 LOC + tests. Branch `fix/T41-bayesian-sqlite-auto-calculator` reserved (off post-T-36 main).
>
> **Previously:** 2026-05-01 (overnight T-36 implementation — Codex-final READY_TO_MERGE_WITH_NITS_CLOSED, awaiting user PR open) —
>
> **T-36 autoscale (UV scaling) preprocessing toggle — IMPLEMENTED, READY FOR PR.**
> Branch `feature/T36-autoscale-toggle` at `2351c3c` (13 commits past `da51f60`).
> Multi-pass review trail (echoing T-11's 7-pass pattern):
> Phase 2 DeepSeek V4 Pro Max → Phase 3 → Phase 4 → Phase 5 → Phase 6 (all READY_TO_PROCEED) → Codex round-1 final review (3 BLOCKERS — Model Dev parser, 3 refinement rebuild paths, code export — all closed in `98fb80f`) → Codex round-2 (READY_TO_MERGE_WITH_NITS — both nits closed in `2351c3c`).
>
> Key wins:
> - **Three pre-existing bugs fixed** (uncovered during plan review, would have shipped silently if autoscale were added naively): `apply_preprocessing` early-return killing autoscale for every preprocessing name, Bayesian cache key omitting autoscale, contamination cache key omitting autoscale.
> - **Bundled one-class metadata-write fix** — `run_one_class_search` result rows now emit `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder` so contamination.py validation rebuild reads real values rather than silent defaults.
> - **Codex caught three downstream silent-mismatch paths DeepSeek missed** — Model Development tab parser dropped `+autoscale`, three refinement rebuild paths didn't pass autoscale, code export emitted no StandardScaler. All three closed.
>
> Test coverage: **62 new T-36 tests + 203 in the targeted regression sweep — all green.** 10 pre-existing I/O test failures (jcamp/spc/opus/perkinelmer) and 2 environmental flakes (subprocess `python` vs `.venv312`) are unrelated.
>
> Audit trail: `docs/bugfix_validation/T36_autoscale_toggle.md`.
>
> Branch is ready for `gh pr create`. NOT pushed yet (per cross-machine doc rule); user should review + push from the originating machine.
>
> **Previously:** 2026-05-01 (early hours — T-11 MERGED via PR #6 after seven reviewer passes) —
>
> **T-11 MERGED to main via PR #6** at `50057af` (rebase merge, linear history).
> Three independent reviewer families converged on READY_TO_MERGE:
> Codex (initial, 4 HIGH + 3 MEDIUM all fixed pre-merge) → Kimi K2.6 (initial,
> 2 MAJOR + 4 MINOR all fixed pre-merge) → DeepSeek V4 Pro pass 1 (24h, 2 HIGH +
> 3 MEDIUM + 9 LOW/INFO all closed) → DeepSeek V4 Pro pass 2 (recheck, 2 NEW HIGH
> closed) → 5 specialist agents post-recheck (code-reviewer, pr-test-analyzer,
> silent-failure-hunter, comment-analyzer, type-design-analyzer; 13 findings,
> 12 closed + 1 deferred) → Codex meta-review (1 NEW critical closed, 1 NEW
> deferred as architectural cost) → DeepSeek V4 Pro pass 3 (high-effort, 0
> blockers + 1 cosmetic). Final: **47/47 T-11 tests passing**, NEW BUG #2
> (per-model Bayesian failure → mark_complete still runs) deferred for a
> follow-up ticket since the fix requires per-model sidecars (architectural
> change). Full audit trail at `docs/reviews/deepseek_v4pro_24h_review_2026-04-30.md`.
>
> **T-33 Gaussian Process Regression** — rough plan filed at
> `docs/plans/2026-04-30-T33-gaussian-process-regression.md`. Not yet
> implemented; awaits prioritization.
>
> **T-36 / T-37 / T-38 preprocessing-discovery roadmap filed (2026-05-01)** —
> three tickets surfaced from a conversation about adding autoscale (UV
> scaling) as a preprocessing option, which led to a deeper audit of the
> preprocessing-discovery surface (basic, GA, exhaustive, learned, ensemble
> modules). Branch `feature/T36-autoscale-toggle` reserved.
>
> | Ticket | Title | Status | Plan |
> |---|---|---|---|
> | **T-36** | Autoscale (UV scaling) preprocessing toggle | IMPLEMENTED on `feature/T36-autoscale-toggle` (tip `2351c3c`); 13 commits, full review trail, awaiting PR | `docs/plans/2026-05-01-T36-autoscale-toggle.md` + `docs/bugfix_validation/T36_autoscale_toggle.md` |
> | **T-37** | TPE quick preprocessing discovery (replaces basic + GA + exhaustive) | **READY_TO_MERGE_WITH_NITS_CLOSED** on `feature/T37-tpe-preprocessing-discovery` (tip `03d95cb`); Phase 7 fix commit closes F2, F6, F9, F10, F12; 33 tests, all green | `docs/plans/2026-05-01-T37-tpe-quick-preprocessing-discovery.md` + `docs/bugfix_validation/T37_tpe_preprocessing_discovery.md` |
> | **T-38** | Dead preprocessing module cleanup | ROUGH_PLAN — three modules deletable now (zero callers); `ga_preprocessing.py` retires after T-37 | `docs/plans/2026-05-01-T38-dead-preprocessing-cleanup.md` |
>
> **T-36 plan caught three pre-existing implementation bugs in a draft autoscale plan (DeepSeek V4 Pro draft, audited):**
> 1. `unified_bayesian.py:apply_preprocessing()` returns early in every named-preprocessing branch — naive autoscale step at end of function would be unreachable. Fix: restructure to assign-not-return.
> 2. `unified_bayesian.py:857-864` preprocessing cache key omits `apply_autoscale` — would cause two trials with same prep but different autoscale to collide on cache. Fix: add to tuple.
> 3. `contamination.py:1109-1112` validation cache key omits `autoscale` — same class of bug. Fix: add to tuple.
>
> **T-36 also bundles a pre-existing one-class metadata gap:** `run_one_class_search` result rows omit `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder` columns; validation rebuild reads them and falls back to defaults silently. Same code surface as the autoscale row writes — low-cost to fix together.
>
> **T-37 background:** Independent audits (Explore agent + DeepSeek V4 Pro via opencode-call) confirmed `ga_preprocessing.py`'s GA mode evaluates ~6× the entire 238-point search space, providing zero benefit over exhaustive on a discrete categorical space with no neighborhood structure. Basic and GA paths cover the same operation space with different search strategies; merge into one TPE-based path covering a richer 5-D space (preproc × window × autoscale × baseline × smoothing) at 75-100 trials.
>
> **T-38 deletes:** `learned_preprocessing.py` (775 LOC, zero callers) and `ensemble_preprocessing.py` (701 LOC, only referenced by the dead `HAS_ENSEMBLE_PREPROCESSING` flag at `gui:236-241`). **Note: this earlier note's claim that T-38 also deletes `preprocessing_wrapper.py` and retires `ga_preprocessing.py` was wrong** — `preprocessing_wrapper.py` is still imported by `ensemble.py:18` (CodeRabbit caught this during T-37 review), and `ga_preprocessing.py` retirement is a separate post-T-37-merge step kept out of T-38's scope. See the top-of-file status entry for the actual T-38 outcome.
>
> **Previously:** 2026-04-30 (evening — T-08 dropped, T-11 staged, T-15 dropped, T-16 reframed, T-19 user-framed) —
>
> **Session close-out:**
>
> | Ticket | Disposition | Branch / commit |
> |---|---|---|
> | T-06 SPA canonical Araújo 2001 enumeration | MERGED + T-06b parallelization | `af44ad4` + `85063b8` |
> | T-08 CARS tree-mode bug | DROPPED — false alarm; algorithm converges; CARS-Tree confirmed dasp-invented | n/a |
> | T-11 Pause/resume + disk logging + Optuna SQLite | **MERGED 2026-05-01 via PR #6** at `50057af` (rebase, 13 commits) — 7-pass review trail | PR #6 |
> | T-33 Gaussian Process Regression | ROUGH_PLAN filed; awaits prioritization | `docs/plans/2026-04-30-T33-gaussian-process-regression.md` |
> | T-12 disk-mirrored logging | CLOSED via T-11 Unit A | (subsumed) |
> | T-15 LeaveOneGroupOut / GroupKFold | DROPPED — user decision; LOGO is footgun in user's data regime (5-100 N per site, 20× ratio); chemometrics literature recommends external test sets, not LOGO | n/a |
> | T-16 model-comparison machinery | REFRAMED — strategic split surfaced. Chemometrics canon (jackknife + Y-permutation = competitor parity) vs ML canon (paired t-test/Wilcoxon = "comparing between models"). Awaiting user decision on Shape A vs B vs hybrid | (investigation only) |
> | T-19 model-native imbalance handling | APPROVED with smaller scope (~2-3 days) — "extend single dropdown to dispatch native kwargs internally + fix 5 PLS-DA sites + close T-32." Math is already statistically equivalent today; gap is audit-trail labels + 5 unwired PLS-DA sites | (awaiting implementation) |
> | T-32 sample_weight length mismatch | DEFERRED into T-19 scope (unchanged) | n/a |
>
> **T-11 detail:** Three sub-units (A: disk-mirrored logging with `RotatingFileHandler` + tee-stdout proxy; B: thread-alive check + UI pause-state honesty with `_actually_paused` event; D: Optuna SQLite persistence + resume-on-startup dialog). Cross-family review pattern caught real bugs that would have shipped: trial-count overrun on resume (silently doubles work), non-atomic sidecar writes, dataset fingerprint stored but never enforced, study names too coarse (load_if_exists would mix incompatible trials), `_TeeStream` thread-unsafe, no log rotation, run-state firing for grid/NSGA-II falsely. All fixed. Pre-existing Nuitka `is_frozen()` bug also fixed (`__compiled__ in dir()` → `__compiled__ in globals()`).
>
> **Memory rules added this session:**
> - GLM 5.1 must NEVER be dispatched through opencode-go (bills against z.ai subscription)
> - CARS-Tree is dasp's invention, not in Li 2009 — don't search for canonical "tree-mode CARS"
> - T-19's user framing is "expose model-native abilities OR auto-detect," not paper-reproducibility
> - T-15 dropped + T-16 reframed (competitive-framework survey)
>
> **Pending user decisions still blocking bigger tickets:**
> - T-31 multi-class SIMCA (yes/no on usefulness)
> - T-22 bootstrap stability diagnostic (yes/no on investment)
> - T-01 reframe (external-test-set workflow yes/no)
> - **T-16 (NEW): pick Shape A (chemometrics canon) vs Shape B (ML canon) vs hybrid**
>
> See `docs/bugfix_validation/` for per-ticket validation notes. See `docs/CONTINUATION_PROMPT_2026-04-30_v2.md` for the next-session pickup prompt with all session lessons applied.
>
> **Previously:** 2026-04-30 (later — T-06 canonical SPA merged + T-06b parallelization follow-up) —
>
> **T-06 SPA canonical seed enumeration MERGED + T-06b parallelization MERGED.** Replaced non-functional `n_random_starts` knob with canonical Araújo 2001 deterministic enumeration over all J variables as candidate first variable. Branch `fix/T06-spa-canonical-seeds` ff-merged into `main`. Validation note: `docs/bugfix_validation/T06_spa_canonical_seeds.md`. Follow-up branch `fix/T06b-spa-parallel` ff-merged immediately after — parallelizes the J-seed loop via `joblib.Parallel(backend='threading')`, capped at min(cpu_count, 8) workers, PyInstaller-bundle-safe. Empirical FTIR-scale benchmark (J=800, n=50, n_features=20): 10.92 sec parallelized vs. ~70-80 sec sequential — ~7-8× speedup as expected for GIL-releasing numpy/sklearn work.
>
> Key gate-finding: the roadmap's proposed `rng.choice()` fix (Option A) was a sklearn-instinct slippage — canonical SPA per Araújo 2001 + auswahl + Galvão 2012 SPA-GUI is deterministic enumeration, not random restarts. No verified chemometrics implementation exposes a "random restart count" parameter. The gate's step-3 field-alignment lookup caught this; without it the fix would have invented a non-canonical pattern.
>
> Cross-family review (Codex + Kimi K2.6) caught: a missed call-site (`tests/gui/test_comprehensive.py:387`), template ↔ in-app divergence on small-sample CV-fold handling, dead `y_norm` code, and a Python-loop projection in the export template that was 100-1000× slower than the vectorized production path. All fixed before merge. Five new T-06 tests + 226-test regression sweep all green (post-parallelization too).
>
> **Previously:** 2026-04-30 (validation gate session — bugfix branches resolved) —
>
> **Bugfix branch validation gate complete.** Five branches that came out of the T-01 roadmap each went through a strict validation gate (literature check + commercial-software comparison + reachability verification + regression test sweep). Final dispositions:
>
> | Branch / Ticket | Disposition | Commit |
> |---|---|---|
> | T-26 SNV near-zero std | DROP / WONT_FIX (current dasp behavior matches PLS_Toolbox default at offset=0; bundled-app distribution makes a backend-only knob useless) | n/a |
> | T-10 PLS components clamp (LOO) | MERGED | `fbeb50c` |
> | T-05 VIP central formula | MERGED | `2c068cd` |
> | T-07 PDS even-window | MERGED | `1b91d93` |
> | T-24 Lin's CCC metric addition | MERGED | `0087cad` |
> | T-05a duplicate VIP formulas (templates + nsga2) | MERGED | `1eb6c06` |
> | T-04 one-class UVE prefilter | MERGED (GUI grey-out matching iPLS pattern) | `6beb5e8` |
> | T-21 SG uniformity guard | PENDING USER DECISION (DROP / DEFER / APPROVE_MINIMAL — radio-button path is safe; only the rarely-used Convert button creates the bug surface) | n/a |
> | T-32 sample_weight length mismatch | DEFERRED to T-19 (current path is unreachable; bug fires only after T-19's planned scope) | n/a |
> | `bayesian_utils.py random_state=42` hardcoding | MERGED (refactored to use the shared `RANDOM_STATE` constant) | `50d5d05` |
> | `search.py:2855 top_n_vars=30` hardcoding | DROPPED (display economy, not a bug — `all_vars` already preserves the full wavelength list for replication) | n/a |
> | Worktree cleanup (5 stale worktrees) | DONE — branch refs preserved locally | n/a |
>
> Per-branch validation notes at `docs/bugfix_validation/`. The gate itself caught two false alarms (T-26 + the two re-evaluation flags) where the underlying claim was real-but-unreachable or really a code-style issue. The recurring pattern: verify reachability + match-the-field BEFORE drafting verdicts. See `docs/bugfix_validation/README.md` "Lessons" section.
>
> **One pending user decision:** T-21 disposition. Investigation findings at `docs/bugfix_validation/T21_findings.md` flagged that the plan as written has citation errors (OpenSpecy `is_evenly_spaced()` is fabricated; PLS_Toolbox `gridcheck` is unverified) and a scope gap (~half of SG paths bypass the proposed guard). The user's typical workflow uses the radio-button relabel (which is safe — verified `_on_x_unit_override` does NOT call `convert_x_axis`); only the explicit "Convert" button creates the non-uniform-grid bug surface, and that button is rarely used in routine analysis.
>
> **Three pending user decisions still from the re-evaluation:** (1) T-31 multi-class SIMCA — confirm "none of the above" output is useful. (2) T-01 reframe scope — confirm external-test-set approach over per-fold varsel. (3) T-22 reframe — confirm bootstrap stability diagnostic investment.

> **Previously:** 2026-04-30 (full roadmap re-evaluation under chemometrics master rule) —
>
> **Roadmap re-evaluation complete.** All 32 tickets + deferred items re-evaluated against chemometrics literature + bone-FTIR application domain. Full results at `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md`. Summary: **27 KEEP, 2 REFRAME (T-01, T-22), 2 DROP (T-02, T-03), 2 DEFER (T-05a, T-10b), 2 NEEDS_USER_DECISION (T-31).** Key findings: T-02 (ensemble OOF preprocessor) and T-03 (preprocessing-discovery full-data) are false alarms — same pattern as T-01 (sklearn-pipeline-purity misapplied to per-spectrum chemometrics operations). T-04 (one-class UVE prefilter on outlier-contaminated labels) is a REAL issue distinct from the leakage question. T-21 (SG uniformity guard) is chemometrics-correct (Savitzky & Golay 1964 assumes uniform sampling; PLS_Toolbox's `gridcheck` is precedent). T-22 reframed as bootstrap stability diagnostic. Top 5 next actions: T-15 (LOGO), T-16 (bootstrap CIs), T-11 (Optuna SQLite), T-01 reframe (external-test-set workflow), T-19 (loss reweighting).
>
> **Worktree disposition:** `fix/varsel-leakage` branch artifacts evaluated. `varsel_transformer.py` → MERGE_AS_OPTIONAL_TOOL (useful for T-22 stability diagnostic, expert mode, paper reproduction). `test_varsel_transformer.py` → MERGE. Audit doc → CHERRY_PICK (rename to VARSEL_ARCHITECTURE_AUDIT.md, fix labels). Plan doc + Codex reviews → DROP. See `docs/varsel_leakage_worktree_disposition_2026-04-30.md`.

> **Last updated:** 2026-04-30 (T-01 framing reconsidered + 5 small tickets implemented) —
>
> **Critical update:** the T-01 audit's "leakage" framing has been **reconsidered after literature validation by both Codex and GLM 5.1** (full analyses at `docs/analysis_vs_chemometrics_lit/`). Verdict: dasp's "varsel-on-full-calibration + CV" pattern is the **standard chemometrics workflow** documented in Li 2009 (CARS), Centner 1996 (UVE), Araujo 2001 (SPA), Norgaard 2000 (iPLS), Wold 2001 (PLS/VIP). The audit was technically correct as a *description of the code* but rhetorically wrong to label it "LEAKY" — that framing applied sklearn-pipeline-purity convention to a domain with its own established methodology. Furthermore, the user flagged that **per-fold varsel actively defeats the scientific purpose** of these methods (which is identifying wavelengths corresponding to real chemistry — stable selection across resamples is *evidence* the chemistry is real; per-fold instability destroys interpretability). The audit doc has been annotated with a banner pointing here. Read `docs/T01_VARSEL_LEAKAGE_AUDIT.md` (banner) + the two validation files for full reasoning.
>
> **Master rule (now memory-pinned):** dasp methodology decisions must be validated against chemometrics literature + the applied domain (bone FTIR / isotopes / paleoanthropology), NOT against sklearn / generic ML / genomics ML conventions. When literatures conflict, chemometrics wins. See `feedback_validate_against_chemometrics_and_application_lit.md` in user's auto-memory.
>
> **Status of in-flight work:**
> - `fix/varsel-leakage` Phase 1 (additive `VarselTransformer` infrastructure) is implemented but **paused — should NOT be merged**. Phases 2–7 paused indefinitely.
> - **T-02, T-03, T-04 in the roadmap need re-evaluation** under the same literature lens before any work. T-04 (one-class UVE prefilter on outlier-contaminated labels) may be a real issue distinct from the selection-on-full question and should be evaluated separately. T-02 / T-03 framing is suspect.
> - **5 implemented branches that came out of the T-01 roadmap (T-05 VIP, T-07 PDS, T-10 PLS clamp, T-24 CCC, T-26 SNV) are unaffected** — those are real bugs verified independently. All on `fix/T<X>-*` branches, none pushed, ready for review/merge:
>   - `fix/T05-vip-formula-fix` — canonical Wold (2001) VIP, 5+2+33+5 tests pass. T-05a deferred (duplicate VIP formulas in `templates/variable_selection.py:54` + `nsga2_search.py:627`).
>   - `fix/T07-pds-even-window` — odd-window enforcement + `apply_pds` hardening, 10+44 tests pass.
>   - `fix/T10-pls-components-clamp` — CV-aware clamp + new `compute_min_train_fold_size` helper, 16+5+1+113 tests pass. T-10b (NSGA-II audit) deferred.
>   - `fix/T24-lins-ccc` — Lin's CCC in 4 result paths + exported template, 14 unit tests pass.
>   - `fix/T26-snv-near-zero-std` — 1e-12 floor + warning, 5+14+31 tests pass.
> - **T-21 SG uniformity guard** has a plan committed on main (`d8c46f0`) but no implementation yet (codex review wedged). Plan applies a uniformity check to SG derivatives — **needs literature check before implementation** (the user flagged this in conversation: applying lessons from other literatures is the recurring failure mode).
>
> **Empty `opencode draft snapshot` commits** (2 in T-05, 1 in T-07, 3 in T-24) auto-created by the dispatcher wrapper after killed/snapshotted opencode sessions. Drop or squash on merge — they contain no diffs.
>
> **Next action recommended:** dispatch a fresh agent to re-evaluate the entire roadmap (`docs/RECONCILED_ROADMAP_2026-04-29.md`) under the master rule. Each ticket must be checked against chemometrics + bone-FTIR application literature before being treated as actionable.
>
> **Previously:** 2026-04-29 — T-10 PLS components clamp (cv_strategy-aware) — IMPLEMENTED on branch `fix/T10-pls-components-clamp` (not yet merged). New helper `compute_min_train_fold_size(cv_strategy, n_samples, n_folds)` in `cv_utils.py` returns the smallest training-fold size for kfold, repeated_kfold, loo, and repeated_loo strategies. `run_search` and `run_bayesian_search` now use it to clamp `max_n_components` before passing to `get_model_grids`. `models.py` contract documented via docstring only (no assert — Codex suggestion #3). Raises `ValueError` when `n_folds > n_samples` for kfold/repeated_kfold (Codex suggestion #1). Deferred NSGA-II audit ticket (T-10b): `_get_constrained_pls_components` clamps by n_samples but some evaluation paths already use min_train_samples from cv_folds; full audit of decode/result-row/reporting consistency needed before declaring same bug present. **Tests added:** 16 helper unit tests (`test_cv_strategy.py`), 5 integration tests (`test_cv_pls_clamp.py` for run_search), 1 Bayesian test (`test_cv_pls_clamp.py` for run_bayesian_search). Full sweep: 113 passed, 0 failed. **Commits:** `64c16d0` (task 1), `8de2cf4` (tasks 2+3), `1384625` (task 4), `0921e77` (task 5).

> **Previously:** 2026-04-29 (T-05 VIP formula fix) — Fixed `compute_vip()` in `models.py` to use canonical Wold (2001) formula (`SSY_a = q_a**2 * sum(T_a**2)` via `y_loadings_`) instead of the buggy `np.var(y)` scalar. Old formula collapsed all components to the same Y-weighting, skewing VIP rankings. 7 new regression tests in `tests/test_vip_formula.py`. All existing tests pass. Deferred follow-up T-05a: apply same fix to duplicated formulas in `templates/variable_selection.py:54` and `nsga2_search.py:627`.
> **Previously:** 2026-04-29 (T-24 Lin's CCC merged on `fix/T24-lins-ccc`) — Lin (1989) Concordance Correlation Coefficient added as an always-computed regression metric. New `lins_ccc()` helper in `scoring.py`; `CCC` (calibration) and `CCCcv` (CV-pooled) columns now emitted by all four regression result paths: grid (`search.py:4744-4748`), Bayesian-path-1 (`unified_bayesian.py:1437-1443` user_attrs), Bayesian-path-2 (`unified_bayesian.py:2056-2059` row dict), NSGA-II (`nsga2_search.py:3655,3690`). Exported validation template (`templates/validation.py`) gets an inline `_lins_ccc()` helper since templates can't import internal scoring. `ddof=0` (population variance) per Lin (1989) closed-form. Display convention: 0.0 for one-sided degeneracy, 1.0 for both-equal-constants. 14 TestLinsCCC tests pass, including `test_ccc_finite_sample_ddof_zero_exact` (CCC=4/7 for `y=[0,1,2], pred=[1,2,3]`) per Codex suggestion. GUI: CCC/CCCcv added to `higher_is_better_cols` for descending default sort + tooltip dict for column header hover.
>
> **Previously:** 2026-04-29 (T-01 audit complete) — Completed T-01 per-fold variable-selection leakage audit. Full report at `docs/T01_VARSEL_LEAKAGE_AUDIT.md`. Result: **60 LEAKY / 0 CLEAN / 35 N/A** across 19 methods x 6 search paths. Every y-using varsel method in every search path computes importances once on full calibration data and passes a frozen `top_indices` into CV — no method refits per-fold. Top 3: (1) grid CARS at `search.py:2552` (inflates R2cv 0.05-0.15), (2) grid `importance` mode at `search.py:2476` (0.02-0.05), (3) one-class UVE prefilter at `search.py:5649` (compounds with T-04). NSGA-II CARS guidance (`nsga2_search.py:1872`) is leaky but low-priority (guidance only, not per-fold varsel). Refinement path is clean (reuses saved wavelength list, no varsel re-run). Recommended fix: `VarselTransformer` sklearn wrapper for grid search, per-fold varsel inside Optuna objectives for Bayesian paths.
>
> **Previously:** 2026-04-29 (latest, post-Codex T-19 review) — Codex did a focused review of T-19 (the imbalance-handling ticket) and found three substantive issues: (1) **PLS-DA inner LR is split across ~10 sites, not 3** — five active sites still missing `class_weight` handling: `search.py:380`, `bayesian_utils.py:400`, `nsga2_search.py:822`, `nsga2_search.py:3453`, `ga_preprocessing.py:428`; (2) **the originally-prescribed "route sample_weight via imblearn fit_params" approach is broken** because imblearn's Pipeline does NOT rewrite fit_params after resampling — the right design is a custom estimator wrapper that recomputes sample_weight from the y received (post-resample). (3) **An active length-mismatch bug at `search.py:3883`** today — passes `sample_weight_train` (computed from pre-resampling `y_train`) against `y_train` instead of `y_train_for_model`, so when a resampler runs the lengths mismatch. New ticket **T-32** added to roadmap. T-19 effort revised up to 5-7 days. MLP support is now conditional on sklearn ≥1.7 PyInstaller bundle compatibility — if incompatible, skip MLP rather than block T-19. Roadmap, design doc, and SESSION_LOG all updated. Total tickets now 32 (P0/P1/P2) plus drop list.
>
> **Previously:** 2026-04-29 (latest) — **Reconciled roadmap from three independent reviews of the FTIR-paper + Unscrambler analyses**: see `docs/RECONCILED_ROADMAP_2026-04-29.md`. Three reviewers (Claude / Codex / DeepSeek V4 Pro, intentionally cross-family) deduplicated ~240 raw findings to 30 tickets across P0/P1/P2, plus 1 PENDING (multi-class SIMCA — needs user decision on whether "none of the above" outputs are scientifically useful) and 1 elevated per user request (T-17: multi-Y / PLS-2 workflow — substantial 2-3 week effort but high-leverage). Top-5 first: T-01 (per-fold varsel leakage audit nobody actually did), T-15 (LeaveOneGroupOut by site), T-11 (Optuna SQLite for crash-survivable resume), T-16 (bootstrap CIs + permutation tests for two-method comparison), T-12 (disk-mirrored logging). Key corrections during reconciliation: (a) "binary specificity bug" in LESSER_FINDINGS 1.1 is a false alarm (current code is correct under standard sklearn convention); (b) FTIR analysis claim that PLS-DA inner LR is stuck at `class_weight=None` is wrong — `search.py:4257-4261` already applies balanced weights when `imbalance_method='class_weight'`, so loss-reweighting design doc scope shrinks to boosting models only; (c) "509 except blocks" stat is unreproducible / scope-inflated; (d) HMAC signing of `.dasp` is technically insufficient (key extractable from app), use safe `extractall` + UX warning instead. New auto-memory: `feedback_chemometrics_conventions.md` — per-spectrum preprocessing (SNV / SG derivatives / baseline) is NOT data leakage by chemometrics community convention, so reviewers should not flag it; cross-sample operations (PCA fit on full data, varsel using y, hyperparameter tuning on full data) ARE real leakage and should be flagged.
>
> **Previously:** 2026-04-29 (later, continued) — Completed quick-wins audit of core modules. **`docs/QUICK_WINS.md`** — 14-item prioritized matrix (P0–P3) with exact line numbers, estimated fix times, and commercial-impact ratings. Highest-payoff items: (1) delete 8 debug prints in `search.py` (5 min), (2) replace ~66 prints with logger in `calibration_transfer.py` (15 min), (3) fix hardcoded version mismatch (`0.5.0b1` backend vs `v3` GUI/templates, 10 min), (4) fix VIP formula bug in `models.py` (15 min), (5) clean 42 prints + 29 bare excepts in `nsga2_search.py` (20 min). Total P0–P2 effort ~80 minutes for measurable commercial polish. See QUICK_WINS.md for full execution order.
>
> **Previously:** 2026-04-29 (later) — Completed gap analysis of dasp vs. the Sponheimer FTIR Bone PLS paper (in prep, working dir `C:\Users\mspon\Desktop\_DeskSync\FTIR Bone PLS\`). Full analysis in **`docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md`** — 10 confirmed-missing items, 5 partially-supported items, ranked priority recommendations. Top 3 gaps for contemplation: (1) **LOGO / GroupKFold CV** by site (paper's transferability claim depends on it; dasp has none), (2) **bootstrap CIs + paired permutation tests** for two-method comparison (paper's only inferential machinery; dasp has none), (3) **model-native loss reweighting for boosting models + PLS-DA inner LR** (resampling-only today; design doc at `docs/plans/2026-04-29-model-native-loss-reweighting.md`). Also flagged: ENLR is genuinely missing as a classifier (the existing `ElasticNet` card is the regression flavor), LDA-on-PLS-scores is missing as a model card, Lin's CCC is missing as a metric, stratified Kennard-Stone is missing for cal/test split. Use the gap analysis as the planning reference when prioritizing feature work.
>
> **Previously:** 2026-04-29 — Completed deep analysis vs. CAMO Unscrambler across all subsystems (8 parallel agents + 4 validation agents + 3 lesser-findings agents). Reports written to `docs/analysis_vs_unscrambler/` (6 documents: MASTER_ANALYSIS.md, TECHNICAL_GAPS.md, CODE_QUALITY.md, COMMERCIAL_READINESS.md, VALIDATION_SUPPLEMENT.md, LESSER_FINDINGS.md). Key findings: analytical engine exceeds Unscrambler in 9/17 categories, but commercial release is blocked by 12 issues. Validation supplement corrected understatements (error swallowing is 509 blocks, not 130; additional security issues: Zip Slip, ReDoS, GUI pickle.load). Lesser findings document catalogs ~200+ non-blocker issues including 11 silent correctness bugs, ~150 UX polish issues, 36 code hygiene items, 8 performance issues, and 14 robustness gaps. See analysis documents for phased release plan.
>
> **Previously:** 2026-04-21 (late evening) — Merged `fix/duplicate-target-col`: Data Management viewer no longer shows the target column twice when loaded from a combined-format CSV/Excel. Four view/export sites updated to filter target out of combined `metadata_cols` iteration (mirroring the already-correct separate-ref branch), plus a target-preservation guard in `_apply_data_viewer_edits` so the target-switch dropdown survives cell edits. Investigation by Codex + Kimi K2.6 in parallel; implementation by GLM-5.1 via z.ai direct (opencode build); independent verification by Codex + Claude; user manual verification with Valley CSV. Also: `9bd8751` handles RepeatedKFold in `compute_pls_complexity_curve` (sklearn's `cross_val_predict` rejects non-partition splitters — now falls back to manual per-fold aggregation) and surfaces validation-curve failures to the Results text instead of console-only. All regression tests still pass (144/144). Pending push.
>
> **Earlier today (same date, evening):** Reverted `_infer_task_type_from_y` to Fix-B1 heuristic (binary OR non-numeric → classification, else regression) after `0ef91e5`'s switch to `sklearn.utils.multiclass.type_of_target` caused integer-valued numeric targets (burn temperatures, 3-value proportion columns) to auto-flip back to classification — the exact bug `41371e0`/Fix-B1 was meant to prevent. Three stale auto-detect sites (`_run_analysis_thread`, `_update_task_type_label`, imbalance thread) that `0ef91e5` missed are now routed through the helper for consistency; user-reported crash on `group.cov.heath` (3 int-valued target, GUI auto-checked PLS-DA, backend resolved regression → `run_search`: "No valid models found") is resolved. 12/12 task-type preservation tests pass (was 10/12 against HEAD). Also: CV strategy tooltip overhaul — consolidated 3 duplicated inline texts into `TOOLTIP_CONTENT['cross_validation']['strategy_detail']`, added Kohavi (1995) / Krstajic et al. (2014) / Hastie et al. / Ezenarro et al. (2025) citations, widened Repeated-K-Fold inline-hint range to n=30–200.
>
> **Earlier today (same date):** Long session of imbalance-handling + task-type refinements. Session commits (in order): `a6955c5`/`b8a70cb`/`f8ab20b`/`ac43289` (imbalance task-type-aware dropdown + banner + logger fallback + banner-ordering fix), `f9fa206` (docs), `8d05dc0` (prediction-tab export duplicate-method fix), `aa73edd`/`8f14511`/`b9d2b39` (CV guard + type_of_target + refine preflight before model creation), `5dd00a0` (imbalance dropdown refreshes on task-type change), `8b2cd74` (don't auto-clear banner on redundant refresh), `18b2244` (classification y_range debug-print guard), `10d97a6` (target variable authoritatively drives task type, overrides sticky Import-tab radio except for one_class), `0ef91e5` (`_infer_task_type_from_y` helper + banner suppressed when imbalance disabled), `0f4a1e6` (Bayesian cost estimate no longer multiplies by phantom 10× preprocessing dimension), `0da5bd4` (LOO warning thresholds raised 10× to match small-spectral-dataset reality). Earlier: Mixed-type target Validation-Wide fix, `7407140` (one-class export support), `de6b618` (2x2 confusion matrix + NaN-safety), PLS-regression hotfix, phantom-class label normalization. Pending push.

> **Previously:** 2026-04-19 by GLM-5.1 — Analysis Subset V1 implemented on branch `glm/analysis-subset-v1` (NOT yet merged). New pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab has "Analysis Subset" card above Holdout Validation. Metadata-only dialog with categorical multi-select. Subset provenance stored in `last_training_config`. Mismatch warnings in Model Development. One-class guardrail blocks when subset removes all inlier samples. C2 missing-column safe handling via `_refresh_active_group_indices`. All changes uncommitted in worktree. 41/41 analysis_subset tests + 61/61 OC regression tests pass. See `docs/SESSION_LOG.md` 2026-04-19 entry for architecture decisions and gotchas.

---

## Distribution path — bundle-only as of 0.5.0b1 (updated 2026-04-21)

**Decision:** the PyInstaller 3.12 bundle is now the only supported distribution path. Nobody is expected to clone and `pip install -e .`; the source-install scaffolding (`install.bat` / `install.sh` / `INSTALL.md`) stays in-repo as a developer convenience but is no longer marketed to end users. Beta version `0.5.0b1` ships exclusively as the bundled installer.

**Implication for parallelism:** the 3.12 bundle still uses the threading-backend fallback (see `src/spectral_predict/search.py:_frozen_needs_threading_fallback` — frozen-state-only, NOT version-gated; the original 3.12 plan to recover loky was wrong). Practical impact: numpy/sklearn/lightgbm/xgboost get thread-parallel speedup (those C extensions release the GIL), but pure-Python parallel loops (pymoo NSGA-II, GA-PLS evaluation) are single-core in the bundle. There is no longer a "use the source install for full multiprocessing" escape hatch for users — what the bundle does is what they get.

**Still in-repo from the source-install era (kept, not deleted):**
- `install.bat` / `install.sh`: detects Python 3.12, creates `.venv312`, runs `pip install -e . --upgrade`. Idempotent. Useful for developer setup.
- `INSTALL.md`: GUI-focused walkthrough — now developer-facing, not user-facing.
- `pyproject.toml` deps audited via AST scan. Added: `Pillow>=10.0.0`, `shap>=0.44.0`. Re-enabled: `jcamp>=1.2.1`. Floors bumped: `numpy>=2.0`, `pandas>=2.0`, `scikit-learn>=1.5`, `scipy>=1.11`.

**Intentionally NOT declared as required:**
- **`torch`** — no in-tree module imports it. T-38 deleted the last importer (`learned_preprocessing.py`); the dead `HAS_ENSEMBLE_PREPROCESSING` flag was removed at the same time. The build still excludes torch defensively in case a transitive PyInstaller import sneaks it in.
- **`agilent-ir-formats`** — no Python 3.12 wheel on PyPI. Stays in optional `[agilent]` extra. .seq file loading raises a clear ImportError from `agilent_reader.py:62` until upstream ships a 3.12 wheel.

**Open follow-up if bundle parallelism becomes a real bottleneck:** fix the PyInstaller spawned-child runtime hook (the argv-parse crash in `multiprocessing.freeze_support()`) so loky can be used in the bundle. Tractable but non-trivial — would need a custom runtime hook + verification across the supported Windows targets.

---

## 3.11 build retirement — DONE 2026-04-21

The 3.11 build path was archived to `archive/build_3_11/` on 2026-04-21 as part of the `0.5.0b1` beta cut. The 3.12 PyInstaller bundle is now the only build path. Files moved (with git history preserved via `git mv`):

- `spectral_predict.spec` → `archive/build_3_11/spectral_predict.spec`
- `spectral_predict_mac.spec` → `archive/build_3_11/spectral_predict_mac.spec`
- `build_installer.py` → `archive/build_3_11/build_installer.py`
- `installer/spectral_predict.iss` → `archive/build_3_11/installer/spectral_predict.iss`
- `docs/BUNDLED_APP_BUILD_GUIDE.md` → `archive/build_3_11/BUNDLED_APP_BUILD_GUIDE.md`
- `run_gui.bat` → `archive/build_3_11/run_gui.bat` (was `.venv311`-based)
- `run_v3.bat` → `archive/build_3_11/run_v3.bat` (deprecated Dear PyGui V3 prototype)

**Kept in repo root** (current `.venv312` dev launchers, NOT 3.11-specific): `RUN_SPECTRAL_PREDICT.bat`, `run_gui.sh`.

**Still TODO** (deferred — cosmetic, not blocking):
- Rename `spectral_predict_py312.spec` → `spectral_predict.spec`, `build_installer_py312.py` → `build_installer.py`, `installer/spectral_predict_py312.iss` → `installer/spectral_predict.iss`, `docs/BUNDLED_APP_BUILD_GUIDE_PY312.md` → `docs/BUNDLED_APP_BUILD_GUIDE.md`. Drop the `_py312` / `-py312` suffixes everywhere they appear (APP_NAME constants, .iss `OutputBaseFilename`, etc.). Save for after `0.5.0b1` ships so installer filenames stay stable through the beta.
- Delete `.venv311/` (or leave for occasional 3.11 reproduction if needed).

**Restore one file from archive:** `git mv archive/build_3_11/<file> <original-path>`.

---

## Tooltip coverage — RESOLVED 2026-04-17 (GLM 5.1 implementation)

Originally an audit-only finding (6 missing-tooltip items + 1 codebase discrepancy); GLM then implemented the fixes in the same session. **All 16 missing `CreateToolTip()` calls added** + new `TOOLTIP_CONTENT['one_class']` section (10 entries) + new `cross_validation['repeats']` key. Tooltips are research-backed (sklearn docs + chemometrics literature with citations: Bouckaert & Frank 2004, Schölkopf et al. 2001, Liu/Ting/Zhou 2008, Breunig et al. 2000, Rousseeuw & Van Driessen 1999, Wold 1976, Oliveri & Downey 2012).

**Coverage now includes:**
- CV repeats label + spinbox (main tab + refine tab)
- CV strategy combobox (main tab + refine tab)
- All 5 one-class model checkboxes (PCA-SIMCA, OneClassSVM, IsolationForest, EllipticEnvelope, LOF)
- Inlier class label + combobox
- All 4 one-class hyperparameter controls (`nu`, `contamination`, `alpha`, `n_components`)

**Verification: passed.** GUI imports cleanly, TOOLTIP_CONTENT['one_class'] dict has all 10 expected keys, no key shorter than 50 chars (sanity guard against placeholder text).

### `repeated_stratified_kfold` — clarified (was a documentation issue, not a bug)
Earlier audit said `RepeatedStratifiedKFold` doesn't appear in the codebase. **Refined finding:** the *string value* `'repeated_stratified_kfold'` is not in the dropdown — but the *class* `RepeatedStratifiedKFold` IS imported and used in `cv_utils.py`. The CV factory `build_cv_splitter` auto-selects between `RepeatedKFold` and `RepeatedStratifiedKFold` based on task type (classification → stratified, regression → unstratified) when the user selects `repeated_kfold`. So users see one dropdown option, get the right splitter under the hood. No user action needed; status doc PR #4 description was technically accurate, just misleading without that context.

---

## Recently resolved

### T-10: PLS component grid CV-strategy-aware clamp — FIXED 2026-04-30

`search.py:1109` and `search.py:3312` previously used `n_samples * (folds - 1) // folds`
to bound the PLS `n_components` grid. Correct for K-fold/RepeatedKFold, under-counts
for LOO (true train-fold size is `n - 1`, not `4n/5`). Added
`cv_utils.compute_min_train_fold_size(cv_strategy, n_samples, n_folds)` and routed
both grid clamps through it. LOO grids now reach the proper `n - 1` ceiling.
Also validates `n_folds > n_samples` geometry (Codex suggestion #1). Group splitters
(`group_kfold`, `leave_one_group_out`) deliberately raise `NotImplementedError`;
T-15 will plumb group-aware sizing.

Test coverage: `tests/test_cv_pls_clamp.py` — 16 unit tests on the helper +
4 grid integration tests on N=10 / N=80 synthetic data + 1 Bayesian integration test.
### T-05: VIP formula corrected to canonical Wold (2001) — FIXED 2026-04-30

Replaced `np.var(y)` per-component weight with canonical `q_a**2 * sum(T_a**2)` (Wold 2001, Mehmood et al. 2012 Eq. 1) in `src/spectral_predict/models.py:compute_vip`. Old formula collapsed all components to the same Y-weighting scalar, skewing VIP rankings whenever components had similar X-score energy but different Y-loading. Added `ssy_total <= 0` guard for degenerate fits. 7 new tests in `tests/test_vip_formula.py`. All 33 preprocessing_discovery tests + 5 PLS-DA importance tests pass unchanged.

**Deferred to T-05a:** Two additional copies of the buggy formula at `templates/variable_selection.py:54` and `nsga2_search.py:627` still need the same correction. See plan at `docs/plans/2026-04-29-T05-vip-formula-fix.md`.
### T-07: PDS even-window arithmetic — FIXED 2026-04-30

`estimate_pds` crashed on even window sizes due to B-allocation overflow (slice has window+1 columns for even window but B has window columns). Fix: reject even windows with clear ValueError. `apply_pds` signature changed to `window: int | None = None`, deriving geometry from `B.shape[1]` with FutureWarning on mismatch. 10 new tests in `tests/test_pds_window_arithmetic.py`. Commits: `d3d1606` (tests), `f438083` (fix).

### PLS regression hotfix: refine tab PLS-DA instead of PLS — FIXED 2026-04-21

Two bugs latent since commit `057d9f6` (2026-04-11), triggered by Burned-temperature data with ≤9 unique values:

1. **Fix A:** `_load_model_for_refinement` now trusts the saved Task type for regression/classification (was only trusting one_class). Extracted pure helper `_detect_refine_task_type(config, y)` for testability. Prevents the PLS → PLS-DA cascade when loading regression results into Model Development.
2. **Fix B1:** Removed `nunique() < 10` heuristic from auto-detect at all 9 sites. Only `nunique() == 2` and non-numeric dtype remain as classification triggers.

Tests: `tests/test_refine_task_type_preservation.py` (12 tests). 376/376 broader suite pass.

### OC validation helper silent-wrong-result paths + PCA-SIMCA default — FIXED 2026-04-18 (9e9ca11 + follow-up)

Three silent-wrong-result paths in `compute_validation_metrics_for_top_one_class_models` (`src/spectral_predict/contamination.py`) eliminated — all now warn and skip instead of producing believable-but-wrong `val_*` metrics:
- Malformed / missing / empty `Params` string → skip (was: default to `{}`, validate with estimator defaults)
- Partial wavelength-map miss → skip (was: silently drop missing wavelengths, validate on truncated feature set)
- Removed fallback to `selected_wavelengths` user-attr (for regression/classification Bayesian, only first 50 are stored there; `all_vars` always holds the full trained subset for all OC code paths)
- Follow-up: missing/empty/NaN `all_vars` also now warns and skips (was: silently validated on full spectrum) — both OC Bayesian (`unified_bayesian.py:1056-1062`) and OC grid (`search.py:5332`) always populate it, so a missing value means row metadata is corrupt

Also bumped PCA-SIMCA `n_components` default from 5 to 10 at `gui:2892` (more sensible starting point for typical NIR datasets). Regression test suite: 6 new parametrized cases in `TestGridSearchValidationMetricsParity::test_corrupt_metadata_row_is_skipped_not_silently_wrong` covering all warn+skip paths. 54/54 `test_contamination_detection.py` pass. Closes three "Follow-Ups (unclaimed)" items flagged MAJOR by Codex in the pre-merge review.

### LOO + Repeated K-Fold cross-validation strategies — MERGED 2026-04-17 (PR #4)

Added `LeaveOneOut`, `RepeatedKFold`, and `RepeatedStratifiedKFold` as first-class CV options across regression / classification / one-class. Backend: `src/spectral_predict/cv_utils.py` — factory `build_cv_splitter`, `validate_cv_strategy_for_task`, `cross_val_predict_pooled`, `reduce_repeated_cv_predictions`. Pooled RMSEcv (regression — fixes the LOO degeneracy where mean-of-fold-RMSE = MAE); pooled sensitivity/specificity (one-class); BER consistency under repeated CV; AUC kept as mean-of-fold-AUCs (scores not comparable across independently-fit folds). GUI: Analysis-tab + Model-Development controls, cost estimator with LOO/Repeated warnings. `training_config` stores `cv_strategy` for save/load. Includes a code-export fallback in `code_generator.py:108-115` so exported scripts/notebooks honor the user's CV regime when callers pass `training_config` rather than top-level keys (caught in pre-merge codex review). 1456-line test suite at `tests/test_cv_strategy.py`, 5 prior review rounds visible in commit history.

### LightGBM "parameter capture" warning + NaN / crash on `.venv311` — FIXED 2026-04-16 (PR #5)

Fix: `clone(model)` at four pipe-construction sites in `search.py` (`:2191`, `:4161`, `:4163` for the regression bug, plus `:4139` PLS-DA defensive). Root cause: shared estimator instance being fit first on full preprocessed X (setting `n_features_in_=2151`), then reused in subset-feature fits; sklearn 1.5.2's pre-fit `_check_n_features(reset=False)` fires before the refit would reset the count, raising. sklearn 1.7.2's pre-fit check is more lenient and doesn't raise on the same pattern. The PLS-DA `:4139` site was added after Claude pr-reviewer flagged it as the same pattern; classification baseline run on `.venv311` with PLS-DA un-cloned proved PLS-DA doesn't actually hit the bug (importance-capture at `:2191` doesn't fire for PLS-DA), so the clone there is purely defensive/symmetric. (The separate sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood seen on `.venv312` comes from a DataFrame-fit / ndarray-predict transition in the pipeline; it's unrelated to the shared-state bug and is NOT changed by this fix — see COMPARISON.md.)

Verification: harness `scripts/verify_shared_model_fix.py` run with GUI defaults on BoneCollagen — 4-run before/after matrix on both venvs. Post-fix `.venv311` produces 9408 LightGBM rows with `best_cv_rmse=0.9702327793086989`, bit-identical to `.venv312`. See `docs/plans/2026-04-16-lightgbm-shared-model-fix.md` for the plan and `docs/plans/artifacts/2026-04-16/COMPARISON.md` for the metrics matrix.

## Known follow-ups (deferred from PR #4 reviews, non-blocking)

- **`run_bayesian_search()` does not call `validate_cv_strategy_for_task()` upfront.** Located in `src/spectral_predict/search.py:3096`. Unlike `run_search()` and `run_unified_bayesian()` which validate the CV-strategy / task-type / class-count combination before any trials run, `run_bayesian_search()` accepts `cv_strategy` / `cv_n_repeats` and proceeds directly. Result: a bad CV config (e.g. LOO requested with task='classification' and a class with <2 samples) would only fail mid-trial inside the Bayesian objective rather than producing a clean preflight error. Codex flagged this as non-blocking in the PR #4 review. Fix is a one-liner: add `validate_cv_strategy_for_task(...)` at the top of the function, mirroring the other two entry points.
- **Style nits in CV code**: `cv_utils.py` duplicates the `RepeatedKFold` import (module-level + inside `build_cv_splitter`); `templates/validation.py` prints `({cv_folds}-fold)` even for `loo`/`repeated_kfold` (misleading once strategy-specific exports are working); `_majority_vote` should default to NaN in the empty-votes else-branch for safety; LOO `splits = list(...)` materialization at `search.py:~4228` is O(n²) memory for very large datasets (fine for n<5000 spectral data but worth a comment).
- **Suppress sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood** on `.venv312` — cosmetic noise, ~12MB stderr per grid run, unrelated to any bug. Consider `warnings.filterwarnings(...)` in the GUI entry point.

---

## Active Branch context (secondary)

## Active Branch: `claude/contamination-detection-model-PrntN`

**Goal:** Add one-class contamination detection models as first-class citizens alongside regression and classification. One-class models learn from clean/inlier samples only and flag anything outside that distribution as suspect.

**Models:** OneClassSVM, IsolationForest, LOF (LocalOutlierFactor), EllipticEnvelope, PCA-SIMCA

**Prediction tab manual verification: PASSED 2026-04-11.** User confirmed one-class predictions correctly show inliers when they should after the 2026-04-10 order-of-operations fix. See SESSION_LOG.md 2026-04-10 entry for the original bug trace.

---

## What Works

- [x] **Analysis Subset V1** (branch `glm/analysis-subset-v1`, uncommitted): Pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab card, metadata-only dialog with categorical multi-select, subset provenance in training config, mismatch warnings, one-class guardrail, C2 missing-column safe handling.
- [x] One-class model implementations in `src/spectral_predict/contamination.py`
- [x] Bayesian optimization for one-class (`unified_bayesian.py` handles `task_type='one_class'`)
- [x] Grid search for one-class (`run_one_class_search` in `search.py`)
- [x] GUI task type selection: "One-Class" radio button shows one-class model checkboxes
- [x] Inlier class selection UI with auto-detection
- [x] Model save/load with scaler + PCA reducer persistence (`model_io.py`)
- [x] Prediction tab: save→load→predict round-trip verified manually 2026-04-11. Training inliers come back as "Inlier (X)" as expected. Regression test: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`.
- [x] External validation: label mapping, confusion matrix, balanced accuracy/sensitivity/specificity
- [x] External validation metrics in Results tab — top N respects `validation_top_n` (default 700), matching classification/regression
- [x] External validation produces full 7-metric set (Sensitivity, Specificity, Precision, F1, Accuracy, BalancedAcc, AUC) via `compute_validation_metrics_for_top_one_class_models()` in `contamination.py` — parity with cal/CV
- [x] External validation metrics in Model Development results text
- [x] Validation checkbox preserved when loading results into Model Development
- [x] Dancing man animation stops on completion
- [x] Model Development: runs complete, buttons re-enable, cursor resets
- [x] Wavelength importance: surrogate LightGBM via `compute_one_class_importances()`
- [x] Code export: one-class models export as standalone Python scripts and Jupyter notebooks with full CV reproduction
- [x] SHAP: permutation importance for most models, TreeExplainer for IsolationForest
- [x] Diagnostics: decision score distribution + sample classification plots
- [x] Residual/leverage diagnostics route to one-class-specific plots (not regression fallback)
- [x] Basic preprocessing discovery works for one-class grid search (callback wrapper fixed)
- [x] Variable selection: 'importance' method works for one-class
- [x] Results Treeview tooltips: all one-class metrics (Sensitivity/AUC/cv) + validation metrics (RMSEP/R2pred/val_*) now covered; jargon-heavy tooltips (ROC_AUC, Kappa, MCC, BER, LogLoss) rewritten for non-technical audience
- [x] CV strategy support: LOO, Repeated K-Fold, and standard K-Fold via `build_cv_splitter()` factory in `cv_utils.py`. Pooled RMSEcv (regression) and pooled sensitivity/specificity (one-class). GUI controls in Analysis tab + Model Development. Cost estimator with LOO/Repeated warnings. training_config stores cv_strategy for model save/load. Post-review fixes (2026-04-12): RepeatedKFold crash via `cross_val_predict_pooled`, one-class Bayesian cv_strategy forwarding, GUI LOO folds metadata, LOO classification minority-class guard, differentiated mixed-regime warnings.

## Known Issues

- [x] ~~**Bayesian one-class results NaN on Results tab**~~ — Fixed: enabled `compute_calibration=True` in Bayesian objective
- [x] ~~**Scaler/PCA not persisted in grid search results**~~ — Fixed: `cal_scaler`, `cal_pca_reducer`, `oc_score_stats` now stored in result dicts
- [x] ~~**LightGBM label encoding bug**~~ — Fixed: `compute_one_class_importances()` now converts +1/-1 to 0/1
- [x] ~~**"Contamination" naming**~~ — Fixed: comments/docstrings updated where "contamination" referred to task type
- [x] ~~**Performance: IsolationForest/LOF grid bloat**~~ — Fixed: reduced IF n_estimators (300→100), removed redundant LOF configs, skipped unused importance in discovery
- [x] ~~**PCA-SIMCA silently produces 0 rows on small training sets**~~ — Fixed 2026-04-10: `PCASIMCA.fit()` had a hardcoded `n_samples < 10` guard that tripped every CV fold when training inliers were scarce (e.g. ~7 inliers + 5-fold CV → 5-6 per training fold). All 5 folds failed → skip guard → `+inf` every trial → 0 rows in final DataFrame. Other one-class models (OCSVM/IF/LOF/EllipticEnvelope) all have explicit small-sample fallbacks. Relaxed SIMCA floor to `n_samples < 3` (the true mathematical minimum for PCA + chi² moment fit). Also added `skip_reason` surfacing through `run_one_class_cv` → `unified_bayesian` progress wrapper so any remaining skips show "SKIPPED (reason)" in the GUI progress tab instead of a silent `-inf`.
- [x] ~~**Every specimen labeled "Outlier" at predict time, including training specimens**~~ — Fixed 2026-04-10 (second session). Root cause: one-class refinement thread in `_run_refined_model_thread` trained with full-spectrum preprocessing first then subset to selected wavelengths, but hardcoded `'use_full_spectrum_preprocessing': False` in the saved metadata and never populated `full_wavelengths`. At predict time `model_io.py` took the subset-first branch, which for SNV / SG derivatives / baseline correction produces feature values mathematically different from training, so the StandardScaler fit inside `run_one_class_cv` received distributionally alien inputs and `model.predict()` returned -1 on everything. Fix: assign `self.refined_full_wavelengths = list(original_wavelengths)` and flip the flag to `True`, matching the regression/classification refinement path. Regression test added: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`. See `docs/SESSION_LOG.md 2026-04-10 "One-Class Prediction Disaster"` for full trace.
- [x] ~~**Uncertainty subtab always shows "Outlier" for one-class**~~ — Fixed 2026-04-10 (second session). `_display_uncertainty()` at line ~38318 read already-stringified predictions (`"Inlier (…)"` / `"Outlier"`) from `predictions_df` but compared them to the integer `1`, which always evaluated False. Would have silently contradicted the Results tab once the primary bug above was fixed. Replaced with an `isinstance(raw, str)` branch.
- [x] ~~**Bayesian trials did not persist `cal_scaler`, `cal_pca_reducer`, `oc_score_stats`**~~ — Closed 2026-04-10 (second session). Latent bug: `unified_bayesian.py` stored metrics and preprocessing config but not the fitted calibration artifacts, so any future direct-save-from-results path would silently save `None`. Added `set_user_attr` calls mirroring `search.py:5171-5173` and propagated the three keys through `convert_study_to_dataframe`. Does not change current user-visible behavior because saves go through refinement, which re-runs CV from scratch.
- [x] ~~**Grid-search one-class validation columns silently empty**~~ — Fixed 2026-04-11. Grid search wrote display name (`'snv_deriv1_w11'`) in `Preprocess` but never set `PreprocessBase`, so the validation helper passed an unparseable name to `build_preprocessing_pipeline`, the resulting `ValueError("Unknown preprocess: …")` was swallowed by the row-level except, and val_* dropped to NaN for every derivative grid-search row. Bayesian was unaffected because it already wrote `PreprocessBase`. Two-layer fix: search.py now writes `PreprocessBase` (mirrors classification at line 4506), and `compute_validation_metrics_for_top_one_class_models` now normalizes display names as a fallback. Regression test: `tests/test_contamination_detection.py::TestGridSearchValidationMetricsParity`. See `docs/SESSION_LOG.md 2026-04-11`.
- [ ] **One-class grid search inherently slower than classification** — Expected due to two-phase CV+calibration design and LOF O(n²) complexity. Optimized but still slower by nature.
- [ ] **Preprocessing importance dropdown has no effect for one-class** — All methods resolve to LightGBM. Per-model refinement never triggers because `models_to_test` isn't passed.
- [ ] **Variable selection limited** — Only 'importance' method works. UVE/SPA/iPLS/CARS are PLS-specific and incompatible. This is by design but could use a UI hint.
- [ ] **Residual correlation overlay** — Disabled for one-class (no continuous residuals). Shows zeros.
- [x] **One-class hyperparameter grids now exposed in Model Config subtab** — Fixed 2026-04-19. Five collapsible cards in Tab 4C (OneClassSVM, IsolationForest, EllipticEnvelope, LOF, PCA-SIMCA) using old-style BooleanVar + Checkbutton pattern. Untouched cards preserve curated default grids. Customized cards use per-model Cartesian product. Bayesian one-class path unchanged.
- [ ] **LOO / Repeated K-Fold results may not auto-populate the Results tab** — observed 2026-04-12 during manual testing of PR #4 (cv-strategy-overhaul). User ran LOO on a 49-sample regression dataset; analysis completed without errors, but the Results tab did not display the completed runs as expected. Repeated K-Fold uncertain — not explicitly tested for the same behavior. Save/load/predict round-trip from a refined LOO model works. Possible causes to investigate: (a) Results tab's Treeview populate path is gated on `Folds` matching a specific integer range or on `training_config['cv_strategy'] == 'kfold'`; (b) progress callback's final "completion" signal isn't wired for non-kfold strategies; (c) sort/filter logic in `_populate_results_treeview` (or equivalent) drops rows where `Folds == n_samples` (LOO case). Needs repro + trace before fixing.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Surrogate LightGBM for wavelength importance | Model-agnostic, fast, already implemented in `compute_one_class_importances()` |
| Permutation importance instead of SHAP KernelExplainer | KernelExplainer is impossibly slow for spectral data (hundreds of wavelengths × thousands of model evals per sample) |
| TreeExplainer only for IsolationForest without scaler/PCA | Only case where SHAP TreeExplainer works directly |
| Validation metrics: balanced accuracy, sensitivity, specificity | Standard one-class metrics. Sensitivity = outlier detection rate, Specificity = inlier retention rate |
| String comparison for inlier labels | Both Bayesian and grid search convert y and inlier_label to strings before comparison to handle numeric/text label mismatches |
| Early return pattern in GUI threads | One-class has separate pipeline from regression/classification. Every early return MUST include cleanup (animation stop, button reset, etc.) |

## Key Files

| File | Role |
|------|------|
| `spectral_predict_gui_optimized.py` | Main GUI (~45K lines). One-class paths scattered throughout. |
| `src/spectral_predict/contamination.py` | One-class model implementations, `run_one_class_cv()`, `compute_one_class_importances()` |
| `src/spectral_predict/search.py` | `run_one_class_search()` for grid search |
| `src/spectral_predict/unified_bayesian.py` | Bayesian optimization, handles `task_type='one_class'` |
| `src/spectral_predict/scoring.py` | Scoring functions with one-class metrics |
| `src/spectral_predict/model_io.py` | Save/load with scaler/PCA persistence |
| `src/spectral_predict/preprocessing_discovery.py` | Smart preprocessing, has one-class path at line ~680 |

## Next Steps

1. ~~End-to-end manual test of save→load→predict~~ ✅ done 2026-04-11
2. ~~Re-run grid-search OC validation in the GUI to confirm `val_*` columns now populate~~ ✅ done 2026-04-11 — user confirmed "the validation stats now appear in grid search"
3. ~~Merge PR #3 to main~~ ✅ pending merge in this session
4. Follow-up PRs (post-merge): see "Follow-Ups (unclaimed)" section below

## Follow-Ups (unclaimed)

- **Model-native loss reweighting for class imbalance — not exposed for boosting models or PLS-DA's inner LR (flagged 2026-04-29).** Today the Data Quality tab handles imbalance via resampling (SMOTE family) and `class_weight='balanced'` is injected only into `RandomForest` / `LogisticRegression` / `SVC`. The boosting model construction sites in `models.py` (XGBoost line 285, LightGBM 307, CatBoost 327) never receive `scale_pos_weight`, `is_unbalance`, `class_weight`, or `auto_class_weights` — verified by grep, zero matches in `src/`. PLS-DA's inner `LogisticRegression` (`search.py:380`, `:4261`, `bayesian_utils.py:400`) is also stuck at default `class_weight=None`. This means standard chemometrics configurations like the FTIR Bone PLS paper's "XGBoost (scale_pos_weight)", "LightGBM (balanced)", and "Logistic regression (EN, balanced)" are not reproducible in dasp without source edits or routing through resampling (which is a different statistical intervention). Proposed fix: per-model "Imbalance handling" dropdown with four options (Equal sample weights / Auto / Class-balanced loss reweighting / scale_pos_weight custom). Auto mode reuses `detect_class_imbalance(y, threshold=3.0)` from `imbalance.py:61` to decide per-fold at fit time. Important UX: must clearly distinguish "Equal sample weights" (current default — does NOT mean balanced; means majority class dominates the loss) from "Class-balanced loss reweighting" (corrects the imbalance), since the standard library defaults are widely misunderstood. Full proposal in `docs/plans/2026-04-29-model-native-loss-reweighting.md`. Effort estimate ~1-2 days, low risk because default preserves all existing behavior.

- **Expose remaining one-class hyperparameters (partial exposure shipped 2026-04-19).** The 2026-04-19 parity work added Tab 4C cards covering the parameters already in the curated `get_one_class_model_grids()` defaults — OCSVM `kernel`/`gamma`/`nu`/`degree`, IF `n_estimators`/`contamination`/`max_features`, EllipticEnvelope `contamination`, LOF `n_neighbors`/`contamination`, PCA-SIMCA `n_components`/`alpha`. Deferred parameters that sklearn supports but the GUI still does not surface: **LOF `metric`**, **IsolationForest `max_samples`**, **OneClassSVM `coef0`** (and additional kernels like `sigmoid`/`linear`), **EllipticEnvelope `support_fraction`**. Also not yet exposed: additional predefined values beyond the shipped curated grid (e.g. OCSVM `nu=0.2`, IF `n_estimators=200/500`, LOF `n_neighbors=50`, PCA-SIMCA `alpha=0.10`) — users can enter these as custom values today, but there's no predefined checkbox. Finally, the one-class **Bayesian** path was deliberately out of scope for 2026-04-19 and still uses the hardcoded parameter ranges in `unified_bayesian.py::suggest_one_class_params()` rather than anything from the new cards; any future work to honor user-customized grids in Bayesian search would need a separate design. Exposing the deferred parameters requires: (a) extending the curated default grid in `get_one_class_model_grids()`, (b) adding checkbox rows to the corresponding Tab 4C card in `spectral_predict_gui_optimized.py::_create_tab4c_model_configuration`, (c) extending the per-model builder in `_resolve_one_class_model_grids()` in `src/spectral_predict/search.py`, and (d) adding default-state comparison entries in `_collect_one_class_model_param_overrides()` so the curated-default preservation invariant still holds. The 2026-04-19 plan's "Non-Goals" section lists these explicitly as future follow-ups.

- **`predict_with_uncertainty` swallows one-class decision_function failures silently.** `model_io.py:854-860`: after `predict_with_model()` returns, the bare `except Exception: decision_scores = None` catches any score-extraction error and returns predictions with empty `uncertainty`/`applicability_domain` payloads. The GUI has no way to surface "predictions worked, uncertainty broke" — exactly the failure mode this PR has already hit once with the order-of-operations bug. Fix: log the exception with `logger.warning(...)` at minimum; ideally return a structured `decision_score_error` flag that the GUI can display alongside the predictions.

- **OC validation helper can't recover the right `polyorder` for 2nd-derivative grid-search rows.** Grid search writes `polyorder=None` (delegating to `polyorder_map` inside `SavgolDerivative` which picks 3 for `deriv=2`), but the validation helper at `contamination.py:922-927` falls back to `min(2, window-1) if window > 2 else 0`, which gives `poly=2` for `deriv=2`. The pipeline then runs with the wrong polyorder and the resulting `val_*` metrics are slightly off vs. what training actually used. Fix: either store the resolved polyorder in the grid-search result dict (most direct), or import the same `polyorder_map` lookup into the validation helper, or have `_maybe_int` fall back to `polyorder_map[deriv]` when poly isn't set.

- **CV Strategy Phase 2: Propagate outer CV strategy into variable selection inner loops.** Currently inner loops (UVE, SPA, iPLS, CARS, GA-PLS) always use hardcoded 5-fold K-fold regardless of the outer CV strategy. A GUI warning is displayed when outer != kfold. Phase 2 would thread `cv_strategy`/`cv_n_repeats` into the variable selection internals.

- **CV Strategy Phase 2: Propagate CV strategy into predictor screening and smart preprocessing.** These internal CV sites are currently hardcoded 5-fold and don't respect the user's chosen strategy.

- **CV Strategy Phase 2: NSGA-II multi-objective search CV strategy support.** Currently falls back to K-fold with a logged warning when non-kfold strategy is selected. Needs native LOO/Repeated K-fold support.

- **One-class search: Add `training_config` to result rows.** Regression/classification grid search writes `training_config` (with `cv_strategy`, `folds`, `cv_n_repeats`) but one-class search does not. This means one-class saved models don't preserve the CV strategy used during training.

- **One-class export lacks automated regression tests.** The existing 33 export tests (`tests/test_code_generator.py`, etc.) do not exercise the `task_type='one_class'` path. Manual verification confirms all 5 models compile, but a future refactor of `templates/validation.py` or `code_generator.py` could silently break one-class export. A single parameterized test generating scripts for all 5 OC models would be cheap insurance.

- **Template string formatting fragility in one-class CV block.** `CROSS_VALIDATION_ONE_CLASS_TEMPLATE` uses `.format()` with `{model_name}` and `{x_var}` variables inside a large multi-line template string. This is correct today but slightly fragile: a future edit that introduces an additional brace pair (e.g. a dictionary literal `{` `}`) would cause a `KeyError` at generation time. This is pre-existing technical debt in the template system, not introduced by this commit, but worth noting if the template engine is ever refactored.

- **One-class export lacks per-fold CV statistics printout (parity bug).** Regression export prints pooled metrics **and** per-fold breakdown (`print(f"\\nPer-fold RMSE: {{[f'{{x:.4f}}' for x in fold_rmse]}}")`). One-class CV template computes `fold_metrics` list but `METRICS_ONE_CLASS_TEMPLATE` prints only pooled metrics, not the per-fold details. This is a **parity mismatch** with the regression export experience. Classification also lacks per-fold printout but gives detailed confusion matrix + classification report instead, which one-class already does via its own confusion matrix plot. Still, for parity with regression's explicit per-fold reporting, one-class should print per-fold sensitivity/specificity/AUC if available.

- **Pause/resume hardening — worth real design work (flagged 2026-04-23).** User lost hours on a CatBoost run after Pause → Resume appeared to do nothing; log was gone by the time we could inspect it (progress pane is widget-only, no disk log). Diagnosis surfaced four real gaps in the current `SearchController` implementation (`src/spectral_predict/search_controller.py` — two `threading.Event`s; worker calls `check_and_wait()` at checkpoints):
  1. **Coarse checkpoint granularity in the Bayesian path.** `unified_bayesian.py:1826` only checks pause in Optuna's `progress_wrapper` callback, which fires *between* trials. A single CatBoost trial on a 19-hour search can be 10–30+ minutes. Pause's UI flips to "paused" instantly but the thread is still burning CPU until the current trial finishes — and if the user Resumes before that happens, pause/resume is a silent no-op.
  2. **UI lies about pause state.** `_pause_search` (gui:22842) calls `_update_search_buttons('paused')` the instant it's clicked, regardless of whether `check_and_wait()` has actually been reached. Should display an intermediate "pausing — waiting for current trial to finish…" until the thread actually blocks.
  3. **No thread-alive check on Resume.** If the worker thread silently errored during the paused window (exception caught by the `except Exception` at gui:27664 which logs `[X] Error:` and exits the thread), `_resume_search` (gui:22849) just sets the pause event — nobody is waiting on it, no feedback to the user, UI returns to "running" state with a dead thread. Resume should call `self.analysis_thread.is_alive()` and surface a clear error if the thread has died.
  4. **Zero persistence.** Pause is purely in-memory. Any crash, OOM, Windows update, or process exit while paused loses everything. Optuna has built-in SQLite persistence (`storage="sqlite:///..."`) — adding this would make resume survive process death and make a 19-hour run recoverable; likely the single highest-value change.
  Bonus gap: the progress log is Tkinter-only (`_log_progress` at gui:27770 only writes to `self.progress_text`), capped at 2000 lines (gui:27782), never mirrored to disk. Post-mortem of any long run is impossible once the app closes. A simple `logging.FileHandler` writing to `outputs/logs/<run-timestamp>.log` would have let us diagnose this incident definitively.

## Analysis Subset V1 — Known Limitations / Risk (2026-04-19)

Blocking bugs (stale `active_indices` after dataset replacement or row deletion) fixed in this session. Remaining deferrals:

- **Integration-level GUI tests not yet written:** reload-with-subset, row-delete-with-subset, revert-after-column-delete, and mismatch-warning text. Pure-logic coverage (41 tests in `test_analysis_subset.py`) is solid, but no automated test drives the GUI through these multi-step scenarios.
- **Manual GUI verification checklist still pending:** end-to-end validation that the Analysis-tab card, data-viewer highlighting, provenance in training config, and one-class guardrail all behave correctly after each dataset-replacement path.
- **`_use_for_analysis()` does not carry `combined_metadata_df`** from data sources. If a user loads data via Data Management → Use for Analysis, metadata columns may be absent, causing the subset dialog to show no columns. This is a pre-existing limitation of the data-source path, not introduced by Analysis Subset V1.
