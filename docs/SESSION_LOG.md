# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

## 2026-05-04 overnight (T-30b + P3 + T-20c PLS-DA/MLP) — author-intent-tagged debug blocks; the asymmetry between two files IS the finding; dead-code defensive scaffolding masquerading as a bug

**Three more PRs added on top of the session-2026-05-04 batch — all pushed, none merged.**

Three non-obvious lessons earned this overnight session:

**1. `calibration_transfer.py` vs `nsga2_search.py` — the asymmetry between the two files IS the T-30b finding.** Continuation prompt assumed both files needed triage. Reality: `calibration_transfer.py`'s CTAI block (~25 prints) was tagged with `# Enhanced validation with debug logging` at the source — the author's stated intent was debug logging but they used `print()` everywhere, including a giant `=== CTAI Debug Information ===` header. Meanwhile `nsga2_search.py`'s 43 prints are ALL properly gated behind `verbose >= 1` / `verbose >= 2` / `gen % 10 == 0` flags or are unconditional warnings (CV-strategy fallback, no-feasible, user-cancelled). The module already has `logger` set up. The author structured nsga2_search correctly from the start. **For files with pre-existing verbosity gating, the right action is "leave alone" — deleting prints would degrade UX.** T-30b's value lives entirely in CTAI; nsga2 was already done. The lesson: read the author's intent in comments and verbosity gates BEFORE bulk-editing prints.

**2. "Looks like a real export bug" can be unreachable defensive scaffolding.** P2 (Ridge classifier export investigation) traced what looked like a clean codegen mismatch: `code_generator._resolve_model_ctor_class` returns `'Ridge'` (regressor) for `task_type='classification'` — should be `'RidgeClassifier'`. The runtime branch in `models.py:471-473` builds `RidgeClassifier(random_state=42, **params)` for `model_name='Ridge', task_type='classification'`, suggesting Ridge classification is a real feature with a broken export path. **But verify reachability before fixing.** `model_registry.CLASSIFICATION_MODELS`, `model_config.CLASSIFICATION_TIERS`, the GUI's `valid_models_classification` list (lines 32479 + 33425), and `is_valid_model('Ridge', 'classification')` ALL exclude Ridge from classification. The runtime branch is dead code — defensive scaffolding for a feature that never shipped. The export-resolver mismatch is consistent-with-dead-code, not a bug. **Per the user's "engineering polish defers" rule, adding Ridge classification as a real feature is out-of-scope; the right move was to file as a deferred ticket and walk away.** Saved ~1-2h that would have gone into a feature-disguised-as-fix.

**3. Cross-family review for tiny test-only PRs is still worth dispatching when the test machinery is non-trivial.** T-20c added a PLS-DA parity row (~75 lines) and an MLP marker test. Both passed locally on first try. The continuation prompt called cross-family review "WORTH IT given the Pipeline complexity." GLM 5.1's review verified the param-routing trace through `_split_pls_da_params` (codegen line 1261) → `_render_pls_da_pipeline` (codegen line 1280) and confirmed the in-process Pipeline lands on identical PLSTransformer + LogisticRegression params. GLM caught a LOW about the codegen's `_lr_filtered = {k: v for k, v in _lr_params.items() if k in _lr_sig.parameters}` filter — pre-existing latent fragility (silent drop if sklearn renames a param), not introduced by this PR, no action required. **For test-machinery tickets, the review value is verifying the test catches the bug class it claims to catch.** GLM's per-axis verification (param routing, PLSTransformer equivalence, determinism, test value) is exactly the right structure for a parity-test review.

**Operational lessons (carried forward):**
- **Commit before yielding control** — followed throughout this session (4 separate commits across 3 branches before any agent dispatch). No losses.
- **For stacked PRs that depend on unmerged parent branches: branch off the parent, not main.** P4's PLS-DA parity row needed the T-20 test-runner helpers from PR #22 → #23 → #24's stack. Branching off `fix/Tcv-catboost-multiclass-cv-pooling-scalarise` (PR #24's branch) was correct; the attempt to branch off main first surfaced the helper absence immediately.
- **DEFAULT_PARAMS merge mirroring is still the parity contract** — MLP marker test required `from spectral_predict.templates.models import DEFAULT_PARAMS` + `params_full = DEFAULT_PARAMS['MLPClassifier'].copy(); params_full.update(test_params)` to reproduce the codegen's full param set in-process. Hardcoding MLP defaults would have been brittle; importing the same source-of-truth is the right pattern.

---

## 2026-05-04 (T-14b + T-20 + T-20b + codegen fix + T-29b — five PRs in flight) — regression-net loop closed in same session; CatBoost multiclass shape quirk; predict_proba is load-bearing for classification parity

**Five tickets shipped in one session via stacked PRs (#21-#25), none merged yet — user opens/merges.**

Five non-obvious discoveries earned this session:

**1. CatBoost multiclass `predict()` returns shape `(n, 1)`, not `(n,)`.** sklearn classifiers return 1-D class labels. CatBoost multiclass returns a 2-D column vector — one column with the predicted class label per sample. The `templates/validation.py` CV majority-vote pooling block (`Counter(preds_per_sample[i]).most_common(1)[0][0]`) couldn't hash 1-element ndarrays → `TypeError: unhashable type: 'numpy.ndarray'` before the exported script reached its metrics section. **Every CatBoost multiclass user got a broken exported script silently — until T-20b's parity test caught it.** Fix in PR #24 is one line: `np.ravel(y_pred_fold[local_i])[0]` at the append site (no-op for sklearn (n,) shape, unwraps the (n, 1) case). DeepSeek empirically reproduced the shape via `CatBoostClassifier(iterations=10, depth=3, verbose=0).fit(X, y).predict(X[:3]).shape` returning `(3, 1)`.

**2. `predict()` vs `predict_proba()` in classification parity tests is load-bearing — hard labels can mask probability drift up to ~0.3 silently.** Initial T-20 commit compared `model.predict(X_test)` (hard class labels) for classification rows. DeepSeek's pr-test-analyzer review ran an empirical check: with XGBoost class_weight handled via `scale_pos_weight` (in-process) vs `sample_weight=compute_sample_weight('balanced', y)` (codegen path), probability divergences hit **0.296** while every single hard label remained identical. Hard labels are quantised at the class boundary; small probability shifts that don't cross the boundary are invisible. The fix-of-fixes switched classification to `predict_proba` with `rtol=1e-3, atol=1e-6`. **A regular test that passes despite a real regression is worse than no test** — false confidence is the worst signal. predict_proba is the right comparison granularity for classifiers; predict is fine for regression (continuous output).

**3. The DEFAULT_PARAMS merge pattern in `code_generator._render_model` is the parity contract.** When the runtime saves a model, the saved object uses whatever params the runtime constructed it with. The exported script reconstructs the model from scratch and merges user params with `templates/models.DEFAULT_PARAMS[<model>]`. So an in-process parity test model must mirror the runtime's param assembly, NOT call sklearn's defaults directly. PLS in this codebase always uses `scale=False` (24+ runtime sites; chemometrics convention to avoid double-scaling SNV-preprocessed data); the codegen mirrors via `DEFAULT_PARAMS['PLS'] = {'n_components': 10, 'scale': False}`. Initial T-20 PLS row used sklearn's default `scale=True` and got divergent predictions — looked like a parity bug, was actually a test bug. **The right reference for a parity test is "what the runtime would have saved," not an arbitrary sklearn instantiation.**

**4. XGBoost has no `class_weight` constructor kwarg — uniform user-facing imbalance handling translates to per-library mechanism translation under the hood.** The codebase's "model-native imbalance handling" (T-19 reframed) uses each library's native mechanism: sklearn classifiers / LightGBM / RidgeClassifier accept `class_weight='balanced'` constructor kwarg; CatBoost uses `auto_class_weights='Balanced'` (different name, different value); XGBoost uses `sample_weight=compute_sample_weight('balanced', y)` at fit time (no native constructor kwarg). MLP has nothing — explicitly unsupported in the codebase. **The user-facing interface is uniform** (one `imbalance_method='class_weight'` dropdown), but the translation under the hood is per-library because the library APIs differ at the call site. This is what T-19 shipped, working as designed. The user reaffirmed mid-session — earlier framing memos that read as "uniform sample_weight everywhere" were superseded by the model-native approach the codebase now uses.

**5. Stacked PRs with chained bases keep cross-family review cost linear.** Five PRs landed in the queue, three of them on a chain: T-20 (PR #22, `base=main`) → T-20b (PR #23, `base=fix/T20-...`) → codegen fix (PR #24, `base=fix/T20b-...`). Reviewers diff each child against its base, not against main, so they only review the new content per layer. Without stacking, T-20b would have to wait for T-20 to merge before being reviewable; same cascade for the codegen fix. After T-20 merges, T-20b rebases onto main and its base auto-updates; same for the codegen fix when T-20b merges. The mechanical rebase chain is the cost, paid once per stacked PR. The benefit is that a regression net (T-20b) and the bug it caught (codegen fix in #24) can be reviewed and shipped in parallel rather than sequentially. **The regression-net loop (T-20 → T-20b → codegen-fix) closed within the same session.**

**Operational lessons (carried forward):**
- **Commit before yielding control** — long pytest runs / agent dispatch / `git push` can collide with parallel sessions. Followed throughout this session, no losses. Memory `feedback_commit_before_yielding_control.md` still load-bearing.
- **`pytest | tail` masks pytest's exit code** — used `2>&1 | grep -E "passed|failed"` consistently this session.
- **opencode-call agents may end on a different branch than they checked out** — verified branch state with `git branch --show-current` after each agent return.
- **Cross-family review (DeepSeek + GLM via opencode-call) earned its slot again** — 4 of 5 PRs in this session had at least one HIGH or MEDIUM finding the reviewers caught. The pr-review-toolkit's specialised agents (test-analyzer + silent-failure-hunter) caught HIGHs that the cross-family pair missed (predict_proba switch, sample_weight matching codegen, balanced-data auto test). Each review layer had a non-overlapping yield.

---

## 2026-05-03 (Quick-wins batch — T-50/T-14/T-29/T-32/T-30) — pipe-tail masks pytest exit code; cross-family review caught 1 bug class GLM missed + 1 false positive

**Five tickets shipped in one session via 5 squash-merge PRs (#16-#20)** to main at `1d3eba0`. Cross-family review pattern (DeepSeek + GLM in parallel via opencode-call) per ticket, fix-of-fixes commit per ticket, GLM consolidated sanity-recheck on the 4 fix-of-fixes commits → all `READY_TO_MERGE` with no HIGH/MEDIUM findings. Post-merge regression sweep 327 + 1 skipped, zero failures.

**Three non-obvious lessons earned this session:**

**1. `pytest | tail` silently masks pytest's non-zero exit code.** During T-29's "atomic burst" pattern (`compile && pytest && commit && push`), I piped pytest output through `tail -10` to limit output length. The pipe inherits `tail`'s exit code (always 0), so the chain proceeded to `git commit && git push` despite a failing test (`test_t29_metric_failure_logs_warning` had a wrong premise about modern sklearn's roc_auc behavior). The fix-of-fixes commit `d0df57b` corrected the test, but only because I noticed the failure in scrollback. **Pattern fix:** drop the `| tail` in chain-commit patterns — use `2>&1 | grep -E "passed|failed"` if you need length-limited output (grep returns 0 when matches found, propagating non-failure correctly), or split into separate steps. Memory: this got captured in T-29 commit message rather than as a separate file.

**2. Cross-family review (DeepSeek + GLM in parallel via opencode-call) earned its slot — 5/5 tickets had at least one HIGH/MEDIUM finding I missed, despite confidence the fixes were complete.** Across the batch:
   - **T-14**: DeepSeek caught 3 HIGH same-class drift sites I missed (`model_io.py`, `templates/header.py`, `export_bundle.py`). Highest blast radius — every exported script claimed "v3" major version.
   - **T-29**: DeepSeek caught 1 HIGH (`balanced_accuracy_score` was the only metric outside the new try/except — half-fixed asymmetry).
   - **T-30**: BOTH reviewers independently caught the missed `[PLS-DA DEBUG]` block (DeepSeek HIGH, GLM MEDIUM). Bracket-text variation `[PLS-DA DEBUG]` vs `[DEBUG]` slipped my regex sweep.
   - **T-32**: GLM caught the early-stopping branch architectural landmine (same bug class, different code path, no crash today but fragile).
   - **T-50**: GLM caught `.sqlite3-journal` sibling (rollback-journal mode fallback when WAL is rejected) and `is_relative_to` defense-in-depth gap.

**3. DeepSeek had 1 false positive in this batch (T-50).** Claimed `bytes_freed += sibling_size` ran *before* `sibling.unlink()`. Actually ran AFTER (line 813 vs 812 in same try block — OSError correctly skips increment via except). Verified by reading the actual code, not just trusting the verdict. **Reviewers aren't infallible** — verify findings against actual source before applying. Net positive: 1 false positive vs 6+ real catches across the batch.

**Operational lessons (carried over from T-19, reaffirmed):**
- **Commit before yielding control.** Long pytest runs / agent dispatch / `git push` can collide with parallel sessions or hooks that swap branches and clobber uncommitted working-tree work. T-50 lost ~30 min to this; T-14/T-29/T-32/T-30 followed an "edit → compile → commit IMMEDIATELY → only then run long sweeps" pattern with no further loss. Memory: `feedback_commit_before_yielding_control.md`.
- **opencode-call agents may end on a different branch than they checked out** (or never check out at all — they often use `git show <sha>:<path>` for read-only review, leaving the worktree on whichever branch was active at dispatch time). Verify branch state after each agent return; don't assume.

---

## 2026-05-02 (T-19 Auto mode) — scope-correction trap: don't conflate "expose" walkback with Auto deferral

T-19's user-framing memo (`project_t19_user_framing.md`) called for either (a) per-model UI dropdowns or (b) Auto mode. Mid-session the user said: "the issue is correctness and speed. if we dont need to expose and do as good with just 2 then that si fine." I read that as deferring Auto mode AND per-model dropdowns. Wrong. The user clarified shortly after: **"no, that memory is wrong. the automode is the highest priority. another agent is also working but it needs to be done."** The walkback was about (a) per-model UI surgery (separate dropdowns for `is_unbalance` / `scale_pos_weight` / `auto_class_weights`), not (b) Auto mode itself. Auto mode IS the simpler-shape interpretation of "expose or auto-detect" — single dropdown option, automatic.

**Lesson for scoping decisions on user feedback:** when a user says "if X isn't needed that's fine," confirm what X refers to before shrinking scope. Pattern-match the comment to the *specific* prior ask it might walk back, not to the whole feature. The framing memo had two acceptable shapes; the walkback only ruled out one of them.

**Auto mode mechanics shipped (commit `0b1d4a2`):** new `'auto'` GUI dropdown option; helper `imbalance.resolve_auto_imbalance(y, task_type, threshold=3.0)` returns `('class_weight', info)` or `(None, info)`; runtime resolution at the entry of `run_search` / `run_nsga2_search` / `run_unified_bayesian` mutates `imbalance_method` in-place so downstream code paths fire correctly without needing to know about `'auto'`. Code generator emits a runtime-resolution block at the end of `_render_imbalance_handling()` that mutates `IMBALANCE_METHOD` post-data-load in the generated script. Per-library kwargs (`auto_class_weights` for CatBoost, `sample_weight` for XGBoost via `fit_kwargs`) are baked at codegen time as if `'auto'` were `'class_weight'`; on balanced data the runtime resolution prevents the `sample_weight` block from firing and the baked `class_weight='balanced'` is mathematically a no-op (uniform per-class weights). Resolution happens once at run-entry against global y, not per-fold — stratified CV preserves global ratios tightly enough that per-fold drift rarely flips the decision.

**Operational lesson (separate but co-occurring this session):** opencode-call agents leave the working tree on the branch they checked out, AND parallel sessions on other machines can stash/checkout/reset the working tree underneath you. **Commit ASAP after implementation, before any operation that yields control** (long pytest, agent dispatch, even `git push`). Memory: `feedback_commit_before_yielding_control.md`. Also lost ~10 minutes mid-session when a parallel session stashed my Auto-mode draft as `user-deferred-T19-auto-mode-draft` and reset HEAD; reflog + stash list recovered it intact.

---

## 2026-05-02 (T-19 reframed — exported-code class_weight bugs) — XGBoost silent no-op + CatBoost/MLP TypeError

T-19 reframed scope (per `project_t19_user_framing.md` — "expose model-native abilities, not paper reproducibility") surfaced three real bug classes in `code_generator.py`'s `imbalance_method='class_weight'` path:

1. **CatBoost catch-all-else branch (`code_generator.py:917` pre-fix):** `params_full.setdefault('class_weight', 'balanced')` for CatBoost → `TypeError: unexpected keyword argument 'class_weight'` on `CatBoostClassifier.__init__`. Hard crash on instantiation. Verified empirically before the fix.

2. **XGBoost catch-all-else branch (same site):** `XGBClassifier(class_weight='balanced')` does NOT raise. `class_weight` lands in `get_params()` but XGBoost's loss function ignores it entirely at fit time. **Silent unweighted training while the user believes imbalance handling was applied.** This is the worst kind of bug — no error, no warning, wrong predictions. Discovered via empirical instantiation test, not docs.

3. **MLP StandardScaler-wrapped branch (`code_generator.py:904` pre-fix):** same `TypeError` as CatBoost. The `_needs_standard_scaler()` branch (`SVC/MLP/MLPClassifier/NeuralBoosted/Ridge/Lasso/ElasticNet`) was unconditionally injecting `class_weight='balanced'`. **I missed this in the initial T-19 fix because I had wrongly asserted to my reviewers that the StandardScaler path was correct-and-untouched.** GLM 5.1 caught it. Lesson: cross-family RLHF orthogonality earns its slot when I'm asserting things I haven't fully verified.

**Why the runtime path doesn't crash for any of these:** `search.py:4418-4435` uses `hasattr(model, 'class_weight')` to gate the kwarg injection. Default `XGBClassifier()` and `CatBoostClassifier()` have NO `class_weight` attribute → falls through to sample_weight at fit() (which works). MLP same path → sklearn≥1.7 sample_weight, otherwise warning. Runtime is correct; only EXPORTED code was broken because the export emits `class_weight='balanced'` directly into the constructor.

**Fix:** library-aware dispatch per kwarg shape:
- CatBoost → `auto_class_weights='Balanced'`
- XGBoost → no `__init__` kwarg; thread `sample_weight=compute_sample_weight('balanced', y)` into per-fold + final fit() calls (uniformly correct for binary AND multiclass; mirrors runtime)
- MLP → no `class_weight` injection; mirror runtime fallback (unweighted, sklearn floor 1.5 below 1.7 sample_weight requirement)
- Others (LightGBM/RandomForest/sklearn LR/SVC/NeuralBoosted) keep `class_weight='balanced'` — works natively.

**Conditional fit_kwargs emission (GLM MEDIUM):** initial T-19 commit emitted `fit_kwargs = {}` and `**fit_kwargs` for ALL classification models even when sample_weight wasn't being threaded. Pure noise in generated code; readers wonder what the empty dict was supposed to carry. Fix-of-fixes commit makes the `fit_kwargs` plumbing conditional on `xgb_sample_weight=True`.

**Indentation discipline noted in code:** `sample_weight_block_cv` prefixes 4 spaces (lands inside for-loop body); `sample_weight_block_final` prefixes 0 spaces (module-level). Comment in template warns against "normalizing" prefixes during refactor — would silently break layout. Tests pin via `exec()` of generated code, so refactor breakage fails fast in CI.

**Cross-family review pattern (3 reviewers, all closed):** GLM 5.1 (twice — once via diff bundle through llm-call/z.ai, once via opencode-call after user routing correction) + DeepSeek V4 Pro Max via opencode-call. GLM caught the MLP gap I had wrongly asserted was correct. DeepSeek validated multiclass `compute_sample_weight` correctness (`n_samples / (n_classes * count)` formula handles n_classes>2 uniformly; XGBoost's `fit(sample_weight=...)` accepts per-sample weights in multiclass mode). Two LOW findings deferred (duplicate `compute_sample_weight` import in CV + final-model sections — Python handles dedup as no-op; `startswith('XGB')` redundant after canonicalization — harmless until a hypothetical `XGBaseline` model is added).

**Operational lesson:** opencode-call agents leave the working tree on the branch they checked out. After the agent returns, **verify branch state before continuing** — I lost a few minutes when an MLP edit landed on `main` instead of `fix/T19-expose-model-native-imbalance` because the GLM agent had `git checkout main`'d during its review. Memory `feedback_parallel_review_reroute.md` captures the routing-correction lesson; the branch-state lesson is implicit in this entry.

---

## 2026-05-02 (open-ticket re-validation pass) — roadmap doc had drifted from memory; reconcile through amendment block, not rewrite

User asked: "look through all of the open tickets and check that they are all 'real', not problems we are trying to solve that are not relevant for chemometric literature." Recurring failure mode in this repo: tickets get filed by importing sklearn-pipeline-purity instincts onto per-spectrum chemometrics ops (T-01/T-02/T-03 leakage panic, T-08 CARS framing). The April 2026-04-30 master-rule re-evaluation cleaned that out, but three ticket dispositions had evolved further in memory without the canonical roadmap doc tracking it: T-15 was DROPPED (memory `project_t15_dropped_t16_reframed.md`), T-16 was REFRAMED to "competitive model-comparison machinery survey" (same memory), T-19 was REFRAMED to "expose model-native abilities, not reproduce a publication framework" (memory `project_t19_user_framing.md`). Roadmap doc still showed all three as KEEP under the original framings.

User confirmed the three pre-existing reframes during this session and made one new decision: **T-31 (Multi-class SIMCA) confirmed as a real need** (was NEEDS_USER_DECISION). Bone-FTIR / diagenesis specimens can legitimately belong to multiple classes or none; discriminant classifiers force every specimen into one trained class with no "none of the above" output. Saved to memory: `project_t31_simca_confirmed.md`. Implementation must be true multi-class SIMCA per Oliveri & Downey 2012 (one PCA model per class + independent membership decisions per specimen) — NOT an extension of the existing one-class `PCASIMCA` in `contamination.py` (single-class membership detector, structurally wrong base for class-modeling).

Also surfaced: T-12 (disk-mirrored logging) is fully **subsumed by T-45** (logger warnings invisible in bundled GUI, current branch `fix/T45-logger-file-handler`). Same underlying need; T-45 has the concrete plan. T-12 closed without separate work.

**Approach for the doc update — preserve audit trail, don't rewrite:** added a top-level `## Amendments — 2026-05-02 (user re-validation pass)` block listing the six dispositions (T-12 SUBSUMED, T-15 DROP, T-16 REFRAMED, T-19 REFRAMED, T-22 REFRAMED-DEFERRED, T-31 KEEP) with summary table and revised post-amendment counts + Top-5 actionable order. Inline `> **Amended 2026-05-02:** ...` markers on each affected ticket entry. Original verdicts and rationale preserved verbatim below — important so a future re-evaluation can see *why* each was originally KEPT and what changed. Just rewriting the original verdicts in place would have lost the audit trail and let the same false-alarm patterns sneak back in next ticket-filing pass.

**Lesson:** memory updates are fine for ad-hoc decisions, but when a disposition lands that overrides a canonical roadmap doc, the doc needs to be amended in the same session — otherwise the next agent reading the doc cold will re-apply the stale verdict. The amendment-block-plus-inline-marker pattern is the right shape: amendment block visible at the top for cold reads, inline markers visible during ticket-by-ticket review, original audit trail preserved underneath.

**Net post-amendment dispositions on currently-open tickets:** zero phantom chemometrics tickets remain. Newer T-33 through T-49 are all GUI/infrastructure (resume lifecycle, logging, type cleanup) — no chemometrics framing at risk. Top 5 actionable: T-19 reframed (~1-2d), T-16 reframed survey, T-31 SIMCA (1-2w), T-17 PLS-2 (2-3w), T-01 reframed (~2-3d).

---

## 2026-05-02 (housekeeping flag) — SESSION_LOG is 1800+ lines past archive threshold

CLAUDE.md says move older entries to `docs/SESSION_LOG_ARCHIVE.md` once SESSION_LOG.md exceeds ~200 lines. File is now 1800+ lines (~9× the threshold). Deferred archiving across the last several sessions because each one had higher-priority ticket work and an archive cut would noise up an unrelated commit. **Recommended next maintenance window:** archive everything before 2026-04-30 (the master-rule re-evaluation entry is the natural cut point — pre-April-30 entries are about phantom-leakage tickets that were retired during that pass, so they're safe to archive). File a separate ticket for it; don't piggyback on a feature/bug PR.

---

## 2026-05-02 (T-44 phantom-hasattr survey) — `hasattr(self, X)` + `self.X.get()` is a silent-no-op trap class

T-44's typo (`n_trials_var` instead of `n_unified_trials`) was invisible at the Python level because `hasattr` makes a wrong-name read a no-op rather than an `AttributeError`. The hasattr guard was added defensively to handle "Tk var might not be created yet on this code path" — but it converts every typo at the same site into a silent fallthrough. Pre-T-44, `RunMetadata.n_trials_per_model` was always None in every sidecar; the resume banner showed "?". Nobody noticed for months.

DeepSeek V4 Pro Max's T-44 sibling-survey audited ~18 inline `hasattr(self, X) + self.X.get()` patterns in the GUI and found **one more phantom**: `task_type_var` at `spectral_predict_gui_optimized.py:30859` (actual var is `task_type` at line 2856). Same class, same shape — folded into the T-44 PR.

**The other ~230 multi-line `hasattr(self, X)` patterns weren't fully audited.** A guard on one line and the `.get()` access on a later line is harder to typo (you'd have to introduce the wrong name twice), but not impossible.

**Recommended permanent fix:** ruff AST visitor that flags `hasattr(obj, name)` where `name` is a string literal that never appears in any `obj.name = ...` assignment in the same class. Closes the class. Out of scope for T-44 itself.

**Rule for future sessions:** when reviewing GUI changes that touch Tk var access, check that the `hasattr` guard name matches the `.get()` access name AND that the var is actually defined in `__init__`. Don't trust the lack of `AttributeError` to mean the code works.

---

## 2026-05-02 (T-45 dedup test) — `isinstance` across module reload silently fails

T-45's MEDIUM fix (Codex finding) was: scan `spectral_predict` logger handlers for an existing `_SafeRotatingFileHandler` pointing at the same `dasp.log` path, reuse if found, only attach a new one otherwise. First implementation used `isinstance(existing, _SafeRotatingFileHandler)` — looks correct.

Test caught it failing: `tests/test_run_logging.py::test_setup_app_logger_dedups_after_module_reload` pops `spectral_predict.run_logging` from `sys.modules`, re-imports, calls `setup_app_logger` again, asserts only one handler attached. **Got 2 handlers.**

Root cause: module reload creates a fresh `_SafeRotatingFileHandler` *class object*. Handlers attached by the old module instance are instances of the OLD class. `isinstance(old_handler, NEW_class)` returns False. The dedup check structurally couldn't find them.

**Fix:** match by `existing.__class__.__name__ == "_SafeRotatingFileHandler"` instead. Class-name strings are reload-stable.

**Rule for future sessions:** any "is this object an instance of MyClass" check across a module-reload boundary needs class-name matching, not `isinstance`. Bites in:
- Logging (handlers survive on the logger after the module reloads)
- Plugin systems (registered classes outlive the plugin module)
- Any test that pops + reimports a module and then checks objects from before the pop

This is also why the `fresh_logging` fixture in `test_run_logging.py` cleans handlers from the `spectral_predict` logger at teardown — leaving them attached to a logger that survives the test would let them be inherited by the next test, which is the failure mode this rule guards.

---

## 2026-05-02 (DeepSeek routing through opencode-call) — pre-emptive routing refusals are subagent misjudgments

The `opencode-call` agent's `deepseek` alias correctly routes to DeepSeek API direct (per `feedback_deepseek_routing.md`). During T-45 review dispatch, the agent pre-emptively refused, claiming opencode's `deepseek/` provider routes through opencode-go (which is forbidden by the routing memory). Result: T-45 shipped with Codex-only review.

User confirmed "deepseek should be working." Retry on T-44 with explicit instruction succeeded.

**Rule for future sessions:** when `opencode-call` (or any subagent) refuses to dispatch a model on routing-policy grounds, **retry with an explicit "the alias is known-working; proceed and report the actual error verbatim if it genuinely fails."** Don't accept pre-emptive refusals as truth. The subagent's understanding of its own routing config can be stale or pessimistic.

Memory at `feedback_deepseek_routing.md` updated with the working-channel confirmation.

---

## 2026-05-02 (user feedback) — bugs vs methodology changes vs polish

User course-corrected mid-session: "remember that for every ticket it has to be evaluated as to whether or not it really is relevant for chemometrics" + "we don't just make changes unthinkingly" + "if it is a real bug that means something does not work do 45. the point of my warning was there were a number of suggested changes to pipeline to deal with leakage that were not relevant for chemometrics on NIR data. i want to avoid changing the way the program works. bugs are fine."

The framing the user wants:
- **Bug fixes** (something is silently broken / metadata is wrong / a code path doesn't do what it claims) → ship.
- **Pipeline methodology changes** (refactoring SNV / autoscale / SG / baseline / variable-selection to address ML "leakage" objections from per-fold-fitting orthodoxy) → STOP. Confirm with user. Default to NO. Chemometrics-community convention is the authority here, not ML orthodoxy.
- **Pure engineering polish** (test infrastructure for already-covered surfaces, comment cleanup, tests pinning low-probability regressions, "defense-in-depth" on already-working code) → defer unless trivial.

Applied: T-45 / T-46 / T-47 / T-44 all shipped (bug fixes / observability / behavioral defaults that materially affect what most users get). T-48 deferred (real-Tk integration tests for surfaces already pinned by `_FakeGUI` shim tests — pure polish).

Memory at `feedback_chemometrics_relevance_per_ticket.md`. The earlier-but-related `feedback_chemometrics_conventions.md` covers the methodology side specifically (don't flag SNV/autoscale-pre-CV/variable-selection-on-full-training as leakage).

---

## 2026-05-02 (T-46 review by DeepSeek) — recent merges can reclassify edge cases as common cases

T-46 plan was filed when `bayesian_persistence_mode='never'` was the default — the auto-migration code path was a rare edge case. Plan said: site 1 (always-on) gets `progress_callback`; site 2 (auto-migration) gets only `logger.warning` because no callback in scope and the failure mode was rare.

T-47 merged earlier the same session and flipped the default to `'auto'`. **Auto-migration is now the most-traveled WAL-touching path.** The plan's "logger.warning is fine, T-45 will surface it" reasoning no longer held: under the new default, the user-visible benefit of T-46 ("OneDrive/Dropbox no longer silently halve throughput") only fires in `'always'` mode — most users on the new `'auto'` default still got silent halving until T-45 ships.

DeepSeek V4 Pro Max caught this in cross-family review by integrating the T-47 context into the T-46 evaluation. The plan author missed it because T-47 didn't exist when the plan was written. Codex caught the test gap at site 2 but didn't connect it to the default-flip context — convergence on the gap, divergence on the why.

**Lesson:** when a recent merge changes the default behavior of an adjacent surface, audit older filed plans against the new default before implementing. A plan's "rare edge case" assumption can flip overnight. Cross-family review with same-session context is the gate that catches this — neither the plan author nor a per-ticket reviewer in isolation would notice.

**Fix applied:** threaded `progress_callback` through `_migrate_study_to_sqlite` as optional kwarg. Distinct event tag `t41_decision: "wal_rejected_at_migration"` (vs site 1's `"wal_rejected"`) so consumers can identify which lifecycle phase rejected from the structured key alone. 264 + 1 skipped after the fix.

---

## 2026-05-02 (T-47 follow-up) — DeepSeek caught the missing absent-key fallback test

DeepSeek V4 Pro Max review of the T-47 default-flip identified a MEDIUM gap: the test suite had no regression covering from_dict with the bayesian_persistence_mode key absent entirely. Every existing from_dict test explicitly included the key (sometimes with garbage, sometimes with a real value), so a future refactor accidentally flipping data.get("bayesian_persistence_mode", "never") to data.get(..., "auto") would have landed silently.

The corruption-coercion test (test_corrupted_sidecar_mode_coerces_to_never) covers the case where the key is present with a junk value — it triggers the validate-and-coerce branch at run_state.py:163-168. But absent-key takes a *different* branch: the data.get(..., "never") default at line 162 is consumed before validation runs, so the coercion log never fires. That distinct path needed its own test.

Added test_legacy_sidecar_without_persistence_mode_defaults_to_never to pin it. Two LOWs closed in the same commit: plan-doc step 3 said "confirm default also auto" for unified_bayesian.py:1771 but it was already auto since T-41 (no change needed, only verification); and added a comment above the GUI hasattr(self, "bayesian_persistence_mode") else "never" fallback in spectral_predict_gui_optimized.py:25266 distinguishing it from a user-facing default — it's a malformed-Tk-state safety net, sibling to the from_dict fallbacks, deliberately at "never" and not to be "fixed" alongside any future field-default flip.

**Lesson reinforced:** the SESSION_LOG entry above ([T-47] "default value vs fallback for malformed input") was the right framing, but the *test coverage* didn't yet pin the absent-key half of the legacy-sidecar story. "One pair of contracts" → "three test cases" (field-default flipped, corruption coerced, absent-key fallback held). Sweep: 261 + 1 skipped (was 260+1 pre-fix).

---

## 2026-05-02 (T-47) — "default value" vs "fallback for malformed input" are distinct concerns

T-47 plan said one-liner: flip `bayesian_persistence_mode` default `'never'` → `'auto'`. The field has FOUR sites in `run_state.py` that look like defaults but split into two semantic categories:

- **Field defaults (flipped to `'auto'`):** dataclass field at `RunMetadata` line 133, `start_run()` parameter at line 349. These represent "what to do when the caller didn't specify". Users today want `'auto'`.
- **Safety fallbacks (kept at `'never'`):** `from_dict` legacy-sidecar fallback at line 162 (sidecar predates T-41, missing the field), corruption-coercion at line 168 (sidecar HAS the field but value is junk like `"GARBAGE"`). These represent "what to do when input is malformed".

Conflating them is the common refactor mistake — and not theoretical. Flipping the corruption-coercion to `'auto'` would silently start writing SQLite based on garbage input. Flipping the legacy fallback would change resume behavior of pre-T-41 sidecars without the user agreeing.

**Lesson:** when flipping a "default", inventory every textual default for the field, then classify each as "unspecified-input default" (safe to flip with the rest) or "malformed-input fallback" (decide separately on what's safest, often distinct). The plan's "one-liner" framing collapses that distinction; the implementation should not.

Two regression tests pin this contract: `test_default_persistence_mode_is_auto` (start_run + RunMetadata both `'auto'` when unspecified) and the existing `test_corrupted_sidecar_mode_coerces_to_never` (corruption stays `'never'`). The pair makes the asymmetry explicit at the test level.

---

## 2026-05-02 (T-49 user-caught) — validation partition is decided AT START not AT END

User asked during the consolidated PR wrap-up: "for the pause function, when it restarts does it also maintain the external validation set? or does that not really matter since only calculated at the end?"

**The framing is the trap.** Validation metrics (RMSEP / R²pred / val_Accuracy) ARE only computed at the end of the search — but the validation *partition* is decided BEFORE the search runs (when the user clicks "Create Validation Set"). The Bayesian trials train on the calibration partition only. If on resume the user creates a different validation set:

- **Deterministic algorithms (SPXY / Kennard-Stone / Stratified):** same data + same algorithm + same % gives the same partition. The captured Tk vars (now in `CAPTURABLE_SETTINGS`) are sufficient — IF the user remembers to click "Create Validation Set" again.
- **Random algorithm:** re-clicking gives a DIFFERENT random draw. Some samples that the resumed Bayesian trials trained on can land in the new "validation" set → silent leakage on RMSEP.
- **Manual algorithm:** there's no algorithmic way to reproduce the partition; the indices must persist or the resumed run is broken.
- **Even for deterministic algorithms:** if the user forgets to click the button, `validation_X` is None and the post-search validation step silently skips. Banner says "settings restored" but key state is missing.

User's judgment: this is a correctness blocker, not a deferred follow-up. Folded into the consolidated PR.

**Architecture:** added `RunMetadata.validation_indices: list[Any] | None` (Any because DataFrame index labels can be int or str). Sidecar persists labels as-is; `from_dict` type-guards against malformed shapes. GUI's `_check_for_incomplete_run` stashes captured indices on `self._pending_validation_indices`; new `_apply_pending_validation_indices` helper re-slices `self.X` / `self.y` after the fingerprint check passes in `_run_analysis_thread`. The helper has three guard cases: respect user's manual re-creation, skip-on-missing-label as defense in depth, no-op when no pending indices.

**Lesson logged:** the question "does it matter only at the end?" is the same shape as "does the per-trial cv_strategy attr matter?" — both look like post-hoc state but are actually preconditions of the search. When auditing a resume flow, list every piece of state the search consumes BEFORE the first trial, not just what it produces. Validation partition belongs in that list; T-43 missed it.

---

## 2026-05-02 (T-38 dead preprocessing cleanup) — plan correction held; test name was misleading

Plan correction from earlier (T-37 review by CodeRabbit) held: `preprocessing_wrapper.py` is alive and stays (imported by `ensemble.py:18`); only `learned_preprocessing.py` (775 LOC) and `ensemble_preprocessing.py` (701 LOC) deleted, plus the dead `HAS_ENSEMBLE_PREPROCESSING` import block at `gui:236-241`.

**Test-name misdirection:** the plan mentioned deleting `tests/test_ensemble_preprocessing.py` along with `ensemble_preprocessing.py`. **Did NOT delete that test file** — despite the matching name, it imports from `preprocessing_wrapper` and `ensemble` (both alive), NOT from `ensemble_preprocessing` (deleted). 19 tests in that file + the integration suite all green post-deletion. The plan's instruction to delete the test was a misread of CodeRabbit's earlier correction; the test name reflects what it tests (preprocessing for ensemble use), not which module it depends on.

**Build-time torch exclusion:** the `spectral_predict_py312.spec` excludes `torch`/`torchvision`/`torchaudio` with a comment that referenced `learned_preprocessing.py` as the rationale. Updated the comment — `learned_preprocessing.py` is gone, but the exclusion stays as belt-and-braces against a transitive PyInstaller import.

Sanity-imported every `spectral_predict.*` submodule via `pkgutil.walk_packages` after deletion — clean. 244/244 across the broader regression sweep.

**DeepSeek V4 Pro post-push review (2026-05-02):** READY_TO_MERGE. Zero HIGH findings. One MEDIUM (M1): the plan doc still said "delete tests/test_ensemble_preprocessing.py" despite the implementation correctly preserving it — plan text never re-synced after the T-37 correction. Three LOW: stale `__pycache__/.pyc` files for the deleted modules (gitignored, harmless), T-21 audit doc references `ensemble_preprocessing.py` as a 28-call-site SG location (now obsolete; T-21 unguarded count drops to ~59), and Feb 2026 archival design docs reference the deleted modules in mapping tables (historical artifacts, no action). Applied M1 (plan doc text) and L2 (T-21 footnote) as follow-up doc-only commit. Confirmed via DeepSeek: no `.dasp` save format, Inno Setup, hiddenimports, datas, or config file references the deleted names; `model_io.py:176` saves only `type(model).__name__` so legacy `.dasp` files cannot encode `StackedPreprocessing*` (the class was never wired into a code path that produced model objects).

---

## 2026-05-02 (T-42 + T-43 cross-family review trail) — Codex caught a T-43 silent-failure that DeepSeek missed; DeepSeek caught a stale T-42 test that pr-review-toolkit / Codex would have missed

Two-ticket overnight queue. After implementation of each ticket, dispatched DeepSeek V4 Pro Max via opencode-call. Per the protocol, after every 2 tickets ALSO ran Codex via the codex-reviewer agent — the cross-family check that has caught real bugs in prior tickets (T-41 phantom resume prompts, docstring drift).

**The Codex-only finding (T-43 BLOCKER):** the GUI has TWO parallel sets of baseline/smoothing/region/UVE Tk vars — the "general" ones (`enable_baseline`, `baseline_method`, `enable_smoothing`, `region_test_all_individual`, `region_test_pairwise`, `enable_uve` if it existed) AND a Bayesian-specific set (`bayes_enable_baseline`, `bayes_baseline_method`, `bayes_enable_smoothing`, `bayes_region_test_all`, `bayes_region_test_pairwise`, `bayes_enable_uve`). The Bayesian path at `spectral_predict_gui_optimized.py:27554-27570` reads the bayes_* set; my whitelist captured only the general set. Resume of a Bayesian run would have silently used bayes_* defaults despite the banner claiming "settings restored" — the exact silent-failure trap T-43 was meant to close. DeepSeek had reviewed the same diff in pass 1 and didn't catch this; Codex did because it traced the actual analysis read path back from the run_unified_bayesian call site.

**The DeepSeek-only finding (T-42 HIGH):** my audit doc claimed "132/132 regression tests green" but I had only run a curated sweep that excluded `tests/test_cv_strategy.py`. That file's `test_one_class_bayesian_writes_cv_strategy_to_trial` still asserted `best.user_attrs.get('cv_strategy')` against the keys I'd just removed from per-trial scope. Codex didn't flag this in its T-42 pass (verdict: READY_TO_MERGE, after empirically verifying user_attrs survive `optuna.copy_study`); DeepSeek flagged it as a sweep-coverage gap. Lesson: for tickets that change shared contracts (in this case, "where to read cv_strategy from"), the regression sweep must include EVERY test file that touches the contract surface, not just the new ticket's own tests.

**Cross-family complementarity confirmed.** Each reviewer caught what the other missed:
- Codex traced runtime read paths through GUI surface code and caught the bayes_* whitelist gap.
- DeepSeek empirically tested Optuna 4.8 behavior (would have caught a copy_study user_attrs trap if it existed; verified ours doesn't) AND caught the test-coverage gap by exhaustively grep'ing for stale assertions on the changed surface.

For overnight protocols: the every-2-tickets Codex pass paired with per-ticket DeepSeek isn't redundancy — it's distinct family-orthogonal coverage.

---

## 2026-05-02 (T-42 baseline measurement) — T-41 already closed the perf gap

Critical empirical finding before implementing Approach C: **the post-T-41 baseline (`tests/_bench_bayesian_per_model.py` on `fix/T42-write-path-plumbing-approach-c` tip = T-43 commit `487b1d9`) shows SQLite WAL ratios already at the T-42 "definition of done" target.** Numbers (n_trials=10, synthetic data n=100, n_features=200):

| Model | In-memory | SQLite WAL | Ratio | Overhead/trial |
|---|---|---|---|---|
| PLS | 0.28s | 0.20s | **0.69x** | -9ms |
| Ridge | 0.18s | 0.19s | **1.06x** | +1ms |
| RandomForest | 14.58s | 3.57s | **0.25x** | -1100ms (caching noise) |
| LightGBM | 2.35s | 2.19s | **0.93x** | -16ms |
| XGBoost | 4.06s | 4.09s | **1.01x** | +3ms |

T-42's plan quoted XGBoost 1.36×, LightGBM 1.89×, PLS 28× from PRE-T-41 measurements. T-41's WAL pragmas + 30s SQLite busy_timeout collapsed the per-trial finalize cost from ~200ms to near-noise. **Approach C is no longer needed to make `'auto'` mode viable** — it already is.

Proceeding with Approach C anyway because:
1. The plan explicitly requested it.
2. It's correct cleanup: `cv_strategy` and `cv_n_repeats` are written every trial (60+ writes per study) but read by nobody — `convert_study_to_dataframe` takes them as function parameters, never reads `trial.user_attrs`.
3. `early_stopping_rounds` is constant per study but currently written per trial; can hoist to `study.user_attrs` and have `convert_study_to_dataframe` read it once.

Per-trial set_user_attr counts (from `tests/_bench_t42_set_user_attr_count.py`):
- PLS regression: 30 calls/trial (29 unique keys)
- Ridge regression: 30 calls/trial (29 unique keys)
- PLS-DA classification: 42 calls/trial (41 unique keys)

Approach C savings: ~3 of 30 calls (10%) for regression, ~3 of 42 (~7%) for classification. At ~2ms WAL each, ~6ms/trial cumulative — modest but real cleanup.

**Filed but deferred:** flipping T-41's default from `'never'` to `'auto'` is now justified by the bench data alone, doesn't depend on Approach C succeeding. Trivial follow-up ticket.

---

## 2026-05-02 (T-43 implementation) — Tk var quirks + GUI introspection trade-offs

**Architectural call: curated whitelist > auto-introspection for GUI settings capture.** The dasp GUI has ~700 Tk vars; auto-introspecting `vars(self)` for Tk types would silently capture display state (filter dropdowns, chart colors, exploratory-tab UI) and restore it on resume — surprising the user. Settled on a ~80-name `CAPTURABLE_SETTINGS` whitelist in `src/spectral_predict/run_gui_settings.py`. Trade-off: adding a new analysis-defining setting requires editing the whitelist (drift risk), but the contract is testable and predictable.

**Tcl quirk: `tk.IntVar.set("not-a-number")` does NOT raise.** Tcl stores the string verbatim; the `_tkinter.TclError: expected integer but got "abc"` surfaces on the next `.get()`. Caught during DeepSeek MEDIUM #3 review. Defense pattern: read back via `.get()` after every `.set()` in `restore_gui_settings`, compare to expected, push mismatch to `report.errors`. This protects against crafted/corrupted sidecars rather than against normal capture (`.get()` returns the right type for the Var).

**Order-dependent contract: restore-then-override-persistence-mode.** In `_check_for_incomplete_run`, we restore captured settings BEFORE forcing `bayesian_persistence_mode='always'`. If a future refactor inverts the order, restore would clobber 'always' back to whatever the previous run was using ('auto'/'never'), and the resumed SQLite URL would be silently ignored. Comment + new test (`test_restore_then_override_persistence_mode_order`) document this.

**`from_dict` filtering pattern for forward-compat:** `dataclasses.fields(cls)` gives the known field set. Filter the input dict before `cls(**filtered)` so future builds adding a new field don't TypeError on older Python. Plus a type guard: `gui_settings` of wrong type (string, list) coerces to None with a logger warning. Without the type guard, a corrupted sidecar passes the field filter and crashes downstream in `restore_gui_settings.items()`.

**Pre-existing bug surfaced (DeepSeek LOW #7, deferred):** `spectral_predict_gui_optimized.py:25123` reads `self.n_trials_var.get()` (guarded by `hasattr`). The actual var is `self.n_unified_trials`. The `hasattr` guard causes `n_trials_per_model=None` in every Bayesian `start_run` call. T-43 captures `n_unified_trials` correctly, so the resume flow has the right value — but the run-state metadata's `n_trials_per_model` field is always None. File a separate ticket.

---

## 2026-05-02 (T-41 final review pass) — multi-pass review trail caught real bugs each round

T-41 took 8 review passes before merge. Each pass caught real bugs the prior passes missed. Lessons worth saving for the overnight-run protocol and future ticket reviews:

**Each reviewer family caught distinct bug classes:**
- **DeepSeek V4 Pro Max (max thinking, full repo via opencode-call):** caught the `_study_ref` mutable-container swap not redirecting Optuna's writes (HIGH#1) and the `cb_study.stop()` vs `_study_ref[0].stop()` post-migration crash (HIGH#2). These are semantic-level bugs about Optuna's optimize-loop reference capture — exactly the kind of trap that requires deep reasoning about library internals. DeepSeek empirically verified Optuna 4.8 behavior via subprocess tests.
- **GLM 5.1 (z.ai sub via opencode-call):** caught the no-integration-test gap for the cb_study.stop()→restart pattern. Tests passed but exercised the helper, not the integration. GLM's domain-knowledge depth on Optuna API contracts (load_if_exists+sampler silently ignoring) was decisive.
- **Codex CLI:** caught phantom resume prompts for in-memory crashes (no SQLite to resume from but sidecar exists), and docstring/tooltip drift after the default flip.
- **pr-review-toolkit (5 specialist agents in parallel):** silent-failure-hunter caught 4 HIGHs the prior passes missed — WAL pragma silent rejection (no return-value check), orphan SQLite from partial-success migration, start_run-failure silent downgrade, resume-override `set("always")` failure but lying banner. type-design-analyzer caught the `enable_sqlite_persistence: str` silent-fallthrough on typos.
- **User runtime verification:** caught the resume-banner-is-a-no-op bug that none of the bot reviews would have caught — required actually clicking through the GUI flow.

**Key meta-lesson:** the silent-failure findings consistently came from ONE reviewer family per pass, not multiple. Single-family review at a single point in time misses bugs that another family or runtime testing would catch. Cross-family review at multiple checkpoints is what produces the convergence.

**Architectural traps worth keeping front-of-mind for similar work:**
1. `optuna.create_study(load_if_exists=True, sampler=...)` SILENTLY IGNORES the sampler kwarg on existing studies. Use `optuna.copy_study` + `optuna.load_study(sampler=TPESampler(...))` for sampler attachment. This trap would have shipped if not for DeepSeek's empirical testing.
2. `study.optimize()` captures the study by reference; swapping a closure-bound mutable container does NOT redirect Optuna's writes. Trials post-swap go to the original object. Pattern: stop the loop, restart on the migrated study from the outer scope.
3. `Study.stop()` requires the study currently being optimized. Calling it on a different study (e.g., a migrated SQLite study after the in-memory study aborted) raises RuntimeError mid-optimize. Pattern: use the cb_study reference Optuna passes to the callback, not a closure-captured reference.
4. `PRAGMA journal_mode=WAL` does NOT raise on rejection — it returns the journal mode that was actually applied. Filesystems that reject WAL (network shares, AV-shimmed paths) silently fall back to DELETE mode. Always read the row.
5. Multi-model Bayesian runs share one SQLite file (one `storage_url` per `start_run`, multiple models per run). File-level cleanup (`Path.unlink`) on a per-model failure can nuke prior models' trials. Use study-scoped operations (`optuna.delete_study`).

**For future ticket reviews:** dispatch the cross-family panel ONCE per significant change (post-implementation, post-fix-of-fixes); don't trust a single family or a single point in time. The pr-review-toolkit parallel-5 fan-out was the highest-yield single review pass.

---

> **Older entries archived to [SESSION_LOG_ARCHIVE.md](SESSION_LOG_ARCHIVE.md)** — second archive batch on 2026-05-02 moved 2026-05-01 and earlier entries out. First batch (2026-04-29) moved entries before 2026-04-15. Grep the archive when you need historical context on a closed bug, decision, or PR.
