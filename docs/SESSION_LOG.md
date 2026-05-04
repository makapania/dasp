# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

## 2026-05-05 wrap-up — PR #41 merged (validation-rebuild MEDIUM); PR #42 framing correction; T-resume-y-variable-persist fully spec'd

PR #41 (`fix/Tclass-weight-validation-rebuild`) merged at commit `44ffefb`. Closes the validation-rebuild MEDIUM. Five reviewer types ran across 7 dispatches: Codex GPT-5.5 (design pass + post-implementation review), GLM 5.1 (z.ai), DeepSeek V4 Pro Max (max-thinking via DeepSeek API), pr-review-toolkit code-reviewer + silent-failure-hunter + comment-analyzer + pr-test-analyzer. **Three saves from the audit pyramid:**

1. Codex's design pass caught 2 GUI direct callers (`gui:27914`, `gui:28069`) the continuation prompt had missed — the prompt only flagged the 2 backend callers in `search.py`. Lesson: continuation prompts under-specify caller surface; independent grep at design time is mandatory.
2. Codex's post-implementation review caught the per-caller pin gap — original tests verified the function signature accepts `imbalance_method` and the helper is referenced from the body, but did NOT pin that the 4 actual callers actually pass `imbalance_method=imbalance_method`. A refactor that drops the kwarg from one caller would silently regress validation rebuild for that path. **Generalisable lesson: structural pins should be one-per-consumer, not one-per-producer.**
3. pr-test-analyzer caught the missing fit-site splat pin — same silent-failure shape as the original bug. The headline behavioral test verified the helper RETURNS `sample_weight`, but nothing pinned that the production fit site actually splatted `**fit_kwargs` into `model.fit(...)`. A refactor dropping the splat would regress XGBoost back to UNWEIGHTED training with all tests still passing.

PR #42 (`docs/T-resume-y-variable-correct-framing`) merged at `2db6311`. The T-resume-y-variable ticket originally filed as a "UX bug" was reframed by the user as a state-restoration feature gap: defaulting to first-column-on-load is correct behavior; the actual issue is that Y variable selection is never written to the sidecar, so resume can't recall it. **Generalisable lesson: when a user reports something that "doesn't work right," the surface symptom is often correct behavior interacting with a missing feature, not a bug per se. First instinct is to file as bug; better instinct is to ask "what's the actual contract, and what part isn't being honored?"**

GLM 5.1 review of PR #42 (z.ai subscription, ~2 min wall-clock) confirmed the framing accuracy AND surfaced two valuable additions:

- **Edge case the framing should mention**: if the restored column name doesn't exist in the newly-loaded dataset (renamed column, different file), the Combobox will display a stale string that isn't in its option list. The existing readback check at `run_gui_settings.py:299-309` only verifies `.set()`/`.get()` round-trip — does NOT validate against the current data's columns. Implementation must validate against `_get_available_target_columns()` (`gui:16849`) and fall back to default on mismatch.
- **Related-state observation**: `dataset_fingerprint` (`run_state.py:138`) already detects dataset swaps on resume. The stale-column scenario would likely co-occur with the existing fingerprint-mismatch warning. The validation-against-current-columns is still required for cases where fingerprint matches but column ordering/naming changed (user added/deleted a column).

Both folded into the deferred-ticket entry in PROJECT_STATUS.md and into the new continuation prompt at `docs/CONTINUATION_PROMPT_2026-05-07_resume_y_variable_persist.md`.

### Generalisable lessons (carried forward)

1. **Continuation prompts are themselves subject to the transferred-justification fallacy.** They under-specify caller surface; verify (a) reachability and (b) failure-mode applicability before implementing. (Repeated lesson — second time this session.)
2. **Codex's `get_params(deep=True)` priority-order probe** is a cleaner Pipeline-discriminator pattern than `model_name` dispatch. Worth adopting in future PRs.
3. **Multi-reviewer audits are complementary, not redundant.** Cross-family panels catch reachability and design issues; pr-review-toolkit catches comment rot, test gaps, and silent-failure shapes. Each layer caught something the others missed across PR #34 → #41 → #42.
4. **Line-number references in code comments are write-only documentation.** They decay monotonically as the file evolves. Use stable function-name identifiers — they're refactor-stable and queryable by grep.
5. **Bug-vs-feature-gap framing matters at ticket-filing time.** "Two correct behaviors composing into a confusing experience" is feature-gap shape (additive fix), not bug shape (corrective fix). Reframing avoids the implementing agent chasing the wrong root cause.

---

## 2026-05-05 late evening — validation-rebuild MEDIUM closed; ensemble-rebuild LOW dropped as moot

PR-NEW (`fix/Tclass-weight-validation-rebuild`) closes the validation-rebuild MEDIUM filed in `docs/CONTINUATION_PROMPT_2026-05-06_validation_rebuild.md`. New helper `_apply_class_weight_discriminator_for_rebuilt_model` at `search.py:408+` mirrors the canonical discriminator pattern from PR #38, applied at the rebuild fit site (`compute_validation_metrics_for_top_models:747+`). The helper uses Codex's `get_params(deep=True)` priority-order probe (`lr__class_weight`, `model__class_weight`, `class_weight`) — cleaner than dispatching on `model_name` because it picks up Pipeline step names (PLS-DA's `lr` step, scale-sensitive's `model` step) without hardcoding.

Caller threading: `imbalance_method` parameter added to `compute_validation_metrics_for_top_models`; threaded through 4 call sites:
- `search.py:3346` (run_search / Grid)
- `search.py:3987` (run_bayesian_search / Bayesian)
- `gui:27914` (GUI Bayesian validation panel)
- `gui:28069` (GUI NSGA-II validation panel)

Codex's design pass (GPT-5.5, full repo access) caught two of the four call sites that the continuation prompt missed — the GUI direct callers at `gui:27904+` and `gui:28058+`. The continuation prompt only flagged the 2 backend callers in `search.py`. **Lesson: when the continuation prompt says "update all callers in run_search and run_bayesian_search," that statement is itself unverified — independent grep is required.**

### Ensemble-rebuild LOW dropped as moot

The continuation prompt's LOW (`_reconstruct_models_from_results` at `gui:23642+` fits unweighted at 4 sites) is **not actually reachable for classification**. Ensemble methods are regression-only:
- `_update_ensemble_controls_state` at `gui:16651+` greys out the ensemble UI controls when `task_type == 'classification'` (gui:16669-16670). Sets the disabled state on the manual-retrain button.
- The auto-flow at `gui:28323+` explicitly skips with a "skipped: only supported for regression" log line.

So `_reconstruct_models_from_results` is never reached with `task_type='classification'` in any production flow, which means the class_weight discriminator (a classification-only mechanism) is irrelevant to this path. The continuation prompt's LOW was the exact same **transferred-justification fallacy** documented in the prior session log entry — shape-matched the rebuild bug pattern without checking whether the *failure mode* applies. Per the user's "double-check fixes are really needed" rule, the LOW was dropped, not patched.

### Generalisable lessons

1. **Continuation prompts are themselves subject to the transferred-justification fallacy.** They're written by the agent that just finished a related ticket, often using the same shape-pattern-matching that produced the original bug. Read them critically — specifically, verify (a) reachability and (b) whether the failure mode applies, before implementing.
2. **Codex's `get_params(deep=True)` priority-order probe is a much cleaner Pipeline-discriminator pattern** than model_name dispatch. The probe finds the right step name (`lr__`, `model__`, bare) automatically without needing the discriminator to know about SCALE_SENSITIVE_MODELS or PLS-DA's structure. Worth adopting in the next PR that adds a discriminator.
3. **Continuation prompts under-specify caller surface.** The MEDIUM said "update all callers in `run_search` and `run_bayesian_search`," missing the 2 GUI direct callers. Independent grep at design time saved a sister-site bug from being introduced.

---

## 2026-05-05 evening — transferred-justification fallacy in `class_weight` defense-in-depth (factory + RegressionResampler)

A prior agent attempted a "trivial defense-in-depth fix" for `RegressionResampler.fit_resample` mirroring the `ClassificationResampler.fit:244` no-op for `'class_weight'`/`'auto'` sentinels. That patch (`f6ccfd1`) was committed locally then reset before push. A four-way investigation (silent-failure-hunter, Codex GPT-5.5, GLM 5.1, DeepSeek V4 Pro Max) unanimously confirmed the no-op is **wrong** — the mirror form was correct but the *safety justification* did not transfer across the classification/regression boundary. Two of three external reviewers voted to fail loud over warn-and-no-op; the dissenting vote (DeepSeek) was on project-coherence grounds, not correctness grounds.

**Why the mirror failed.** The line-244 ClassificationResampler no-op is safe because the classifier itself receives `class_weight='balanced'` at construction time on three integrated paths (`search.py:4418-4421`, `unified_bayesian.py:1238-1241`, `nsga2_search.py:1401-1405`) — the resampler is no-op'd, but the model is still weighted. The regression side has **zero compensating mechanism**: sklearn regressors do not accept `class_weight`, and there is no project-level sentinel-to-sample-weight mapping. A regression no-op produces an unweighted-AND-unresampled model that ranks against properly-handled siblings in CV results — silent wrong scientific output.

**The same fallacy lived at two layers**, not one: the original `RegressionResampler.fit_resample` patch (now reverted) AND the factory `build_imbalance_transformer` (lines 947-956 pre-fix, which silently routed `(task_type='regression', method='class_weight')` to ClassificationResampler's no-op regardless of task_type — same shape-pattern-match across the boundary, same broken safety justification). The factory layer was actually *more* reachable than the inner-class layer: GUI ensemble-reload at `spectral_predict_gui_optimized.py:37587` calls the factory with `loaded_imbalance_method` from saved configs, which can plausibly carry a stale classification-trained `'class_weight'` value into a regression context.

**Fix shape (this session, on top of `c372ab2`):**
1. **Factory split by task_type**: classification path keeps the route-to-no-op (compensation is real); regression path raises `ValueError` with diagnostic message naming the classification-only nature of the sentinel and listing valid regression methods. (`imbalance.py:947-967`.)
2. **`_needs_resampling_pipeline` consistency**: `unified_bayesian.py:156` and `nsga2_search.py:142` previously guarded `'class_weight'` only; brought into line with `search.py:289` which guards both `('class_weight', 'auto')`. Per the search.py comment, the second guard prevents a future refactor that delays `'auto'` resolution from accidentally wrapping it in ImbPipeline.
3. **Test contract update**: `tests/test_imbalance.py` parametrized regression-task no-op test split into two — `test_classification_sentinels_no_op` (4 cases, no-op preserved) and `test_regression_sentinels_raise` (2 cases, `pytest.raises(ValueError, match="classification-only")`).

**Sites NOT touched (deliberate):** `ClassificationResampler.fit:244` no-op stays as-is. It is architecturally fragile (a future bypass-the-router caller would silently train an unweighted classifier) but produces correct scientific output today because every integrated caller compensates at construction. Per "fix what's wrong, don't redesign around it" — file as a future hardening item, don't expand the present fix scope.

**Generalisable lessons:**

1. **Transferred-justification fallacy.** When mirroring a defensive pattern across modules, you must verify the *safety conditions* that made the pattern correct in the original location also hold in the new location. Pattern shape doesn't carry safety with it. The chemometrics-specific sharpening: `class_weight` is a classification-only mechanism in sklearn; ANY defensive code that treats `'class_weight'` as a no-oppable input on the regression side is implicitly claiming a compensation path exists when it doesn't.

2. **Cross-family panel reveals values disagreements that single-reviewer panels can't.** The 4-way verdict was 4/4 on substance (no-op is wrong) but split 2-1 on remedy (raise vs warn). The split itself was informative — it surfaced that "raise" and "warn" express different priors (programmer-discipline vs chemometrics-pragmatism). Single-reviewer would have hidden the values dimension behind one verdict.

3. **5-minute "verification" reviews check reachability, not remedy-fitness.** The original DeepSeek-V4-Pro pass that "verified" the bad fix in 5 minutes was checking whether the branch was reachable (correct: it wasn't). It did NOT evaluate whether the chosen remedy was appropriate given the asymmetric compensation. The fresh DeepSeek-V4-Pro-Max max-thinking pass (~6 min) caught the asymmetry immediately — different question, different review depth.

4. **The "skips the misleading print()" argument is circular.** The original commit message argued the no-op "correctly skips" the post-resample print() at imbalance.py:545-546. This argument presupposes the conclusion — it only applies if you've decided to no-op. If the remedy is `raise ValueError`, the print() never runs because the function exits via the exception. Watch for this shape of self-justifying defensive-code commit message.

---

## 2026-05-05 afternoon — class_weight discriminator sister-site bug class fully closed (PR #38, four-way convergent review)

The c395317 GUI fix from earlier 2026-05-05 closed the `_run_refined_model_thread` instance of a defective `hasattr(model, 'class_weight')`-only check. PR #38 closed FOUR more sister sites of the same bug class — `unified_bayesian.objective`, `nsga2_search._evaluate`, `nsga2_search._compute_classification_cv_metrics`, `nsga2_search._compute_calibration_metrics`. Pre-fix, every Bayesian trial AND every NSGA-II individual evaluation for CatBoost/XGBoost classifier under `imbalance_method='class_weight'` (or `'auto'` resolving to it) trained UNWEIGHTED — silent contract violation, wrong user-visible Accuracy / F1 / AUC / etc.

Three high-leverage lessons earned in this session:

**1. Four-way convergent review on a fix-of-fixes is the strongest possible signal of correctness.** Codex + GLM + DeepSeek on PR #38 round 1 all independently flagged the SAME 4th sister site (`_compute_calibration_metrics`) as HIGH severity. Codex via call-graph reasoning on the dispatcher, GLM via grep-then-read enumeration of `imbalance_method == 'class_weight'` consumers, DeepSeek via tracing the `convert_nsga2_to_v1_format → cal_metrics` flow. Then Kimi K2.6 on round 2 did the canonical sister-site sweep and confirmed exhaustive (no 5th site). Four orthogonal RLHF traditions (US-trained, Chinese-trained, two different code-tuning regimes) converging on the same conclusion is essentially a proof. **Generalisable**: when a fix-of-fixes commit closes a HIGH from cross-family review, the standard play is one Kimi sister-site sweep before merge — Kimi's enumeration recall catches the residue when the others stop after their convergent finding.

**2. `set_fit_request` mutation is sticky on the estimator instance even after `sklearn.config_context(enable_metadata_routing=True)` exits.** The metadata-routing API in sklearn 1.4+ uses `inner_model.set_fit_request(sample_weight=True)` to opt the estimator into receiving sample_weight via `params=`. The setter mutates the estimator's metadata-routing state, which persists past the config context manager. Caught by GLM's grep-then-read sweep on PR #38 round 1; Codex's call-graph reasoning did NOT surface this. Fix: clone the model first via `sklearn.base.clone()` before set_fit_request so the original caller's estimator is untouched. **Generalisable**: any time you write `inner.set_fit_request(...)` in a context manager, you need to clone first OR save/restore the metadata-routing state.

**3. The 4th sister site (`_compute_calibration_metrics`) was the most insidious of the four.** It computes the user-visible CALIBRATION metrics shown right next to the `_cv` columns in the NSGA-II Results panel. Pre-fix, the panel could show `Accuracy=0.85 / Accuracycv=0.78` where Accuracy came from an unweighted refit (looks better) and Accuracycv came from a weighted CV (the round-1 fix). Visually the mismatch is a red flag — but only if you know to look. Most users would rank Pareto individuals by the larger Accuracy column and pick a model whose displayed score doesn't match its true behavior. **Generalisable**: when fixing a discriminator-style bug class, search for ALL functions that build + fit a model from scratch (not just the CV ones — also the calibration / refit / display-metric ones).

**Operational lessons (carried forward)**:
- **`sklearn.base.clone()` before `set_fit_request()` is the canonical pattern** for the metadata-routing API. Documented in PR #38 round-2 fix-of-fixes commits at `cv_utils.py:813-820` and `nsga2_search.py:1531-1538`.
- **Manual CV loop is the right fallback** when sklearn drops `fit_params=` (sklearn 1.8 removed it). `cross_val_predict_pooled` forces its manual loop when `fit_params` is provided, sidestepping the metadata-routing API entirely for the cross_val_predict path.
- **`pytest.importorskip` inside `pytest.param` marks raises `Skipped` at parametrize-collection time, dropping the entire parametrize set.** Use `importlib.util.find_spec(...) is not None` as a bool probe in skipif marks instead. Caught by GLM as MEDIUM on PR #38 round 1; the symptom was the XGBoost row over-skipping when CatBoost was absent.
- **`git add -A` is dangerous when scratch directories exist in the working tree.** PR #38 had a self-inflicted accident where `.review-tmp/` (39k+ LOC of patches/diffs/agent transcripts) got swept into a fix-of-fixes commit. Recovered via a follow-up cleanup commit (`6ad32bb`) plus a `.gitignore` entry to prevent recurrence. **Generalisable**: always stage explicitly by file (`git add path/to/file1 path/to/file2`) for code commits, OR check `git status --short` after the add and before the commit.

**Pre-existing related gaps surfaced by Kimi (not regressions, deferred)**:
- `compute_validation_metrics_for_top_models` at `search.py:408` rebuilds models via `_rebuild_model_from_row` for `val_*` columns but doesn't thread `imbalance_method`. XGBoost trains unweighted there; PLS-DA's `class_weight='balanced'` is lost through the rebuild. Filed in `docs/CONTINUATION_PROMPT_2026-05-06_validation_rebuild.md`.
- `_reconstruct_models_from_results` at `gui:23643` ensemble-reconstruction path has the same shape, affecting only "Train Ensemble" feature.

---

## 2026-05-05 morning ticket-closing run — three orthogonal lessons earned (parallel-session collision, line-ending drift in Edit tool, three-way reviewer convergence on sister sites)

Five PRs opened (#32–#36) closing the deferred backlog from the 2026-05-04 batch, plus a docs-summary PR (#37). Three non-obvious lessons:

**1. Parallel-session collision is not theoretical.** While I was investigating the P0 GUI wording, `origin/main` advanced from `b0d031f` to `0576358` — five commits including a real regression fix (`4dcedbc`) and its round-2 cross-family fix-of-fixes (`c395317`). My initial `git fetch --all --prune` at session start was already stale by the time I started branching for P1 work. Local `main` was 1 commit ahead of `origin/main` for a while (had unpushed `0bb7e19` from the parallel agent's docs+wording bundle), then origin/main re-fetched past it. The mitigation that actually worked: branching every new PR off `origin/main` directly (not local `main`) and running `git fetch --all --prune` before each new branch.

**2. The Edit tool can introduce phantom whitespace hunks on files with mixed CRLF/LF endings.** The Ridge dead-code deletion on `models.py` produced a clean 4-line deletion AND an unintended whitespace-only hunk at lines 1807-1817 (the `get_feature_importances` Pipeline-unwrap block). The cause: that block had LF endings while the rest of the file was CRLF, and the Edit tool's write step normalized SOME lines but not all. The recovery was: `git reset --soft HEAD~1`, `git restore --staged`, `git checkout origin/main -- src/spectral_predict/models.py`, then re-apply the edit via a binary-mode Python script (`Path.read_bytes()` → `bytes.replace()` → `Path.write_bytes()`) which preserves byte-for-byte line endings. Memory entry: `feedback_edit_tool_line_endings.md`.

**3. Three-way reviewer convergence on a sister-site bug is essentially proof.** PR #33 fixed `_PIPELINE_PARAMS` stripping `n_jobs` in `code_generator.py`. GLM, Codex, AND DeepSeek all independently flagged the SAME sister site at `unified_bayesian.py:88` (identical `PIPELINE_PARAMS` set used by `_capture_serializable_params` to strip params from Optuna trials) as HIGH severity. The codex-round-2 lesson from PR #24 (call-graph reasoning catches sister sites the cross-family pair misses) reaffirmed but with a twist: GLM's grep sweep was extensive enough to catch this one without needing call-graph reasoning. **Generalisable**: GLM's grep-then-read pattern is roughly equivalent to Codex's call-graph reasoning for sister-site sweeps when the sister site shares a literal string match. Codex's specific advantage shows when the sister site is mediated by a non-obvious dispatcher branch (PR #24's case) where literal grep doesn't surface it.

**Operational notes (carried forward)**:
- Three reviewers in parallel on a 39-line test PR (PR #32) returned three convergent READY_TO_MERGE verdicts. The marginal cost was real but the verdict-agreement signal is strong; for high-stakes test additions, the prompt's prescription stands.
- `pr-test-analyzer` flagged a real follow-up gap (LightGBM/CatBoost auto-with-correction rows missing for symmetry) that the cross-family pair did not surface. Specialised agents have non-overlapping yields with cross-family reviewers — the right structure is "test addition → cross-family + pr-test-analyzer."

---

## 2026-05-05 Round-2 fix-of-fixes: silent-unweighted CatBoost/XGBoost class_weight (commit `c395317`)

After commit `4dcedbc` shipped, dispatched two cross-family reviewers (DeepSeek V4 Pro Max via opencode-call, Codex via codex-reviewer agent) to verify the fix held across all model types — user's emphasis: "the key is to make sure this works for all model types and analysis types." **Both independently flagged the same HIGH:** the round-1 fix used `hasattr(model, 'class_weight')` to gate routing, which returns False for CatBoostClassifier (uses `auto_class_weights`/`class_weights`) and XGBClassifier (only supports `sample_weight` at fit time). Round-1 fell into the warning branch and silently cleared `loaded_imbalance_method`, so both classifiers trained UNWEIGHTED whenever a user picked `class_weight` or `auto` (resolved to `class_weight`) for them. **No crash — silently wrong numbers**, violating the foundational "Refine reproduces Results" contract.

Convergent HIGH from two independent reviewers is the user's standing signal that a finding is real. Round-2 fix in `c395317`:
- GUI dispatcher branches on `model_name` first: PLS-DA → `_lr_kwargs` LR tail; CatBoost → `model.set_params(auto_class_weights='Balanced')` (parity with `code_generator.py:946`); `hasattr` → `set_params(class_weight='balanced')`; sample_weight-fallback (XGBoost / RidgeClassifier-style — detected via `inspect.signature(model.fit)`) → sets `use_sample_weight_for_classification` flag.
- The flag drives `compute_sample_weight('balanced', y_train)` at three fit sites: early-stopping fold via new `sample_weight=` kwarg on `_fit_with_early_stopping` (extended in `cv_utils.py`); non-early-stopping fold via `pipe_fold.fit(X, y, model__sample_weight=...)` sklearn fit_param routing; final-pipeline fit on full data via the same routing. The `'model'` step name is consistent across all three GUI pipeline-build paths (GA / derivative+subset / raw) for non-PLS-DA models — Codex confirmed via call-graph trace at lines 37608-37614 / 37736-37742 / 37791-37797.
- All `warnings.warn` calls in the discriminator replaced by `self._log_progress` so messages reach users in the bundled .exe (PyInstaller detaches stderr to nul; warnings.warn is invisible there). MLP message rewritten to be explicitly directive ("switch to SMOTE / ADASYN / SMOTE-Tomek, or pick RandomForest/SVC/LightGBM/CatBoost/XGBoost/NeuralBoosted/PLS-DA").
- Defense-in-depth additions from DeepSeek MEDIUMs: `_needs_resampling_pipeline` early-return now explicitly guards `'auto'` (was correct-by-accident); `build_imbalance_transformer` routes both sentinels through `ClassificationResampler` regardless of task_type so the no-op short-circuits even on the regression path.

Round-2 review: GLM 5.1 max via z.ai (user's explicit reviewer pick) returned READY_TO_PUSH with 5 non-blocking observations (engineering polish only). Codex via codex-reviewer for cross-family convergence — CONFIRM_READY_TO_PUSH with 10-point call-graph trace and one stale-comment nit (line 37414 still mentions RidgeClassifier under sample_weight fallback though actual code correctly routes it via the hasattr branch — non-blocking, deferred per engineering-polish rule).

**Generalisable lesson #2** (extending the round-1 lesson about dropdown-mixes-flags-and-methods): when applying a "model-parameter flag" to a model, `hasattr(model, '<flag_name>')` is NOT a sufficient discriminator across the sklearn ecosystem. CatBoost uses `auto_class_weights`, XGBoost uses `sample_weight` at fit time, MLP uses neither. The canonical pattern (now enshrined in `search.py:4411-4448` and mirrored in the GUI's refined-model thread) is to branch on model_name FIRST for known per-library kwargs, fall back to `hasattr` for sklearn-conformant models, then fall back to `inspect.signature(model.fit)` for sample_weight-only models, then warn for the no-support case. Any future code path that applies model kwargs across heterogeneous estimators must follow this discriminator order — `hasattr` alone produces silent unweighted training for the boosting libraries.

**Test sweep**: `tests/test_imbalance.py` 62 passed + 1 skipped (was 58 + 1 after round-1; +4 from round-2's regression-task sentinel parametrize). py_compile clean across all 4 src files + GUI.

**Push status (after this entry):** `main` at `c395317` (round-2 fix) + docs commits, all pushed to `origin/main`.

---

## 2026-05-05 Model Dev "refined model" crash on imbalance_method='class_weight' / 'auto' (commit `4dcedbc`)

**Symptom (user report):** running any classifier from Model Development → Refine raised `ValueError: Unknown resampling method: class_weight. Available: ['smote', ...]` from `imbalance.py:252`, traceback through `spectral_predict_gui_optimized.py:37849` (`step.fit_resample(...)` in the early-stopping fold loop). Confirmed broken for **all** classifier model types, not model-specific.

**Root cause:** `_run_refined_model_thread` constructed `ClassificationResampler('class_weight')` and inserted it as a Pipeline step. But `class_weight` is a model-parameter sentinel (per the contracts in `_needs_resampling_pipeline` at `search.py:286` and `validate_classification_config` at `imbalance.py:1259`), not an actual resampler — so when CV called `step.fit_resample()`, the resampler factory exploded on the unknown method name. The canonical pattern in `search.py:4411-4470` is: for `'class_weight'`, call `model.set_params(class_weight='balanced')` (with `hasattr` guard + sample_weight fallback + MLP warning), then **do not** pass `'class_weight'` into `build_imbalance_transformer` / `build_preprocessing_pipeline`. The refined-model thread skipped the entire model-kwarg routing step and just shoved the sentinel through the resampler pipeline.

**Why it became visible recently:** T-19 (commit `1d2bf6d`, "expose model-native imbalance handling — bug-fix + Auto mode") added runtime resolution where `imbalance_method='auto'` mutates to `'class_weight'` (or None) at run-entry inside the search modules. The refined-model thread *consumes* `selected_model_config['imbalance_method']` from saved results — so any saved-then-reloaded model whose original search resolved auto → class_weight (or whose user picked class_weight directly) feeds the sentinel into the broken path. Pre-T-19 this was less likely to trigger because auto-resolution didn't exist; post-T-19 every Auto-mode user with imbalanced data hits it on the first refined-model run.

**Fix shape (two layers, mirroring `search.py:4411-4470`):**
1. `_run_refined_model_thread`: after reading `loaded_imbalance_method`, resolve `'auto'` via `resolve_auto_imbalance(y_array, task_type)`, then for `'class_weight'` call `model.set_params(class_weight='balanced')` with the standard guards, and clear `loaded_imbalance_method = None` so no resampler step is appended downstream. PLS-DA routes the kwarg to its `LogisticRegression` tail via a shared `_lr_kwargs` dict consumed at all three pipeline-build sites (GA / derivative+subset / raw).
2. `ClassificationResampler.fit / fit_resample`: defense-in-depth no-op for `'class_weight'` / `'auto'` sentinels (case-insensitive). Ensures any future caller that misroutes them returns input unchanged instead of crashing mid-CV.

**Lesson for similar bug shape:** "model-parameter flag listed in the same dropdown as resampler methods" is a recurring gotcha pattern. Whenever the codepath branches on `imbalance_method`, the discriminator must be: `None` → nothing; `'auto'` → resolve first; `'class_weight'` → model kwarg, NOT pipeline step; resampler-method-name → pipeline step. The GUI's dropdown deliberately mixes both kinds for UI parity, so every consuming code path must handle the discrimination — `search.py` does, the refined-model thread did not. Grep for `imbalance_method ==` and `imbalance_method !=` to find any future site that's missing the discriminator.

**Regression test:** `tests/test_imbalance.py::TestClassificationResampler::test_class_weight_and_auto_are_no_ops` parametrizes `'class_weight'` / `'auto'` (plus uppercase variants for case-insensitivity) and asserts `fit_resample` returns the input by identity (`X_res is X`). Catches both the "resampler explodes on sentinel" regression and any future "resampler quietly mangles input on sentinel" regression.

---

## 2026-05-04 stacked-PR merge cascade — GitHub auto-closes stacked PRs the moment their base branch is deleted, and they cannot be reopened

When the 8-PR queue went through squash-merge, the four stacked PRs (#23 → #24 → #28 on top of #22) hit a workflow gotcha that's not obvious from the GitHub docs:

1. Squash-merge PR #22. Its branch `fix/T20-...` is deleted from origin (gh's `--delete-branch` flag).
2. PR #23 had `baseRefName=fix/T20-...`. When that base branch is deleted, **GitHub does NOT auto-redirect the base to main** — it auto-closes the PR.
3. **The closed PR cannot be reopened** (`gh pr reopen 23` returns `Could not open the pull request`).
4. **The closed PR's base cannot be changed** (`gh pr edit 23 --base main` returns `Cannot change the base branch of a closed pull request`).
5. The recovery is: rebase the head branch locally onto post-parent main with `git rebase --onto origin/main <last-parent-commit> <head-branch>`, force-push, then `gh pr create --base main --head <head>` to open a NEW PR with a new number. The old PR stays closed.

This happened three times tonight — PRs #23 / #24 / #28 became #29 / #30 / #31 after the cascade. Each rebase-and-reopen was a 2-3 minute mechanical step but the *first* one is a stop-and-figure-it-out moment if you don't know the pattern.

**Generalisable pattern**: when squash-merging a stacked queue, after each parent merges, immediately:
1. `git fetch --all --prune` to drop the deleted base from local refs.
2. `git checkout <child-branch>` and `git rebase --onto origin/main <pre-rebase-parent-tip-SHA> <child-branch>`. Use the parent's pre-rebase tip SHA, not the squashed commit SHA — git skips the parent's commits cleanly when their content is now in main as a single squashed commit.
3. Force-push the child with `--force-with-lease`.
4. **Open a new PR** (the old PR is dead). Title and body can be reused verbatim.

**Bonus side-finding during the cascade**: an `xfail(strict=True)` test that PR #29 (T-20b) inherited from PR #23 went XPASS-strict after rebase — PR #26's defensive `.ravel()` had landed before the T-20 stack and quietly fixed the bug class the xfail was waiting for. The XPASS-strict failure was actually a signal that the fix had cross-pollinated; flipping the marker (commit `792ce20`) was the right resolution. Lesson: when rebasing a stacked test PR onto a moved main, run the test suite immediately after rebase to catch xfail markers that have become stale.

---

## 2026-05-04 codex round 2 — call-graph reasoning catches a cross-file bug that within-file cross-family review missed

**Single non-obvious lesson, but a high-leverage one:**

**Cross-family LLM review (DeepSeek + GLM) and codex are complementary, not redundant.** Dispatched codex against all 8 outstanding PRs after the cross-family + pr-review-toolkit passes had cleared everything. Codex caught a real production bug on PR #24 that both DeepSeek and GLM missed:

- PR #24's title is `fix(codegen): scalarise predictions in classification CV pooling`. The fix added `np.ravel(...)[0]` to `templates/validation.py`'s CV pooling block to handle CatBoost multiclass's `(n, 1)` predict() shape.
- DeepSeek + GLM both reviewed only `templates/validation.py` and confirmed the fix was correct in that file.
- **Codex traced the call graph** from the codegen dispatcher: `code_generator._render_cross_validation` at line 1435-1436 routes to `_render_cross_validation_with_imbalance()` whenever `self.imbalance_method` is set, **bypassing `templates/validation.py` entirely**. That alternate inline template at line ~1761 had the SAME pre-PR-#24 unfixed pooling pattern.
- **Result**: every CatBoost multiclass export with `class_weight` / `auto` / SMOTE / random_undersampler / etc. was silently broken before codex's catch — the script would crash before reaching the parity-test appendix with `TypeError: unhashable type: 'numpy.ndarray'`.

**Why DeepSeek + GLM missed it**: both reviewers are strong at within-file reasoning (logic, types, edge cases in the diff they're given). They don't typically grep the surrounding codebase for *other* sites that share the bug class. Codex's tool-use loop (it runs `git grep` / `rg` / file-reads as part of investigation) catches sister-site regressions that pure-text review misses.

**Generalisable pattern**: when a fix targets a specific code path (e.g., one of two CV templates), **search for sister sites of the same bug class** before declaring the fix complete. The grep query that would have caught this: `rg "y_pred_fold\[local_i\]" src/spectral_predict/` returns 2 hits — `templates/validation.py:163` (PR #24 fixes this) and `code_generator.py:1764` (PR #24 misses this). One grep, two seconds.

**Other lessons from the codex round**:
- **Codex's PowerShell sandbox is fragile on Windows.** The Windows ConstrainedLanguage-mode policy blocked Select-String multi-path queries on PR #25 + PR #27 reviews; the helper recovered by switching to `codex exec` with the diff captured into the prompt directly. For complex investigations where this matters, prefer the `codex exec` path from the start.
- **Codex's docstring-vs-coverage HIGH on PR #22 was meta-review value.** The file's docstring claimed T-32 coverage, but no row in the matrix actually exercised T-32's surface (which lives in `tests/test_t32_sample_weight_resampling.py`). Within-file reviewers won't flag a documentation/coverage mismatch like this — they're reasoning about what the test does, not what it claims to do.

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
