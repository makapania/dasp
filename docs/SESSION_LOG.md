# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

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
