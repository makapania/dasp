# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

## 2026-05-01 (evening) — Final adversarial sweep caught T-37's missing-merge-base; merged T-36 fixes into T-37

**Trace pattern worth logging so it doesn't get re-made:**

When two feature branches are developed in parallel and one keeps getting fix commits while the other has already branched off the older tip, the second branch will inherit silent-mismatch bugs that look fixed in the first branch's history. Standard PR review tools (Codex, gemini-code-assist, CodeRabbit, pr-review-toolkit) operating on a single branch's diff against `main` cannot catch this — they don't see the sister branch.

What happened: T-37 was branched off T-36 at `1f49d73` (T-36's then-tip). T-36 then received `b92274c` (6 cross-family review findings) + `afb52d9` (4 pr-review-toolkit findings) + `4d4c542` (docs). T-37 had its own independent post-merge review iteration (`6a0eb28`) but never absorbed T-36's. Three HIGH + four MEDIUM + two LOW T-37 bugs all traced to "T-36 fix not in T-37."

How it was caught: a final adversarial sweep dispatched **two cross-family reviewers in parallel via `opencode-call`** with full repo access — DeepSeek V4 Pro Max (DeepSeek API direct, max thinking) + MiMo 2.5 Pro Max (opencode-go subscription). Both reviewers, independently, identified the merge-base gap as the root cause of every T-37 finding. The convergence is high-signal — when family-orthogonal models agree on a structural finding, the bar for false positive is low.

What the in-band reviews missed: each in-band review (the cross-family bot panel from earlier in the session, plus pr-review-toolkit) operated on diff-against-main scope. They couldn't see "T-37 lacks T-36's fixes" because that's not a diff-against-main fact — it's a relative-to-sister-branch fact. The final sweep with full repo access was what surfaced it.

Fix: `git merge feature/T36-autoscale-toggle` into T-37 (commit `78998d4`). Source merged cleanly (the only changes T-36 made post-`1f49d73` were targeted line additions to functions T-37 also touched but in non-overlapping regions). Doc files (PROJECT_STATUS.md, SESSION_LOG.md) needed manual conflict resolution — took T-37's already-comprehensive versions which were the superset.

**Lessons:**

1. When developing sister branches in parallel and one accumulates fix commits, the other should be rebased onto the first's tip (or merged from it) before final review. Otherwise the in-band review bots will miss the inheritance gap.

2. A final cross-family adversarial sweep with full repo access is structurally different from in-band PR review. Use it as the last gate when sister branches have been evolving in parallel.

3. Convergence between family-orthogonal models on the SAME finding is much higher-signal than either model alone. Two independent panels catching the same structural pattern is a "definitely broken" indicator without needing further verification beyond the initial sanity check.

---

## 2026-05-01 (afternoon, late) — Architectural correction: grid path does NOT do per-fold autoscale via Pipeline mechanic

**Trace error worth logging so it doesn't get re-made:**

I (and Gemini's PR #7 review summary) initially believed the grid-search path applied autoscale per-fold via the sklearn Pipeline mechanic, while the Bayesian path applied it pre-CV global. We claimed this was a real divergence between the two paths. **It is not.**

The actual flow:
1. `search.py:2061` — `prep_pipeline = Pipeline(prep_pipe_steps)` then `X_preprocessed = prep_pipeline.fit_transform(X_np, y_np)` on **full** training data. The autoscale `StandardScaler` step (added by `build_preprocessing_pipeline` when `autoscale=True`) is fit AND transformed here. Result is `X_preprocessed`.
2. `search.py:2276` — `_run_single_config(X=X_for_models, ..., skip_spectral_preprocessing=True, ...)` — the **already-preprocessed** `X_for_models` (column-filtered `X_preprocessed`) is passed in.
3. `_run_single_config` at `search.py:4243-4259` — when `skip_spectral_preprocessing=True` (which **every live caller** passes — verified by `grep skip_spectral_preprocessing=True src/spectral_predict/search.py`), `pipe_steps = []` and only an imbalance transformer is added if needed. **No spectral preprocessing in the inner CV pipeline.**
4. `_run_single_fold` calls `pipe_clone.fit(X_train, y_train)` on the training fold of the **already-autoscaled** `X_for_models`. Pipeline only contains the model (and maybe imbalance) — autoscale is already baked in.

The misleading code path is `search.py:4260-4286` (the `else` branch with the full `build_preprocessing_pipeline` call inside `_run_single_config`). That branch is reachable in principle but no live caller passes `skip_spectral_preprocessing=False` for spectral preprocessing. Reading that branch as if it were the canonical flow leads to the wrong conclusion.

**Implication:** Both grid and Bayesian apply autoscale **pre-CV global**, by design, matching PLS_Toolbox / Unscrambler / SIMCA-P / Pirouette chemometrics convention. The "Gemini Bayesian leakage" finding is a chemometrics-vs-ML-convention question, not a path-divergence bug. There is nothing to "fix" to make grid and Bayesian agree — they already agree.

**Why the Pipeline mechanic doesn't bite for the per-spectrum ops (SNV, SG, baseline) elsewhere in the codebase:** those ops compute within-row math, so per-fold-via-Pipeline gives the same numerical result as global. Autoscale is the first cross-sample step in `build_preprocessing_pipeline`, so per-fold-via-Pipeline would diverge from global — but the live grid path never actually goes through the per-fold-via-Pipeline branch because of `skip_spectral_preprocessing=True`.

**Where per-fold-via-Pipeline DOES happen:** the `_rebuild_model_from_row` validation rebuild path at `search.py:305-394`. That is a separate flow (used by `compute_validation_metrics_for_top_models` to rebuild a fitted estimator from a saved row for held-out validation). The T-36 fix-of-fixes commit `afb52d9` added an `autoscale=False` keyword arg to that function so the per-model `StandardScaler` is skipped when autoscale was already applied during search — preserving the pre-CV-global semantics through the rebuild path.

**Verification command:** `grep -n "skip_spectral_preprocessing=True" src/spectral_predict/search.py | head` — every live call site passes `True`.

---

## 2026-05-01 (afternoon, late) — Sibling silent-failure-fix gap: pr-review-toolkit caught a fix-that-was-dead-code

**Trace error worth logging:**

T-36 commit `b92274c` added a display-name fallback parser at `contamination.py:1118-1136` to extract baseline / smoothing / autoscale flags from suffixed pipeline names like `"als+sg0+snv+autoscale"` for legacy `.dasp` files without explicit columns. **The fix was dead code on the legacy path it claimed to handle.**

Root cause: `_normalize_preprocess_for_pipeline()` at `contamination.py:1081` ran BEFORE the new parser. That helper collapses any unbuildable name to `'raw'` (the last-resort fallback at `contamination.py:930`). So for a legacy row with `Preprocess="als+sg0+snv+autoscale"` and no `PreprocessBase` column, by the time the new `'+'`-parser ran, `preprocess_name` was already `'raw'` and there was no `'+'` to split on. The fix never fired for the case it was designed for.

Fix in `afb52d9`: move the `'+'`-parser BEFORE `_normalize_preprocess_for_pipeline()`. Now the parser sees the original suffixed name, extracts the baseline/smoothing/autoscale flags, rebuilds `preprocess_name` from the residual core parts, and only THEN normalizes. Mirrors the order used in `search.py:548-562` for the same kind of legacy fallback.

**Why my own tests didn't catch it:** the new tests use rows that already have explicit columns (`Autoscale=True`, `baseline_method="als"`), so the parser branch was never the source of truth for them. A test that explicitly constructs a legacy row (only `Preprocess="als+sg0+snv+autoscale"`, no other columns) would have caught it. None exists.

**Lesson for future fix passes:** when adding a fallback parser meant to handle a legacy data shape, write a test that exercises ONLY the fallback path (drop all the columns the modern path uses). Otherwise the test exercises the explicit-column path and the fallback parser is never reached.

---

## 2026-05-01 (afternoon) — DeepSeek V4 Pro external reviews of T-41 plan and T-37 Phase 1; Bayesian SQLite slowdown root-caused

### T-41 Bayesian SQLite slowdown — root cause and plan

User reported Bayesian search 5× slower than pre-T-11. Codex investigation pointed at T-11's per-trial Optuna SQLite writes. Empirical benchmarks (`tests/_bench_bayesian_sqlite.py` and `tests/_bench_bayesian_per_model.py`) confirmed:

| Model | In-memory | SQLite default | SQLite WAL |
|---|---|---|---|
| PLS | 30 ms | 600 ms | 244 ms (8.13×) |
| Ridge | 21 ms | n/a | 215 ms (10.16×) |
| LightGBM | 276 ms | n/a | 520 ms (1.89×) |
| XGBoost | 534 ms | n/a | 728 ms (1.36×) |

**Key finding:** SQLite overhead is roughly constant ~200ms/trial regardless of model. Ratio = `1 + (200ms / fit_time_ms)`. Fast models (PLS, Ridge) get crushed; heavy models (XGBoost, LightGBM) tolerate it.

**Architecture decision:** per-model auto-calculator. First N trials in-memory, measure fit time, migrate to SQLite+WAL via `optuna.copy_study` if median > 1.0s. Mixed-model runs naturally per-model because GUI loops over `selected_models` calling `run_unified_bayesian` once per model. 3-way GUI override (Auto/Always-on/Always-off). T-41 plan filed at `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md`.

### DeepSeek V4 Pro pre-implementation review of T-41 plan

DeepSeek V4 Pro Max thinking (direct API per routing rule, NOT opencode-go) ran an empirical adversarial review of the T-41 plan. Verdict: READY_WITH_PLAN_REVISIONS. **6 findings, all applied in commit `7861a81`:**

- **HIGH #1 (Task 4 — sampler attach):** plan's handwavy "recreate the TPE sampler if needed" was a footgun. `optuna.create_study(load_if_exists=True, sampler=...)` SILENTLY IGNORES the sampler kwarg on existing studies. Plan now explicitly prescribes `optuna.copy_study` followed by `optuna.load_study(sampler=TPESampler(...))` (the only way to attach a sampler to an existing study). DeepSeek empirically tested this on Optuna 4.8 — TPE state preserved across migration, post-migration trials show clear TPE-guided exploitation (best=3.93 startup → best=0.043 trial 9).

- **HIGH #2 (Task 2 — "release the SQLite handle"):** ambiguous and incorrect. `start_run()` doesn't actually create the SQLite file (Optuna does so lazily on first access). No "handle" exists to release. Replaced with "do not pass `storage` kwarg to `optuna.create_study`; the `_active_storage_url` global stays set for subsequent models."

- **MEDIUM:** warmup window 5→10 trials with median (was mean). TPE's `n_startup_trials=20` random-sampling regime can produce wildly different first-5 trial times (e.g., n_components=1 PLS vs n_components=20). 10 trials with median is robust. Added fallback: if completed trials < 3, default SQLite ON conservatively.

- **MEDIUM:** WAL durability claim "still crash-safe under WAL semantics" was overstated. Corrected: WAL+sync=NORMAL prevents database CORRUPTION but may lose UNCHECKPOINTED trials. Set `PRAGMA wal_autocheckpoint=50` to bound loss to ~50 trials per crash event.

- **LOW:** WAL pragma application moved from `start_run` to migration time (file doesn't exist until then), Always-off branch skips URL generation entirely, stale sidecar cleanup added in `mark_complete`, 4 tests added (TPE-continues-learning, mixed-model-independence, auto-decision-surfacing, one-class-uses-same-path).

**Lesson:** `optuna.copy_study` mid-run migration WORKS, but the canonical pattern requires `load_study(sampler=...)` not `create_study(load_if_exists=True, sampler=...)`. The latter silently no-ops the sampler arg on existing studies — the kind of bug that ships and hides for months.

### GLM 5.1 external review of T-37 Phase 1 commit `ef5f61e` (attribution corrected)

User dispatched T-37 implementation in parallel; Phase 1-6 commits landed (`ef5f61e` through `7d8f940`) before main Claude reviewed. External read-only review of Phase 1 returned READY_WITH_REVISIONS — refined ratings vs an earlier draft analysis. Initial main-Claude commit (`fd0072a`) misattributed this review to DeepSeek V4 Pro; corrected by user — these were **GLM 5.1's findings**.

**GLM's most important contribution: a critical correction on the proposed fix.** The dead-code `_resolve_window_choices` originally looked like a missing window-constraint enforcement that should be wired into `_objective`. **GLM caught that this would break Optuna:**

> `trial.suggest_categorical` requires the search space to be CONSTANT across trials per Optuna's ask/tell interface contract. Varying window choices per preprocessing type per trial would corrupt TPE's KDE-based posterior models. The current "suggest from union, ignore invalid combos" approach is the correct workaround.

So the dead-code finding is real (remove or comment as reserved) but the apparent "fix" is wrong-headed. The startup-trial waste (~30-40% of 20 random startup trials hit invalid combos) is the unavoidable tradeoff of operating TPE on a 5-D space with mostly-categorical axes. If startup efficiency becomes a concern, bump `n_startup_trials`, don't restrict the search space.

**GLM's confirmed Phase 7 fix list (smaller than the original draft suggested):**

- **MEDIUM F2:** `_tpe_baseline_params` never written to output dicts (currently safe by coincidence)
- **MEDIUM F6:** `_quick_evaluate` PLS fallback uses RMSE for all task types
- **MEDIUM F9:** Mutual exclusion test only checks `results is not None`, doesn't verify TPE actually ran
- **MEDIUM F12:** Test coverage gaps (`_apply_full_preprocessing` not unit-tested, `_quick_evaluate` not direct-tested, empty-TPE-result fallback untested, roundtrip-through-`build_preprocessing_pipeline` unverified)
- **LOW:** F7 mutation hygiene, F8 print-vs-logger, F10 dead `_resolve_window_choices` (remove or comment)
- **NOT-AN-ISSUE:** F4 polyorder for raw/snv (correctly None), F5 one-class baseline/smoothing reset (no doubling blocks exist), F11 display name order (matches between paths)

Original "F1: derivative-window enforcement missing — fix by per-trial dynamic search space" downgraded to LOW after GLM's Optuna-semantics correction.

**Phase 7 fix commit:** ~110 LOC total (5 + 10 + 15 + 80 lines for F2/F6/F9/F12). None falsify Architecture A.

### T-37 Phase 7 fix commit (`03d95cb`) — F2, F6, F9, F10, F12 closed

All GLM 5.1 findings from the Phase 1 review are now resolved:
- **F2** (`_tpe_baseline_params`): added to TPE output dict; downstream `build_preprocessing_pipeline` now receives explicit params instead of coincidentally-aligned defaults.
- **F6** (`_quick_evaluate` fallback): branches on `task_type` — regression keeps `neg_root_mean_squared_error`, classification uses `accuracy`, one_class returns `-np.inf` (don't pollute TPE with garbage).
- **F9** (mutual exclusion test): `tpe_score` propagated to result rows in `search.py` (3 result-path sites); integration tests now assert `tpe_score` column presence + non-null + `smart_score` absence.
- **F10** (`_resolve_window_choices`): commented as RESERVED with explanation that per-trial dynamic search space would break Optuna's ask/tell contract.
- **F12** (coverage gaps): 12 new tests added — `TestApplyFullPreprocessing` (5), `TestQuickEvaluateDirect` (4), `TestEmptyTPEFallback` (1), `TestPipelineRoundtrip` (2). All green.

**Surprise during implementation:** `run_search` returns `(df_ranked, label_encoder)`, not a plain list. The original F9 test iterated over the tuple and hit `TypeError: argument of type 'NoneType' is not iterable` when `label_encoder` was `None`. Fixed by unpacking the tuple and using DataFrame column assertions.

**Self-review verdict:** READY_TO_PR. No silent-mismatch paths detected in downstream consumers (model_io doesn't read baseline_params directly; validation rebuild threads it through `build_preprocessing_pipeline` correctly). `tpe_score=None` on non-TPE rows is harmless data inflation.

### Cross-cutting takeaway

Two patterns recurred across all three reviews (DeepSeek on T-41, self-review + GLM on T-37):

1. **Silent degradation through coincidental alignment.** `create_study(load_if_exists=True, sampler=...)` would have aligned with the loaded study's pre-existing sampler IF Optuna re-attached it (it doesn't). `_tpe_baseline_params` aligns with `preprocess.py` defaults today (might not tomorrow). The kind of failure that ships and breaks months later.

2. **Domain-knowledge gaps in code review.** GLM caught the Optuna ask/tell-interface constraint that an initial diff-only read missed — an external reviewer with deep knowledge of the specific library can catch failure modes that whole-codebase but library-shallow reviewers can't. Cross-family review also matters: DeepSeek and GLM agreed on most findings but GLM's domain expertise on Optuna's API contract was decisive on F1/F10. Family-orthogonal panels (Anthropic + DeepSeek + Zhipu/GLM) catch what a single-family review misses.

---

## 2026-05-01 (midday) — T-37 TPE preprocessing discovery implemented; self-review caught 2 silent-mismatch bugs

**Tip:** `5ca7080` on `feature/T37-tpe-preprocessing-discovery` (5 commits past T-36 tip `1f49d73`).

### What shipped

A new TPE-based preprocessing discovery mode that replaces the basic exhaustive + GA paths with a smarter Optuna TPE search:

- **Architecture A** (model-agnostic surrogate): LightGBM proxy evaluates preprocessing quality, returns top-N diverse configs, search loop tests ALL enabled models against each — preserves model diversity.
- **5-D search space**: preproc (14 cat) x window (derivative-aware, ported from ga_preprocessing) x autoscale (T-36, bool) x baseline (5 cat) x smoothing (bool).
- **multivariate=True** TPESampler exploits the ordered-window dimension — justifies TPE over exhaustive on an otherwise-categorical space.
- **75-trial default** (50/75/100/150 GUI dropdown), 20 random startup.
- **Diversity selection** ported from preprocessing_discovery.py's `select_diverse_configs` — ensures top-N configs span different preprocessing families, not all variants of one.
- **Output contract identical to basic discovery** — the search loop doesn't know whether configs came from TPE or exhaustive.

### Self-review caught 2 silent-mismatch bugs

Following the T-36 lesson (audit ALL 6 consumer surfaces when adding a new flag):

1. **preprocess_cfg["name"] used pipeline_name instead of display_name.** The `name` field was set to the clean pipeline name (e.g. "deriv") instead of the display name with all prefix cascades (e.g. "als+sg0+deriv_snv_w23+autoscale"). Result rows and validation rebuild parse the Preprocess column for baseline/smoothing/autoscale prefixes — missing prefixes would silently lose those downstream settings. Fixed by using the already-computed `display_name`.

2. **Baseline/smoothing/autoscale doubling blocks re-doubled TPE-discovered per-config settings.** The doubling blocks (search.py ~1893-1888 and ~5521-5535) run unconditionally after the config-building phase. TPE configs already have per-config baseline/smoothing/autoscale values — the global doubling would add extra config variants with overridden values. Fixed by nulling global flags (`baseline_method = None; autoscale = False; smoothing = False`) inside the TPE success block so doubling blocks are no-ops.

### Architecture insight: TPE is model-agnostic

The TPE path does NOT use the user's selected models to evaluate preprocessing quality — it uses a LightGBM proxy (same surrogate as basic discovery). This is by design (Architecture A from the user's constraint). Per-model parallel mini-studies (Architecture B) would produce per-model preprocessing configs, collapsing model x preprocessing diversity. The current approach preserves the same output contract as basic discovery.

### Test coverage

21 new tests in `tests/test_tpe_preprocessing_discovery.py` — all green:
- 3 module structure (import, DERIVATIVE_WINDOW_RANGES, BASELINE_METHODS)
- 5 end-to-end (regression/classification/one_class/diversity/dimensions)
- 4 dimension disablement (individual axes + all-disabled)
- 5 edge cases (tiny dataset, constant columns, n_top, windows, progress)
- 3 integration (run_search + run_one_class_search + mutual exclusion with smart)
- 1 reproducibility (deterministic with seed=42)

### Next session

T-37 is ready for review. T-38 (dead preprocessing module cleanup) is unblocked.

---
## 2026-05-01 (overnight) — T-36 autoscale toggle implemented end-to-end; Codex caught 3 downstream silent-mismatch paths DeepSeek missed

**Tip:** `2351c3c` on `feature/T36-autoscale-toggle` (13 commits past `da51f60`). Branch ready for PR.

### What shipped

A user-selectable autoscale (UV scaling) preprocessing toggle that:
- **Grid path** (`run_search`, `run_one_class_search`) — doubles `preprocess_configs` so every selected combo runs both with and without autoscale; `+autoscale` suffix on display name; `Autoscale` boolean column on result rows.
- **Bayesian path** (`run_unified_bayesian`) — `apply_autoscale` becomes a per-trial Optuna categorical when `enable_autoscale=True` is set. TPE learns whether autoscale helps for the dataset.
- **Validation rebuild** (`compute_validation_metrics_for_top_models` + one-class twin) — reads `Autoscale` column from result rows, threads it into `build_preprocessing_pipeline`. Cache keys updated.
- **Per-model `StandardScaler` skip** for `SCALE_SENSITIVE_MODELS` when autoscale is on, in both grid and Bayesian paths. PLS-DA's internal scaler (operating on PLS scores) preserved.
- **GUI** — checkbox in Preprocessing tab + tooltip; flag threaded to all four search call sites.
- **Model Dev tab + code export** — refinement and exported scripts now reproduce the autoscaled pipeline (Codex catch — see below).

### Three pre-existing bugs fixed (uncovered during plan review)

1. **`unified_bayesian.py:apply_preprocessing` early-return.** Function returned early in every named-prep branch, so a naive autoscale step at the end would have been UNREACHABLE for every preprocessing name. Fix: assign-then-return restructure with single trailing autoscale block.
2. **Bayesian preprocessing cache key omitted `apply_autoscale`.** Two trials with same prep but different autoscale would collide on the cache, second trial silently receiving the first's `X_prep`. Fix: 7th element added to cache_key tuple.
3. **`contamination.py` validation cache key omitted autoscale.** Same class as #2 for one-class. Fix: 9th element added.

### Bundled adjacent fix

`run_one_class_search` result rows previously omitted `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder`. Validation rebuild used silent defaults. Fixed in the same commits as the autoscale row writes — same code surface, near-zero risk.

### Codex caught three downstream silent-mismatch paths DeepSeek missed

DeepSeek's per-phase reviews focused on the diff itself + immediate test surface. **Codex's final cross-family review (commit 8e5137c) found three silent-mismatch paths in downstream consumers** — all HIGH/MEDIUM, all real, all closed in `98fb80f` and `2351c3c`:

1. **HIGH** — Model Development tab `Preprocess` parser at `spectral_predict_gui_optimized.py:33045` assumed the LAST `+` segment is the core preprocessing name. For `snv+autoscale` the parser set `core='autoscale'` and fell through to `'raw'`, silently building a wrong refined pipeline. Fix: strip trailing `+autoscale` first, prefer explicit `Autoscale` column over suffix parsing, set `self.use_autoscale` for downstream.
2. **HIGH** — Three Model Dev refinement `build_preprocessing_pipeline()` calls (one-class refinement, Path A full-spectrum-derivative, Path B raw/SNV) didn't pass `autoscale=...`. Selecting an autoscale=True search row and refining produced a non-autoscaled pipeline. Fix: every refinement call now passes the flag.
3. **MEDIUM** — Code export had no autoscale contract. `model_config` didn't carry the flag, and `code_generator._render_preprocessing_application` emitted no StandardScaler step. Exported Python scripts couldn't reproduce autoscaled training pipelines. Fix: GUI passes `autoscale` into `model_config`, renderer emits `from sklearn.preprocessing import StandardScaler` + fit_transform after the spectral block.

### Lesson for future T-36-shaped tickets

When a new flag is added to a search loop, audit ALL downstream consumers — Model Development tab refinement, code export, model save/load metadata — for parallel changes. DeepSeek's diff-scoped review correctly verified the search-loop changes; only Codex's broader cross-family pass spotted that the same flag wasn't being threaded into the four refinement call sites and the code generator. Pattern: **new flags travel through (a) search loop, (b) result row, (c) validation rebuild, (d) Model Dev refinement rebuild, (e) saved-model metadata, (f) exported-script rendering**. T-36's plan covered (a)-(c); Codex caught the gap on (d)-(f).

### Review-pass discipline

| Phase | DeepSeek verdict | Findings |
|---|---|---|
| 2 (preprocess.py) | READY_TO_PROCEED | 3 LOW (test ordering tightened) |
| 3 (search.py grid) | READY_TO_PROCEED | 1 MEDIUM (string-bool parse) + 2 LOW |
| 4 (one-class + contamination) | READY_TO_PROCEED | 1 MEDIUM (silent-skip assertion hardened) |
| 5 (Bayesian + 3 bugs) | READY_TO_PROCEED | 1 MEDIUM (display-name suffix) + 1 LOW (deriv3/4 test coverage) |
| 6 (GUI) | READY_TO_PROCEED | 0 findings |
| 7 — Codex round-1 | BLOCKERS_FOUND | 2 HIGH + 1 MEDIUM (all closed in `98fb80f`) |
| 7 — Codex round-2 | READY_TO_MERGE_WITH_NITS | 1 MEDIUM + 1 LOW (closed in `2351c3c`) |

Total: 13 commits (12 review + fix iterations + 1 final nits commit). Mirrors T-11's seven-pass pattern at smaller scale.

### Test coverage

- 62 new T-36 tests across 4 new files: `test_preprocess_extended.py::TestAutoscaleStep` (9), `test_autoscale_grid_doubling.py` (5), `test_autoscale_one_class.py` (4), `test_autoscale_bayesian.py` (44).
- 203 in the targeted regression sweep (autoscale + preprocess + Bayesian + cv_pls_clamp + contamination + export) — all green.
- 10 pre-existing I/O test failures (jcamp/spc/opus/perkinelmer reader modules) are unrelated; verified by stash-and-rerun against `da51f60`.

### Next session

T-36 is ready for `gh pr create`. T-37 (TPE quick preprocessing discovery) is unblocked.

---

## 2026-05-01 (early hours) — T-11 MERGED via PR #6 after seven reviewer passes; 4 deferral tickets filed

T-11 went from APPROVED-but-not-pushed to MERGED via the project's first
GitHub PR (#6) instead of the prior fast-forward-from-local pattern.
Decision driver: T-11 is the largest single feature on this codebase
(~7000 LOC, new modules) with non-trivial threading + persistence + GUI
state surfaces. A PR URL is the cheap audit-trail artifact for future
bisects.

**Seven independent reviewer passes converged on READY_TO_MERGE:**

1. Codex initial (4 HIGH + 3 MEDIUM, all fixed pre-PR)
2. Kimi K2.6 initial (2 MAJOR + 4 MINOR, all fixed pre-PR)
3. DeepSeek V4 Pro pass-1 (24h sweep, 2 HIGH + 3 MEDIUM + 9 LOW/INFO,
   all closed in `446465c` / `9107a68` / `53b3078`)
4. DeepSeek V4 Pro pass-2 (recheck, found 2 NEW HIGH — same
   `__compiled__ in dir()` pattern in 2 GUI sites the pass-1 grep
   missed because it was scoped to `src/spectral_predict/`. Closed
   in `e62c15d`)
5. Five specialist agents in parallel (code-reviewer, pr-test-analyzer,
   silent-failure-hunter, comment-analyzer, type-design-analyzer).
   13 findings: Cluster A (3 interlocking sidecar-lifecycle bugs in
   `run_state.py`), Cluster B (3 logger-failure-isolation bugs in
   `run_logging.py`), Cluster C (start_run Frankenstein metadata),
   Cluster D (5 test-coverage gaps for headline contracts), plus 8
   "Important" + 4 "Suggestion" items. 12 closed in commits
   `37a71ea` / `5db0b40` / `4dc7ec3` / `6fd8dab` / `29fe489`. 1
   deferred → T-34.
6. Codex meta-review of the synthesis. Found NEW BUG #1 (mark_complete
   deletes unrelated sidecars — GUI calls it after every successful
   analysis, so a paused Bayesian sidecar was destroyed by a fresh
   grid run; closed in the run_state.py rewrite) AND deferred NEW
   BUG #2 (per-model Bayesian failure swallowing → mark_complete
   still runs, can erase resume state for partially-completed
   multi-model runs; filed as T-34).
7. DeepSeek V4 Pro pass-3 with `--variant high` (high-effort
   reasoning). Verdict: READY_TO_MERGE. 0 blockers, 1 cosmetic
   observation explicitly accepted ("not worth fixing").

**Final T-11 test count:** 47 (34 original + 13 added in `29fe489`
covering the cluster-A/B/C contracts and the rotation regression).

**Lessons that recur:**

- "Diff-only" review misses pre-existing copies of patterns. The
  `__compiled__ in dir()` Nuitka bug existed in 4 sites; pass-1 grep
  scoped to `src/` and pass-2 broadened to the GUI monolith caught
  the remaining 2. Lesson: when a finding surfaces a *class* of bug
  not an isolated incident, ask the next pass to grep repo-wide for
  the entire pattern.
- Multi-agent convergence as a defect-confidence signal. The
  `start_run` Frankenstein-metadata bug was flagged independently
  by type-design-analyzer ("dataclass returns inconsistent state")
  and code-reviewer ("API silently lies about which copy is
  authoritative"). Two agents arriving at the same defect through
  orthogonal lenses = near-zero false-positive probability.
- The chemometrics master rule scaling. Five new agents + Codex,
  zero false-positives flagging SNV/varsel/CARS-Tree as bugs. Rule
  was embedded in every prompt with concrete examples. Without it,
  ~30% of output would have been sklearn-instinct noise.
- "High-effort" reasoning mode for the gate-pass review. Pass-3
  used opencode `--variant high` (provider-specific reasoning
  effort), 12 minutes wall time, extensive `Thinking:` traces showing
  actual code-tracing work. Worth the budget for the final pre-merge
  pass.

**Four new tickets filed for deferrals:**

- **T-34** per-model Bayesian failure handling (NEW BUG #2). Requires
  per-model sidecars — architectural change. `docs/plans/2026-04-30-T34-per-model-bayesian-failure-handling.md`
- **T-35** T-11 type-design follow-ups. `RunMetadata` `frozen=True,
  slots=True` + `__post_init__`, `_TeeStream` extract `_LineBuffer`
  helper, `_TeeStream` add missing file-protocol attrs.
  `docs/plans/2026-04-30-T35-t11-type-design-followups.md`
- **T-39** `fingerprint_dataset` numpy 2.x repr stability. `str(X.flat[idx])`
  format differs across numpy versions; spurious "data has changed"
  rejections on resume after numpy upgrade.
  `docs/plans/2026-04-30-T39-fingerprint-numpy-stability.md` (originally
  drafted as T-36; renumbered to avoid collision with parallel session's
  T-36 autoscale-toggle reservation)
- **T-40** Stop-vs-Complete + concurrent-instance footgun. Stop button
  silently kills resume option; two app instances can race on resume
  dialog. `docs/plans/2026-04-30-T40-stop-vs-complete.md` (originally
  drafted as T-37; renumbered to avoid collision with parallel session's
  T-37 TPE-quick-preprocessing-discovery reservation)

T-33 GP regression rough plan was filed earlier in this session via
user request. Still ROUGH_PLAN, awaits prioritization.

**Merge details:** PR #6 rebase-merged at `50057af` (this commit's
parent in `git log main`). 13 commits + 1 PROJECT_STATUS update.
Original branch SHAs (`4084352`, etc.) became new SHAs after rebase
replay; GitHub keeps both for history queries.

---

## 2026-04-30 (evening) — T-08 dropped, T-11 shipped, T-15 dropped, T-16 reframed, T-19 user-framed

Continuation session after the morning's T-06 + T-06b merges. User asked
for parallel reality-checks on T-15, T-19, T-11, T-08, then T-16
(reframed). Verdicts:

**T-08 CARS tree-mode bug — DROPPED.** Third gate-caught false alarm of
the same shape (after T-26 SNV, search.py:2855 top_n_vars hardcoding):
prior agent cited line numbers in the wrong control-flow branch
(1519-1522 are PLS-mode; tree-mode is at 1499-1507). Empirical
reproducer disproved all three claims (oscillation, bias, persistent
tiny weights). CARS-Tree converges with std 0.0007-0.0038 over last 10
iterations. **User confirmed: CARS-Tree is dasp's invention** for tree
models that lack PLS coefficients (canonical CARS depends on those);
saved to project memory so future agents don't search Li 2009 for a
nonexistent canonical "tree-mode CARS."

**T-11 Pause/resume + Optuna SQLite + disk logging — SHIPPED LOCALLY**
(committed, not pushed per user request). Three sub-units:
- A: `RotatingFileHandler` writing to `<user_data_dir>/dasp/logs/` +
  thread-safe tee proxy over stdout/stderr capturing backend prints that
  hit /dev/null in the bundle. Closes T-12 simultaneously.
- B: `_actually_paused` event in `SearchController` set inside
  `check_and_wait`'s `try/finally`. New 'pausing' UI state. Resume checks
  `analysis_thread.is_alive()`.
- D: per-run UUID + SQLite storage URL + sidecar JSON tracking active
  runs. Atomic sidecar writes via `tempfile + fsync + os.replace`.
  Resume-on-startup dialog. Run-state ONLY for Bayesian (`optimization_method == "unified"`).

**Cross-family review caught real bugs that would have shipped:**
- Codex HIGH #3: `study.optimize(n_trials=N)` runs N MORE trials, not
  until total reaches N. Crashed 80/100 study would have resumed with
  100 fresh trials = 180 total. Fixed via terminal-state-filtered remaining-trial clamp.
- Codex HIGH #5: `write_text()` could leave partial JSON on crash. Fixed
  via atomic-write-replace.
- Codex HIGH #7: dataset fingerprint stored but unused — user could
  resume on different data and Optuna would silently pick up trials with
  stale objective values. Fixed via GUI-side `verify_resume_fingerprint`
  check before `start_run`.
- Codex HIGH (additional): study_name was just `unified_bayesian_{model_name}` —
  `load_if_exists=True` would silently mix trials across runs with
  different task_type / CV / seed / etc. Fixed via config-fingerprint hash
  in study_name.
- Codex MEDIUMs: `_TeeStream` thread-unsafe, no log rotation, run-state
  firing for grid/NSGA-II (misleading resume prompts). All fixed.
- Kimi MAJOR #2: `verify_resume_fingerprint` didn't check run_id —
  second app instance could overwrite sidecar between resume and check.
  Fixed: assert `data["run_id"] == _active_run_id`.
- Kimi MAJOR #3b: GUI used `clear_resume_state` on fingerprint mismatch
  → sidecar persisted → re-prompted forever. Fixed: GUI calls
  `discard_incomplete_run` instead.
- Kimi MINOR #4: `_TeeStream.splitlines()` splits on `\r` AND `\n`;
  tqdm `\r<bar>\r<bar>\n` would emit one log line per bar update. Fixed
  via `replace("\r", "")` before split.
- Kimi MINOR #5: pre-existing Nuitka detection bug. `"__compiled__" in
  dir()` checks function-local namespace, not module globals. Fixed at
  3 sites in `resource_paths.py`.
- Kimi MINOR #6: `n_trials` in `study_name` orphans old studies on
  target change. Removed; rely on trial-count clamp.
- Kimi MINOR #7: SQLite default short busy-timeout → "database is locked"
  on Windows under contention. Fixed: `?check_same_thread=False&timeout=30`.

Documented as known limitations: cross-process sidecar locking
(deferred — would need filelock dep, GUI is single-instance in practice);
search-space hyperparameter bounds NOT in config_hash (theoretical for
now since custom ranges from Tab 4C don't currently flow into Bayesian);
GUI thread-integration test (Tk threading tests are flaky).

Test sweep: 260/260 pass.

**T-15 LeaveOneGroupOut — DROPPED** by user decision after gate
investigation validated their pre-investigation skepticism. The user's
own paper data (5-100 N per site, 20× ratio) makes LOGO a footgun
without uncertainty quantification. Chemometrics literature (Westad &
Marini 2015, Workman 2018) recommends external test sets, NOT LOGO. No
single canonical citation mandates LOGO over external sets. Competitor
parity mixed (only PLS_Toolbox exposes group-aware CV). The user's paper
§2.6 was the only load-bearing reason; since that paper isn't actively
shipping, dropped.

**T-16 model-comparison machinery — REFRAMED.** User asked "what do
competitors do" and "compare between models." Investigation surfaced a
strategic split: chemometrics tools (Unscrambler/SIMCA/PLS_Toolbox/OPUS)
ship single-model validation only — Y-permutation (PLS_Toolbox) +
coefficient jackknife (Unscrambler). They DO NOT ship two-model paired
comparison. ML frameworks (caret/mlr3/tidymodels) ship the actual
between-model machinery (paired t-test, Wilcoxon, Friedman+Nemenyi).
Four candidate shapes catalogued; Shape A (jackknife + Y-permutation)
gets PLS_Toolbox parity + Unscrambler-adjacent parity for ~3-5 days +
closes T-13 simultaneously. Shape B requires per-fold metric storage
schema upgrade (hidden infra cost ~2-3 days on top). User decision
pending: A vs B vs hybrid.

**T-19 model-native imbalance handling — REFRAMED.** User clarified the
framing: "expose model-native abilities OR auto-detect," not the
roadmap's "FTIR Bone PLS paper reproducibility." Investigation
confirmed: math is already statistically equivalent today (existing
imbalance dropdown's `class_weight` selection routes through
`compute_sample_weight('balanced')` for boosting models). Gap is
audit-trail labels (result CSVs say "sample_weight" not
"scale_pos_weight") + 5 unwired PLS-DA inner-LR sites + T-32 closure.
Smaller scope confirmed: ~2-3 days vs. design doc's 5-7 days. Yesterday's
design doc + Codex review captured the detail; do not re-investigate
from scratch. User affirmed via course-correction: "make sure the
imbalance thing is not another ticket, as we dealt with this in real
detail yesterday."

**Memory rules saved this session:**
- `feedback_glm_routing.md` — never dispatch GLM 5.1 via opencode-go
  (bills user's z.ai subscription; flat-rate opencode-go plan covers
  Kimi/DeepSeek/MiniMax/Qwen/MiMo only).
- `project_cars_tree_origin.md` — CARS-Tree is dasp invention; canonical
  CARS doesn't run on tree models that lack PLS coefficients.
- `project_t19_user_framing.md` — T-19 is one ticket, not a fork.
  User's lens: expose existing abilities + auto-detect.
- `project_t15_dropped_t16_reframed.md` — T-15 closed, T-16 standalone
  competitive-machinery survey.

**Lessons reinforced this session:**
1. **Reality check (gate step 1) keeps catching framings that don't survive direct line-number inspection.** T-08 was the third in this pattern (after T-26 SNV and search.py:2855). Future ticket framings citing "buggy" code at specific line numbers should be expected to have this kind of error.
2. **Cross-family review (Codex US + Kimi K2.6 Moonshot) catches different blind spots.** This session: Codex caught Optuna-specific gotchas (trial-count overrun, study-name collisions). Kimi caught Python/threading gotchas (`splitlines` on `\r`, `dir()` vs `globals()` for Nuitka, run_id race, SQLite Windows locking).
3. **The "competitors don't ship X" finding cuts two ways.** For T-11: it means dasp is the outlier among AutoML-adjacent tools (auto-sklearn / H2O / FLAML all do checkpoint-with-resume), so ship parity. For T-15: it means dasp dropping LOGO matches OPUS/SIMCA/Unscrambler (only PLS_Toolbox exposes it), so dropping isn't an outlier choice. Not all "field doesn't do X" findings are alike.
4. **Domain pushback ("group composition is variable in practice") was empirically validated.** The user's instinct caught a real chemometrics issue (LOGO footgun on uneven group sizes) before any work was done. The gate's job was to test the instinct against actual data + literature; both confirmed it.

---

## 2026-04-30 (later) — T-06 SPA canonical Araújo 2001 enumeration

Picked up DASP validation-gate ticket triage on a different machine (the
2026-04-30 morning bugfix run was completed on the prior machine). Selected
T-06 (SPA `n_random_starts` non-functionality) as the smaller-win-to-verify-
gate-still-works candidate per `docs/CONTINUATION_PROMPT_2026-04-30.md`.

**Investigation phase** (parallel general-purpose agent, findings at
`docs/bugfix_validation/T06_findings.md`):
- Confirmed roadmap framing on code-reading: every iteration of `for start_idx
  in range(n_random_starts)` produces byte-identical output. The function's
  own docstring at line 322-324 already conceded: "currently SPA is
  deterministic, but this parameter is included for API consistency and
  future enhancements." Empirical confirmation: `n_random_starts ∈ {1, 5,
  10}` and `random_state ∈ {42, 123}` produce identical importances.
- **Key gate-finding that flipped the fix path:** the roadmap proposed
  `rng.choice()` for random first-variable selection (Option A). Field
  alignment check killed this. Canonical Araújo 2001 SPA is **deterministic
  enumeration over every variable as candidate seed**, not random restarts.
  `auswahl` (modern Python reference) explicitly enumerates every variable
  with no `n_random_starts` and no `random_state`. Galvão 2012 SPA-GUI
  (Araújo's own group) follows the same pattern. No verified chemometrics
  implementation exposes random restarts for SPA. Option A would have been
  a textbook sklearn-instinct-on-chemometrics-domain failure — the recurring
  master-rule violation.
- **GUI reachability confirmed:** unlike T-26's backend-only-knob trap, T-06
  has a real Spinbox (`gui:12085`, range 1-50, default 10) plumbed through 6+
  `search.py` call sites. The fix surface IS reachable to bundled-app users.

**Disposition:** Option B — canonical Araújo 2001 enumeration. Replaced
`for start_idx in range(n_random_starts)` with `for first_var in
range(n_vars)` in production + the export `SPA_TEMPLATE`. Dropped
`n_random_starts` and `random_state` from `spa_selection` signature plus
the three hybrid signatures (`uve_spa_selection`, `uve_cars_spa_selection`,
`fipls_spa_selection`). Removed the GUI Spinbox + IntVar + 2 plumbing call
sites. Removed 6 `search.py` call-site usages + signature parameter on
`run_search` and `run_one_class_search`. Removed 2 hardcoded
`spa_n_random_starts = 10` blocks in `bayesian_utils.py`.

**Cross-family review caught real bugs the verdict-author missed:**
- Codex (US-trained) found a missed call-site at
  `tests/gui/test_comprehensive.py:387` still passing `random_state=42` →
  fixed. Plus a template ↔ in-app divergence on small-sample CV-fold
  reduction and failure-fallback semantics → fixed (template now mirrors
  production's small-sample fold adjustment + uniform fallback).
- Kimi K2.6 (Moonshot, Chinese-trained, via opencode-go subscription) found
  a dead `y_norm` computation in production (leftover from the argmax-
  correlation seed path) → removed. Plus a Python-loop projection in the
  template (~2-3 orders of magnitude slower than production's vectorized
  matmul on FTIR-scale J=2000) → vectorized.
- Both reviewers independently flagged the original soft
  `test_spa_explores_multiple_seeds` test as too weak — it would have
  passed under the pre-T-06 implementation. Replaced with a deterministic
  pinned-data fixture (`np.random.default_rng(0)`, 30×15 noise) where
  canonical SPA picks chain `[0, 5, 6, 7, 10]` from seed 6 while argmax-
  only would pick `[3, 5, 8, 10, 11]` from seed 11. Added Kimi's suggested
  call-count invariant test as additive coverage (`test_spa_evaluates_all_j_seeds`).

**Performance follow-up implemented in same session (T-06b):** the
sequential canonical enumeration was 100×–1500× the pre-fix work
(~30-750 sec on bone-FTIR-scale data). Parallelized in branch
`fix/T06b-spa-parallel` via `joblib.Parallel(backend='threading')`
across J independent seed evaluations. Threading (not loky) for
PyInstaller-bundle safety per the existing `_frozen_needs_threading_fallback`
pattern. Inner `cross_val_score` keeps `n_jobs=1` to avoid nested
parallelism. Worker count capped at min(cpu_count, 8). Empirical
FTIR-scale benchmark (J=800, n=50): 10.92 sec parallelized vs. ~70-80
sec sequential — ~7-8× speedup. Export `SPA_TEMPLATE` parallelized
identically so exported user scripts match in-app SPA performance.

**Lesson reinforced (master-rule, again):** The roadmap's `rng.choice()`
fix would have been a textbook sklearn-instinct slippage —
chemometrics-aligned reviewers would catch it; sklearn-trained reviewers
would not. The gate methodology's step 3 (field alignment via
documentation lookup, not generic intuition) caught it. Future agents
encountering "SPA random starts is broken" tickets MUST do the field-
alignment lookup first.

**Commits / merge:** `fix/T06-spa-canonical-seeds` ff-merged into main.
Validation note: `docs/bugfix_validation/T06_spa_canonical_seeds.md`.

---

## 2026-04-30 — Bugfix branch validation gate session (closes the previous-day's run)

Each of the 5 implemented bugfix branches went through a strict validation gate
(literature check + commercial-software comparison + reachability verification +
regression test sweep). Plus the two re-evaluation flags + T-32 + T-04 + T-21 were
investigated using the same gate. The gate caught two categories of overzealous flag
that would have wasted implementation effort:

**False-alarm patterns the gate caught:**
- **T-26 SNV near-zero std** — main behavior already matches PLS_Toolbox default at
  offset=0 (both produce unit-normalized round-off on degenerate spectra). The user
  has run thousands of analyses without hitting it. Verdict: DROP. Tests would have
  passed; field-alignment check showed dasp wasn't actually misaligned.
- **`search.py:2855 top_n_vars=30`** flagged as "count shown disagrees with count
  used." Turned out to be display economy: `all_vars` column already preserves the
  full wavelength list for replication; `top_vars` is a separate display-only
  truncated list. Not a correctness bug; the model uses the right number of features.
  Verdict: DROP.
- **T-32 sample_weight length mismatch at search.py:3883** — defensive code for
  resampler+sample_weight combinations that's currently unreachable because
  `_needs_resampling_pipeline` returns False when imbalance_method='class_weight'.
  Codex correctly identified the future-bug under T-19's planned scope. Verdict:
  DEFER to T-19.
- **`bayesian_utils.py random_state=42`** flagged as "ignores user setting." Turned
  out: no user setting exists (random_state isn't exposed anywhere in the codebase).
  Real issue is just code-style (use the shared `RANDOM_STATE` constant from
  `constants.py` instead of literal 42). Refactored as `50d5d05`.

**Real bugs the gate confirmed and fixed:**
- **T-10 PLS components clamp** — main's `n_samples * (folds-1) // folds` formula is
  K-fold-correct but over-clamps for LOO + spinbox-default-folds (gives n*4//5
  instead of n-1). Real bug, small impact for typical bone-FTIR n_components <= 15.
  Merged at `fbeb50c`.
- **T-05 VIP formula** — main's `compute_vip` used `var(y) * (T'T)` instead of
  canonical Wold 2001 `q_a^2 * (T'T)`. On NIR/FTIR-realistic data with structured
  X-noise (interferents, scattering), main picks wrong wavelengths in top-N. Field
  consensus is universal — every commercial + open-source chemometrics package uses
  canonical. dasp internally inconsistent (contaminant_analysis already canonical).
  Merged at `2c068cd`. Plus T-05a duplicate fixes at `1eb6c06` for
  `templates/variable_selection.py:54` + `nsga2_search.py:627`.
- **T-07 PDS even-window** — main's `estimate_pds(window=even)` crashes with cryptic
  numpy traceback. Universal field consensus on odd-only windows (Wang 1991 +
  Bouveresse 1996 + Feudale 2002 + RNIR + specProc + PLS_Toolbox). Plus apply_pds
  hardening: derive geometry from B.shape[1] not caller arg. Merged at `1b91d93`.
- **T-24 Lin's CCC metric addition** — formula bit-correct vs Lin 1989. Field
  alignment soft-flagged (CCC standard for method-comparison since 1989, but
  chemometrics-specific tools rarely report it by default). User explicit override:
  "EVEN IF ccc IS NOT SUPER common in my domain... it is clearly relevant and a
  small addition so i would say go ahead." GUI plumbing fixed pre-merge per user
  requirement (CCC/CCCcv added to higher_is_better_cols + tooltip dict). Merged at
  `0087cad`.
- **T-04 one-class UVE prefilter** — UVE-on-y_oc returns wavelengths that
  discriminate outliers from inliers, not wavelengths defining the inlier class
  structure. Pomerantsev, Kucheryavskiy & Rodionova (2025) "Variable selection for
  one class classifiers. Introduction of LOVE" opens with exactly this critique.
  Empirical bone-FTIR demo: main's UVE prefilter selected 5/5 consolidant peak
  wavelengths, 0/20 phosphate or carbonate (the actual bone chemistry). Verdict:
  GUI grey-out for one-class mode matching the existing iPLS pattern. Merged at
  `6beb5e8`. T-04b/c follow-ups deferred (broader y_oc-as-target audit + LOVE-style
  native one-class varsel).
- **T-21 SG wavelength uniformity guard** — Savitzky-Golay 1964 assumes uniform
  spacing. dasp's "Convert to other unit" button (`_convert_x_unit_and_replot`)
  numerically inverts column values via 1e7/x, producing a non-uniform grid; SG
  on that data is silently miscomputed (verified empirically: median 22% / max 60%
  relative error in peak regions on a representative NIR spectrum). User-verified
  reachability finding: the radio button (`_on_x_unit_override`) is RELABEL-ONLY
  (no value modification — verified at gui:19007-19029); only the Convert button
  creates the bug surface. Disposition: hide the Convert button (don't .pack() it),
  preserve the function in code for a future resample-on-convert fix. Resolved at
  `a5eef70`. The original T-21 implementation plan
  (`docs/plans/2026-04-29-T21-sg-wavelength-uniformity-guard.md`) had factual
  errors flagged by the gate (fabricated OpenSpecy `is_evenly_spaced()`, unverified
  PLS_Toolbox `gridcheck` tolerance, scope gap with 30+ direct SavgolDerivative
  callsites bypassing the planned guard) — annotated with a banner pointing to the
  findings doc.

**Lessons learned (now documented in `docs/bugfix_validation/README.md`):**
1. **Verify reachability before drafting verdicts.** T-26, T-32, and the two
   re-evaluation flags all turned out to be reachable-only-in-theory. The gate's
   recurring test: "is the buggy code path actually hit by the GUI's bundled-app
   user?"
2. **Verify commercial-software behavior with documentation lookup, not generic
   intuition.** T-26's first-pass verdict appealed to "universal numerical-computing
   practice" (sklearn-instinct). Actual PLS_Toolbox / SIMCA documentation showed the
   field uses a different pattern (continuous additive `offset` parameter, default 0).
3. **Bundled-app distribution matters.** A "fix" reachable only from a Python REPL
   is dead code for non-technical users. T-26's hardcoded threshold and T-04's
   programmatic-only mitigation would have failed this test.
4. **A real finding can warrant zero action.** T-26's "current behavior already
   matches PLS_Toolbox default" justified a DROP rather than a code change. T-21's
   "rarely used button creates the bug surface" justified hiding the button rather
   than fixing every SG callsite.
5. **Match-the-field cuts both ways.** When the field has converged on a pattern
   (T-05 VIP canonical formula, T-07 PDS odd-only windows), matching it is high
   confidence. When the field is actively developing replacements (T-04 one-class
   variable selection, the 2024-2025 LOVE / MPS-SIMCA literature), being the outlier
   is a strong signal something's wrong.

**Outstanding decisions still pending from the broader roadmap re-evaluation:**
1. T-31 multi-class SIMCA — confirm "none of the above" output is useful for
   bone-FTIR/diagenesis science.
2. T-01 reframe scope — confirm external-test-set workflow over per-fold varsel.
3. T-22 reframe — confirm bootstrap stability diagnostic investment.
4. T-04 follow-ups — T-04b broader y_oc-as-target audit (compute_one_class_importances
   has the same fundamental issue), T-04c proper one-class-native varsel
   (Forina modeling power / LOVE / OGA).

---

## 2026-04-30 — Full roadmap re-evaluation under chemometrics master rule

**Re-evaluation complete.** All 35 items (32 tickets + T-05a, T-10b, T-31 PENDING + P3 drop list) re-evaluated against chemometrics literature + bone-FTIR application domain.

**Results:** 27 KEEP, 2 REFRAME, 2 DROP, 2 DEFER, 2 NEEDS_USER_DECISION.

**Key findings that contradicted prior agent framing:**
- **T-02 (ensemble OOF preprocessor) is a FALSE ALARM.** `PreprocessorConfig` at `preprocessing_wrapper.py:15-100` only applies per-spectrum operations (SNV, SG derivatives, wavelength subsetting). No cross-sample statistics. Same false-alarm pattern as T-01 — prior agents assumed the preprocessor learns cross-sample statistics. It doesn't.
- **T-03 (preprocessing-discovery full-data) is a FALSE ALARM.** The ticket misread the code. Importances do NOT feed into the ranking score at `preprocessing_discovery.py:570-662`; `_quick_evaluate()` uses honest 5-fold `cross_val_score()`. The importance computation is a separate side output. Furthermore, preprocessing choice ranking by CV performance is standard chemometrics practice (The Unscrambler's "Preprocessing Advisor" does exactly this).
- **T-04 (one-class UVE prefilter) is REAL** — distinct from leakage question. UVE with inlier/outlier y selects wavelengths that distinguish outliers from inliers, the opposite of what one-class screening needs. Centner et al. 1996 UVE uses y for PLS coefficient reliability; for one-class, the relevant y is "stable within inliers," not "distinguishes inliers from outliers."
- **T-21 (SG uniformity guard) is chemometrics-correct.** Savitzky & Golay 1964 assumes uniform sampling. PLS_Toolbox's `gridcheck` and OpenSpecy's `is_evenly_spaced()` are commercial/academic precedent. The warn-and-proceed design is appropriate.
- **T-22 should be reframed** as bootstrap stability diagnostic — the right answer to "is this wavelength real chemistry?"

**Worktree disposition:** `varsel_transformer.py` is MERGE_AS_OPTIONAL_TOOL (useful for T-22 stability diagnostic, expert mode, paper reproduction). Plan doc + Codex reviews are DROP. Audit doc is CHERRY_PICK (rename, fix labels).

**New potential tickets found:**
1. `bayesian_utils.py:261` hardcodes `random_state=42` in varsel calls (lines 442, 455, 469, 488, 502, 520) — ignores user's setting. Correctness issue independent of leakage framing.
2. `search.py:2855` hardcodes `top_n_vars=30` regardless of actual `n_top` — reporting mismatch.

**Deliverables:**
- `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` — full per-ticket verdicts
- `docs/varsel_leakage_worktree_disposition_2026-04-30.md` — worktree artifact disposition
- `docs/PROJECT_STATUS.md` — updated with re-evaluation summary

**User decisions needed before further work:**
1. T-31 (multi-class SIMCA): confirm "none of the above" output is useful for bone-FTIR/diagenesis science
2. T-01 reframe scope: confirm external-test-set approach over per-fold varsel
3. T-22 reframe: confirm bootstrap stability diagnostic investment

---

## 2026-04-30 — T-01 audit framing reconsidered after literature validation

**The recurring failure mode named.** Multiple sessions of agents working on dasp have flagged "data leakage" findings that turned out to be standard chemometrics workflow. Tonight this happened again at scale: the T-01 audit labeled 49 method × path combinations as LEAKY, a 2900-line implementation plan was written, 4 Codex review cycles were burned, Phase 1 of the refactor (`VarselTransformer` infrastructure) was implemented — all before the user pointed out that varsel-on-full-calibration is the published methodology in Li 2009 (CARS), Centner 1996 (UVE), Araujo 2001 (SPA), Norgaard 2000 (iPLS), Wold 2001 (PLS/VIP).

**Two independent literature-validation passes** (Codex CLI reading the Li 2009 PDF + web search; GLM 5.1 reading same + 7 chemometrics-specific bias references including Filzmoser 2009, Westad & Marini 2015, Shi 2019) converged: the bias is real, but dasp's pattern matches canonical chemometrics. The published papers escape the bias by reporting RMSEP on a separate held-out test set, NOT by per-fold varsel.

**The user's deeper point:** the *purpose* of CARS/UVE/SPA in this domain is identifying wavelengths that correspond to real chemistry. Stable selection across resamples is *evidence the chemistry is real*. Per-fold varsel that picks different wavelengths per fold destroys interpretability — "you can't say 1650 cm⁻¹ is the diagnostic amide I band if fold 2 picked 1620 and fold 3 picked 1700." The audit's proposed fix would actively damage the science it's meant to support.

**Master rule now memory-pinned at `feedback_validate_against_chemometrics_and_application_lit.md`:** every methodology decision in dasp must be validated against chemometrics literature + the applied domain (bone FTIR / isotopes / paleoanthropology), NOT against sklearn / generic ML / genomics ML conventions. When literatures conflict, chemometrics wins.

**Provider failures observed during the night** (worth logging since they may recur):
- `zai-coding-plan/glm-5.1` wedged twice in retry loops (16 min for varsel Phase 1 attempt 1, 30+ min for T-24 Tasks 8-9). `opencode-go/glm-5.1` worked first attempt for the same workloads. **Default to opencode-go for GLM going forward.**
- Codex CLI wedged once (T-21 plan review, killed after 1 hour with 39-byte output).
- Watchdog: when a long-running tool produces zero output for >2 min, kill and report — don't sit on the connection.

**State of the world after tonight:**
- 5 small ticket branches landed cleanly (`fix/T05-*`, `fix/T07-*`, `fix/T10-*`, `fix/T24-*`, `fix/T26-*`) — these are real bugs verified independently of the audit framing. Ready to merge.
- 1 ticket plan committed but not implemented (`fix/T21-sg-uniformity` plan only, codex review wedged) — needs literature check before implementation.
- 1 worktree paused (`fix/varsel-leakage` Phase 1 done, Phases 2-7 paused) — disposition pending re-evaluation agent.
- Roadmap doc + audit doc + project-status doc all annotated with reconsideration banners pointing to the master rule.
- New re-evaluation agent prompt at `docs/plans/2026-04-30-roadmap-reevaluation-prompt.md` ready to dispatch.
---

## 2026-04-30 — T-10 PLS component clamp

**What broke today:** roadmap T-10 surfaced that `min_train_samples = n_samples * (folds - 1) // folds`
at `search.py:1109` and `:3312` was K-fold-only. Under LOO the formula under-counts
the train-fold size by 1 (`4n/5` vs `n-1`). Conservative, so not a silent failure
producer in itself, but it shrinks the LOO PLS grid by 1 component on small datasets,
and any future caller who forgot to clamp would crash inside CV.

**Design call:** clamp at the call site (where `cv_strategy` is in scope) rather than
inside `models.py`. `models.py` does NOT silently re-clamp by `n_samples` because that
would mask a caller bug — the contract is documented via docstring only (no assert per
Codex suggestion #3).

**Files touched:**
- `src/spectral_predict/cv_utils.py` (new helper `compute_min_train_fold_size`)
- `src/spectral_predict/search.py` (two call sites: 1109, 3312)
- `src/spectral_predict/models.py` (docstring + comment only)
- `tests/test_cv_pls_clamp.py` (new, 21 tests)
- `docs/plans/2026-04-29-T10-pls-components-clamp.md` (NSGA-II deferred wording)

**Codex review suggestions applied:**
1. `n_folds > n_samples` validation in helper (2 extra tests).
2. Docstring says "exact" not "conservative" — the formula `n*(k-1)//k` equals `n - ceil(n/k)`.
3. No assert in models.py — docstring-only defense-in-depth.
4. `_extract_n_components_seen` uses LVs column as canonical, falls back to `ast.literal_eval` (not `json.loads`) because Params is Python repr with single quotes.
5. NSGA-II deferred ticket is an AUDIT, not an assumed identical bug.

**Group-splitter handling:** `group_kfold` and `leave_one_group_out` raise `NotImplementedError` from the helper. T-15 will route these through a separate group-aware sizing path.

**Empty-DataFrame edge case:** `compute_min_train_fold_size` rejects `n_samples < 2`. Both call sites guard with `if n_samples >= 2` before calling the helper, preserving the existing graceful-empty-return behavior in `run_search`.
---

## 2026-04-30 — T-05 VIP formula fix

### T-05: VIP formula corrected to canonical Wold (2001)
Replaced `np.var(y)` per-component weight with canonical
`q_a**2 * sum(T_a**2)` (Wold 2001, Mehmood et al. 2012 Eq. 1) in
`src/spectral_predict/models.py:compute_vip`. Old formula collapsed all
components to the same Y-weighting scalar, skewing VIP rankings whenever
components had similar X-score energy but different Y-loading. New tests
in `tests/test_vip_formula.py` lock the formula in. Existing PLS-DA
importance tests pass unchanged. See `docs/plans/2026-04-29-T05-vip-formula-fix.md`.

### Pre-fix code had no ssy_total guard
The old `compute_vip()` at `models.py:1738` used `np.var(y, axis=0)` as a scalar weight for all components. It also had no guard for `ssy_total <= 0`, meaning degenerate fits (e.g. zero y_loadings_) would produce NaN from division by zero rather than a clean all-zeros return. The fix adds both the canonical formula and the degenerate-fit guard.

### Deferred: T-05a (duplicate VIP formulas)
Two additional copies of the buggy formula exist at:
- `src/spectral_predict/templates/variable_selection.py:54`
- `src/spectral_predict/nsga2_search.py:627`
These are out of scope for T-05 and deferred to a follow-up ticket. They continue to produce mis-weighted VIP scores.
---

## 2026-04-30 (T-24 Lin's CCC) — non-obvious findings

**Exported validation template required deeper refactor than plan described.** `get_final_model_template()` in `templates/validation.py` previously returned a static `FINAL_MODEL_TEMPLATE` constant (no f-string substitution available). To inject the inline `_lins_ccc()` helper AND the `x_var` substitution it needed, the function had to be rewritten as an f-string-returning helper. Commit `e01f477` carries this refactor — it's NOT a behavior change, just turns a constant into a parameterized return.

**Templates can't import internal scoring.** Source modules can `from spectral_predict.scoring import lins_ccc`, but exported user scripts (which the templates produce) ship as standalone `.py` files. So the template inlines a local copy of `_lins_ccc()` at the top of the generated script.

**ddof=0 vs ddof=1 — only one closed-form test discriminates.** The plan's `y_pred = 2*y_true → CCC = 0.8` test does NOT distinguish ddof=0 from ddof=1 because the variance/covariance scale cancels. Added `test_ccc_finite_sample_ddof_zero_exact` per Codex suggestion: `y=[0,1,2], pred=[1,2,3]` gives `CCC = 4/7` under ddof=0 (different value under ddof=1). This is the protective test against silently switching estimator types.

**`opencode draft snapshot` empty commits** (`7d8e27d`, `8ccbe30`, `cb7d3f9`) auto-created by the dispatcher wrapper after killed opencode sessions. They contain no diffs. Drop or squash on merge.

**zai-coding-plan provider hung twice during T-24 Tasks 8-9** (each session ran 30+ min producing zero stdout / zero file changes / zero git activity, then required `kill -9`). Tasks 1-7 all completed via opencode normally on the same provider. opencode-go provider was used successfully for varsel-leakage Phase 1 around the same time, so when this happens again, fail over to opencode-go rather than retry zai-coding-plan.

---

## 2026-04-29 (T-01 audit) — per-fold variable selection leakage

Full audit at `docs/T01_VARSEL_LEAKAGE_AUDIT.md`. Non-obvious findings:

### NSGA-II CARS guidance is NOT per-fold varsel
`nsga2_search.py:1872` calls `cars_selection(X, y, ...)` but this is OUTSIDE `_evaluate()`. The result biases the initial population's wavelength sampling probability only — the per-individual CV at `_evaluate():1310` uses the chromosome's own wavelength mask with `cross_val_score`. So NSGA-II's CARS is not per-fold varsel leakage in the traditional sense; it's population-initialization bias. Low priority.

### Unified Bayesian has only 3-4 varsel methods, not the full set
`unified_bayesian.py:827` defines `available_methods = ['importance', 'cars', 'region']` with optional `uve`. No `spa`, `ga`, `ipls`, `uve_spa`, etc. The old Bayesian path (`bayesian_utils.py`) has more methods (`spa`, `uve`, `uve_spa`, `ipls`, `cars`, `vcpa-iriv`) but they're all cached per `(preprocess, method)` across trials, compounding leakage.

### Refinement path is N/A — it reuses saved wavelengths
`_run_refined_model_thread()` in the GUI (line 35273) parses wavelengths from the saved result row's `all_vars` field or from `_original_wavelength_order`. It does NOT re-run any varsel algorithm. So the refinement path inherits whatever leakage the search path had, but doesn't introduce new leakage.

### `bayesian_utils.py` hardcodes `random_state=42` inside varsel
Lines 442, 455, 469, 488, 502, 520 all use `random_state = 42` instead of threading through the user's random_state. Grid search correctly threads the seed. This is a reproducibility bug, not leakage.

### Region selection (`create_region_subsets`) is y-using
`regions.py:170` calls `compute_region_correlations(X, y, wavelengths)` which uses `pearsonr` per-region with y. Region selection is used in both Bayesian paths but not in grid search. Marked LEAKY in the audit.

### One-class grid varsel methods are a subset of regression
`implemented_oc_varsel` at `search.py:5299-5302` lists 11 methods. Notably absent: `ipls`, `ipls_forward`, `ipls_backward`, `mc_sipls`, `mwpls`, `fipls_spa`, `fipls_cars`. These interval-based methods are PLS-specific and not applicable to one-class models.

---

## 2026-04-29 (post-Codex T-19 review) — imbalance design corrections

Codex did a focused mechanical-verification pass on T-19 and the original loss-reweighting design doc. Three substantive findings worth preserving for future sessions:

### imblearn Pipeline does NOT propagate fit_params through resampling

Originally the loss-reweighting design proposed "for multiclass boosting and MLP, route `sample_weight=compute_sample_weight('balanced', y_train)` via `Pipeline.fit(..., classifier__sample_weight=w)`." Codex verified this is broken as designed:
- imblearn's `Pipeline._fit()` updates only X and y in `_fit_resample_one()` (`imblearn/pipeline.py:435-437`, helper `pipeline.py:1330-1334`).
- `fit_params` are passed through unchanged to the final estimator.
- If `sample_weight` was computed from pre-resampling y and the resampler changes y size, the sample_weight array length mismatches the resampled y at fit time.

**The correct design** is a custom thin estimator wrapper that intercepts `fit(X, y)` *inside* the Pipeline and computes `sample_weight = compute_sample_weight('balanced', y)` from the y RECEIVED (post-resampling). Wrapper then calls `inner.fit(X, y, sample_weight=sample_weight)`. This bypasses the imblearn fit_params problem and is automatically resampling-aware.

For sklearn (`LogisticRegression`, `RandomForestClassifier`, `SVC`), LightGBM, and CatBoost, the native `class_weight='balanced'` / `auto_class_weights='Balanced'` kwargs are already lazy at fit and don't need a wrapper. Only XGBoost (binary `scale_pos_weight` is constructor-frozen; multiclass needs sample_weight) and MLP need the wrapper.

### Active length-mismatch bug at search.py:3883 (now T-32)

While verifying T-19, Codex found: `search.py:3869-3873` computes `sample_weight_train` from pre-resampling `y_train`, then `:3883` passes it to `final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)` against `y_train` rather than `y_train_for_model` (the resampled y). When a resampler is active, lengths mismatch — this is a live bug today, not a hypothetical risk.

### PLS-DA inner LR is ~10 sites, not 3

Codex audit found ~10 PLS-DA inner LR construction sites in `src/`. Five are missing `class_weight` handling and need fixing:
- `search.py:380` (rebuild-from-row)
- `bayesian_utils.py:400` (importance/full-fit utility)
- `nsga2_search.py:822` (`_build_model()` PLS-DA branch)
- `nsga2_search.py:3453` (calibration metrics conversion)
- `ga_preprocessing.py:428` (GA preprocessing fitness)

Five are already correct: `search.py:4261`, `unified_bayesian.py:1220`, `nsga2_search.py:1424`, `nsga2_search.py:3078`, `code_generator.py:880-884` (exported code).

The originally-cited 3-site claim (search.py:380, :4261, bayesian_utils.py:400) was incomplete. The auxiliary sites (rebuild-from-row, importance utility) aren't the main CV but they still produce results the user sees, so silent inconsistency is still wrong.

### MLP sample_weight needs sklearn ≥1.7 — bundle compatibility unknown

`MLPClassifier.fit()` accepts `sample_weight` only as of sklearn 1.7 (`sklearn/neural_network/_multilayer_perceptron.py:842-845`). Current `pyproject.toml:29` floor is `>=1.5.0`. Bumping to `>=1.7.0` could break the PyInstaller bundle — needs testing before committing. If incompatible, skip MLP from the imbalance dropdown rather than block T-19; MLP is among the least-used models in the user's workflow.

---

## 2026-04-29 (latest) — Reconciled roadmap from three independent reviews

### Scope
User asked for a critical analysis of `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` + six docs in `docs/analysis_vs_unscrambler/`, with pushback on items that are wrong / not useful / don't belong. Then asked for codex independent review, then DeepSeek V4 Pro fresh-eyes review (briefed with prior conclusions + chemometrics-domain rule). Final deliverable: `docs/RECONCILED_ROADMAP_2026-04-29.md`.

### Key findings worth preserving (so future sessions don't re-derive)

**1. The "audit nobody did" meta-finding (DeepSeek).** All seven analysis docs say variants of "audit per-fold variable selection for leakage" but none ever performed the audit. T-01 in the roadmap IS that audit. CARS / UVE / VIP / SPA / iPLS may fit on full calibration data and pass a frozen variable list into CV — would produce optimistic CV estimates. DeepSeek points to `search.py:5400-5460` as a likely leakage site, but the audit needs to actually trace each varsel × each search path before fixes can be designed.

**2. Three false alarms that should not be re-flagged:**
- LESSER_FINDINGS 1.1 "binary specificity bug" — false alarm. `scoring.py:344-348` is correct under standard sklearn convention (positive class at index 1: TN=cm[0,0], FP=cm[0,1]). The reviewer flipped row/column indexing.
- FTIR Gap Analysis #10 "PLS-DA inner LR stuck at class_weight=None" — wrong. `search.py:4257-4261` and `unified_bayesian.py:1220-1222` apply balanced class weights when `imbalance_method='class_weight'`. Loss-reweighting design doc at `docs/plans/2026-04-29-model-native-loss-reweighting.md` needs scope reduction: only XGB/LightGBM/CatBoost need new wiring.
- VALIDATION_SUPPLEMENT "509 except blocks" stat — unreproducible / scope-inflated when re-counted. CODE_QUALITY also self-contradicts (asserts "no path traversal risk" while flagging Zip Slip in same doc).

**3. HMAC signing of `.dasp` is technically insufficient.** If the HMAC key ships inside the bundled app, an attacker can extract it and sign malicious models. Better fix is safe `zipfile.extractall()` (validate entry names against `..` and absolute paths) + UX warning at load time. See T-25.

**4. Chemometrics community convention ≠ ML convention.** Per-spectrum operations (SNV, SG derivatives, baseline correction) are NOT leakage even when computed on full data before CV. They use only within-spectrum statistics. Cross-sample operations (PCA fit on full data, variable selection using y, hyperparameter tuning on full data) ARE real leakage. Naive ML reviewers/agents will misclassify the first as bugs. Saved to auto-memory at `feedback_chemometrics_conventions.md`.

**5. PLS-2 framing correction (Codex).** sklearn's `PLSRegression` is multi-output capable. The gap is the workflow assuming 1D y everywhere (`np.ravel()` at `search.py:695`, results DataFrame schema, plot generation, code export). Real ticket is "multi-Y workflow," not "PLS-2 model card." User confirmed multi-Y prediction is highly useful → T-17 elevated to P1, 2-3 weeks effort.

**6. Multi-class SIMCA is a research-design question, not a coding question.** dasp's "PCA-SIMCA" is one-class anomaly detection (DD-SIMCA). True multi-class SIMCA is class-modeling — each test specimen tested against each class's PCA model independently, can output "none of the above" or "multiple of the above." Different statistical model from PLS-DA / RF / XGB which are discriminant classifiers. Tracked as T-31 PENDING in the roadmap — needs user decision on whether unknowns / continuum samples are real for their bone-FTIR work.

### Three-reviewer methodology worked
Claude / Codex / DeepSeek V4 Pro are deliberately family-orthogonal (Anthropic / OpenAI / Chinese-trained). Each independently caught issues the others missed; Codex caught the PLS-DA inner LR claim, DeepSeek caught the meta-finding plus three under-prioritized leakage issues. When all three converged, confidence is high. When they disagreed, the strongest argument won (e.g., DeepSeek's elevation of ensemble OOF leakage over my mid-priority ranking — DeepSeek's reasoning that academic publication CV scores must be honest is correct).

---

## 2026-04-29 (evening) — Validation supplement to Unscrambler analysis

### Scope
User asked to orchestrate additional agents for deep analysis. Since the 8-agent Unscrambler analysis was already completed earlier today, I dispatched 4 validation agents to spot-check findings and hunt for gaps: (1) commercial blocker verification, (2) quick-wins audit, (3) missed Unscrambler features check, (4) security/compliance audit.

### Output
- **`docs/analysis_vs_unscrambler/VALIDATION_SUPPLEMENT.md`** — new document with corrections, new findings, and quick-wins matrix.
- **`docs/analysis_vs_unscrambler/MASTER_ANALYSIS.md`** — updated with corrected severity figures and new blockers.
- **`docs/PROJECT_STATUS.md`** — updated to reference 5 documents (added supplement).

### Key corrections to original analysis

**1. Error swallowing is 3.9x worse than reported.**
Original: "88+ backend, 42 GUI". Verified: **198 backend, 311 GUI = 509 total**. The GUI count alone exceeds the original total claim.

**2. Pickle attack surface is larger.**
Original analysis flagged `model_io.py:432-481`. Also found: GUI does `pickle.load()` at lines 41606, 44070, 49038; `library_search.py:542` uses `np.load(..., allow_pickle=True)`.

**3. New security issues found.**
- **Zip Slip**: `.dasp` extraction uses `zf.extractall()` without path validation (`model_io.py:416-417`).
- **ReDoS**: Data Management filter compiles user regex without timeout (`data_management.py:637`).
- **Tkinter thread-safety violations**: Worker threads access GUI widgets without `root.after()` mediation — nondeterministic crash risk.

**4. Multi-class SIMCA clarification.**
Original analysis presented "PCA-SIMCA" as a competitive advantage. **This is misleading.** Unscrambler's SIMCA is multi-class class modeling. SP's `PCA-SIMCA` is one-class anomaly detection only (DD-SIMCA). They share a name but serve different purposes.

**5. 12 additional Unscrambler features were missed in original comparison.**
Including: clustering (HCA/k-means), 3D score plots, batch trajectory/MSPC, ASCA, DoE integration, LWR, model compression, PLS-PM.

**6. Version string chaos.**
Three different versions in same product: `0.5.0b1` (backend), `v0.4.0` (report.py), `3.9.0` (code_generator.py).

### Bottom line
The original 8-agent analysis was directionally excellent but understated severity. Revised commercial readiness score: **3/10** (down from 4/10). See VALIDATION_SUPPLEMENT.md for full details.

---

## 2026-04-29 (late evening) — Lesser findings audit (3 agents)

### Scope
User asked: "that document only includes the severe findings? what about lesser ones that are still worth dealing with?" Dispatched 3 parallel agents: (1) code hygiene + performance + robustness, (2) UX/QA polish, (3) silent correctness bugs.

### Output
- **`docs/analysis_vs_unscrambler/LESSER_FINDINGS.md`** — comprehensive catalog of ~200+ non-blocker issues across 5 categories.

### Key discoveries

**11 silent correctness bugs** (most dangerous — wrong results users trust):
1. Binary specificity formula computes recall of negative class instead of TNR (`scoring.py:346`). **HIGH.**
2. VIP uses `np.var(y)` instead of per-component explained variance (`models.py:1738`).
3. SPA deterministic despite `n_random_starts` — `random_state` unused (`variable_selection.py:407`).
4. Ensemble OOF predictions use preprocessor fitted on full dataset — data leakage inflates R² ~0.01-0.03.
5. Preprocessing discovery computes importances on full data before CV — optimistically biased.
6. PDS window arithmetic fails for even window sizes — buffer overflow (`calibration_transfer.py:193-221`).
7. PLS grid can exceed CV fold training samples — no `n_samples` clamp (`models.py:842-843`).
8. One-class UVE prefilter trained on binary labels including outliers — contradicts one-class philosophy.
9. CARS tree-mode weight update biases toward unselected variables — oscillation instead of convergence.
10. SNV exact equality guard misses near-zero std — numerical artifacts.
11. Scoring metrics silently return 0.0 on failure — bare except blocks.

**~150+ UX polish issues** including: 209 "specimen" vs "sample" inconsistencies, 9 "check console" messages in windowed app, 161 generic `showerror("Error",...)`, 184 "Please..." messages, zero progress bars on Analysis tab, 15+ uncentered dialogs, 3 arrow styles for Run buttons, inconsistent CSV exports, no keyboard shortcuts, no help system, emoji breaking screen readers.

**36 code hygiene issues**: commented-out Instrument Lab code blocks, 20+ public APIs without docstrings, missing `__all__` in 7 modules, dozens of `open()` without encoding, `models.py` imports `logging` but never uses it.

**8 performance issues**: triple `.copy()` in `search.py`, `iterrows()` in `ensemble.py` and `io.py`, per-sample `interp1d` in calibration transfer, per-sample baseline loops.

**14 robustness/edge cases**: `pd.read_csv` without encoding detection, division by zero in VIP for constant y, SNV silent skip of constant spectra, `min_wavelengths=100` rejecting valid datasets, unvalidated `simpledialog` inputs.

### Bottom line
Top 12 blockers prevent commercial release. These ~200+ lesser findings determine whether early adopters say "promising" or "amateur." The 4-week roadmap in LESSER_FINDINGS.md (~40 hours) would eliminate silent wrong-result bugs, prevent international data crashes, and remove the "student project" impression.

---

## 2026-04-29 (later, continued) — Quick-wins audit across core modules

### Scope
User asked for additional high-impact, low-effort quick wins beyond the documented Unscrambler analysis gaps. Audited `spectral_predict_gui_optimized.py`, `src/spectral_predict/search.py`, `unified_bayesian.py`, `report.py`, `models.py`, and supporting modules.

### Output
- **`docs/QUICK_WINS.md`** — prioritized P0/P1/P2/P3 matrix with 14 items, exact line numbers, estimated fix times, and commercial-impact ratings.

### Key findings (new, not in prior analysis docs)

**1. `search.py` has 8 debug prints that bypass the logger entirely.**
- `[PLS-DA DEBUG]` at line 3991; `[DEBUG]` at lines 2313, 2789, 2834, 3091, 3093, 3098, 3100.
- These are not `logger.debug()` — they are bare `print()` calls. In the PyInstaller bundle (only distribution path), stdout is invisible to the user but any console window or redirected log captures them, making the app look unfinished.

**2. `calibration_transfer.py` has ~66 prints and no logger.**
- Lines 755+ start with `print("=== CTAI Debug Information ===")` and flood the console.
- No `import logging` in the file. Adding a logger and downgrading all prints to `logger.debug()` is a 15-minute fix with immediate professional polish.

**3. `nsga2_search.py` has 42 prints and 29 bare `except Exception:` blocks.**
- Same pattern: no logger, broad exception swallowing. The prints are progress messages that should be `logger.info`; the bare excepts hide real bugs.

**4. `models.py` imports `logging` (line 3) but never instantiates a logger.**
- The module has zero `logger = logging.getLogger(__name__)` usage. Any diagnostic intent in this 1,800-line module is either a print (elsewhere) or silent.
- **VIP formula bug confirmed still present at line ~1738.** Uses `np.var(y)` as denominator approximation instead of per-component explained variance. This was flagged in the Unscrambler analysis but is worth repeating because it directly affects variable-selection correctness and has a 15-minute fix.

**5. `interference.py` has a duplicate `EPO` class — placeholder + real.**
- Lines 564-600: stub class with `raise NotImplementedError("TO BE IMPLEMENTED IN PHASE 2")`.
- Lines 820+: real implementation.
- The placeholder is dead code that confuses anyone grepping for EPO.

**6. `code_generator.py` hardcodes Python 3.9.0 at line 373.**
- `PYTHON_VERSION = '3.9.0'` is baked into generated scripts regardless of the runtime Python. Generated scripts should either use `sys.version_info` or omit the claim.

**7. `export_bundle.py` has a placeholder GitHub URL at line 244.**
- `https://github.com/yourusername/spectral-predict` appears in exported citation bundles. Looks unprofessional.

**8. Version string is hardcoded in 3 places with mismatching semantics.**
- `src/spectral_predict/__init__.py:3`: `__version__ = '0.5.0b1'`
- `src/spectral_predict/model_io.py:48`: same string, duplicated
- `spectral_predict_gui_optimized.py:2541`: title says "Spectral Predict v3 (Beta)"
- `templates/header.py`: claims "Spectral Predict v3"
- The backend says 0.5.0b1, the GUI says v3, and the template says v3. A single source of truth + dynamic interpolation would resolve this in 10 minutes.

**9. `baseline.py:139` vs `preprocess.py:482` ALS default `p` mismatch.**
- `BaselineALS` class default: `p=0.001`
- `apply_baseline_als` pipeline builder default: `p=0.01`
- 10× difference. Users get different behavior depending on whether they call the class directly or go through the preprocessing pipeline. 0.001 is more standard in chemometrics literature.

**10. `unified_bayesian.py` has verbose trial progress prints (lines 1748-1773, 1882, 1911-1943) that bypass its own logger.**
- A logger IS imported at the top of the file, but the Bayesian progress loop uses bare `print(f"Trial {trial.number}...")`.
- These prints are especially problematic because Bayesian runs can have hundreds of trials, generating hundreds of lines of stdout.

**11. Missing `__all__` in ~10 public modules.**
- `models.py`, `search.py`, `ensemble.py`, `scoring.py`, `baseline.py`, `preprocess.py`, `variable_selection.py`, `contamination.py`, `io.py`, `calibration_transfer.py`.
- No `__all__` means `from module import *` pulls in dependencies and internals. Low commercial impact but cheap to fix (20 min total).

---

## 2026-04-29 (later) — FTIR Bone PLS paper gap analysis + loss-reweighting design

### Scope
User asked for an analysis of the Sponheimer FTIR Bone PLS paper (in prep, working dir `C:\Users\mspon\Desktop\_DeskSync\FTIR Bone PLS\`) and what among its analyses dasp cannot currently reproduce. Analysis spans the v5/v7 manuscript, supplementary tables, and the `paper/scripts/analysis/03q_bootstrap_inference.py` inferential pipeline.

### Outputs
- **`docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md`** — structured 10-item confirmed-missing list + 5-item partially-supported list + already-supported coverage + ranked priority recommendations. Mirrors the `docs/analysis_vs_unscrambler/` convention.
- **`docs/plans/2026-04-29-model-native-loss-reweighting.md`** — design proposal for the imbalance-handling gap (item #10 in the gap analysis). Per-model "Imbalance handling" dropdown with four options including Auto-mode that reuses `detect_class_imbalance(y, threshold=3.0)` from `imbalance.py:61`.
- **PROJECT_STATUS.md** updated with prominent reference to the gap analysis at the top of the file + new Follow-Up entry for the loss-reweighting design doc.

### Key non-obvious discoveries from this analysis (worth preserving)

**1. dasp's `ElasticNet` model card is the regression flavor, not the classifier.** Initially listed it as "ENLR supported" — wrong. `models.py:140` and `:407` instantiate `sklearn.linear_model.ElasticNet` (squared-error loss). The classifier `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=...)` is constructed inside `nsga2_search.py:856` but is not exposed as a standalone model card. The only direct LogisticRegression entry point for classification is the post-PLS LR inside the PLS-DA pipeline, which uses default L2. So ENLR is genuinely missing as a user-pickable classifier — not just nominally present.

**2. Boosting models receive zero imbalance kwargs in dasp.** Verified by grep: `scale_pos_weight`, `is_unbalance`, and `auto_class_weights` appear NOWHERE in `src/`. The imbalance system's `class_weight='balanced'` injection is gated by `code_generator.py:866` to only `RandomForest`, `LogisticRegression`, and `SVC`. XGBoost/LightGBM/CatBoost classifier construction in `models.py` (lines 285, 307, 327) never receives any imbalance-aware kwarg. Users wanting to reproduce the paper's "XGBoost (scale_pos_weight)" or "LightGBM (balanced)" configs cannot do so today without source edits or routing through SMOTE-style resampling (which is a different statistical intervention).

**3. The "nothing vs balanced" misconception.** When constructing the design doc, realized this is the most counterintuitive default in scikit-learn / XGBoost / LightGBM / CatBoost. Default `class_weight=None` (or omitted `scale_pos_weight`) does NOT mean "balanced data assumed" — it means *all samples receive equal weight in the loss function*. With 100 positives and 10 negatives, the loss is dominated 10:1 by positives. The model learns the majority class well and the minority poorly. This is the opposite of what most users intuitively expect from the default. The design doc calls this out with recommended GUI labels (`Equal sample weights (no correction)` instead of `none`) so the distinction is impossible to misread.

**4. Auto-mode for loss reweighting has a per-fold-correctness subtlety.** sklearn's `class_weight='balanced'` and LightGBM's `class_weight='balanced'` compute weights at fit time on whatever training subset is passed in — automatically per-fold-correct in CV. But XGBoost's `scale_pos_weight` is a constructor argument and *cannot* be made per-fold-aware unless you wrap the estimator. The design doc resolves this by recommending Auto mode route through `class_weight`/`sample_weight` kwargs only, side-stepping `scale_pos_weight` for auto and reserving it for the explicit-custom case.

**5. dasp DOES have a peak-ratio calculator with Bone FTIR presets.** `src/spectral_predict/peak_calculator.py` ships built-in presets for Bone FTIR, Collagen & Tissue, etc. with local linear baseline correction at user-specified troughs. Initially assumed the Pal Chowdhury 10-index recipe (IRSF, C/P, Am/P, OrgInorg, etc.) wasn't reproducible in dasp — wrong, it is. Only the surrounding statistical infrastructure (LOGO CV, bootstrap CIs, permutation tests on the index-baseline-vs-multivariate gap) is missing.

**6. The paper's `paired_permutation` and site-level `block_bootstrap` are the load-bearing inferential machinery.** Without them the paper has two MCC numbers and no defensible claim that one is better than the other. dasp has neither. This is gap #2 and #3 in `GAP_ANALYSIS.md` and is probably the highest-leverage *new* statistical infrastructure dasp could add — no other chemometrics tool surfaces these in a usable way (could be a differentiator).

### Pointers for next session
- If implementing any of these gaps, start with `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` for the priority-ranked roadmap.
- The loss-reweighting design (`docs/plans/2026-04-29-model-native-loss-reweighting.md`) is the most ready-to-implement of the items — backend changes are localized to `models.py`, `imbalance.py`, `code_generator.py`, and the relevant Bayesian-search sites; default preserves all existing behavior.
- LOGO CV (gap #1) is the highest-leverage gap but also the largest implementation surface — touches `cv_utils.py`, `search.py`, the cost estimator, the GUI Analysis tab + Model Development, AND the metadata-as-group-column concept (currently only the unmerged Analysis Subset V1 branch reads metadata categoricals).

---

## 2026-04-29 — Deep Analysis vs. CAMO Unscrambler

### Scope
8 parallel specialized agents analyzed all subsystems of Spectral Predict against CAMO Unscrambler (industry-standard chemometrics software). Full reports in `docs/analysis_vs_unscrambler/`.

### Key Discoveries

**Missing fundamentals (would stop a commercial demo):**
- PCR (Principal Component Regression) completely absent from codebase -- zero grep hits. Every spectroscopist expects this.
- PLS-2 (multi-Y) not supported -- search pipeline assumes 1D y (`search.py:4086+`).
- EMSC (Extended Multiplicative Scatter Correction) completely absent -- no file references found. Gold standard for NIR scatter correction.
- Report generation is Markdown-only (146 lines in `report.py`) -- no PDF, no figures, no regulatory formatting.

**Confirmed bugs:**
- VIP formula at `models.py:1738` uses `np.var(y)` approximation instead of PLS y-loadings. Canonical Wold (2001) formula uses `y_loadings_`.
- UVE docstring at `variable_selection.py:44` says cutoff_multiplier > 1.0 is "more conservative" but actually > 1.0 eliminates more variables (more aggressive). Code at line 160 confirms.
- SPA at `variable_selection.py:407` is deterministic despite `n_random_starts` parameter.
- Jackknife prediction intervals fully implemented at `diagnostics.py:143-230` but **never called** from prediction workflow or GUI.
- ALS default `p` inconsistency: class default 0.001 (`baseline.py:139`) vs pipeline builder 0.01 (`preprocess.py:483`).
- Duplicate EPO class: stub at `interference.py:565` overwritten by real implementation at `interference.py:820`.
- CTAI docstring claims unpaired samples but code requires paired (`calibration_transfer.py:653 vs 803`).

**Architecture concerns:**
- `spectral_predict_gui_optimized.py` is 57,116 lines in a single class -- unmaintainable, no MVC, no GUI tests.
- 3% test coverage on core modules (`search.py` 5,516 lines at 3%, `models.py` at 2%).
- `.dasp` files load 7+ pickle files with zero integrity checks -- arbitrary code execution risk.
- 1,214+ `print()` calls vs ~100 `logger.*` calls in backend. Bayesian optimization output invisible in windowed build.

**Genuine advantages over Unscrambler:**
- 19 variable selection methods (Unscrambler has 3-4)
- 6 ensemble types including stacking, MoE, regional specialists
- 6 calibration transfer methods (3 beyond industry standard)
- 5 one-class contamination models (no equivalent in Unscrambler)
- Code export to Python/Jupyter/R with embedded data
- 4-zone applicability domain with per-sample reliability scores
- Batch multi-model prediction with consensus

### Verdict
Analytical engine exceeds Unscrambler in 9/17 categories. Commercial release blocked by presentation-layer gaps (missing PCR/PLS-2, no PDF reports, GUI architecture, security). Fixable in 3-6 months of focused effort. See `docs/analysis_vs_unscrambler/MASTER_ANALYSIS.md` for phased release plan.

---

## 2026-04-21 (late evening) — Data viewer duplicate-target fix + RepeatedKFold complexity curve

### Discovery: Data Management viewer rendered the target column twice under combined-CSV load

**Symptom:** With a combined-format CSV (metadata + spectra in one file), the Data Management viewer showed the target column in two places: once in the metadata section and once at the dedicated target position. Switching the target dropdown made both instances update to the new column name (still two copies). CSV/Excel exports also produced duplicate columns (or `ValueError: cannot insert, already exists` if pandas enforced uniqueness). Worst: `_apply_data_viewer_edits` silently removed the target from `combined_metadata_df` after any cell edit (because both target-named headers matched the target branch in its header scan and neither was captured as metadata), which broke the target-switch dropdown until the user reloaded the file.

**Root cause:** Low-level readers (`read_combined_csv`/`read_combined_excel` in `src/spectral_predict/io.py`) correctly strip the target from `metadata_df`. But the GUI loader intentionally re-adds it at `spectral_predict_gui_optimized.py:17985-17990` so `_get_available_target_columns` can list it in the target-switch dropdown. Four render/export paths then iterated `self.combined_metadata_df.columns` AND appended `[target_col]` separately — producing the duplicate. The separate-ref branch had the correct `if col == target_col: continue` guard but combined did not.

**Investigation approach:** Dispatched Codex and Kimi K2.6 in parallel (via opencode Go's Moonshot integration, NOT OpenRouter) for independent deep investigations. Both agreed on the fix shape. Codex's report was slightly more complete — caught two older export helpers Kimi missed (`_export_to_csv` / `_export_to_excel` at ~17830, ~17870) and recommended an explicit target-preservation guard in `_apply_data_viewer_edits` rather than relying on the "accidental heal" behavior.

**Fix (4 view/export projections + 1 state guard):**
1. `_populate_data_viewer` combined branch (~30287): `metadata_cols = [c for c in self.combined_metadata_df.columns if c != target_col]` — mirrors the separate-ref branch at ~30295.
2. `_export_data_viewer_to_csv` combined branch (~30491): `if col == target_col: continue`.
3. `_export_to_csv` combined branch (~17838): same skip.
4. `_export_to_excel` combined branch (~17880): same skip.
5. `_apply_data_viewer_edits` (~30629-30640): after rebuilding `combined_metadata_df` from metadata-indexed headers, re-attach `self.y` under `target_col_name` so target-switching still works. Guarded by `target_idx is not None and self.y is not None and target_col_name`.

**Workflow (worth repeating):** worktree → detailed plan in `docs/plans/2026-04-21-duplicate-target-col-fix.md` → GLM-5.1 (z.ai direct subscription) implemented via opencode `build` agent → Codex + Claude both verified PASS independently → user manual verification of Valley CSV → `--no-ff` merge via `cb0aa8b`.

**Lesson:** When `self.combined_metadata_df` is re-inflated with data that logically belongs somewhere else (like `self.y` stored separately), every consumer of `combined_metadata_df.columns` becomes a potential duplicate-render site. Search for all `combined_metadata_df.columns` uses when you add any such re-injection. Better yet: don't re-inject; use a dedicated "target_name" attribute for the dropdown instead of repurposing the metadata frame. That's a wider refactor and out of scope for this fix.

### Discovery: `cross_val_predict only works for partitions` on RepeatedKFold

**Symptom:** User ran Bayesian refinement with RepeatedKFold and the Model Complexity Analysis plot was empty. Console showed:

```
Error computing validation curve for PLS: cross_val_predict only works for partitions
```

The exception was caught but swallowed — no user-facing message.

**Root cause:** `sklearn.model_selection.cross_val_predict` requires a partition CV (each sample tested exactly once). Under RepeatedKFold / RepeatedStratifiedKFold each sample is tested multiple times (once per repeat), so sklearn refuses. LOO and plain KFold are partitions and work fine. The offending call is `compute_pls_complexity_curve` at `src/spectral_predict/diagnostics.py:309`. Other validation-curve functions in the same module use `cv.split()` manually and are unaffected.

**Fix:** `compute_pls_complexity_curve` now catches the specific ValueError and falls back to a manual per-fold aggregation that averages the repeat predictions per sample. Verified on n=40/p=30 synthetic data with KFold(5), RepeatedKFold(5,3), LOO — all three now produce non-NaN curves. Also surfaced the failure in the Results text via `_log_progress` so users see a note instead of staring at an empty plot. `9bd8751`.

**Lesson:** Any `cross_val_predict` call path must handle non-partition splitters explicitly. Grep for `cross_val_predict` across the codebase when adding a new CV strategy — if the result isn't a single pooled prediction, `validation_curve` or `cross_validate` (per-fold) is the correct alternative.

---

## 2026-04-21 (evening follow-up) — Task-type helper reverted + CV tooltip overhaul

### Discovery: `type_of_target`-based helper is too aggressive on integer-valued numeric targets
**Symptom:** User loaded a CSV with target `group.cov.heath` (3 integer-valued numeric values). Analysis tab auto-checked PLS-DA. On Run, crashed with `ValueError: No valid models found. Available: ['PLS', 'Ridge', 'ElasticNet', 'RandomForest', 'LightGBM'], Requested: ['PLS-DA']` from `run_search` at `src/spectral_predict/search.py:1192`.

**Root cause (two-part):**
1. Earlier-today commit `0ef91e5` switched `_infer_task_type_from_y` from the Fix-B1 heuristic (`nunique()==2` OR non-numeric → classification, else regression) to `sklearn.utils.multiclass.type_of_target`. The theory: float-typed numeric columns would be called `'continuous'` and only true int-typed multiclass would be `'multiclass'`. **Not true:** `type_of_target` uses `_is_integral_float` on float arrays, so any integer-valued float array — burn temperatures `{150.0, 250.0, ..., 850.0}`, proportion columns `{0.0, 1.0, 2.0}` — is called `'multiclass'`. This pushed the `41371e0`/Fix-B1 intent (numeric-with-few-values defaults to regression) back into the broken state it was meant to fix.
2. `0ef91e5` centralized 6 sites through the helper but missed three: `_run_analysis_thread` (the crash site), `_update_task_type_label` (display label), and the imbalance detection thread at `~22278`. These still used the Fix-B1 heuristic inline. So the GUI model-picker path (helper → classification → PLS-DA checked) and the run path (inline heuristic → regression) disagreed, producing the "No valid models found" crash.

**Evidence:** `tests/test_refine_task_type_preservation.py` already had 2 failing tests against HEAD (`test_no_task_fallback_regression_few_unique_numeric`, `test_empty_task_falls_through`) that encoded Fix-B1's intent — they were written against commit `41371e0` and broke silently at `0ef91e5`/`10d97a6`. Codex review (`019db36b`) recommended reverting the helper rather than keeping the minimal 3-site consistency fix.

**Fix:**
- Revert `_infer_task_type_from_y` to Fix-B1 semantics: `nunique()==2` OR non-numeric dtype → `'classification'`, else `'regression'`. Doc comment updated to explain the reversion.
- Route the 3 stale sites through the now-restored helper for consistency.
- Result: integer-valued numeric targets default to regression; user overrides to classification via the radio button. Matches user's stated preference. True string/categorical columns still auto-detect classification.
- Defense-in-depth: `build_cv_splitter(..., y=y)` and the Model Development `type_of_target` preflight (`aa73edd`) still catch any remaining `classification + continuous y` via user error or stale config.

**Lesson:** `type_of_target`'s `'multiclass'` kind is too inclusive for regression-vs-classification disambiguation on numeric chemometric data — it triggers on any integer-valued array regardless of dtype, which is the common regression-target shape in spectroscopy (concentrations rounded to integers, discrete temperatures, burn degrees). The heuristic must be more conservative: treat numeric as regression unless it's binary or explicitly non-numeric. Let the user's radio button cover the integer-multiclass case.

### Tooltip: CV strategy guidance overhaul
**Problem:** Same tooltip text ("K-Fold standard / Repeated lower variance / LOO best for n<30") duplicated inline at three call sites. Lacked citations. Inline hint (`n>100 K-Fold | 30<n<100 Repeated | n<30 LOO`) capped Repeated K-Fold usefulness at 100, which is too low for an app whose core use-case is *ranking* many models — repeated CV stabilizes ranks up to ~200 samples.

**Fix:** Added `TOOLTIP_CONTENT['cross_validation']['strategy_detail']` as a single shared tooltip string. Replaced all 3 inline duplicates with the shared key (Analysis-tab label + combobox, Model Development combobox). Inline hint updated to `n<30 LOO | 30–200 Repeated | n>200 K-Fold (hover for details)`. Detailed tooltip cites Kohavi (1995) for 10-fold default, Krstajic et al. (2014) for repeated-K-Fold benefit on model selection, Hastie et al. for LOOCV high-variance mechanism (training sets overlap by n−2), and notes the chemometrics-specific caution that LOOCV RMSEP reads optimistically on NIR spectra due to collinearity/overlap. Cross-checked against Wikipedia CV article, scikit-learn user guide, Raschka's evaluation survey (arXiv 1811.12808), and Ezenarro et al. 2025 NIR systematic review. The user's originally-pasted table (specific n-bands like 30–80, 80–200) is preserved as the sample-size guide section, labeled explicitly as a heuristic.

---

## 2026-04-21 — Imbalance-handling + Task-type + Cost-estimate session (long)

### Discovery: Prediction-tab "Export to CSV" said "please run prediction first" even after prediction ran
**Root cause:** Duplicate method name `_export_predictions` — one at `spectral_predict_gui_optimized.py:40771` (Prediction tab, checks `self.predictions_df`), the second at `:45984` (Calibration Transfer workflow, checks `self.ct_pred_y_pred`). Python class resolution uses the LAST definition, so the second method silently shadowed the first. The Prediction tab's button was calling the CT method, which checks a variable the normal prediction path never sets. **Lesson:** When adding methods to long class files, grep for the method name to check for shadowing — this codebase is too big to spot visually. Fix: `8d05dc0` renamed the CT one to `_export_ct_workflow_predictions`.

### Discovery: Substitution banner set, then immediately wiped
**Root cause:** Initial banner-ordering bug (fixed by `ac43289`): `_refresh_imbalance_methods` called `_set_imbalance_banner` then `_update_imbalance_method_description`, which internally calls `_clear_imbalance_banner`. Then a DIFFERENT ordering bug (fixed by `8b2cd74`): `_detect_and_display_imbalance` internally calls `_refresh_imbalance_methods` a second time — once `_on_task_type_changed` set the banner for a substitution, the second refresh found the method valid (already substituted) and hit the else-branch `_clear_imbalance_banner`. Both bugs: banner silently wiped in same call stack. **Lesson:** When two code paths both refresh the same UI element, the second call can inadvertently wipe state the first just set. The fix is not to have the else-branch auto-clear; let banner persist until user-driven acknowledgment (manual method pick or disable toggle).

### Discovery: StratifiedKFold on continuous y in refinement (Codex's diagnosis)
**Symptom:** User ran Bayesian regression, loaded result into Model Development, clicked Run → `ValueError: Supported target types are: ('binary', 'multiclass'). Got 'continuous' instead.` Traceback showed `_run_refined_model_thread` line 36790: `_final = pipe.steps[-1][1]`.

**Codex's diagnosis (`docs/plans/2026-04-21-refine-task-type-root-cause.md`):** The crash traceback line number matched the PRE-`aa73edd` file exactly. In current HEAD, that same statement has moved to `:36816`. So the user's crash came from a **stale Python process running old code** — their GUI was launched before the CV-guard commits landed. No code path in current HEAD could plausibly reset `refine_task_type` to `'classification'` for a regression result. **Lesson:** When investigating reported traceback line numbers, cross-check against the current source — a moved line is strong evidence the user's environment is stale. Always suggest a GUI restart as first diagnostic step.

**Secondary finding:** The CV guard (now at `~36119`) ran AFTER model creation. If a stale `task_type='classification'` slipped through, the estimator was built as a classifier before the guard corrected for CV. Fix `b9d2b39` moved the preflight BEFORE model creation AND also prefers `selected_model_config['Task']` over the radio value (radio can drift between load and run).

### Discovery: Integer-encoded multiclass targets lost classification detection
**Root cause:** Commit `41371e0` (earlier today) dropped the `or self.y.nunique() < 10` clause from auto-detect at 9 sites to fix a Burned-temperature regression dataset that was being mis-classified. Side effect: integer-encoded classification targets (e.g. `collagencat` with values `{1, 2, 3}`) are no longer auto-detected — they're numeric with `nunique > 2`, so everything falls through to regression. User reported "select categorical variable → stays regression." **Lesson:** Fixing an over-eager heuristic at all sites is tempting but risks under-eager detection. The better tool is `sklearn.utils.multiclass.type_of_target` — distinguishes `'multiclass'` (integer or integer-valued float, discrete) from `'continuous'` (floats spanning a range). Fix `0ef91e5` added `_infer_task_type_from_y` helper using `type_of_target` and replaced 6 sites. Note: temperature data with 8 unique integer values still gets flagged as `'multiclass'` by sklearn, so `41371e0`'s Fix A (refinement honors saved `Task`) is what actually protects that workflow, not the initial auto-detect.

### Discovery: Import-tab Task Type radio silently overrides variable-driven intent
**Symptom:** User had radio pinned to "Regression" on the Import tab, then picked a categorical target at Configuration → Basic Settings. Models stayed as regression because `_on_task_type_changed` reads the radio and respects the explicit "Regression" choice. **Lesson:** Target variable selection is a fundamental-intent signal — the user is saying "I want to predict THIS" — and should override any sticky radio state from a previous task. One-class is the exception: a categorical column could be either classification or one-class (user must pick). Fix `10d97a6` in `_on_target_column_changed` runs `type_of_target` on the new y and sets `self.task_type` explicitly, except when current radio is `one_class`.

### Discovery: Bayesian cost estimator inflated 10× by phantom preprocessing dimension
**Symptom:** User saw "Very High Compute Cost" warning for a Bayesian LOO run that finished in 15 seconds. **Root cause:** `estimate_total_cv_fits` was called with `n_preprocessing=10` unconditionally. Grid search DOES have 10 preprocessing configs as an outer loop dimension. Bayesian samples preprocessing as a hyperparameter INSIDE each trial, so effective `n_preprocessing=1`. The 10× inflator pushed 43×100×1×1 = 4,300 fits up to 43,000, crossing the "High Compute Cost" (>10k) and nearly the "Very High" (>50k) thresholds. Fix `0f4a1e6`. **Also:** the 10k/50k fit thresholds were calibrated for slow fits on large datasets. On small spectral data (~50 samples, PLS/Ridge), each fit is milliseconds — 20-40k fits finishes in under a minute. Bumped to 100k/500k in `0da5bd4`.

---

## 2026-04-21 — Phantom rare-class drop + Refine 91-vs-94 mismatch (label normalization)

**Bug:** User had a 94-row classification target column where every cell visually showed `250`. Analysis tab's rare-class auto-drop (from commit `51c50cc`) flagged 3 of them as a different class with count < n_folds and dropped those rows — model trained on 91 samples. Then loading that model into Model Development showed a training-mismatch warning: "trained with 91, current has 94", because `_run_refined_model_thread` doesn't replicate the auto-drop.

**Root cause:** Earlier-today's mixed-type coercion (commit `382b7e6`) uses `astype(str)`, which preserves type-distinct representations of the same number. If 3 cells were stored as `str("250")` (Text-formatted Excel cell, or apostrophe prefix, or trailing whitespace — all invisible in Excel) while others are `int(250)`, `.astype(str)` produces `"250"` for both forms visually but `np.unique` still sees them as distinct classes. The 3 minority-form rows then get auto-dropped as a phantom rare class.

`_run_refined_model_thread` at `gui:35271-35285` also `astype(str)`-coerces but lacks the auto-drop (per `51c50cc`'s own scope note), so Refine sees all 94 rows. `_validate_training_configuration` compares saved 91 vs current 94 → mismatch warning.

**Fix:** Replace `astype(str)` with a new `_normalize_mixed_type_labels()` helper at all 12 mixed-type coercion sites added by `382b7e6`. The helper collapses numerically-equivalent values to canonical strings:
- `250` (int), `250.0` (float), `"250"` (str), `"250 "` (whitespace), `"2.5e2"` (scientific) → all `"250"`
- `250.5` (non-integer float), `"grass"` → unchanged
- `np.nan` → preserved (NOT `"nan"` string)
- Python 3's `str.strip()` handles non-breaking space `\xa0` and other unicode whitespace

Implementation pattern:
```python
def _norm(v):
    if pd.isna(v): return v
    if isinstance(v, str): v = v.strip()
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return str(v)
```

Accepts `pd.Series`, `np.ndarray`, or `list`; returns same container type. Module-level helper in `spectral_predict_gui_optimized.py`; duplicated in `src/spectral_predict/search.py` to avoid circular import (acceptable — 8 lines).

**Enhanced diagnostic log** at the primary Analysis-tab site: prints per-type row counts AND the specific "collapsed labels" list so next time this comes up the user sees e.g. `Target column has mixed Python types: {'int': 91, 'str': 3}` and `Collapsed labels: ['250']` (the str form that got merged with the int form).

**Both reported bugs resolve with this single change** — once all `250`-forms normalize to one class, the auto-drop never fires, and the Refine mismatch never triggers. No need to replicate the auto-drop in Refine (that was Option B, rejected because it would lose 3 real specimens for a spurious reason).

**Tests:** 9 unit tests for the helper (int/float/str collapse, scientific notation, whitespace, numpy-array input, genuine-strings-unchanged, bool collapse with int, empty string, NaN preservation) + 1 integration test (phantom class not dropped from 94-row synthetic dataset). 17/17 total pass.

**Known edge cases not covered by this helper** (documented for future iteration if needed):
- Apostrophe-prefixed Excel cells (`'250` → `"'250"` — stays distinct). Would need explicit `if s.startswith("'"): s = s[1:]`.
- Fullwidth Unicode digits (`２５０`). Would need `unicodedata.normalize('NFKC', s)`.
- Zero-width / Em-space / other non-standard Unicode whitespace. Python's default `.strip()` catches most but not all.

If this fix doesn't resolve the user's specific case, enhance the helper with NFKC + apostrophe stripping. Most likely cause based on the user's report ("exactly 250 no decimals, all look identical") is plain Text-formatted Excel cells (int vs str) or trailing ASCII whitespace — both handled.

**Plan:** `docs/plans/2026-04-21-phantom-class-normalize-FINAL.md`.

Cherry-picked from `glm/label-normalize` worktree as `b754354` + `9d55c8d` (dropped two transient opencode-draft snapshot commits).

---

## 2026-04-21 — Mixed-type target coercion (Validation-Wide fix for 3 sibling bugs)

**Bug cluster:** When classification target column is object-dtype with heterogeneous Python types (e.g. `[1, 2, "3", "4", NaN]` from mixed Excel cells), three independent call sites crashed with `TypeError: '<' not supported between instances of 'str' and 'int/float'`:

1. Bayesian validation metrics — `gui:26849` `np.unique(y_val_np)` during post-Bayesian metrics computation.
2. Stratified validation-set creation — `gui:19609` `StratifiedShuffleSplit` inside `train_test_split(stratify=...)`.
3. SPXY validation-set creation — `LabelEncoder.fit_transform` on raw categorical y.

**Root cause:** Commit `2a1c77c` coerced the **training** target `y_filtered` at `_run_analysis_thread`:25986-26001, but `self.validation_y` and the `_create_validation_set` path had no coercion. Each downstream consumer that sorts or encodes the target re-discovers the crash.

User confirmed manually resetting validation doesn't fix bug #1 — so bug is reproduced **within a single classification run**, not carried over from a prior regression run.

**Why the narrow fix isn't enough (Codex review):** Narrow (only Bayesian site) leaves bugs #2 and #3 broken. Centralized (mutate `self.y` / `self.validation_y` at assignment time) has a blocking NaN regression — `astype(str)` converts `np.nan` to `"nan"`, which silently breaks `self.y.isna()` at `gui:16487` plus downstream NaN filters at `search.py:419/3070/3701` and `gui:26836/27060/37101`.

**Fix (Validation-Wide):** Coerce on **local copies** strictly **after** NaN filter at every site that sorts/encodes validation labels. Never mutate `self.y` or `self.validation_y`.

Sites patched:
- `_validation_stratified` (`gui:19609`) — replaced stale `len(y.unique()) < 10` heuristic with `y.nunique() == 2` (commit `41371e0` had missed this site), plus coercion on stratify vector.
- `_validation_spxy` (`gui:19510`) — coerce before `LabelEncoder.fit_transform`.
- `_create_validation_set` class distribution (`gui:19760`) — coerce before `y_val_set.unique()` / `y_train_set.unique()`.
- Bayesian validation (`gui:26876`) — coerce after NaN filter at 26836, before `np.unique`.
- NSGA-II validation (`gui:27108`) — coerce before `label_encoder.transform`.
- Refine validation (`gui:37155`) — coerce after NaN filter at 37101.
- `compute_validation_metrics_for_top_models` (`search.py:446`) — coerce after NaN filter for classification/one_class.
- Prediction display paths (`gui:40298, 40514, 40684, 40702`) — coerce before `np.unique` / confusion matrix / scatter plot.
- `_on_task_type_changed` (`gui:16306`) — warn in progress log when regression ↔ classification boundary is crossed with a validation set loaded. Does NOT auto-clear (holdout indices still valid; Codex recommendation).

**Tests:** `tests/test_mixed_type_target_coercion.py` — 7 tests:
- Stratified split with `pd.Series([1, "1", 2, "2", 1, "2"], dtype=object)` — no TypeError
- SPXY split with same mixed-type Series — no TypeError
- `compute_validation_metrics_for_top_models` with mixed `y_val` — returns DataFrame with `val_Accuracy`
- NaN drop before coercion — asserts `"nan"` string never appears in post-processing
- Task-type change warning fires on regression → classification with validation loaded
- Task-type change does NOT warn on same-category transition
- Task-type change does NOT warn when no validation set

151/151 tests pass (7 new + 144 regression across contamination/cv_strategy/refine_task_type_preservation).

**Known follow-ups (deferred):**
- `search.py:975` and `search.py:3232` — `LabelEncoder.fit_transform(y_np)` in training-path (not validation). Would crash for direct library callers with mixed-type targets. Out of scope for this fix.
- NaN preservation is explicit only at sites with pre-existing NaN filters (Bayesian, NSGA-II, Refine). At `_validation_spxy`, `_validation_stratified`, `_create_validation_set` class distribution, and prediction display paths, NaN could theoretically reach the coercion and become `"nan"`. In practice these sites are fed from `_create_validation_set` which drops NaN upstream, or would have failed pre-fix on NaN anyway (LabelEncoder rejects NaN). Risk is cosmetic (spurious `"nan"` class in stratification/encoding), not a silent correctness bug.

**Plans:**
- Kimi's diagnosis: `docs/plans/2026-04-21-mixed-type-target-kimi-plan.md`
- Codex review: `docs/plans/2026-04-21-codex-review.md`
- Final consolidated plan: `docs/plans/2026-04-21-mixed-type-target-FINAL.md`

Merged from `glm/mixed-type-target-validation-wide` via `--no-ff`.

---

## 2026-04-21 — PLS regression hotfix: refine tab silently runs PLS-DA

**Bug:** Loading a PLS regression result row into Model Development (Tab 7 "Refine") and clicking Run executes PLS-DA (classification) instead of PLS regression. The downstream cascade `_on_refine_task_type_changed` (line ~16493) replaces PLS with PLS-DA when the task type flips to classification.

**Root cause:** `_load_model_for_refinement` (introduced by commit `057d9f6` on 2026-04-11) only honored `config.get('Task')` when it equaled `'one_class'`. For saved `'regression'` or `'classification'` results, it silently discarded the saved task and re-ran an `nunique() < 10` auto-detect heuristic on `self.y`. Temperature data with discrete step values (e.g. 100/200/300/400/500/600/700/800°C = 8 unique) tripped the heuristic, flipping the task to classification, which triggered the PLS → PLS-DA cascade.

**Secondary bug (same session):** The same `nunique() < 10` heuristic at 8 other auto-detect sites misclassified numeric y targets with few unique values (≤9) as classification. This caused the Analysis tab auto-detect to show "classification" for clearly numeric temperature data.

**Fix A (required hotfix):** `_load_model_for_refinement` now trusts the saved Task for all three values (`regression`, `classification`, `one_class`). Only falls back to auto-detect when Task is missing/empty. Extracted a pure helper `_detect_refine_task_type(config, y)` at module level for testability; `_load_model_for_refinement` calls it instead of inline logic.

**Fix B1 (recommended, applied at all 9 sites):** Dropped `or self.y.nunique() < 10` from the auto-detect heuristic. Only `nunique() == 2` (binary) and `not is_numeric_dtype(...)` (categorical/text) remain as classification triggers. Sites: lines ~16174, ~16230, ~16354, ~18325, ~19626, ~22113, ~22834, ~24325, plus the refine fallback (in the helper). Integer-coded multi-class labels (e.g. {0,1,2,3}) no longer auto-detect as classification — users must explicitly pick classification or store labels as strings.

**Tests:** `tests/test_refine_task_type_preservation.py` — 12 tests in `TestDetectRefineTaskType` covering: saved regression/classification/one_class with inlier labels, no-task fallback, binary detection, non-numeric detection, empty task, and the exact Burned-temperature scenario. 376/376 broader suite pass.

**Caveat:** Line ~19626 (`y_available` validation-path heuristic) was not in the original plan's 8-site list but was fixed for consistency — it uses the same flawed heuristic.

---

## 2026-04-21 — Fix: `np.unique()` crash on mixed str/NaN categorical targets

**Bug:** Loading a combined Excel file with a categorical string target column (e.g., `Habitat` = 'Upland_Tundra'/'Valley') containing NaN values caused `TypeError: '<' not supported between instances of 'str' and 'float'`. This crashed the GUI during import when task_type was set to one_class.

**Root cause:** `np.unique()` sorts its input to find unique values. When `self.y` has object dtype with both strings and float NaN, Python 3 cannot compare them with `<`. This is triggered by `drop_na_y=False` during import (intentionally keeps NaN rows for prediction).

**Fix pattern:** Replace `np.unique(self.y.values)` with `self.y.dropna().unique()` or `self.y.dropna().value_counts()` — pandas `.unique()` is hash-based (no sorting), and `.dropna()` removes NaN before any comparison. Consistent with 3 existing safe sites at lines 3945, 20575, 32633 that already use `pd.notna()` filtering.

**Sites fixed (6 total):**
1. GUI line 16297 — inlier class combo population (`_update_one_class_controls_visibility`)
2. GUI line 22740 — auto-detect inlier class in run analysis (added empty-result guard)
3. GUI line 21080 — y distribution plot fallback in quality check tab
4. GUI line 29734 — task type heuristic for ensemble/export
5. GUI line 3992 — scatter plot coloring (`_apply_color_to_scatter`)
6. `outlier_detection.py` line 419 — `check_y_data_consistency()` categorical path

**Tests added:** 2 new tests in `test_outlier_detection.py::TestYDataConsistency`:
- `test_categorical_with_nan_values` — mixed str/NaN Series
- `test_categorical_all_nan` — all-NaN Series (float64 by pandas inference)

**Gotcha:** `pd.Series([np.nan, np.nan, np.nan])` gets dtype float64 (not object), so all-NaN takes the numeric path in `check_y_data_consistency`, not categorical. The test accounts for this.

**Peer review note:** DeepSeek V3.2 raised concern about empty results after `dropna()`. Fix 2 (line 22740) now has an explicit guard: `if len(y_clean) == 0: show error, return`.

---

## 2026-04-21 — One-class model support in export system (code_generator + templates)

**What:** Added `task_type='one_class'` as a third branch throughout the code export system (6 files). Previously, exporting a one-class model (IsolationForest, OCSVM, etc.) crashed with `NameError: name 'IsolationForest' is not defined`.

**Files changed:**
- `src/spectral_predict/templates/models.py` — ONE_CLASS_MODELS set, ONE_CLASS_NEEDS_SCALING set, PCASIMCA_CLASS_TEMPLATE, one-class MODEL_IMPORTS/MODEL_TEMPLATES/DEFAULT_PARAMS entries
- `src/spectral_predict/templates/header.py` — DATA_LOADING_ONE_CLASS_TEMPLATE (y_oc + inlier/outlier indices)
- `src/spectral_predict/templates/validation.py` — CROSS_VALIDATION_ONE_CLASS_TEMPLATE (mirrors contamination.py:run_one_class_cv exactly — inlier-only KFold, majority vote under repeated CV, mean-of-fold AUCs), METRICS_ONE_CLASS_TEMPLATE, FINAL_MODEL_ONE_CLASS_TEMPLATE, PREDICTION_ONE_CLASS_TEMPLATE, get_final_model_template(), get_prediction_template()
- `src/spectral_predict/templates/visualization.py` — ONE_CLASS_SCORE_DISTRIBUTION_TEMPLATE, ONE_CLASS_CONFUSION_TEMPLATE, updated get_visualization_code()
- `src/spectral_predict/code_generator.py` — one-class branches in __init__, _render_header, _get_imports_code, _generate_embedded_data_section, _render_data_loading, _render_model (_render_one_class_model), _render_cross_validation, _render_final_model, _resolve_model_ctor_class, _resolve_model_class_name, _resolve_default_param_key
- `spectral_predict_gui_optimized.py` — one-class metrics dict + inlier_class_label in model_config

**Gotchas found:**
- PCASIMCA and LocalOutlierFactor do NOT accept `random_state` parameter. The `_render_one_class_model` method must exclude these from random_state injection alongside OneClassSVM.
- One-class models use manual StandardScaler (not sklearn Pipeline) — scaling is handled inline in the CV/final-model templates, NOT via `_needs_standard_scaler()`.
- The CV template uses `{x_var}` and `{model_name}` format variables that must be substituted at generation time (not replaced post-hoc like regression/classification's X_final).

**Verified:** All 5 one-class models (IsolationForest, OneClassSVM, EllipticEnvelope, LOF, PCA-SIMCA) generate syntactically valid scripts that execute correctly end-to-end. 33/33 export tests pass. 61/61 contamination tests pass. No regressions in regression/classification export paths.

---

## 2026-04-19 — PLS-DA classification wavelength importance fix

**Bug:** After running a PLS-DA classification model in Model Development, no wavelength importance figure was displayed. Regression PLS and one-class models showed it correctly. Non-PLS-DA classification (LightGBM, RandomForest, etc.) also worked.

**Root cause (two interacting bugs):**

1. `get_feature_importances()` (`models.py:1802-1808`): When `self.refined_model` is the full PLS-DA pipeline `[imbalance?, pls, scaler, lr]`, the Pipeline unwrapping logic checked for `'model'` in `named_steps` (not found), then took `steps[-1][1]` (LogisticRegression). It then called `compute_vip(LogisticRegression, X, y)` which failed silently because LR lacks `x_weights_`/`x_scores_`.

2. PLS-DA preprocessor extraction (`gui:37086-37094`): `pipe_steps[:-2]` removed scaler+lr but kept the PLS step and any imbalance step. When `_plot_wavelength_importance()` applied this preprocessor, it collapsed data from wavelength space (hundreds of features) to PLS score space (n_components), causing dimension mismatch with wavelengths.

**Fix:**
- `models.py:1805`: Added `elif model_name == "PLS-DA" and "pls" in model.named_steps: model = model.named_steps["pls"]` before the generic fallback.
- `gui:37087`: Replaced `pipe_steps[:-2]` with set-based filtering that excludes `{'pls', 'scaler', 'lr', 'imbalance'}` — only true spectral preprocessing steps survive.
- `gui:33071-33080`: Removed the old wavelength-importance mismatch heuristics (trim wavelengths / fall back to index positions). The figure now only renders when `len(importances) == len(refined_wavelengths)`. A mismatch is treated as a feature-space bug, not something to guess around.

**Failed approach (same session):** An intermediate attempt built `Pipeline(_preproc_steps)` from bare transformer objects (`[s for n, s in pipe_steps ...]`) instead of `(name, transformer)` tuples. sklearn accepts that constructor but `fit()` fails with `TypeError: cannot unpack non-iterable <Transformer> object`. The fix must preserve the `(name, step)` tuples.

**Architecture note:** The PLS-DA pipeline structure `[pls, scaler, lr]` is unique among classification models. Other models either have `'model'` as a named step (tree models, SVM, MLP) or have a single step. Only PLS-DA uses multi-step pipelines without a `'model'` step.

**Contract clarified:** Wavelength-importance figures must obey a strict one-to-one mapping: only wavelengths actually used by the fitted model may be shown, and the importance values must come directly from that same fitted model/feature space. Any count mismatch means the code mixed feature spaces.

**Tests:** `tests/test_plsda_importance.py` — 5 parametrized tests covering: pipeline extraction, VIP parity with direct compute, feature-space dimensionality, bare PLSTransformer, and `'model'`-step pipeline.

---

## 2026-04-19 — LOO cancellation UI bug fix

**Bug:** When user cancels at the LOO / high-compute-cost warning dialog in `_run_analysis()`, the code correctly avoids starting the analysis thread (all 5 `return` paths were correct), but the UI was left in a misleading "running" state. The Progress tab was already switched to, progress labels said "Analysis in progress...", the running-figure animation was spinning, and only the pause/resume/stop buttons got reset to idle. This made it look like the analysis was running despite the cancellation.

**Root cause:** `_run_analysis()` sets up all the "running" UI state (tab switch at 22702, progress labels at 22706-22708, animation at 22717, buttons at 22731) *before* the cost-warning dialogs fire (22794+). The cancellation `return` paths only called `_update_search_buttons('idle')` which only touches the 3 control buttons — nothing else.

**Fix:** Added `_cancel_search_ui(reason)` helper that also stops the running-figure animation, resets progress_status to "Ready", shows the cancellation reason in progress_info, and clears best_model_info / time_estimate. All 5 cancellation points in `_run_analysis()` now call this instead of bare `_update_search_buttons('idle')`.

**Files changed:** `spectral_predict_gui_optimized.py` (lines ~22619-22637 new helper, 5 call sites updated).

---

## 2026-04-19 — Auto Bone FTIR index extraction (main branch)

**Feature:** Added fixed-method Auto Bone FTIR mode to Explore-tab Peak Calculator. Produces 10 diagenesis indices (IRSF, PO4/CO3/AmideI positions & intensities, Am_P, C_P, OrgInorg) via Kontopoulos et al. 2018 method.

**Architecture decisions:**
- NOT modeled as a normal PeakPreset. Auto mode returns 10 outputs per sample vs 1 scalar per sample for manual presets. Dedicated backend functions `extract_bone_ftir_indices()` and `extract_bone_ftir_indices_all_samples()` keep the two paths cleanly separated.
- The fixed method's baseline regions differ from existing Bone FTIR manual presets (e.g., IRSF baseline 400-420 to 630-670 vs existing 490-510 to 690-750). A dedicated implementation was necessary rather than reusing existing helpers.
- GUI mode toggle added at top of dialog with Manual / Auto Bone FTIR radio buttons. Manual-only sections (Presets, Local Baseline, Expression Builder) are pack_forget'd in auto mode and repacked when switching back.
- Auto mode now reads stored spectra directly (`self.X.values` + the normal scope mask) rather than using `_get_peak_calc_data("raw", scope)`, because that helper still routes through the legacy transform-capable Explore raw path.
- Validates x-unit (must be cm-1), data type (must be absorbance), and wavelength coverage (400-1720 cm-1) before calculation.
- Copy in auto mode copies the full CSV table; in manual mode copies the stats text (preserving existing behavior).

**Gotchas:**
- The Amide I baseline uses fixed anchors at exactly 1590 and 1720 cm-1 (not trough search), per the external spec. When AmideI_intensity is NaN (no detectable peak above baseline), the derived ratios Am_P and OrgInorg become NaN (NaN/positive = NaN), not 0.0. This matches the reference implementation.
- Tkinter `pack(before=...)` is needed to reinsert manual frames in the correct position when switching back from auto mode. The `content` frame (scroll_frame) must be passed as `in_=` parameter.

**Bugs found and fixed in review pass:**
- **Peak C state regression**: `_load_preset()` else-branch (no peak_c) was hiding widgets but not resetting `dialog._peak_c_visible = False`. This meant `_peak_calc_build_preset()` could still include a stale hidden Peak C in manual calculations. Fixed by adding the flag reset at `spectral_predict_gui_optimized.py:9057`.
- **Plot clicks leak into auto mode**: `_on_peak_calc_plot_click()` had no guard for auto mode, so clicking the Explore plot while in auto mode silently mutated hidden manual A/B/C fields. Fixed by adding an early return when `_calc_mode_var == "Auto Bone FTIR"`.
- **Stale markers on mode switch**: Switching to auto mode left manual plot markers visible. Fixed by clearing `_found_positions` and calling `_remove_peak_markers()` in both `_peak_calc_mode_changed()` (auto branch) and `_peak_calc_run_auto()`.
- **Auto mode raw-data blocker**: Review found that `_peak_calc_run_auto()` was asking for `_get_peak_calc_data("raw", scope)`, but that helper's raw branch still calls `_apply_transformation(self.X.values)`. Fixed by adding `_get_peak_calc_scope_mask()` and having auto mode filter `self.X.values` directly so the fixed-method workflow never goes through the transform-capable Explore raw path.
- **Single-sample override UX**: Added sample-name scope entries to the Peak Calculator and a `Sample:` dropdown next to Explore Sets. The simplest clean behavior was to treat a selected sample as an explicit set-of-one override: selecting a sample clears the active set selection, disables assign mode, and reuses the existing Exclude / Keep Only actions against that single index. This avoids parallel set+sample state ambiguity.
- **Reset path for override**: The Explore `Sample:` dropdown now includes an explicit `All Samples` entry as the default/reset state. Without this, once a single sample was chosen there was no clear UI path back to the non-override state.
- **Explore single-sample override root cause**: The first implementation only updated UI state (`_current_sample_var`, labels, Exclude / Keep Only target resolution). It did not affect the actual Explore plots because `_create_explore_plot_in_frame()` still iterated over every row in `data`, and the sample-selection handlers did not trigger plot regeneration. The correct fix is in the shared Explore plot renderer so the override applies consistently to raw, derivative, and baseline subtabs.

**Files changed:**
- `src/spectral_predict/peak_calculator.py` — added ~130 lines: backend functions + constants
- `spectral_predict_gui_optimized.py` — mode toggle, auto calculate, auto export/copy, bug fixes
- `tests/test_bone_ftir_auto.py` — 24 tests (17 backend + 7 GUI-path)

**Tests:** 24/24 new, 117/117 existing peak-calc/bone = 141/141 total. Reference comparison: all 10 indices match ftir_indices.py exactly.

---

## 2026-04-19 — Analysis Subset V1 implementation (branch glm/analysis-subset-v1)

**Feature:** Added metadata-driven Analysis Subset feature. Users can restrict analysis, validation, and refinement to a metadata-defined cohort (e.g., "only grasses"). Implemented on the existing `active_group_filter` / `active_indices` internal runtime path with user-facing rename to "Analysis Subset".

**Architecture decisions:**
- Pure matching/summary logic extracted into `src/spectral_predict/analysis_subset.py` (C1 requirement). GUI holds only thin glue. 41 unit tests cover matching, formatting, categorical detection, training metadata, and one-class guardrails — no Tk required.
- `_compute_active_group_matches` and `_format_active_group_condition` now delegate to the pure module.
- New `_refresh_active_group_indices()` helper recomputes `active_indices` from `active_group_filter` (C2: clears filter if column is missing from metadata).
- Subset provenance (`analysis_subset_*` keys) added to `last_training_config` in both one-class and regression/classification storage sites via `_get_analysis_subset_training_metadata()` helper (C3: `n_samples=None` when inactive).
- `_validate_training_configuration` extended with subset mismatch warnings.
- One-class guardrail (`check_one_class_inlier_guard`) blocks analysis/refinement when zero inlier samples remain after subset filtering.
- Dialog now metadata-only (excludes target column), with categorical multi-select via Tk Listbox for columns with <=20 unique non-null values. Supports "in" condition stored as `{"column": ..., "condition": "in", "values": [...]}`.
- `contains` uses `regex=False` (literal substring matching).

**Gotcha:** `_compute_active_group_matches` receives column name + separate args (not a filter dict), but `compute_matches` takes a DataFrame + filter dict. The GUI wrapper builds a temporary filter dict and calls `compute_matches(series.to_frame(), filter_def)`. This works because `compute_matches` matches on `df[col]`, which is the same Series regardless of whether the DataFrame has one column or many.

**Gotcha (C2):** The `_revert_data_viewer` method previously cached `active_indices` in the snapshot and restored the cached set. Now it only restores `active_group_filter` and calls `_refresh_active_group_indices()` to recompute indices from the filter. This handles the case where the snapshot's metadata state differs from current state (e.g., user deleted a column, then reverted — the column is back so the filter is valid again).

---

## 2026-04-19 — `self.inlier_class_label` shadowed: StringVar clobbered by tooltip Label

**Bug:** Attempting to run any one-class analysis after commit `fd376b4` (2026-04-17 tooltip PR) raised `AttributeError: 'Label' object has no attribute 'get'` at `spectral_predict_gui_optimized.py:_run_analysis` when the code reached `self.inlier_class_label.get().strip()`. GUI progress tab would say "Analysis in progress" but never produce output.

**Root cause:** The tooltip PR reassigned `self.inlier_class_label` from the `tk.StringVar` created in `__init__` (line 2886) to a `ttk.Label` widget, purely so `CreateToolTip(self.inlier_class_label, ...)` could attach. That one line silently clobbered the StringVar used by 18+ call sites across the file (`.get()`, `.set()`, model save/load metadata, uncertainty display, external-validation config, etc.). The bug was latent because no one ran a one-class analysis between 2026-04-17 and 2026-04-19. First triggered by testing the Tab 4C OC config feature.

**Secondary consequence:** The Combobox at line 6172 was passing `textvariable=self.inlier_class_label` — but at that point `self.inlier_class_label` was a Label widget, not a StringVar. Tkinter silently accepted the Label's widget-path as a Tcl variable name, so the combobox *looked* fine in the UI, but the Python-side StringVar stayed empty. Anyone who thought inlier-class selection was working was actually hitting the auto-detect fallback ("no inlier class specified → auto-detect most frequent class") every time.

**Fix (`59124ff`):** Renamed the Label to a local `inlier_class_label_widget`, attached the tooltip to the local, restored `self.inlier_class_label` as the StringVar from `__init__`. Three-line change. Landed on main as a standalone commit before the OC config feature merge so it can be cherry-picked to any in-flight branches that predate the OC work.

**Lesson:** Reusing `self.X` as both a widget attribute AND a StringVar/BooleanVar is easy to miss at review time because the line that clobbers reads like a normal widget creation. Any future tooltip-retrofit work in this GUI should grep for the target attribute name in `__init__` first. There are ~150 `tk.StringVar` / `tk.BooleanVar` assignments in `__init__` (lines 2822-3120) — every one is a potential shadow target.

---

## 2026-04-19 — Relative imports inside `spectral_predict_gui_optimized.py` (not a package module)

**Bug:** The OC config parity work (`8e1538d`, `afdeca1`) added five `from .contamination import get_one_class_model_grids` calls inside the GUI collectors (`_collect_ocsvm_overrides`, `_collect_if_overrides`, etc.). These would raise `ImportError: attempted relative import with no known parent package` at runtime whenever a user customized any Tab 4C one-class card and clicked Run.

**Root cause:** `spectral_predict_gui_optimized.py` is a top-level script, not a module inside the `spectral_predict` package. Every other import in that file uses absolute `from spectral_predict.contamination import ...` (confirmed at lines 26082, 32469, 35257, 35430). GLM defaulted to the relative form without checking the surrounding file's convention.

**Symptom visibility:** The bug only fires when a card is customized — default (untouched) cards never call the collector, so the curated-grid path was unaffected. 60/60 tests still passed because the test suite hits `_resolve_one_class_model_grids()` directly without going through the GUI collectors.

**Fix (`3641c78`):** Replaced all 5 occurrences with the absolute form. Caught by GLM-5.1's adversarial review of its own diff — blocking issue flagged as "MERGE AFTER MINOR FIXES".

---

## 2026-04-19 — One-class model config parity: per-model grid overrides with curated-default preservation

**Design:** The core invariant is that untouched model cards map back to the curated grids from `get_one_class_model_grids()`. The GUI compares current widget state against shipped defaults; only models whose state differs get a user-defined Cartesian product grid. This avoids silently expanding the default search space.

**Backend (`search.py`):** Added `_resolve_one_class_model_grids()` private helper that centralizes grid resolution. It handles three paths: (1) no overrides → curated grids, (2) legacy `oc_hyperparams` → curated grids with flat parameter overrides, (3) per-model `oc_model_param_overrides` → Cartesian product for customized models, curated defaults for the rest. Per-model builder functions (`_build_ocsvm_custom_grid`, etc.) handle model-specific logic like pruning `degree` from non-`poly` OneClassSVM combos.

**GUI (`spectral_predict_gui_optimized.py`):** Five collapsible cards in Tab 4C using the old-style `BooleanVar` + `ttk.Checkbutton` + `ttk.Entry` pattern (same as Ridge/RF cards). Widget vars added around line 2893. Cards constructed in a dedicated `oc_model_config_container` frame, shown/hidden via `_update_one_class_controls_visibility`. The `_collect_one_class_model_param_overrides()` method reads widget state, compares against defaults, and only emits overrides for customized models.

**Gotcha:** The plan explicitly forbids using `_create_parameter_grid_control()` — it has a wiring bug. The old-style direct BooleanVar/Checkbutton pattern is used instead.

**Backward compatibility:** The `oc_hyperparams` parameter is still accepted by `run_one_class_search()`. The GUI now passes `oc_hyperparams=None` and uses `oc_model_param_overrides` instead. Legacy callers unaffected.

**Tests:** 6 new tests in `TestOneClassModelConfigParity` covering default preservation, per-model override isolation, OCSVM degree pruning, empty-row fallback, and legacy `oc_hyperparams` compatibility. All 60/60 `test_contamination_detection.py` tests pass.

---

## 2026-04-17 — PyInstaller 3.12 bundle: pandas/util/__init__.py intermittently corrupted by TOC collision

**Bug:** 3.12 bundle launched with `--test` crashed with `ImportError: cannot import name 'capitalize_first_letter' from 'pandas.util'`. `capitalize_first_letter` is a real function in pandas 2.3.x — so the bundled `pandas/util/__init__.py` is the wrong file.

**Root cause:** PyInstaller's dist-info collection has a duplicate-key race where `pandas/util/__init__.py` gets overwritten with the contents of `packaging/_structures.py` (the pypa/packaging vendored copy that setuptools aliases as `setuptools._vendor.packaging`). Happens roughly 1 in 3 rebuilds — non-deterministic based on file system iteration order and PyInstaller TOC resolution. Earlier opencode/GLM rebuild also hit this class of bug when they attempted `a.binaries`/`a.datas` filtering inside the spec; reverting that didn't fully eliminate the underlying collision risk.

**Symptom check:** bundled `dist/SpectralPredict-py312/_internal/pandas/util/__init__.py` starts with `"# Vendored from https://github.com/pypa/packaging/blob/main/packaging/_structures.py"` instead of the real pandas content.

**Fix:** Post-build self-healing step in `build_installer_py312.py` after the COLLECT phase — byte-compare `pandas/util/__init__.py` in the bundle against `.venv312/Lib/site-packages/pandas/util/__init__.py`; if mismatched, `shutil.copy2` the venv version over the bundled one. Logs `[REPAIR] Restored pandas/util/__init__.py` when triggered. If other files start exhibiting the same corruption pattern, add them to the `repair_targets` list at `build_installer_py312.py:~225`.

**Upstream:** This is PyInstaller 6.18.0 behavior — no obvious open issue. The clean fix would be for PyInstaller to detect package-name collisions at TOC-insertion time, but the workaround is cheap.

---

## 2026-04-17 — PyInstaller 3.12 bundle: loky spawn crash → threading fallback revert

**Bug:** The Python 3.12 PyInstaller windowed bundle crashed when LightGBM (or any joblib/loky parallel code) tried to spawn workers. Child process ran `multiprocessing.freeze_support()` inside the frozen runtime hook and crashed on argv parsing: `ValueError: not enough values to unpack (expected 2, got 1)`. The parent kept retrying spawn → fork-bomb of GUI windows.

**Root cause:** The version gate `is_frozen and sys.version_info < (3, 12)` in `_frozen_needs_threading_fallback()` was incorrect. Loky's spawn method is broken in ALL PyInstaller windowed bundles, not just 3.11. The frozen runtime hook's `multiprocessing.freeze_support()` crashes on argv parsing regardless of Python version.

**Fix (two files):**
1. `src/spectral_predict/search.py`: Reverted `_frozen_needs_threading_fallback()` to return `is_frozen` (any frozen build → threading backend). Updated docstring.
2. `spectral_predict_gui_optimized.py`: Wrapped bare `multiprocessing.freeze_support()` in `if __name__ == "__main__": try/except` so child-process argv parsing failures exit silently instead of cascading into GUI windows.

**Result:** 3.12 bundle rebuilt, smoke test 31/31 imports OK, bundle size ~1.4 GB (after torch removal).

---

## 2026-04-17 — PR #4 (CV strategies) merged after pre-merge review surfaced a real export-path bug

**Context:** PR #4 (`claude/cv-strategy-overhaul`) had been through 5 prior review rounds. After PR #5 (LightGBM clone fix) merged, this branch was brought up to date by merging `main` back in (commit `058e110`) — clean except for two doc files (PROJECT_STATUS.md and SESSION_LOG.md) where conflicts were resolved by taking main's "fixed" status note + keeping both 2026-04-15 and 2026-04-16 SESSION_LOG entries chronologically.

**Pre-merge dual review (codex + GLM-5.1 via direct z.ai subscription):**
- GLM approved outright. Verified all 4 PR #5 clone sites intact at the merged offsets (`search.py:2214`, `:4185` PLS-DA, `:4210`, `:4212`), confirmed merge integrity, validated LOO/RepeatedKFold mechanics (early-stopping guard, single-class fold guard, BER consistency, AUC mean-of-folds vs pooled distinction), found only minor non-blocking nits (dead `KFold` import in `contamination.py:42`, LOO O(n²) memory edge case, duplicate import).
- Codex requested changes — found a real blocker GLM missed: `code_generator.py:108-110` only read `cv_strategy`/`cv_n_repeats` from top-level config keys, but `search.py:4656-4665` stores the canonical CV metadata under `training_config`. Codex verified directly: `CodeGenerator(model_config={'training_config': {'cv_strategy': 'loo'}}, options)` produced `cv_strategy='kfold'` (the default fallback). Result: any caller passing a search-results dict directly to `CodeGenerator` would get the wrong CV regime in exported scripts/notebooks. The current GUI export path at `gui:36412` happens to dodge this because it builds model_config explicitly with top-level keys from `self.refined_config`, but the library API was incorrect. Fix in commit `ac19363`: 3-line training_config fallback in the constructor. Verified safe with three test cases (training_config-only, legacy top-level, default fallback) — all pass.

**Reviewer-finding tally on PR #4 across the day:**
| Reviewer | Verdict | Notes |
|---|---|---|
| codex (gpt-5.4) | request changes → resolved | Caught the export-path bug + 4 non-blocking nits |
| GLM-5.1 (direct z.ai sub) | approve | Thorough merge integrity check + correctness review |

**Merged via merge-commit `6357aeb` (24 cv-strategy commits + merge from main + export fix). Cleanup: local worktree `.worktrees/cv-strategy-overhaul/` removed, branch `claude/cv-strategy-overhaul` deleted both locally and on remote. Stale local untracked `tests/test_cv_strategy.py` (dated 2026-04-14, missing the `TestPostMergeReviewFixes` class added later in the branch) was deleted before pulling — was a leftover from an earlier session.**

**Known follow-ups deferred to subsequent PRs** (see `docs/PROJECT_STATUS.md` "Known follow-ups" section): `run_bayesian_search()` missing preflight `validate_cv_strategy_for_task()` call (one-line fix at `search.py:3096`); minor style nits in `cv_utils.py` / `templates/validation.py` / `_majority_vote`; LOO O(n²) memory note for very large datasets; sklearn 1.7.2 UserWarning flood suppression on `.venv312`.

---

## 2026-04-15 — PR #4 RMSEcv pooling is the only numeric drift vs main

**Context:** Overnight parity validation between `main` (`fa39504`) and `claude/cv-strategy-overhaul` (`c091c93`). Ran PLS/PLS-DA/LightGBM × regression/classification with `folds=5`, plain K-Fold, `preprocessing_methods={'raw': True}`.

**Finding:** 3 of 4 combos bit-identical on 34+ numeric keys. Both regression combos differ *only* in `RMSEcv` (and derived `RPD`, `RER`). `R2cv`, `MAEcv`, `Bias`, `CompositeScore`, all calibration metrics, all classification metrics, per-quartile `RMSE_Q1..Q4`: identical to float64 precision. Predictions themselves are byte-identical (proven by R2cv matching to 16dp).

**Root cause (search.py):**
- `main:4214` — `mean_rmse = np.mean([m["RMSE"] for m in cv_metrics])` (mean of per-fold RMSEs)
- `branch:4293` — `mean_rmse = float(np.sqrt(mean_squared_error(all_y_test, all_y_pred)))` (pooled RMSE)

Branch's inline comment documents the motivation: pooled RMSE matches Unscrambler/PLS_Toolbox/SIMCA/IUPAC convention, and under LOO per-fold RMSE on a 1-sample fold mathematically degenerates to `|y-ŷ|` (making mean-of-fold RMSE actually equal MAE). Change was bundled into the round-2 "backend Repeated-KFold pooling" commit but applies to plain K-Fold too.

**Important implication:** `main` had an internal inconsistency — `R2cv`, `MAEcv`, `Bias` all used concatenated predictions, but `RMSEcv` used mean-of-fold. Branch makes it uniform.

**Gotcha for future parity work:** The validation plan only anticipated classification-metric drift (F1 averaging, specificity labels) under repeated CV. It missed the regression RMSE pooling change because that commit landed separately. Enumerate *all* commits in the PR range and reason about each one's numeric surface — don't rely on the PR description's "expected diffs" list.

**LOO classification degeneracy (branch only):** PLS-DA and LightGBM collapse to majority-class predictions on this 3-class imbalanced dataset (Kappa/MCC/Specificity=0, F1=Accuracy≈base rate). Not a plumbing bug — tiny 48-sample training folds with imbalanced 3-class targets don't leave room for minority-class learning at default hyperparameters. Worth a GUI tooltip warning.

**Full report:** `docs/pr4_parity_report.md`. JSON artifacts in `docs/pr4_parity/`.

---

## 2026-04-29 — T-10 PLS components clamp implemented

**Problem:** `min_train_samples = n_samples * (folds - 1) // folds` at `search.py:1109` and `:3312`
was a K-fold-only formula applied unconditionally even when `cv_strategy` was `'loo'` or
`'repeated_kfold'`. Under LOO with N=10, it returned 8 instead of 9. This shrank the LOO PLS
grid by 1 component on small datasets. Any future caller who forgot to clamp would crash inside CV.

**Solution:** New helper `cv_utils.compute_min_train_fold_size(cv_strategy, n_samples, n_folds)`
returns the smallest training-fold size for each CV strategy:
- kfold: `n_samples * (k-1) // k` (exact for sklearn KFold with no shuffle)
- repeated_kfold: same as kfold (each repeat uses same fold sizes)
- loo / repeated_loo: `n_samples - 1`
- group_kfold / leave_one_group_out: raises `NotImplementedError` (deferred to T-15)

Both `run_search` and `run_bayesian_search` call this helper to clamp `max_n_components` before
passing it to `get_model_grids`. The helper raises `ValueError` when `n_folds > n_samples` for
kfold/repeated_kfold strategies.

**Codex review:** APPROVE_WITH_CHANGES. All 5 suggestions applied:
1. `n_folds > n_samples` ValueError in helper + 2 extra tests covering the guard.
2. Docstring math wording tightened — "exact" not "conservative" since the formula
   `n*(k-1)//k` equals `n - ceil(n/k)` for sklearn KFold.
3. Task 5 was docstring/comment only in models.py — no assert added. An assert would have
   broken existing tests that call `get_model_grids` without clamping.
4. `_extract_n_components_seen` helper uses the LVs column as canonical source, with
   `ast.literal_eval` fallback (not `json.loads`) because Params is `str(dict)` Python repr
   with single quotes, not JSON.
5. NSGA-II deferred ticket reframed as AUDIT (T-10b), not assumed-bug.
   `_get_constrained_pls_components` clamps by n_samples but some evaluation paths already
   use min_train_samples from cv_folds; need full audit of decode/result-row/reporting
   consistency before declaring same bug present.

**DEFERRED:** NSGA-II audit ticket (T-10b). Full audit of `_get_constrained_pls_components`
and all evaluation paths needed before asserting identical bug.
---

## 2026-04-30 — T-07 PDS even-window arithmetic

**Symptom:** `estimate_pds(window=10)` raised
`ValueError: could not broadcast input array from shape (11,) into shape (10,)`
at the first interior wavelength.

**Root cause:** B allocated as `(p, window)` but the X-slice spans
`2*(window//2)+1` columns. For odd window, slice = window. For even
window, slice = window+1. The assignment `B[i, 0:window+1] = b` overflows.
This implementation uses a centered, odd-width local window of size 2k+1
(channels i-k to i+k), so it must be odd.

**Fix:** Reject even windows with ValueError citing Wang, Veltkamp, &
Kowalski (1991), Anal. Chem. 63(23), 2750-2756. Also harden apply_pds:
- Signature changed from `window: int = 11` to `window: int | None = None`
- Geometry derived from B.shape[1], not caller's window arg
- FutureWarning (not DeprecationWarning) on mismatching window —
  user-visible, not silenced in non-__main__
- ValueError on even-width B and B.shape[0]/X.shape[1] mismatch

**Why reject vs coerce:** Coercing silently would create contract drift
between requested `window` and returned `B.shape[1]`. The canonical
centered-local-regression definition (2k+1) is confirmed in Wang et al.
(1991), the RNIR package, and the specProc R package.

**Codex review modifications applied:**
1. Literature claim tightened to "this implementation's window is the
   full centered width 2k+1, so it must be odd"
2. apply_pds signature default changed to None (plan had 11)
3. FutureWarning instead of DeprecationWarning
4. Two extra tests: even-width B rejection and shape compatibility
5. Test count: 10 (plan said 7-8)

**Commits:** `d3d1606` (RED tests), `f438083` (fix). See
`docs/plans/2026-04-29-T07-pds-even-window-fix.md` and
`tests/test_pds_window_arithmetic.py`.

---

> **Older entries archived to [`SESSION_LOG_ARCHIVE.md`](SESSION_LOG_ARCHIVE.md)** as of 2026-04-29 — entries before 2026-04-15 moved out to keep this log lean. Grep the archive when you need historical context.
