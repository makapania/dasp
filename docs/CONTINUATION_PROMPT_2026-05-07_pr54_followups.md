# Continuation prompt — PR #54 follow-ups

> **STATUS — 2026-05-07 late evening:** Items 1-7 are CLOSED. Item 7 (delete legacy Bayesian path) shipped in this session — see `docs/PROJECT_STATUS.md` header and `docs/SESSION_LOG.md` 2026-05-07 late evening entry. **Item 8 (eight methodology / production-behavior changes) is the only remaining queue and needs explicit user approval per item — DO NOT pick up autonomously.**

---

**Filed:** 2026-05-06 late evening, after PR #54 merge (`4aef396`)
**Pickup:** next session (likely 2026-05-07)
**Last updated:** 2026-05-07 evening

---

## Current state — READ THIS FIRST (2026-05-07 evening)

**Item 7 (delete legacy Bayesian path) is being handled in a separate parallel session — DO NOT pick it up.** A plan was filed at `docs/plans/2026-05-07-delete-legacy-bayesian-path.md` (commit `43607be`) and another worker is executing it. Stay out of `src/spectral_predict/search.py` (the `run_bayesian_search` function and surroundings), `src/spectral_predict/bayesian_utils.py` (`create_objective_function`, `convert_optuna_result_to_dasp_format`), and the three legacy-path test files (`test_class_weight_validation_rebuild.py`, `test_cv_pls_clamp.py` legacy sections, `test_golden_standard_performance.py`) until that session lands.

**Items 1–6 below are DONE but UNCOMMITTED in the working tree.** A prior session this week implemented them and updated `PROJECT_STATUS.md` to reflect completion, but never committed the work. Verify with `git status` before doing anything — you should see these modified files:

```
M  tests/test_bayesian_dedup.py            <- item 1 (SQLite resume rehydration)
M  tests/test_t44_autoscale_wiring.py      <- item 2 (end-to-end Bayesian autoscale)
M  src/spectral_predict/unified_bayesian.py <- items 3, 4, partial 5 (comment polish)
M  src/spectral_predict/bayesian_utils.py  <- item 5 (line-number ref cleanup)
M  src/spectral_predict/search.py          <- item 5 + item 6 (black pass)
M  spectral_predict_gui_optimized.py       <- item 6 (black pass)
```

Plus already-landed commits: `9a299a2` (autoscale Bayesian comparison tool + BoneCollagen 12-cell sweep), `2d2ab3a` (SESSION_LOG entry on PLS-DA validation rebuild needing int-encoded labels), `43607be` (legacy-deletion plan), `98de4bd` (PROJECT_STATUS autoscale entry + this prompt update).

---

## Your session: PRIMARY WORK

### Step 0 — commit items 1–6 in five logical chunks (MANDATORY, do this first)

This is now safety-critical, not just polish. Item 7's parallel session will eventually merge changes that touch some of the same files (especially `search.py` and `bayesian_utils.py`). If items 1–6 stay uncommitted, a rebase/merge collision is guaranteed. Land them in this order so each commit is reviewable independently and the `black` pass is isolated from the source-edit commits:

1. `tests(bayesian-dedup): SQLite resume rehydration round-trip` — `tests/test_bayesian_dedup.py` only
2. `tests(autoscale): end-to-end run_unified_bayesian wiring` — `tests/test_t44_autoscale_wiring.py` only
3. `chore(unified_bayesian, bayesian_utils): comment polish + stale search.py ref cleanup` — `src/spectral_predict/unified_bayesian.py` + `src/spectral_predict/bayesian_utils.py` (items 3/4/5)
4. `style: black pass on search.py and gui` — `src/spectral_predict/search.py` + `spectral_predict_gui_optimized.py` (item 6 — ~27K whitespace-only lines, MUST be its own commit)
5. (Skip — `PROJECT_STATUS.md` already committed today as `98de4bd`.)

**Verification before each commit:** `pytest tests/test_bayesian_dedup.py tests/test_t44_autoscale_wiring.py -v` for the test commits; `python -m py_compile src/spectral_predict/{search,unified_bayesian,bayesian_utils}.py spectral_predict_gui_optimized.py` for the source/style commits. Targeted, not full suite — per project rule "don't run full test suite for small changes."

**After Step 0, choose ONE of the two primary work items below.** Both are independent of item 7 and won't collide.

---

### Option A — harden `compute_validation_metrics_for_top_models` against string-typed classification labels (RECOMMENDED, ~30 LOC)

**The bug.** Today's session (commit `2d2ab3a` SESSION_LOG entry) discovered that `compute_validation_metrics_for_top_models(task_type='classification')` at `src/spectral_predict/search.py:568` silently fails when given raw string class labels (`'Low'`/`'Medium'`/`'High'`). PLS.fit chokes trying to convert `'Medium'` to float; the rebuild loop catches the ValueError, logs `[Warning] Failed to compute validation for model 1: could not convert string to float: 'Medium'` to stdout, and leaves `val_Accuracy`/`val_F1`/`val_ROC_AUC` as NaN for every model. A user reading the Validation tab sees blank metrics with no on-screen explanation.

**Why the GUI doesn't currently hit this:** the GUI wraps a `LabelEncoder` around classification targets before passing them to the search machinery, so `y_train`/`y_val` reaching the rebuild are already integer-encoded. Anyone scripting against this function directly (or any future tool like `tools/autoscale_bayesian_compare.py`) hits it.

**The fix.** At the top of `compute_validation_metrics_for_top_models` (around line 660 where `_normalize_mixed_type_labels` is already called for mixed-type object arrays), add a sibling branch that handles pure-string label arrays:

```python
# Coerce string-typed labels to int for classification rebuild — PLS.fit
# can't ingest object/string arrays, and the existing mixed-type normalizer
# only fires when len({type(v).__name__ for v in y}) > 1.
if task_type in ("classification", "one_class"):
    if (
        getattr(y_train, "dtype", None) == object
        and all(isinstance(v, str) for v in y_train)
    ):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(np.concatenate([y_train, y_val]))
        y_train = le.transform(y_train)
        y_val = le.transform(y_val)
```

Slot this immediately AFTER the existing `_normalize_mixed_type_labels` block (search.py:660-668) so mixed-type arrays still take the existing path; the new branch only catches the pure-string case the existing check skips.

**Tests.** Add to `tests/test_class_weight_validation_rebuild.py` (or a new `tests/test_validation_rebuild_label_encoding.py` if the existing file is being touched by item 7's parallel session — verify with `git log --oneline -- tests/test_class_weight_validation_rebuild.py` before deciding):

1. `test_string_labels_get_encoded` — pass `y_train=np.array(['A','B','A','C'])` etc., assert no warning, assert val_Accuracy is finite.
2. `test_mixed_type_labels_still_normalized` — regression pin that the existing `_normalize_mixed_type_labels` path still fires for mixed `[1, 'B', 1, 'C']`.
3. `test_int_encoded_labels_unchanged` — assert that already-int `y_train`/`y_val` pass through with no transformation (use `id()` or array equality).

**Verification.** `pytest tests/test_class_weight_validation_rebuild.py -v` (or whichever test file you used).

**Risk profile.** Low — purely additive, only fires on a code path the GUI never reaches today. No production behavior change for any GUI user. Only effect on scripted use is "you now get real metrics instead of silent NaNs."

**Why this is bug-fix-grade and not polish:** the current behavior is a silent failure (fits the recurring "silent failure hunter" theme of the last several reviews). The user-facing symptom (blank Validation tab) is indistinguishable from "model genuinely failed" without scrolling stdout for the Warning line.

---

### Option B — CatBoost `thread_count` survival test (~10 LOC, item 8 backup)

If Option A is already in flight elsewhere or you'd rather do something smaller after Step 0:

**The pin.** Mirror the existing `n_jobs` survival test pattern for the one model that uses a different kwarg name. CatBoost's parallelism control is `thread_count`, not `n_jobs`; if the dedup/rebuild paths drop it during model rebuild, CatBoost trains single-threaded and gets ~10× slower with no warning.

**Where.** Add `test_catboost_thread_count_survives_rebuild` to whichever test file currently exercises CatBoost survival (likely `tests/test_unified_bayesian_baseline.py` — grep for `n_jobs` in the test directory to find the canonical pattern).

**Test sketch.**
```python
def test_catboost_thread_count_survives_rebuild():
    # mirror the n_jobs pattern but assert thread_count=2 propagates through
    # the rebuild Path → final fit kwargs.
    # ...10 LOC...
```

**Risk profile.** Zero — pure test addition, pins existing behavior, no production changes.

---

## What's NOT for this session

- **Item 7** (delete legacy Bayesian path) — parallel session, hands off.
- **Item 8** items other than CatBoost test — all need explicit user approval before any agent acts. If you finish Step 0 + Option A early and want more work, surface the item 8 list to the user and ask which they want unblocked. Do not act autonomously on any of: `verbose`-strip-set change, broad-except log-level upgrade, Option C n_components range, code_generator y_transform, K-Fold one-class repeated-CV pooling parity, mock-Tk banner-render test for PR #45.
- **Today's autoscale-default investigation** — concluded "leave default ON, don't change anything." `tools/autoscale_bayesian_compare.py` is reusable if you ever want to replicate on more datasets/models, but no source change is currently warranted. Don't re-run the BoneCollagen sweep — it's in `tools/_autoscale_bayes_compare_full.json` and PROJECT_STATUS top entry.

---

## Original queue (filed 2026-05-06 late evening)

PR #54 shipped autoscale decoupling (T-44) + bayesian dedup hardening + PR-#52 follow-ups. Cross-family review (Codex re-review + DeepSeek V4 Pro Max + 4-agent toolkit panel) signed off; main is at `b014831` after the doc updates.

This queue is what the multi-agent review surfaced as deferrable. Nothing here is bug-fix-grade. Pickup order is roughly leverage-descending.

---

## Cheap, valuable, do these first — pure test additions

### 1. SQLite-backed resume rehydration round-trip test (~50 LOC)

**Where:** new test class in `tests/test_bayesian_dedup.py`.

**What it pins.** `_rehydrate_seen_fingerprints` at `unified_bayesian.py:339-368` is currently only tested against in-memory studies. Production resume goes through SQLite. Two specific contracts are unenforced:

- The non-finite-float sentinels at `unified_bayesian.py:173-177` (`__nan_sentinel__`, `__pos_inf_sentinel__`, `__neg_inf_sentinel__`) exist *only* because `ast.literal_eval(repr(float('nan')))` raises SyntaxError, breaking resume rehydration of fingerprints that include non-finite values (e.g. `imbalance_params` resolving to inf in some configs). No test exercises this round-trip.
- A study with N COMPLETE trials carrying `fingerprint` user_attrs serialized via `_freeze_for_fingerprint` should rehydrate to a `seen_fingerprints` dict of size N when reloaded from SQLite via `optuna.load_study`.

**Sketch.** Use `tmp_path` for the SQLite URL. Build a study with 3-5 trials including one with a fingerprint containing `float('nan')` or `float('inf')` in an imbalance_params slot (force-construct via `_freeze_for_fingerprint({'param': float('inf')})` — confirm `ast.literal_eval(repr(...))` succeeds end-to-end). Reload the study via `optuna.load_study(storage="sqlite:///" + str(tmp_path/"resume.db"), study_name="...")`. Assert `_rehydrate_seen_fingerprints(study, seen) == N` and the dict contains the inf-bearing fingerprint.

### 2. End-to-end `run_unified_bayesian(enable_autoscale=True)` test (~40 LOC)

**Where:** new test in `tests/test_t44_autoscale_wiring.py`, parallel to the existing `TestTPEDiscoveryExploresAutoscale` class.

**What it pins.** TPE side has integration coverage (`test_enable_autoscale_true_produces_both_values` at lines 111-135). Bayesian side currently only tests `suggest_preprocessing` directly — never `run_unified_bayesian` end-to-end. A regression that broke the wiring between `bayes_enable_autoscale` flag and the actual `apply_autoscale` exploration would slip through.

**Sketch.** Call `run_unified_bayesian(X, y, wl, model_name='PLS', task_type='regression', n_trials=20, enable_autoscale=True, ...)` with synthetic data (`np.random.RandomState(42).randn(30, 50)` etc.). Inspect the returned dataframe's `Apply_autoscale` column (or whatever it's named — verify against `unified_bayesian.py` ~line 3043 in convert_study_to_dataframe). Assert the column contains both True and False. Mirror with a second test for `enable_autoscale=False` asserting all values are False.

---

## Comment polish — batch in one pass

These three items are pure inline-comment edits. A worker model (e.g. GLM 5.1 fast-path per the global agent guide Tier 1 dispatch) could do them mechanically with a single instruction.

### 3. Reword stale "dedup pruning bursts" comment

`unified_bayesian.py:2794-2796` currently says the `try/except ValueError` around `best_trial` exists to handle "dedup pruning bursts." Under value-cache-and-replay, dedup'd trials are COMPLETE with cached values — there is no pruning. Reword to mention the actual penalty paths:

```python
# best_trial raises ValueError when no COMPLETE trial has a finite-better-than-1e10
# value yet — possible early in a run if the first trials all hit penalty paths
# (PLS clamp at :1500, OC skip at :1334, generic exception fallthrough at :2006).
```

### 4. Drop reviewer-pseudonym citations

`unified_bayesian.py:319` (`Defense-in-depth per DeepSeek STRONG-2 review.`) and `:1326-1328` (`(Kimi BLOCKER closure)`). Per project rule (CLAUDE.md "Don't reference the current task, fix, or callers"), reviewer pseudonyms rot when reviewers churn. Keep the rationale, lose the attribution. Example for `:1326-1328`:

```python
# Cache the skip sentinel so a future identical OC config replays
# immediately instead of re-running the whole CV+skip detection.
```

T-XX ticket refs ARE durable per project convention — leave those alone.

### 5. Stale `search.py` line-number references

Pre-existing (NOT introduced by PR #54 but worth fixing while in the area):

- `unified_bayesian.py:1504-1505, 1522-1523, 1535, 1542` — references to `search.py` lines that no longer match (claimed line ranges contain different code now)
- `bayesian_utils.py:494-495` — same pattern
- `unified_bayesian.py:1523` — `# the GUI dispatcher fix from c395317` — short SHA reference is rot-prone after rebases

Either remove the line refs (let the description stand alone) or update to current line numbers. The class_weight discriminator now lives in `_apply_class_weight_discriminator_for_rebuilt_model` at `search.py:418-490`.

---

## Cosmetic — trivial

### 6. `black` pass on PR #54 indentation drift

1-space indentation drift inside argument lists at `search.py:1102-1104` and `spectral_predict_gui_optimized.py:27518-27525`. Inert (inside parens) but visually inconsistent. `black src/spectral_predict/search.py spectral_predict_gui_optimized.py` would clear it.

---

## Major refactor — delete the legacy Bayesian path

### 7. Delete `run_bayesian_search` and the legacy `bayesian_utils` machinery (~1–2 hours)

**Decided by user (2026-05-06):** the legacy path is going. Production Bayesian is `run_unified_bayesian`; production preprocessing-discovery uses TPE in `tpe_preprocessing_discovery.py`. `run_bayesian_search` is documented as test-only at `search.py:3880` (`NOTE (T-36): run_bayesian_search is test-only — no GUI caller`) and has been forcing every dedup change (value-cache-and-replay, the PR #54 fingerprint-stamp redesign) to be applied symmetrically across both paths.

**What gets deleted:**

1. `src/spectral_predict/search.py:3491` — the `def run_bayesian_search(...)` function and its body (extends to roughly line ~4040 — verify end via syntax).
2. `src/spectral_predict/bayesian_utils.py:203` — `def create_objective_function(...)`. Verify nothing else in the production path imports it (current grep shows only `search.py:3953` does).
3. `src/spectral_predict/bayesian_utils.py:815` — `def convert_optuna_result_to_dasp_format(...)`. Same verification.
4. Any helpers in `bayesian_utils.py` that become unreachable after removing the above. Run `grep -rn "from .bayesian_utils import" src/` to enumerate residual imports; preserve anything still used.

**What stays in `bayesian_utils.py`:** Anything imported by `unified_bayesian.py` or other production modules. Notably the dedup helpers (`_make_fp` integration, `_register_or_replay_fingerprint`/`_record_fingerprint_value`/`_resolved_weighting_fingerprint`) live in `unified_bayesian.py` and are imported into `bayesian_utils.py`, not the other way around — so deleting the legacy `create_objective_function` doesn't disturb them. Verify before deleting.

**Test migration:**

Three test files reference `run_bayesian_search`. For each, decide: migrate the assertion to `run_unified_bayesian`, or delete the test entirely if the assertion is already covered elsewhere.

- **`tests/test_class_weight_validation_rebuild.py:334`** — parametrized over `["run_search", "run_bayesian_search"]`. Drop `run_bayesian_search` from the parametrize list. The grid-path assertion via `run_search` stays. Verify the Bayesian-side equivalent of this contract is pinned in `tests/test_cv_pls_clamp.py` or `tests/test_unified_bayesian_baseline.py`; if not, add a parallel test against `run_unified_bayesian`.
- **`tests/test_cv_pls_clamp.py:266-302` (`Black-box: run_bayesian_search must clamp the PLS LV upper bound`) and `:306-...` (legacy-path consumer-side pin)** — the production-path equivalent already exists at `TestRunBayesianSearchPLSGridClamping` (which uses `run_search`, not `run_bayesian_search`) and via the `test_unified_bayesian_*` tests. Most of these can be deleted outright. Check `git log -p tests/test_cv_pls_clamp.py | head -200` for the assertion intent before deleting; if any assertion is unique to the legacy path, port it to `run_unified_bayesian`.
- **`tests/test_golden_standard_performance.py:213, 236`** — the assertion here is "golden-standard performance number on a known dataset." If the test is real performance pinning (not just smoke), port to `run_unified_bayesian` with the equivalent metric assertion. If it's a smoke-test of the dispatch surface, delete.

**Verification after deletion:**

```bash
# 1. No imports left:
grep -rn "run_bayesian_search\|create_objective_function\|convert_optuna_result_to_dasp_format" src/ tests/ spectral_predict_gui_optimized.py

# 2. Targeted test pass:
.venv312\Scripts\python.exe -m pytest tests/test_bayesian_dedup.py tests/test_cv_pls_clamp.py tests/test_t44_autoscale_wiring.py tests/test_class_weight_validation_rebuild.py tests/test_unified_bayesian_baseline.py -v

# 3. Full smoke pass to catch indirect import breaks:
.venv312\Scripts\python.exe -m py_compile src/spectral_predict/search.py src/spectral_predict/bayesian_utils.py src/spectral_predict/unified_bayesian.py spectral_predict_gui_optimized.py
```

**Risk profile.** Low — the GUI never calls these functions, so removing them cannot break user-facing behavior. The only risk is a stale test-import that the agent misses (caught by step 1 above) or a helper in `bayesian_utils.py` that turns out to also be imported by `unified_bayesian.py` (caught by step 3). Both are mechanical to fix when surfaced.

**Stale comments to clean up at the same time** (currently rotting because they reference the legacy path's structure): pre-existing `search.py` line-number references in `unified_bayesian.py:1504-1505, 1522-1523, 1535, 1542` and `bayesian_utils.py:494-495`. Item 5 above already enumerates these — fold that cleanup into this PR rather than separately, since deleting the legacy path will likely shift line numbers anyway.

**Sequencing.** Do this AFTER items 1–2 (the test additions on the production path), because deleting the legacy path while writing tests is more error-prone than doing each separately. Items 3–6 can be folded in or done separately, agent's choice.

---

## User-decision-needed — DO NOT touch autonomously

These need explicit approval from the user before any agent acts. Each is a methodology change or production behavior change.

### 8. Other methodology / production-behavior items

These were each previously deferred with "needs user confirmation" framing in PROJECT_STATUS.md:

- **`verbose` in `code_generator._PIPELINE_PARAMS`** strip set — Kimi MEDIUM from PR #49 review. Architectural sister of `n_jobs`; cleanest fix removes it from strip set and handles contextually. Codegen production behavior change.
- **Booster end-to-end forced-duplicate test** — Codex NEEDS_DISCUSSION from prior dedup review.
- **Broad `except Exception → 1e10` log-level upgrade** in trial body — silent-failure WEAK; changes user log signal-to-noise.
- **Option C: data-aware `n_components` suggest range** — methodology question, separate from dedup.
- **`y_transform` handling in `code_generator.py`** — Codex flagged that regression `YTransformWrapper`/TTR is wrapped in-app at GUI line ~37860 but codegen has no `y_transform` handling. Production codegen change.
- **Plain K-Fold one-class repeated-CV pooling parity** — TODO at `contamination.py:721`. Would change user-memorized numbers from existing models — explicitly deferred in the codebase comment as "rebaseline in a separate PR."
- **CatBoost `thread_count` survival test** — mirrors n_jobs survival pin for the one model using a different kwarg name. ~10 LOC. Deferrable polish but technically a test addition.
- **Mock-Tk banner-render test for PR #45** — needs Tk dialog mocking infra. Medium effort.

---

## Cleanup — update PROJECT_STATUS.md when picking up

`docs/PROJECT_STATUS.md` line ~50 lists "narrow `_resolved_weighting_fingerprint` exception scope to `AttributeError` only (silent-failure WEAK)" as a "possible future follow-up." **PR #54 closed this finding** by deleting the entire except per project policy (sklearn `BaseEstimator.get_params(deep=True)` cannot raise for conforming estimators). Remove from the deferred-followups list when next updating that file.

---

## Briefing for the next agent picking this up

For items 1–2: pure test additions, full repo access useful for sanity-checking against existing tests in `tests/test_bayesian_dedup.py` and `tests/test_t44_autoscale_wiring.py`. Read those files for the existing test patterns before writing new ones.

For items 3–5 (comment polish): mechanical edit batch. Suitable for a worker model dispatch (GLM 5.1 fast-path per the global agent guide, since the task is "given file + spec, produce file with edits applied").

For item 6: literally `black src/spectral_predict/search.py spectral_predict_gui_optimized.py && git diff` to inspect, then commit if the changes are limited to whitespace.

For items 7–8: present to user, await explicit approval, do not act autonomously.
