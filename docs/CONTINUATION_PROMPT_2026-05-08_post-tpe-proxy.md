# Continuation handoff — post-TPE-proxy ship

**Anchor:** `origin/main` at `24efa39` (2026-05-08). TPE family-aware proxy shipped at `a33d956` and pushed.

**Filter applied:** Tickets that affect *results* (R²/F1/accuracy of leaderboard rows) or *utility* (what users can do that they couldn't before). Engineering polish (CI hygiene, test-coverage gaps, naming consistency, bare-Tk AST sweep) is excluded — see §Excluded at the bottom.

---

## Tier 1 — Empirical verification gap on the just-shipped fix

**Status:** Code is green (82/82 unit + end-to-end smoke on PLS/LightGBM/Mixed arms), but the original handoff's verification battery on real data was NOT run. Until it is, we don't know if the fix is a real win, a no-op, or a regression on the user's actual workflow.

### T-TPE-VERIFY (HIGH, ~1-2h)

Run the SPXY 20% A/B harness on real BoneCollagen data. The harness already exists at `tools/_repro_tpe_fix_downstream_ab.py` (untracked working tree); the monkeypatch was updated to accept `**kwargs` so it survives the new `proxy_family` kwarg.

**Pass thresholds (from plan §4):**
- BoneCollagen + PLS-only at SPXY 20%: **best passing R²pred at gap≤0.02 ≥ 0.97** (matches or beats pre-fix 0.9722).
- Top-N includes `snv_deriv2_w15+autoscale` (the canonical PLS winner the original LightGBM proxy demoted).
- Wall-clock not significantly worse than pre-fix (< 1.5× pre-fix).
- Cross-split sanity: stratified 20% R²pred ≥ 0.95, random 20% R²pred ≥ 0.95 (pre-fix figures: 0.9520 / 0.9526, ±0.005 noise band).

**Tree-arm verification:** `models_to_test=['LightGBM']` on the same data; assert `proxy_family='tree'` lands in CSV (already confirmed by smoke), no degenerate scoring, audit RMSE column meaningful.

**Decision after running:** If thresholds hit → close TPE-proxy work. If miss → either tune the resolver default (rare — would need user buy-in for non-linear default) or add a per-trial fallback heuristic (e.g., switch to linear if tree-side scores tie within 1e-9).

**Saved artifacts** for reference: `tools/_tpe_fix_ab_arm_PRE_*.csv` and `tools/_tpe_fix_ab_arm_POST_*.csv` (pre-fix and post-`9b9d244` arms; both untracked).

---

## Tier 2 — Real bugs that ship wrong rankings

### T-BASIC-PROXY (HIGH, ~3-4h after T-TPE-VERIFY confirms the design works)

**Bug:** `src/spectral_predict/preprocessing_discovery.py:669` has the same hardcoded LightGBM `_quick_evaluate` that the TPE path had — same n<50 mean-prediction collapse, same proxy/downstream-mismatch when downstream is PLS. Affects every user who picks **Basic Preprocessing Discovery** in the GUI (`enable_smart_preprocessing` / `smart_preprocess=True` per `gui:11836`).

**User-scoped OUT of `a33d956`** per 2026-05-08 direction ("the whole point was to change behavior by being model specific" — TPE only). The bug remains.

**Fix shape:** identical to TPE path. The resolver `resolve_tpe_proxy_family` already exists in `tpe_preprocessing_discovery.py` and can be imported. Apply the same family-branching pattern to `preprocessing_discovery.py`'s `_quick_evaluate` at line 669, and thread `proxy_family` through `evaluate_preprocessing_config:570` → `discover_preprocessing:825` → `search.py:1801` (the smart-preprocessing call site).

**Why fix this:** Exhaustive Preprocessing already exposes `fitness_model='pls'` default + `_evaluate_with_actual_model` opt-in path (`ga_preprocessing.py:289`), so users hitting the Basic-path bug have a workaround — but only if they know about it. Not all users know to switch from Basic to Exhaustive.

**Skip if:** user prefers to deprecate the smart-preprocessing module entirely (TPE was designed to replace it per its own docstring). That's a strategic decision, not a code one. Ask before fixing.

---

## Tier 3 — Methodology features the user has confirmed they want

### T-19 Auto mode for imbalance handling (HIGH, partially-drafted)

**Per `feedback_t19_auto_mode_deferred.md`:** Auto mode (`imbalance_method='auto'` calling `detect_class_imbalance` and applying per-library balanced-loss on imbalanced data) is **required** for T-19 to be considered shipped. User clarified 2026-05-02: "the automode is the highest priority. another agent is also working but it needs to be done."

**Status:** Draft was stashed as `user-deferred-T19-auto-mode-draft` (6 files: gui_optimized + code_generator + imbalance + nsga2_search + search + unified_bayesian). **Coordination:** check whether the parallel agent ever finished — `git stash list` and `git branch -a | grep -i t19` before starting. If stashed work exists, pop and finish; if another branch exists, ask user which to advance.

**Scope confirmed:**
- IN: `imbalance_method='auto'` GUI dropdown option that detects imbalance via `imbalance.detect_class_imbalance(threshold=3.0)` and applies the correct native knob per model (`scale_pos_weight` for XGBoost, `class_weight='balanced'` for sklearn, `is_unbalance` for LightGBM, `auto_class_weights` for CatBoost).
- OUT (deferred per user 2026-05-02): per-model dropdowns exposing each native kwarg separately. Single Auto option is the priority.

**Why:** The user's bone-FTIR / paleoanthropology workflow has imbalanced classes; today the GUI doesn't expose any imbalance-handling, so users get unweighted models even when the data screams for re-weighting. Auto mode is "expose existing model abilities to GUI users without source edits."

### T-31 multi-class SIMCA (MEDIUM-HIGH, 1-2 weeks)

**Per `project_t31_simca_confirmed.md`:** User confirmed 2026-05-02 — true multi-class class-modeling per Oliveri & Downey 2012 LOVE, NOT a one-class extension of the existing PCA-SIMCA contamination model.

**Why:** Discriminant classifiers (PLS-DA, RandomForest, XGBoost) force every specimen into one of the trained classes — no "none of the above" output. SIMCA gives **independent per-class membership decisions** with chi²/F-test thresholds. User's domain (fossil bone, diagenesis continuum, unknown consolidants, "every site is its own thing") is the textbook case for class-modeling rather than discrimination.

**Implementation note:** the existing one-class `PCA-SIMCA` in `contamination.py` is a *single-class* membership detector and is the wrong base. Build a new multi-class implementation that fits an independent PCA model per class on inliers-only, then computes Q (orthogonal distance) + T² (in-model distance) per test sample × per class, with chi²-based membership thresholds.

**Status:** PLANNED — no implementation work has started.

### T-16 competitive model-comparison machinery (MEDIUM, survey-then-implement)

**Per `project_t15_dropped_t16_reframed.md`:** User reframed 2026-04-30 from "block bootstrap CIs + paired permutation" to "**competitive model-comparison machinery survey + implementation**." User's framing: *"ways of really comparing between models are warranted and have to be looked at from a competitive framework. there was jackknifing and permutations and i don't know what we should do, but what our competitors do is a head start."*

**Why:** Today dasp doesn't expose any pairwise-model-comparison machinery. Two models with R²pred 0.95 vs 0.94 — is that real or noise? Users can't answer this in dasp, so they fall back to "pick the higher number," which is overfitting-prone.

**Status:** Survey not started. Catalog needed:
- What does Unscrambler expose? (CAMO Software)
- What does SIMCA-P expose? (Sartorius)
- What does OPUS expose? (Bruker)
- What does PLS_Toolbox / Solo expose? (Eigenvector)
- What does mdatools (R) expose?

After survey, scope: smallest defensible shape for dasp (likely paired bootstrap CI + paired permutation test as a starting point; extend to Wilcoxon signed-rank if needed).

**Effort:** survey ~1 day, implementation ~3-5 days.

---

## Excluded — engineering polish (not the user's framing)

These are real but don't affect leaderboard outputs or what users can do:

- **T-CI-1 / T-CI-2 / T-CI-3 / T-CI-4** — CI rot fixes. Main has been silently red since 2025-10-27. Does not affect runtime correctness.
- **Bare-Tk-in-worker AST sweep** — DeepSeek/pr-review-toolkit MEDIUM, ~30 LOC. Defensive coverage.
- **`_run_predictions` test coverage for `decision_score_error`** — pr-test-analyzer CRITICAL-2 by their rubric, but no user-facing impact today.
- **Parameterized polyorder test across deriv=0..4** — pr-test-analyzer M-2.
- **Stale runtime label `gui:11925`** — one-line fix, cosmetic.

If the user re-prioritizes any of these (e.g., CI rot blocks PR-merge confidence), they move up. Default is to skip until they bite.

---

## Recommended order

1. **T-TPE-VERIFY** first — it's the cheapest test of whether the just-shipped fix is real.
2. **T-19 Auto mode** next if the parallel-agent stash still exists; this is the longest-pending high-priority methodology gap.
3. **T-BASIC-PROXY** when convenient (after T-TPE-VERIFY confirms the resolver design empirically).
4. **T-31 SIMCA** when ready to invest 1-2 weeks. New methodology.
5. **T-16 survey** as a parallel side-task; it's read-only research that doesn't conflict with other work.

Each item above is independently mergeable. Don't bundle.
