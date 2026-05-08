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

> **Audit correction 2026-05-08:** an earlier draft of this file claimed T-19 Auto mode was "partially-drafted" and pending. That was based on a stale 2026-05-02 memory pin. **T-19 is shipped** — PR #15 (`1d2bf6d`) landed Auto mode + bug fix; PR #38 closed sister sites in Bayesian + NSGA-II; PR #41 closed the validation-rebuild class_weight gap. PROJECT_STATUS bullet was explicitly unrotted at `965dee1` (2026-05-07): user verdict "niche enough to leave open until a specific paper-reproduction asks for it" for the few remaining edge cases (non-balanced ratios, ElasticNet PLS-DA inner LR, per-model override). T-19 entry removed from this list.

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

### T-17 PLS-2 multi-Y workflow (HIGH-LEVERAGE, 2-3 weeks)

**Per PROJECT_STATUS open-ticket roster (line 565):** "T-17 PLS-2 (2-3w)" — the multi-Y / PLS-2 workflow. User-elevated 2026-04-29 from the original reconciled-roadmap pass: *"high-leverage but substantial effort."*

**Why:** Today dasp models one Y at a time. PLS-2 fits multiple correlated Y-variables jointly in a single PLS decomposition, exploiting cross-Y covariance for better latent-variable estimation. Standard chemometrics workflow for multi-target spectroscopy (e.g., joint prediction of multiple bone-collagen indicators from one FTIR run). Today users either run dasp N times (no information sharing across Ys) or fall back to Unscrambler/SIMCA-P for joint modeling.

**Status:** PLANNED — no implementation yet. Status doc references it consistently across multiple sessions as an open ticket.

**Scope considerations:** PLS-2 changes the validation-metrics shape (per-Y RMSEcv/R²cv plus pooled), the result-CSV schema (per-Y columns), the GUI Y-target picker (multi-select), and prediction outputs (per-Y predictions). Cross-cuts most of `search.py` — the 2-3 week estimate is real.

### T-01 reframed — external-test-set workflow (~2-3 days)

**Per PROJECT_STATUS line 565 + 1017:** the reconciled-roadmap pass reframed T-01 from "per-fold varsel leakage audit" (false alarm — chemometrics convention allows full-data SNV/SG-deriv per `feedback_chemometrics_conventions.md`) to **external-test-set workflow capability**.

**Why:** Westad & Marini 2015 + Workman 2018 (canonical chemometrics validation refs) recommend external test sets over LOGO/group-aware CV for the user's data regime. dasp today supports SPXY 20% partition (which is what motivated the TPE proxy fix), but the broader "designate an external test set, train on the rest, score the model on the external set with full audit trail" is not first-class — users currently do it via manual file splits.

**Status:** PLANNED. Effort small (~2-3 days) because the validation-rebuild path already exists; this is mostly GUI surface (test-set designation) + result-CSV plumbing for the external-set columns.

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

1. **T-TPE-VERIFY** first — it's the cheapest test of whether the just-shipped fix is real. Determines whether `a33d956` is a real win or needs tuning.
2. **T-BASIC-PROXY** if T-TPE-VERIFY confirms the resolver design empirically — same fix, second module, ~3-4h.
3. **T-01 reframed (external test set)** — small (~2-3d) and complements T-TPE-VERIFY (the SPXY 20% workflow IS an external test set; making it first-class GUI surface closes the loop).
4. **T-16 survey** as a parallel side-task — read-only research, doesn't conflict.
5. **T-31 multi-class SIMCA** when ready to invest 1-2 weeks. New methodology, real chemometrics need confirmed.
6. **T-17 PLS-2** when ready to invest 2-3 weeks. High-leverage but biggest scope; cross-cuts the most files.

Each item above is independently mergeable. Don't bundle.
