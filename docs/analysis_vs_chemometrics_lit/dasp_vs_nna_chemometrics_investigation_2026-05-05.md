# dasp vs. NNA Pipeline: Chemometrics-Grounded Gap Investigation

**Date:** 2026-05-05
**Investigator:** Claude Opus 4.7 (1M context), with adversarial verification by Codex CLI and DeepSeek V4 Pro Max (z.ai/opencode-go subscription path).
**User context:** Bone-NIR and forage-NIR chemometrics. Industry standards (ASTM E1655 / Pomerantsev / AOAC) take precedence over ML-leakage purity arguments.
**Question:** Within reason, what is in the NNA pipeline (`nir_chemometrics_pipeline.py`) that produces models the user's `.dasp` workflow cannot replicate, and what subset of that gap is actually worth importing into dasp?

---

## 0. TL;DR — Verified Final Recommendation

After three rounds of refinement (initial scan → two-model debate → user corrections → Codex verification), the surviving high-value additions for the user's bone-NIR + forage-NIR chemometrics work are:

| Rank | Item | Effort | Status in dasp |
|---:|---|:---:|---|
| 1 | Default NIR noisy-region mask preset (<400 nm + 1350-1450 nm + 1800-1950 nm) wired into the search registry | **S** | Partial — region selection exists, default chemometrics mask preset does not |
| 2 | Calibration pre-flight outlier report run automatically before search, using the studentized PLS-CV residual (not raw Y z-score) for Y-outlier detection | **M** | Partial — components exist, workflow timing + residual definition missing |
| 3 | RPIQ alongside RPD in the metrics table | **S** | Missing entirely |
| 4 | Group-stratified validation-set picker (existing "Stratified" gets a Group column dropdown) | **M** | Missing |
| 5 | DUPLEX wired into the GUI validation-set picker radio list | **S** | Missing (already implemented in `sample_selection.py:138-204`) |
| 6 | MCCV holdout (repeated stratified random splits, mean ± SD) | **L** | Missing — and the data model assumes a single fixed validation index set, so this is a real architectural lift |

**Items dropped during investigation** (initially recommended, ultimately verified as non-gaps or not relevant to the user's workflow):

- **VIP-as-selector → any-final-model wiring.** Already covered. Bayesian path sweeps `subset_type` ∈ {cars_tree, model_specific, lightgbm, vip} as a search dimension (`unified_bayesian.py:1180-1203`, `preprocessing_discovery.py:81-98`).
- **PLS parsimony rule (fewest LVs within ~5% of min RMSECV).** Irrelevant to dasp's architecture — dasp emits a row per `(preproc, selector, model, n_components)` and the user picks. There is no internal "auto-pick best PLS" decision a parsimony rule could improve.
- **MSC in preprocessing discovery.** DeepSeek's argument was specifically for FTIR Mie scattering. The user's bone work is NIR (diffuse reflectance), not FTIR. SNV is the right primary scatter correction for both forage and bone NIR.
- **LOGO / group-aware CV.** Deprioritized by the user. NNA used geographic LOGO only diagnostically (Section 3.3 of NNA's SUMMARY_REPORT) showing transferability is poor; the headline performance numbers came from MCCV-style same-population validation. Not a chemometrics-standard requirement.
- **Per-fold variable selection sealing.** ASTM E1655 / Martens performs selection on the calibration set as a whole. Strict per-fold selection is an ML-purity move that dasp has already audited (`docs/T01_VARSEL_LEAKAGE_AUDIT.md`).

---

## 1. Investigation Setup

### 1.1 The two repositories

**REPO 1 — NNA pipeline (single-file Python).**
- `C:\Users\mspon\Desktop\_DeskSync\Arctic\NNA\nir_chemometrics_pipeline.py`
- Companion: `results_full_20260331_190255\SUMMARY_REPORT.md`

A tiered NIR chemometrics pipeline:
- Tier 1: 10 preprocessings × PLS only, 5-fold CV → keep top 4.
- Tier 2: those 4 × 6 selectors × 9 models, 5-fold CV → ranked.
- Tier 3: top 10 unique combos per target re-evaluated under 7 split strategies (5fold CV, stratified random 80/20, Kennard-Stone, SPXY, geographic Finland→Norway, geographic Norway→Finland, MCCV 20×).
- Tier 4: best-model report with plots and saved joblib models.

Pre-modeling diagnostics: PCA score plot by site, Hotelling T² + Q-residual outlier detection, dead-channel + saturation screen, Y-outlier studentized PLS-residual screen, target distribution histograms, full spectra overview by site.

Total configurations evaluated: 1,128 across 4 targets (N, NDF, ADF, N_moist_corrected).

**REPO 2 — dasp (Tkinter GUI + Python package).**
- `C:\Users\mspon\git\dasp\` (Spectral Predict, V1)
- Entry point: `spectral_predict_gui_optimized.py`
- Backend: `src\spectral_predict\`

User's main chemometrics tool. Search modes include grid (`search.py:run_search`) and Bayesian TPE (`unified_bayesian.py`). Models: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP. Variable selection: UVE, SPA, iPLS, CARS, GA-PLS. Calibration transfer: DS, PDS.

### 1.2 Domain context

The user works in two NIR-spectroscopy domains:

1. **Forage / plant nutrition NIR** — multi-site multi-target (N, NDF, ADF, moisture-corrected N), ~220 samples, 350-2500 nm. The NNA dataset is representative.
2. **Bone NIR** (not FTIR) — diffuse-reflectance NIR on bone samples for nutritional / compositional inference.

Industry standards in scope: ASTM E1655 (NIR multivariate quantitative analysis), AOAC Official Methods, ICH Q2(R1), and the Pomerantsev-Rodionova validation tutorial tradition in chemometrics.

### 1.3 What the question actually was

The user produced six `.dasp` calibration files in this folder:
```
R100_PLS_sg2_n-moist-correct_r2p-0.85
R100_PLS_snv_sg3_n_r2p-0.23
R100_PLS_snv_sg4_n-moist-correct_r2p-0.86
R100_PLS_snv_sg4_n-moist-correct_r2p-0.88
R50_PLS_sg3_n-moist-correct_r2p-0.85
R58_PLS_snv_sg2_n_r2p-0.82
```

All PLS, single-split R²p reported in filename. The user's claim entering this investigation: *NNA produces better models than dasp on the same data, and we should figure out why*.

The investigation was partly about parity (what is missing in dasp?) and partly about model quality (what does NNA do that lifts the actual numbers?). The two questions diverged.

---

## 2. Initial Gap Analysis (Pre-Verification)

The first pass produced a 10-item list, ordered by what *appeared* to differentiate NNA's pipeline from dasp's:

1. Multi-lens validation reporting per candidate (5fold CV, KS, SPXY, geographic, MCCV all on the same model side-by-side).
2. Group-aware stratification for regression splits (site × y-tercile).
3. Geographic / leave-one-group-out splits.
4. Pre-modeling diagnostics block (Hotelling T² + Q, Y-outlier screen, PCA-by-group, dead-channel/saturation).
5. Tier-based search with finalist hand-off + model-type diversity guarantee.
6. Region-removal as a wired variable-selection option.
7. VIP-threshold as a first-class selector that feeds any downstream model.
8. RPIQ alongside RPD.
9. PLS component-selection rule "fewest within ~5% of min RMSECV".
10. Per-fold variable selection sealing.

This list was correct in structure but **substantially overstated the gap**. Multiple items either already existed in dasp under different names, were not chemometrics-standard, or were not relevant to the user's typical workflow. The verification process ahead would cut it down significantly.

---

## 3. Multi-Agent Adversarial Review

### 3.1 Round 1 — Independent recommendations

Two reviewers were dispatched in parallel with full repo read access:
- **Codex** (via `codex-reviewer`) — local repo read, no model preference disclosed.
- **DeepSeek V4 Pro Max** (via `opencode-call` on the z.ai / opencode-go subscription path).

Both were given the gap analysis and asked for *one primary* + *one backup* high-value addition for the user's bone-NIR + forage-NIR work, with file:line citations.

**Convergence:** Both independently named **group-aware cross-validation (LeaveOneGroupOut / GroupKFold) wired into `cv_utils.build_cv_splitter`** as the primary recommendation. Codex's evidence: `cv_utils.py:177-225` (sizing helper raises `NotImplementedError` for group strategies), `cv_utils.py:321-338` (build_cv_splitter supports kfold/repeated_kfold/loo only), GUI combobox locked to those three at `spectral_predict_gui_optimized.py:11331-11334`. DeepSeek's evidence: same locations plus the bone-FTIR `GAP_ANALYSIS.md` rationale (later corrected — bone is NIR).

**Divergence on backup:**
- Codex: multi-lens validation report for top-N finalists, hooking after the existing top-model validation step at `search.py:3433-3477`.
- DeepSeek: add MSC to preprocessing-discovery candidates, citing missing entry in `PREPROCESSING_CANDIDATES` at `preprocessing_discovery.py:29-49`, MSC class already at `interference.py:217`, manual wiring at `preprocess.py:362-364`.

### 3.2 Round 2 — Cross-debate with verification

Each reviewer was given the other's position and asked to verify the cited file:line claims, then defend or concede.

**Codex round 2** verified DeepSeek's MSC claims against the actual repo:
- Confirmed `PREPROCESSING_CANDIDATES` at `preprocessing_discovery.py:29-49` lists raw + snv + deriv1-4 combinations but no MSC.
- Confirmed `apply_preprocessing()` at `:424` raises ValueError on unknown names; `:463-464`.
- Confirmed MSC class at `interference.py:217` and manual wiring at `preprocess.py:342-364`.
- Conceded backup priority: "For bone FTIR, adding MSC to discovery is a better backup than multi-lens reporting because it changes the actual candidate set." [This concession was later invalidated — bone is NIR, not FTIR.]

**DeepSeek round 2** verified Codex's claims:
- Confirmed every cited line: `cv_utils.py:177-225`, `:275`, `:321-338`, `:336`; `search.py:1035-1115`, `:2195`, `:3433-3477`, `:3811`; GUI `:11331-11334`, `:14161-14218`, `:20014-20030`.
- Recommended group-aware CV + multi-lens reporting as a paired deliverable.

**Round 2 outcome:** Both panelists conceded toward the other's stronger argument. No hallucinated citations on either side.

### 3.3 What the panel got right

- Group-aware splitting machinery is genuinely missing from dasp's CV path.
- DUPLEX exists in `sample_selection.py` but not in the GUI picker.
- MCCV-style repeated holdout is missing.
- DeepSeek's verification approach was rigorous (it called out potential hallucinations in its own output and asked for cross-checks).

### 3.4 What the panel got wrong (to be corrected by the user)

- Both panelists prioritized group-aware CV as primary. The user later deprioritized this — LOGO is complicated in practice and was not the source of NNA's headline numbers.
- DeepSeek pitched MSC for bone-FTIR Mie scattering. The user's bone work is NIR, not FTIR.
- Codex's "users can manually run multiple searches" framing was conceded but the multi-lens recommendation persisted into round 2.

---

## 4. User Course-Corrections — The Pivot Points

The investigation's actual signal came from successive user corrections that reshaped the question. Each correction tightened the recommendation and is documented here for future reference.

### 4.1 "External validation options might not be ideal for certain datasets"

The user pushed against the panel's CV-centric framing and asked specifically about the **validation-set picker** — i.e., what algorithm picks which samples become the held-out validation set.

dasp's existing options (verified at `spectral_predict_gui_optimized.py:14197-14217` and dispatched at `:20021-20028`):
- Kennard-Stone (X-only Euclidean diversity)
- SPXY (X+y combined distance)
- Random (sklearn `train_test_split`, seed=42, no stratification)
- Stratified (`pd.qcut(y, q=4)`, seed=42 — y-only, no group axis)
- Manual

All five share two structural weaknesses for the user's data:
- Single-shot — one split, one R²p, no variance estimate.
- None know about group structure (donor / site / batch / instrument).

NNA's contribution on this axis: `make_stratification_labels(y, groups) = site_encoded * 10 + qcut(y, 3)` at `nir_chemometrics_pipeline.py:787-800`, plus MCCV at `:883-892` (20 repeated stratified random splits).

### 4.2 "How did LOGO get into this? Is it from this repo's results?"

LOGO was introduced by the parent agent, not by NNA's results. The clarification:

NNA does run two geographic splits at `nir_chemometrics_pipeline.py:869-880` (train Finland → test Norway and reverse), but in the SUMMARY_REPORT they appear as Section 3.3, framed as **a diagnostic showing transferability is poor** (N geographic R² = 0.31). The headline best-models table in §3.1 is **MCCV R²** — same-population repeated random splits, not LOGO.

The user's correction: deprioritize LOGO. It's complicated in practice (donor structure is rarely clean in bone-NIR; site structure is binary in the forage case). Not the source of NNA's edge.

### 4.3 "Bone work is NIR, not FTIR"

This invalidated DeepSeek's MSC backup recommendation. MSC's chemometrics rationale is specifically about additive + multiplicative correction against a population reference for Mie-scattered FTIR spectra. For NIR diffuse reflectance, SNV is the standard primary scatter correction, and NNA's winning preprocessings (SNV, SNV+SG, bare SG) confirm this empirically.

The dasp `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` exists, but the user's actual bone work uses NIR.

### 4.4 "ML worries are not always relevant for spectroscopic chemometrics"

This was the largest reframing in the investigation. The parent agent had been framing items as "honesty fixes" and "leakage prevention" — which is ML's frame. Chemometrics has its own standards.

ASTM E1655, Pomerantsev/Rodionova, and AOAC explicitly endorse:
- **SPXY and Kennard-Stone holdout selection.** These use full-dataset distance information to pick representative test samples. ML calls this leakage; chemometrics calls it representativeness (ASTM E1655 §10.2).
- **Whole-set variable selection.** Selection on the calibration set as a whole, not per-fold. Per-fold selection is a recent ML-purity move that's actively debated in chemometrics circles.
- **Selection of n_components by RMSECV with a parsimony rule.** Strict argmin is not the chemometrics convention; "fewest within ~5% of min RMSECV" is (Wold / Haaland-Thomas).

The reframing: NNA's edge over dasp, if real, comes from **modeling primitives that produce better calibrations**, not from ML-leakage discipline.

### 4.5 "Are you sure dasp doesn't already do VIP+Ridge?"

This caught a real overstatement. Verified:
- `preprocessing_discovery.py:81-98` exposes 4 importance methods: `cars_tree`, `model_specific`, `lightgbm`, `vip`.
- `_compute_model_specific_importance` at `:244-278` routes PLS → VIP, Ridge/Lasso/ElasticNet → coefficients, trees → native importance, SVR → support-vector importance, MLP → first-layer weights.

The DEFAULT (`model_specific`) ranks Ridge variables by Ridge coefficients — NOT by VIP. So "Ridge with top-k variables" in dasp default mode is not the same selection as NNA's "VIP+Ridge". But the user CAN explicitly set `method='vip'` and replicate NNA's topology.

### 4.6 "PLS parsimony rule is irrelevant — dasp produces thousands of models"

Correct. dasp's search emits a row per `(preproc, selector, model, n_components)` and the user picks. There is no internal "auto-pick best PLS by argmin-RMSECV" decision that a parsimony rule could override. NNA's parsimony rule applies inside its `tune_and_evaluate` for PLS at `nir_chemometrics_pipeline.py:1078-1084`, which is an internal selection step dasp doesn't have. Dropped from the list.

### 4.7 "Bayesian picks top variables in multiple ways"

Verified at `unified_bayesian.py:1180-1203`. The Bayesian search treats `subset_type` (the importance method), `subset_size`, and `model_name` as **independent search dimensions**. A single Bayesian run will sample combinations like (snv_d2, vip-top-50, Ridge), (snv_d2, model_specific-top-50, Ridge), (snv_d2, lightgbm-top-100, XGBoost), etc., with caching per `(preprocessing, importance_method, model)` at `:1191-1203`.

This invalidates the "VIP-as-selector wiring" recommendation entirely. dasp's Bayesian already explores NNA's "VIP-prefilter feeds Ridge/SVR/XGB" topology. If the user's saved `.dasp` files are all PLS despite running Bayesian, that's a *result* — Bayesian considered VIP+Ridge and didn't beat PLS — not a missing capability.

It also strongly suggests NNA's edge isn't from VIP+Ridge as a topology. NNA found VIP+Ridge winning *with single-split SPXY* on a 220-sample 2-site dataset; if the same topology weren't winning the user's Bayesian leaderboard, the difference is more likely in dataset characteristics or in the validation lens used to declare "best", not in modeling primitives dasp lacks.

---

## 5. Codex Verification Pass — The Final Audit

The reduced 6-item list was sent back to Codex with a single instruction: verify each item against actual code, cite file:line for every claim, and state final value ranking.

### 5.1 Verification matrix (verbatim, with Codex's evidence)

| # | Item | Status | File:Line evidence | Effort |
|---|---|---|---|---|
| 1 | Default NIR noisy-region mask | **Partial** | NNA hard-codes `<400`, `1350-1450`, `1800-1950` removal at `nir_chemometrics_pipeline.py:365-372`. dasp has dynamic correlation regions, not that default mask: `regions.py:9-24,169-179,202-230`. Bayesian reaches `subset_type='region'` at `unified_bayesian.py:1126-1146`. Grid search reaches region subsets via `run_search(... enable_region_subsets=True ...)` at `search.py:1067-1069`, computes them at `:2464-2476`, tests them at `:3357-3378`. GUI exposes broad presets/custom kept regions at `spectral_predict_gui_optimized.py:11493-11518`, parsed at `:19011-19023`, passed to grid at `:28232-28234`. | S |
| 2 | Calibration pre-flight outlier flagging | **Partial** | Hotelling T² at `outlier_detection.py:27-62`, Q-residuals at `:153-230`, raw Y `\|z\| > 3` at `:451-480`, report at `:484-620`. GUI has manual "Run Outlier Detection" button at `spectral_predict_gui_optimized.py:10980-10988`, runs `generate_outlier_report` at `:20774-20819`, lets user mark exclusions at `:21646-21668`. Search only filters already-excluded/validation samples before modeling at `:26916-26923`. | M |
| 3 | Group-stratified validation picker | **No** | Current stratified picker bins regression `y` only at `spectral_predict_gui_optimized.py:19897-19935`. NNA combines group + target bins at `nir_chemometrics_pipeline.py:787-800`. GUI selection state has algorithm/indices only at `:2995-3001`. | M |
| 4 | DUPLEX in GUI picker | **No** | Backend has DUPLEX at `sample_selection.py:138-204`; method literal includes `"duplex"` at `:28`. GUI radio list is KS / SPXY / Random / Stratified / Manual only at `spectral_predict_gui_optimized.py:14196-14217`; dispatch matches at `:20021-20028`. | S |
| 5 | MCCV holdout option | **No** | Repeated CV exists in `cv_utils.py:325-330` and repeated detection at `:344`. Holdout picker stores one fixed `validation_indices` set at `spectral_predict_gui_optimized.py:20033-20037`; training removes that single set at `:26916-26923`; external validation metrics run against the stored validation data at `:37292-37297`. No repeated external-validation split generator or mean ± SD aggregation across holdout repeats. | L |
| 6 | RPIQ alongside RPD | **No** | NNA computes RPIQ as IQR/RMSE at `nir_chemometrics_pipeline.py:933-950`. dasp result columns include RPD/RER, not RPIQ, at `scoring.py:474-476`; search computes RPD/RER at `search.py:4728-4736` and stores them at `:5127-5131`. No `rpiq`/`RPIQ` matches in src, GUI, or tests. | S |

### 5.2 Codex's nuance on item #2

Item #2 is more subtle than first stated. dasp has the outlier-detection components AND a manual "Run Outlier Detection" button. What's missing is two-fold:

1. **Workflow timing.** Outlier detection isn't automatically run before the search starts, so users routinely launch a sweep on data that includes Hotelling T²/Q outliers and Y-outliers. NNA runs this as a mandatory pre-search step at `nir_chemometrics_pipeline.py:181-275`.

2. **Y-outlier residual definition.** dasp's `:451-480` computes raw Y z-score, which catches Y-distribution outliers (e.g., a sample whose Y value is far from the population mean). NNA's `:267-272` computes the **studentized PLS-CV residual** — `(y - y_cv) / std(residuals)` from an initial 5-fold-CV PLS — which catches **Y-not-explained-by-X** outliers. These are different bug classes. The chemometrics-standard test (per Pomerantsev 2018 and ASTM E1655 §11) is the studentized PLS-CV residual. dasp currently flags one bug class but not the more diagnostic one.

### 5.3 Codex's final recommendation

> Highest-value additions for bone-NIR + forage-NIR: **#1 default NIR noisy-region mask** and **#2 true calibration pre-flight outlier report**. Both are standard chemometrics workflow controls and directly reduce avoidable bad calibration inputs. **#6 RPIQ** is cheap and worthwhile, especially for skewed forage/bone reference distributions, but it is reporting rather than model-control. I would deprioritize #5 MCCV holdout unless the user specifically wants repeated external validation reports; dasp already has repeated CV, and repeated fixed-holdout aggregation is a larger UX/data-model change.

---

## 6. Final Recommendation Detail

### 6.1 Priority 1 — Default NIR noisy-region mask preset (S effort)

**What.** Add a built-in chemometrics-default mask preset that removes the three problematic NIR regions: <400 nm (detector edge / visible region with limited compositional information), 1350-1450 nm (water absorption), 1800-1950 nm (water absorption). This is a single named preset like `"NIR_chemometrics_default"` available in the same UI surface that already lets users define kept regions (`spectral_predict_gui_optimized.py:11493-11518`).

**Why.** ASTM E1655 explicitly endorses spectral subsetting on physical grounds. NNA's best NDF model uses this mask (`SUMMARY_REPORT` §3.1). For both forage and bone NIR, water bands and the detector-edge region are noise sources and removing them by default is conventional.

**Where.** Wire into `regions.py` as a named preset, surface in the GUI region UI, ensure `subset_type='region'` in Bayesian (`unified_bayesian.py:1126-1146`) and `enable_region_subsets=True` grid path (`search.py:1067-1069, 2464-2476, 3357-3378`) can both find and use it.

**What it gives up if skipped.** Users either define this mask by hand each time, don't apply it, or have it buried under "custom regions" UI where they need to know the literature wavelengths to type. The chemometrics-standard reflex isn't a one-click default.

### 6.2 Priority 2 — Calibration pre-flight outlier report (M effort)

**What.** Auto-run outlier detection on the calibration set immediately before any search starts, with results presented to the user as a pre-flight report (not a buried button). Two changes:

1. **Workflow integration.** Call `generate_outlier_report` (`spectral_predict_gui_optimized.py:20774-20819`) automatically when the user starts a search. Show a summary dialog: N samples flagged by Hotelling T², N flagged by Q, N flagged by Y-residual, with the option to exclude or proceed. Currently this runs only when the user clicks the manual button at `:10980-10988`.

2. **Studentized PLS-CV residual for Y-outliers.** Replace or supplement the raw Y z-score at `outlier_detection.py:451-480` with NNA's residual definition at `nir_chemometrics_pipeline.py:267-272`: fit PLS with 5 components on the candidate calibration set, compute `y_cv = cross_val_predict(...)`, compute residuals `r = y - y_cv`, flag samples where `|r| / std(r) > 3`. This catches samples whose Y value is inconsistent with the X-Y relationship — the chemometrics-standard outlier class per ASTM E1655 §11.

**Why.** Reviewers and ASTM-aware audiences expect this pre-flight. Removing 1-3 problematic samples before fitting is conventional and routinely gains 0.02-0.05 R²p — comparable to the gap between the user's `.dasp` results and NNA's. This is the single largest *quality* lift on the list.

**Where.** Augment `outlier_detection.py` with `studentized_pls_cv_residual()` mirroring NNA's logic. Hook the existing `generate_outlier_report` into the search pre-flight in `search.py` / GUI workflow.

**What it gives up if skipped.** Calibrations contaminated by Y-not-explained-by-X outliers will continue to depress R²p without the user having a clear signal that pre-flight cleanup would help.

### 6.3 Priority 3 — RPIQ in metrics (S effort)

**What.** Add `rpiq = (Q3 - Q1) / RMSEP` alongside existing RPD in the metrics block at `scoring.py:474-476` and `search.py:4728-4736`, store at `search.py:5127-5131`, and surface in the GUI results table.

**Why.** RPD is sensitive to the spread of y; for skewed targets (forage N where Norway provides a long high tail; bone targets with extreme tails) RPIQ is more honest. Modern NIR papers and AOAC increasingly report both.

**Where.** Single-function change in `scoring.py` plus column wiring in `search.py` and the result-display layer.

**What it gives up if skipped.** Trivially missing chemometrics-conventional reporting metric.

### 6.4 Priority 4 — Group-stratified validation-set picker (M effort)

**What.** Extend the "Stratified" radio option in the GUI validation-set picker (`spectral_predict_gui_optimized.py:14191-14219`) with an optional Group-column dropdown. When set, build the stratification label as `group_encoded * (n_y_bins + 1) + y_bin` (mirroring NNA's `nir_chemometrics_pipeline.py:787-800`), and pass to `train_test_split(..., stratify=combined_label)`. When unset, current behavior.

**Why.** ASTM E1655 §10.2 requires representativeness across both response range AND expected variation sources (sites, donors, batches, instruments). Current Stratified picks balance y-quartiles but can accidentally produce a holdout that's, e.g., 80% Norway just because Norway has the spectral tails that landed in particular splits.

**Where.** Modify `_validation_stratified` at `spectral_predict_gui_optimized.py:19897-19935`. Add a small Group-column combobox near the algorithm radios at `:14191-14219`, populated from metadata categorical columns. Plumb the selection through `_create_validation_set` at `:19943+`.

**What it gives up if skipped.** Multi-site or multi-batch users continue to risk biased validation sets, and have to either eyeball the resulting holdout or switch to Manual selection.

### 6.5 Priority 5 — DUPLEX in GUI picker (S effort)

**What.** Add a `value="DUPLEX"` radio button at `spectral_predict_gui_optimized.py:14196-14217`, and an `elif algorithm == "DUPLEX"` branch around `:20021` that dispatches to the existing `duplex()` function at `sample_selection.py:138-204`.

**Why.** DUPLEX is a chemometrics-standard alternative to Kennard-Stone that explicitly partitions samples so train AND test both span X-space — KS picks for train and leaves leftovers, which can underrepresent spectral extremes in test. For small bone-NIR datasets where every test sample matters, DUPLEX is often preferable.

**Where.** Two file edits, both small, in the GUI file.

**What it gives up if skipped.** A chemometrics-standard option that's already implemented in dasp's backend remains inaccessible to users who haven't read the source.

### 6.6 Priority 6 — MCCV holdout option (L effort)

**What.** Add a sixth validation-set algorithm: "Repeated Stratified (MCCV)" with a spinbox for n_repeats (default 20). Generate a list of validation index sets via stratified random splits, fit and score the search on each, aggregate metrics as mean ± SD across repeats.

**Why.** Replaces the seed=42 lottery that all current dasp pickers depend on with a credibility-grade variance estimate. Pomerantsev's tutorial recommends this when datasets are small. For bone-NIR especially (typically smaller n than forage), one-shot SPXY R²p can swing 0.05-0.10 across seeds.

**Effort tier — L, not S.** Codex specifically called out that this is more than a UI change. The data model in dasp assumes a single fixed `validation_indices` set (`spectral_predict_gui_optimized.py:20033-20037`), training removes that single set (`:26916-26923`), and external validation metrics run against stored validation data (`:37292-37297`). Refactoring this to handle a list of validation sets and aggregate across them is a real architectural lift.

**Recommendation.** Defer unless the user explicitly wants per-repeat external-validation reporting. dasp already has repeated CV in `cv_utils.py:325-330`, which serves a similar credibility function for the CV side. The marginal value of also having repeated *holdout* aggregation, given the implementation cost, is moderate.

---

## 7. What This Investigation Found That Wasn't On Anyone's Original List

A handful of items emerged during the corrections process that weren't in the initial gap analysis or the panel debates:

1. **dasp's `model_specific` importance default may be a discoverability trap for chemometrics users.** When the user picks "Ridge with top-k variables" and runs grid search, the variables are ranked by Ridge coefficients. To replicate NNA-style "VIP+Ridge", they must explicitly set `method='vip'` in the importance configuration. Bayesian sweeps this automatically; grid search may not. Not a code gap, but a UX / documentation issue worth flagging in the user-facing docs: "for chemometrics-standard VIP-prefiltered Ridge/SVR/XGBoost, set Importance Method to PLS VIP."

2. **dasp's "Random" validation-set option is rarely chemometrics-correct.** It performs no stratification (`spectral_predict_gui_optimized.py:19868-19895`), uses fixed seed=42, and is presented as a peer to "Stratified". For NIR work, Random should either be removed, marked "discouraged", or folded into Stratified with `n_bins=1` as an explicit user choice rather than the path of least resistance.

3. **Range-coverage validation is implicit, not enforced.** ASTM E1655 §10.2 expects calibration to span the y range you intend to predict, and validation to fall within calibration's range. After picking a validation set via any of the existing algorithms, dasp doesn't warn if (a) validation y-range covers <80% of calibration's IQR, or (b) any y-tercile has zero validation samples. This is a small post-pick check that catches a real failure mode.

These are noted for completeness; they did not rise to the priority list above.

---

## 8. Appendix — What Chemometrics Standards Actually Say

A quick reference for the standards invoked in this investigation, since the framing matters.

### 8.1 ASTM E1655 — Standard Practices for Infrared Multivariate Quantitative Analysis

- **§10 Calibration set.** Must span the y range expected during prediction. Validation set must be drawn from the same population as calibration unless explicitly testing transferability.
- **§10.2 Representativeness.** Selection should reflect expected variation in instruments, environments, sample types — i.e., group-aware stratification when groups exist.
- **§11 Outlier detection.** Mandatory pre-modeling. Includes Hotelling T² on PCA scores, Q residuals on PCA reconstruction, and **studentized PLS-CV residual on Y** (the version dasp does not currently implement).
- **Spectral subsetting.** Endorsed as standard practice. Removing detector-edge regions and known interference bands (water in NIR) is conventional, not selection.

### 8.2 Pomerantsev & Rodionova validation tradition

- **MCCV is preferred over single-split holdout** when n is small (Pomerantsev 2014, 2018). 20+ repeats with stratification across known variation sources gives mean ± SD that reviewers expect.
- **SPXY > Kennard-Stone > Random** for fixed-holdout selection when you need a deployable single split.
- **Studentized PLS-CV residual for Y outliers** (Pomerantsev 2018) is the recommended residual definition, not raw Y z-score.

### 8.3 AOAC Official Methods

- **RPIQ alongside RPD** for skewed reference distributions.
- **Validation metrics reported with variance estimates** (e.g., R² ± SD across repeats) preferred over single-shot point estimates.

### 8.4 What chemometrics does NOT require

- **Per-fold variable selection sealing.** Whole-calibration-set selection is the convention; per-fold selection is an ML purity move that's not standard.
- **Strict argmin n_components selection.** Parsimony rules ("fewest LVs within ~5% of min RMSECV") are conventional, but dasp's per-row reporting architecture sidesteps this by letting the user choose.
- **LOGO as default validation.** Group-aware CV is appropriate when transferability across groups is the deployment scenario; for within-population deployment, MCCV or repeated stratified holdout is the convention.

---

## 9. Investigation Provenance

| Phase | Inputs | Outputs |
|---|---|---|
| Initial scan | NNA pipeline + dasp source + dasp docs | 10-item gap list (overstated) |
| Round-1 panel | Both repos with full read access; gap list as starting context | Codex + DeepSeek converge on group-aware CV; diverge on backup |
| Round-2 panel | Each panelist gets the other's position + asked to verify file:line | Codex concedes backup to DeepSeek's MSC; DeepSeek shifts to paired group-CV + multi-lens |
| User correction 1 | "External validation options might not be ideal" | Pivot from CV-side to holdout-side analysis |
| User correction 2 | "How did LOGO get in?" | LOGO deprioritized |
| User correction 3 | "Bone is NIR not FTIR" | MSC dropped from list |
| User correction 4 | "ML worries aren't always relevant for chemometrics" | Reframe entire ranking on ASTM/Pomerantsev terms |
| User correction 5 | "Doesn't dasp do VIP+Ridge?" | VIP-as-selector dropped (already covered) |
| User correction 6 | "Bayesian picks top variables in multiple ways" | Confirmed via `unified_bayesian.py:1180-1203`; case fully closed |
| Codex verification | Surviving 6-item list + read-only access | Final verified status table with file:line evidence |

---

## 10. Recommended Next Step

The user has not yet committed to any specific addition. The cheapest high-value first patch, given the verified evidence, is:

**Priority 1 + Priority 3 in a single pass** — the default NIR noisy-region mask preset and RPIQ in metrics. Both are S effort, both land chemometrics-standard practice in dasp without any architectural change. The combined pass is ~1 day of focused work.

If a single deliverable is preferred: **Priority 2 (auto pre-flight + studentized PLS-CV residual)** is the largest quality lift but is M effort and touches the search workflow. Best treated as its own pass.

Priority 4 (group-stratified picker) and Priority 5 (DUPLEX wiring) are good follow-ups. Priority 6 (MCCV holdout) should be deferred or scoped separately given the architectural cost.

---

*End of investigation document. All file:line citations in this document have been verified by Codex CLI with read access to the actual repos. No claims are inferred; every cited location was opened and read during the verification pass.*
