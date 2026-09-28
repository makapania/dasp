> Generated 2026-09-28 by a Claude multi-agent review (7 specialist lenses + synthesis) of `main` at `449dfb1`. Scratch scripts/screenshots/timings it cites lived in a local temp dir and are NOT in the repo; re-run them if you need the numbers. Companion bug list: `docs/reviews/2026-09-28-adversarial-review.md`. Combined fix order: PROJECT_STATUS §0.

# dasp improvement roadmap

*Built from seven specialist reviews: calibration transfer, contamination, modelling workflow, speed, GUI usability, GUI visuals and architecture. Prepared for Matt, 2026-09-28.*

## How to read this

Over the last few months most of the effort went into the Bayesian search. The reviews agree that the science modules around it now lag behind: calibration transfer, contaminant removal and one-class screening. They also agree that some of what those modules show on screen is misleading, not just unpolished. The roadmap tackles that first, then the missing validation design (grouped CV) and the "can I defend this in a paper" layer. Speed and appearance come in as cheap wins alongside.

Effort: **S** is a day or two, **M** is about a week, **L** is a multi-PR series, **XL** is a long campaign. Impact is judged for this group's real use (small-n bone, FTIR, leaf NIR, Glyptal) and for any future release.

---

## 0. Claims I checked myself

I re-read the code at the cited lines for the five claims with the biggest consequences. All five hold.

| # | Claim | What the code shows |
|---|---|---|
| 1 | **Nested thread oversubscription**: boosters use every core inside a CV fold loop that also uses every core | `models.py:636-744` builds RF/XGB/LGBM with `n_jobs=-1`. `search.py:3184` passes `n_jobs_cv = n_jobs_default` (= -1, `search.py:1580`) for every model not in `MODELS_PREFER_SERIAL_CV` (`search.py:174`), into `Parallel(n_jobs=n_jobs_cv, backend="loky")` (`search.py:4935`). `threadpool_limits` is used nowhere in `src/`. *I did not re-run the timings; the reviewers measured on a machine other agents were loading.* |
| 2 | **The default "EPO" contaminant correction does not remove the contaminant** | `EstimatedEPO` defaults to `estimation_method='pca_diff'` (`contaminant_analysis.py:466`). `pca_diff` builds its library as the mean difference plus 10% noise copies of it (`:636-645`). `_build_projection_matrix` then subtracts the library mean (`:677-679`), which removes the difference itself and leaves only the noise to project out. The GUI builds `EstimatedEPO(n_components=...)` with that default (GUI ~`:60168`). `transform` returns mean-centred spectra (`:763-766`). |
| 3 | **The Kennard-Stone/SPXY holdout puts the KS-selected samples into validation** | GUI `:20859-20861` gets `selected_indices` from `_validation_kennard_stone`/`_validation_spxy`, and `:20871-20873` stores them as `self.validation_indices` / `validation_X`. |
| 4 | **Calibration Transfer misleads** | TSR uses the first n rows (`transfer_indices = np.arange(n_samples)`, GUI `:49350`). The quality R² is `r2_score` on the ravelled primary vs transferred values of the standards used for the fit (`:48339-48344`). The default method is `'nspfce'` (`:54854`). The guide recommends DS for fewer than 10 standards and "Feature-based matching methods" (`:54644-54647`). |
| 5 | **Contaminant and interference corrections never reach a CV fold or a saved model** | `interference_settings` is commented out at the `run_search` call ("DISABLED: Code stashed (broke R² reproducibility)", GUI `:30853`), yet `apply_to_analysis` defaults to `True` (`:2901`). "Apply to Main Dataset" overwrites `self.X` in place (`:60088-60104`). `_contam_export_corrected_spectra` is a placeholder (`:60262`). |

I also confirmed three more claims while checking:
- Grouped CV raises `NotImplementedError` (`cv_utils.py:221-225`).
- `PCASIMCA` fits its chi² limits on the training T² and Q (`contamination.py:~158-167`) and combines them with Fisher's method (`:263-294`).
- The Bayesian classification objective still runs a second `predict_proba` CV pass (`unified_bayesian.py:~1928-1936`), although the comment at `:1911` says that pass was removed.

**Caveats on the evidence.**
- The calibration-transfer RMSEP numbers come from a **simulated** satellite instrument built from the 49 example spectra: a 6 nm shift, 8 nm blur, gain tilt, curved baseline and added noise. The ranking of methods is informative. The absolute values would change on a real second instrument.
- Speed figures were measured on a shared 24-thread machine. The before/after ratios are reliable; the absolute seconds are not.
- Three references were flagged by their reviewers as not fully checked: Zeaiter et al. 2005 (cited from memory), the Folch-Fortuny 2017 author list, and the Mikulasek 2023 volume and pages. Check them before citing.

---

## (a) Quick wins: small effort, noticeable benefit

| ID | What | Why it matters to you | Evidence | Impact | Effort | Depends on |
|---|---|---|---|---|---|---|
| **QW1** | **One thread-budget rule.** Models run single-threaded inside parallel CV folds (`n_jobs=1`, CatBoost `thread_count=1`). The outer pool is `min(n_splits, physical cores)`. One-class CV and `simca._cross_fit_null` are wrapped in `threadpool_limits(1, user_api='openmp')`. Tiny jobs run serially. One helper holds the rule (`resolve_threads` / `parallel_policy`). Merges three reviewers' ideas (speed, contamination, architecture). | Booster grids go from "leave it overnight" to minutes. This probably explains the "CatBoost trial runs 20+ min" complaint. Pause responds sooner, and the GUI test suite gets faster too. | Spot-check #1. One LightGBM config in `run_search`: 25-28 s as shipped vs 0.35-0.47 s with model `n_jobs=1`. One-class LOF grid: 47.6 s vs 0.96 s with the OpenMP cap; the "LOF is O(n²)" known issue is a misdiagnosis. `ipls_selection` first call: 6.0 s vs 0.12 s. XGBoost 60×30 refit test: 14.9 s vs 0.11 s. | **High** | S | none |
| **QW2** | **Stop the Calibration Transfer tab misleading.** Rename CTAI to "PC-DS (truncated-SVD DS)" and NS-PFCE to "Iterative ridge DS". Hide JYPLS-inv or mark it experimental with a hard error when y is missing. Make TSR the default. Rewrite the guide and tooltips (drop "Feature-based", drop "DS for <10 standards", drop "No transfer samples required"). Fix the TSR citation (use Shenk & Westerhaus 1991, *Crop Sci* 31(6):1694-1696). Relabel the R² on the quality plot "fit to standards (not a validation)" until F2 replaces it. | A reviewer who looks up the CTAI or PFCE reference currently finds a different algorithm. The default method (NS-PFCE) was worse than no correction with 3 standards. | Spot-check #4. `calibration_transfer.py:696-964` (CTAI is a paired, truncated DS), `:1119` ("Literature reference needed"), `:1428-1663`. Simulated RMSEP with no correction 3.91: dasp-CTAI 3.73-4.49; JYPLS-inv 9.4-11.7; NS-PFCE 4.26 with 3 standards. Refs: Zhao et al. 2019 *Molecules* 24(9):1802; Zhang et al. 2021 *Anal Chim Acta* 1142:169-178. | **High** | S | none |
| **QW3** | **Fix the EPO contaminant-removal maths.** Do not centre the interferent library. Make `mean_diff` the default and rewrite or remove `pca_diff`. Return `X @ P` uncentred so the output is still a spectrum. Remove the OPLS-DA filter from "Apply Correction" or make it remove the *predictive* component; today it keeps the contaminant and strips the analyte. Add behavioural tests: a synthetic band is removed to <5% while >90% of the analyte signal is retained. | Today "Apply EPO correction" to Glyptal-treated bone leaves about 99.8% of the Glyptal difference in place. | Spot-check #2. `epo_check.py`: fraction of the contaminant direction removed is 0.045 for `pca_diff` vs 0.997 for `mean_diff`. Residual group difference 3.879 of 3.886. Influence map correlates 0.04 with the true band. The current tests only check array shapes (`tests/test_contaminant_analysis.py:225-270`). Ref: Roger, Chauchard & Bellon-Maurel 2003 *Chemom Intell Lab Syst* 66(2):191-204. | **High** | S | none |
| **QW4** | **Fix the holdout-split direction.** Route the split through `sample_selection.kennard_stone/spxy/duplex`. KS/SPXY choose the **calibration** set and the remainder validates. Compute distances on preprocessed spectra or PCA scores, not raw X. Add stratified KS and DUPLEX. | External validation stops forcing extrapolation, and the split follows the convention reviewers expect. | Spot-check #3. On the example data (10 held out) with the current direction, 2 validation samples fall outside the calibration y range and RMSEP is 3.47. With the textbook direction RMSEP is 2.14, against a mean of 2.34 over 200 random splits. Refs: Kennard & Stone 1969 *Technometrics* 11(1):137-148; Galvão et al. 2005 *Talanta* 67(4):736-740. | **High** | S | none (group-aware later via F1) |
| **QW5** | **Full figures-of-merit function.** Add `scoring.regression_figures_of_merit(y, yhat)` returning bias + t-test, SEP/SECV, slope and intercept + test, RPIQ, RPD, RER and CCC. Call it for calibration, CV **and** test. Add RPIQ and SEP as leaderboard columns. | These are the numbers ISO 12099 and journal reviewers ask for, and the bias and slope tests say when a slope/bias correction is justified. External validation currently stores only RMSEP and R²pred (`search.py:1186-1189`). | Example data: SECV 2.40, RPIQ 4.88, slope 0.949 (none of these is currently shown). Ref: Bellon-Maurel et al. 2010 *TrAC* 29(9):1073-1081 (in Paperpile). | **High** | S | none; feeds F5 |
| **QW6** | **Remove or label controls that do nothing.** Grey out the Interference "Method Configuration" EPO/DOSC controls and set `apply_to_analysis` to False with a "not applied during analysis" banner. Make "Export Corrected Spectra" work (it is a few lines) or remove it. Disable the one-class preprocessing-importance dropdown. Remove the dead `use_msc` var. | Every visible control should do what it says. Today you can tick EPO, run, and get results computed without it. | Spot-check #5. `use_msc` is created at GUI `:3339` and never read. | Medium | S | none |
| **QW7** | **DPI awareness and a correct font.** Call `SetProcessDpiAwareness(1)` before `tk.Tk()` and scale the pixel constants; add a DPI-aware manifest to the spec. Fix `('Segoe UI','Arial')`: Tk does not treat a tuple as a fallback list, so every ttk widget renders in Arial. Use 5 named fonts. | This is the most visible "looks professional" change. The UI is currently bitmap-stretched and blurry on 125% displays, with two typefaces on every screen. | The process reports DPI awareness 0; Tk sees a 2048×1152 screen at 96 dpi. `tkfont.Font(font=(('Segoe UI','Arial'),10)).actual()` returns Arial (GUI `:4665-4825`). A before/after crop is in `gui-visual/hidpi_before_after.png`. | **High** | S | none |
| **QW8** | **Flatten cards and let pages fill the window.** Cards become one frame with a 1 px outline and no nested shadow/accent frames, plus a `Card.*` ttk style so labels stop painting grey boxes. One `ScrollableFrame` class that tracks the window width replaces the 30 hand-rolled canvases. | Calmer screens, and no more empty right half on a maximised window. | GUI `:5339-5374` (three nested frames). 30 `create_window((0,0)…)` calls vs 1 width binding (`:52432`). | Medium | S | none |
| **QW9** | **Help and wayfinding that match the app.** Rewrite Quick Start from the real click path. Replace about 21 stale "Tab 11A"-style strings with page names held in a constant map. Pick one product name (the window says "ASP", GUI `:2672`). Fix the two Chapter 4s. Remove the README's CLI/`--target`, v0.1.0 roadmap and "30 tests" material. Correct CLAUDE.md's "7 tabs". | New users can trust the help. | `docs/UserGuide.md:181-280` describes actions that do not exist (a single click opens a plot; Export → .pkl/PDF). | Medium | S | ideally after QW2 and F2 for Ch. 4 |
| **QW10** | **Faster test loop.** Set `addopts = -m 'not comprehensive and not slow'`. Move the 34 comprehensive tests, which are backend benchmarks that never touch GUI code (`harness.py:454`), to a nightly job. Set `OMP_NUM_THREADS=1` in the test conftest. Snapshot and restore all 771 Tk vars around each GUI test. | Every PR waits on the suite: about 38 min today, likely under 10 min after this. Order-dependent GUI failures go away. | 264 fast GUI tests took 768 s, and the top 4 tests took 500 s of that (XGBoost refit 254 s on 60×30 data). | **High** (for development speed) | S | pairs with QW1 |
| **QW11** | **Bayesian classification: one CV pass per trial.** Fit each fold once and call both `predict` and `predict_proba`. Move the per-trial full refit out of the objective and do it for the top-K trials afterwards. | Classification Bayesian runs (CollagenCat-style) take about half the time; regression about 15% less. | Verified: `unified_bayesian.py:1911` comment vs the second pass at `:1936`. Measured 2 `cross_val_predict` calls per trial and 11→6 fits. | Medium | S | none |

---

## (b) Flagship improvements: one PR series each

### F1. Specimen- and site-aware validation (grouped CV everywhere)

**What.**
- A "Specimen/Group ID" column picked at Import, stored aligned to X.
- A frozen `CVPlan(strategy, n_folds, n_repeats, seed, groups)` object with `.split()`, `.inner()` (the nested plan that keeps group labels) and `.min_train_size()`. It is passed through `run_search`, `unified_bayesian`, NSGA-II, the one-class search, the selectors, predictor screening and smart preprocessing.
- It replaces the 36 hard-coded `KFold`/`StratifiedKFold` sites. Once they all take a plan, add `GroupKFold`, `StratifiedGroupKFold` and `LeaveOneGroupOut` in one place.
- Add "average replicates by specimen" and whole-group holdout (KS on group-mean spectra).
- A grep test fails on any new `KFold(` outside `cv_utils`.

**Why it matters.** Every real dataset in the logs has replicate or site structure: Border Cave, the multi-site ATR-FTIR bone paper, leaf NIR. When replicates of one bone fall on both sides of a fold, R²cv is inflated, and today the only fix is hand-written Python (AGENT_COMPOSITION §4). The same work closes "CV Strategy Phase 2": your chosen CV currently never reaches the variable-selection inner loops, which are hard-wired to 5-fold. It also gives contaminant correction (F3) and stability selection a place to run per fold.

**Evidence.** `cv_utils.py:221-225` (verified); `build_cv_splitter` knows only kfold/repeated/LOO (`cv_utils.py:275-340`). 42 `KFold(`/`LeaveOneOut(` hits in `src/`, 6 of them in `cv_utils`. GAP_ANALYSIS.md #1/#12 ("largest single gap").

**Impact / effort.** High / L (roughly 3 PRs: CVPlan plus the grid path and group column; selectors and Bayesian; NSGA-II and one-class).
**Dependencies.** Easier after ST1 (RunRequest), but it does not need it. Unblocks F4's leave-one-contaminant-group-out check, F5's block bootstrap, and group-aware stability selection.

### F2. Calibration transfer that tells you whether it worked

**What.** Four linked pieces:
1. **Validate** by leave-one-standard-out. The table shows RMSEP (in %Collagen) with no correction vs after transfer, bias, slope and SEP; spectral RMSE when there is no y; class agreement or inlier rate for classifiers. Plots: residual spectra of the standards over the model's b-vector, T²/Q of the transferred spectra in the model's space, and difference spectra before and after on shared axes. The resubstitution R² goes.
2. **Compare methods.** A bake-off leaderboard with a "no correction" baseline always shown. Candidates: TSR, regularised PDS over a small window grid, dual-form DS, **slope/bias of predictions from satellite standards**, bias-only, TOP/EPO, and augmenting the calibration set. Double-clicking a row builds and saves that transfer. A three-question situation picker replaces the static guide, and only the selected method's parameters are shown.
3. **Standards done right.** Pair by **sample ID** (the ID-preserving loader already exists at GUI `:48656`) with matched/unmatched lists. A "Plan transfer standards" step suggests which specimens to rescan (KS/SPXY from `sample_selection.py`), shows them on a PCA plot and exports a CSV checklist. Delete the dead `_build_ct_transfer_model` / `_load_ct_paired_spectra`, about 400 lines with no callers.
4. **Store the transfer in the `.dasp` file** as per-instrument adapters (transfer model and/or slope/bias, validation RMSEP, n standards, date), with an Instrument dropdown in Model Prediction. `bias_correction` is already saved and applied (`model_io.py:97, :821-824`). Only its source changes: satellite standards instead of the model's own CV.

**Why it matters.** Your usual case is a handful of re-scannable bones and a small-n model. In the benchmark the simple methods won (TSR 2.62-2.64, slope/bias 2.62-2.71, centred low-rank PDS 2.58-2.61, against 3.91 uncorrected and 2.62 ideal). Meanwhile the tab's default was worse than no correction, and its plot showed R² = 1.00 even when DS raised RMSEP to 4.94. This mirrors the main leaderboard: evidence first, with a visible "nothing helped" row. It also gives you a number to report in a paper, following Workman 2018 *Appl Spectrosc* 72(3):340-365 (success is judged by prediction error on the child instrument).

**Evidence.** Spot-check #4. GUI `:48954-49040` (IDs dropped; pairing by row order), `:49480-49481` (instrument IDs hard-coded as "primary"/"satellite"). UserGuide §4.6 describes a before/after workflow that does not exist. All fits take 0.1-1.5 s at 2151 bands with 12 standards, so a full comparison runs in seconds.

**Impact / effort.** High / M-L (3 PRs: backend `evaluate_transfer()` with PDS/DS fixes and slope/bias adapter; GUI validate and compare panel with ID pairing; `.dasp` adapters and Prediction dropdown).
**Dependencies.** QW2 first. CT1 and CT2 (section c) supply the methods. Merging the three prediction pages (LF8) is easier afterwards.

### F3. Contaminant removal as a validated, saved pipeline step

**What.**
- **ContaminantCorrector**: an sklearn transformer (fixed EPO, EMSC-with-interferent, or real GLSW) fitted **inside each CV fold** from a group column in the main dataset. It is saved in the `.dasp` file and replayed by `predict_with_model`.
- Find and fix the "broke R² reproducibility" bug that caused the interference plumbing to be stashed, instead of leaving it off.
- A **Correction A/B** check: same CV splits with and without correction, RMSECV split by group (clean vs each contaminant), and a paired per-specimen difference.
- Before/after diagnostics on shared axes: group means ± IQR, |contaminated − clean| per wavelength, PCA scores coloured by group, centroid Mahalanobis distance, and an **"analyte retained"** figure (the share of the clean-class or regression subspace that survives the projection, a net-analyte-signal idea).
- A persistent "Correction active: EPO k=2 (Glyptal)" banner with Remove and Undo.
- Later, merge Data Quality screening, Contaminant Analysis, Interference Removal and Spectral Library into one **Screen & Clean** workspace (Groups → Screen → Correct → Use) with a shared interferent library.

**Why it matters.** Glyptal-consolidated bone is the motivating case (UserGuide.md:8652-8655). Today the correction is fitted on all the data and written over `self.X`. That leaks information into CV (optimistic RMSECV), is never applied to new specimens at prediction, and you cannot see whether it helped. With this in place you could write "Glyptal correction lowers RMSECV on consolidated specimens from X to Y at no cost on clean bone" and defend it.

**Evidence.** Spot-check #5. `preprocess.py:345-475` and `search.py:1882-1917` already provide per-fold EPO/DOSC/GLSW steps. `model_io.py` has no contaminant or interference hooks (grep). The groups are reloaded from separate files (GUI `:58606`) instead of taken from the main dataset.

**Impact / effort.** High / L.
**Dependencies.** QW3 (correct maths) and CS1 (EMSC) first. F1's CVPlan helps. ST4 (tab extraction) makes this far cheaper to work on. ST6 (PreprocessSpec) is the tidy home for the step.

### F4. One-class screening you can quote in a paper

**What.**
- Real **DD-SIMCA**: full distance f = N_h·h/h₀ + N_q·q/q₀ ~ χ²(N_h+N_q), an extreme (α) limit and a family-wise outlier (γ) limit, classic and robust estimators, and an option to estimate the parameters from out-of-fold distances. Add a parity test against the MIT `ddsimca` package. Keep the current Fisher variant as "legacy" so saved models still load.
- A **"Rigorous (target class only)"** mode as the default. You fix α; complexity is chosen from inlier-only CV specificity. Contaminated samples are held back for one final sensitivity estimate. Report leave-one-contaminant-group-out sensitivity. The current behaviour stays available, labelled "Compliant (uses alternative class), optimistic".
- **Calibrated out-of-fold p-values for all five engines**, reusing `simca._cross_fit_null`/`_empirical_p` (`simca.py:946-1072`).
- **Acceptance and extreme plots**, and per-contaminant sensitivity and specificity with Wilson CIs (`simca.wilson_ci`, `:1197`).
- The prediction output becomes p-value, regular/extreme/outlier, plus the α and γ used.
- Rebuild the **Data Quality outlier screen** on the same machinery.

**Why it matters.** The rejection of genuine clean specimens has to match α. At α = 0.05 the current PCA-SIMCA rejects 14-36% of held-out clean bone spectra, depending on k. Thresholds tuned on the contaminants at hand also overstate sensitivity to *new* consolidants (0.744 in CV vs 0.683 on fresh ones in simulation). Data Quality currently counts T² and Mahalanobis as two votes when they are the same statistic, and the Q flag always flags about 5% of any dataset.

**Evidence.**
- Checked: `contamination.py:~158-167` fits the χ² limits on the training data, and `:263-294` combines T² and Q with Fisher's method.
- `unified_bayesian.py:964-990` tunes α, contamination and ν on balanced accuracy.
- `outlier_detection.py:221` sets the Q limit at the in-sample 95th percentile; `:589-600` does the voting.
- `model_io.py:979-1011` derives prediction "confidence" from a sigmoid and applicability-domain status from percentiles.

Refs:
- Pomerantsev 2008, *J Chemom* 22:601-609.
- Pomerantsev & Rodionova 2014, *J Chemom* 28:429-438.
- Kucheryavskiy, Rodionova & Pomerantsev 2024, *J Chemom* 38(7):e3556.
- Rodionova, Oliveri & Pomerantsev 2016, *Chemom Intell Lab Syst* 159:89-96.

**Impact / effort.** High / M-L (3 PRs: DD-SIMCA estimator; rigorous mode and OOF p-values; plots and Data Quality rebuild).
**Dependencies.** QW1 (OpenMP cap) makes the extra CV affordable. F1 supplies group-held-out sensitivity. CS4 (Wold modelling power) keeps variable selection one-class.

### F5. Defensible model choice and publication-ready output

**What.**
- **Results that admit ties.** A Summary view with the best row per model × preprocessing × subset family, a "statistically tied with #1" badge (one-SE rule under repeated CV), a "simplest in the tied set" marker, filter chips, "N configurations compared" next to the top score, and a Compare panel for 2-4 rows.
- **Robustness.** On a selected row: N repeated or group-aware splits, or repeated double CV (Filzmoser et al. 2009). For two rows: a paired randomization test on pooled CV errors (van der Voet 1994) with a specimen-block bootstrap CI.
- **Parsimonious LV rule** (min / 1-SE / randomization), with an RMSECV-vs-LV sparkline.
- **Publication report.** A self-contained HTML file (PDF optional) with an auto-written methods paragraph, the QW5 figures-of-merit table for calibration, CV and test, the standard figures (parity with 1:1 line, RMSECV vs LV, residuals, signed b-coefficients with jack-knife CIs plus VIP over the mean spectrum, PCA with the cal/val split), and a per-sample CSV.
- **"Export for publication…"** on every plot: 85/180 mm widths, 7-8 pt text, editable text in PDF/SVG, and a CSV of the plotted data.
- **One matplotlib house style** (`plot_style.py`: constrained layout, Okabe-Ito colours, viridis for continuous values) applied at start-up.

**Why it matters.**
- *Near-ties:* on a Quick run of the example data, the top 8 of 1,915 rows are within 2.5% of each other, and #1 is a 325-variable subset because the parsimony penalties default to 0.
- *Split luck:* a single 10-sample holdout on n = 49 gives an RMSEP anywhere from 1.43 to 3.43 depending on the split.
- *Reporting:* the current report is 156 lines of top-5 Markdown text (`report.py:10-156`), so a paper needs about 10 manual figure exports and hand-written methods text.
- This matches your recorded position that selection bias is handled by an external test or double CV (SESSION_LOG_ARCHIVE.md:4018). rdCV is that route.

**Evidence.** `scoring.py:15-60`; GUI `:2977-2978` (penalties = 0), `:16880-16902` (export keeps the GUI theme background); zero `rcParams` or style calls in the code base; GAP_ANALYSIS #2, #3, #5. Refs: Filzmoser, Liebmann & Varmuza 2009 *J Chemom* 23(4):160-171; van der Voet 1994 *Chemom Intell Lab Syst* 25(2):313-323; Martens & Martens 2000 *Food Qual Prefer* 11:5-16.

**Impact / effort.** High / L (4 PRs: tie-aware Results; robustness and paired test; house style and publication export; report).
**Dependencies.** QW5. F1 for group-aware resampling. ST5 (shared evaluator) makes the out-of-fold predictions uniform across engines, but the grid path can go first.

---

## (c) Method additions

### Calibration transfer

| ID | What | Why | Evidence | Impact | Effort | Depends on |
|---|---|---|---|---|---|---|
| CT1 | **PDS with centring, an offset and low-rank windows** (truncated SVD or local PLS with rank ≤ n−1), with window width and rank chosen by leave-one-standard-out; double-window PDS as an option. Add as options, keeping backward compatibility (`estimate_pds` is on the agent-composition surface). | Performance with 3 standards matches performance with 12, which is the realistic number of re-scannable bones. | dasp PDS gets worse as standards are added (3.46 → 5.08 from 5 to 12). The fixed version holds at 2.58-2.61. `calibration_transfer.py:173-245`. Wang, Veltkamp & Kowalski 1991 *Anal Chem* 63(23):2750-2756. | High | S | none |
| CT2 | **Dual-form DS**, A = Xsᵀ(XsXsᵀ+λI)⁻¹Xp, with λ chosen by leave-one-standard-out. | DS stops being worse than no correction, and fitting becomes cheap enough to cross-validate. | DS at λ = 1e-3 (the GUI default, `:54906`) gives 4.94 with 3 standards vs 3.91 uncorrected. The current p×p fit takes 1.5 s and 37 MB. | High | S | none |
| CT3 | **Slope/bias correction from satellite standards**, written into a copy of the `.dasp`, with a warning when there are fewer than 3 standards or the y range is narrow. | This was the best or equal-best method, and it uses machinery that already exists. | 2.62-2.71 across 3-12 standards. `model_io.py:332-359`. Bouveresse et al. 1996 *Anal Chem* 68(6):982-990. | High | S | part of F2 |
| CT4 | **Model-side options**: TOP/EPO on primary−satellite difference spectra, and augmenting the calibration set with standards, both routed through the existing refit path. | Best in the mild case (2.42-2.59), worse in the severe case, which is why they belong in the F2 bake-off rather than being used blindly. | Andrew & Fearn 2004 *Chemom Intell Lab Syst* 72(1):51-56. | Medium | M | F2, QW3 (EPO class) |
| CT5 | **Different grids handled properly**: use the overlap on the coarser grid, refuse to extrapolate, add "match resolution" using the unused `instrument_profiles.estimate_smoothing_between_instruments` (`:412-455`), retrain on the overlap in one click, and allow the 100-band minimum to be overridden for satellite data. | Moving a benchtop ASD model to a portable or handheld unit (museum and field work) without silently fabricated data. | GUI `:49299` extrapolates linearly. Band minimums at `io.py:140, 316, 773, 1109, 1486`. Riu et al., *Anal Chem*, doi:10.1021/acs.analchem.5c06767 (volume and pages not on page 1). | Medium | M | F2 |
| CT6 | **Check-standard drift chart**: predict a stable reference specimen each session and plot it against ±2·SEP limits, with a one-click rebuild of the slope/bias correction. | Covers the commonest real case: the same instrument after a lamp change, a new sample cup or drift. | Workman 2018. | Medium | M | F2 adapters |
| CT7 | **Transfer-robustness column** in Results: RMSEP on satellite standards without transfer, computed from models that are already fitted, optionally usable as a tie-break. | Lets you pick a model that needs no transfer, or only a slope/bias fix. | Mild case: 2.83 uncorrected vs 2.62 ideal, showing the sensitivity depends on the model. | Medium | M | F2 |
| CT8 | **di-PLS** (domain-invariant PLS) as a model type that takes unlabelled target-domain spectra; it can be implemented natively in about 150 lines. Always shown against the no-correction baseline. | Pooled multi-lab datasets and specimens that cannot be re-scanned, the honest version of what "CTAI, no standards" promises today. | Simple per-instrument mean-centring did not help (3.77). Nikzad-Langerodi et al. 2018 *Anal Chem* 90(11):6693-6701. | Medium | L | QW2 |

### Contamination screening and removal

| ID | What | Why | Evidence | Impact | Effort | Depends on |
|---|---|---|---|---|---|---|
| CS1 | **EMSC with interferent spectra**: x = a + b·m + polynomial + Σ cⱼkⱼ, with kⱼ from the Spectral Library, the uncentred group difference, or residual PCs. Outputs the corrected spectrum **and a per-sample contaminant load cⱼ ± SE**. The same transformer is the missing EMSC preprocessing. | Flags and removes consolidant in one step, and gives an interpretable per-specimen amount of Glyptal for the archaeology write-up. | Not present in `src` (grep). Martens & Stark 1991 *J Pharm Biomed Anal* 9(8):625-635; Afseth & Kohler 2012 *Chemom Intell Lab Syst* 117:92-99. | High | M | none; feeds F3 and MW2 |
| CS2 | **Real GLSW**, G = V(S²/a + I)^(-½)Vᵀ, built from the clutter covariance of contaminated-minus-clean residuals, with a single parameter a. | Unlike the current diagonal weighting, it can down-weight a contaminant direction that overlaps analyte bands. | `interference.py:704-718` (inverse variance per wavelength); `contaminant_analysis.py:1271-1303`. Martens et al. 2003 *J Chemom* 17:153-165. | Medium | M | none |
| CS3 | **PLS-space applicability domain plus an attachable "clean bone" screen**: use the regression model's own score and residual distances with regular/extreme/outlier categories, and allow a saved DD-SIMCA model to be attached to a regression model. Output reads like "%Collagen = 12.3; clean-class p = 0.002 (possible consolidant)". | One consistent rule for trusting a prediction, and consolidated specimens caught before a wrong value is reported. | `model_io.py:1136-1235` (a separate PCA with mixed status vocabularies). Rodionova & Pomerantsev 2020 *Anal Chem* 92(3):2656-2664. | Medium | M | F4 |
| CS4 | **One-class variable selection**: Wold modelling power on the target class only, already in `simca.py:1324/:1627`; LOVE as an option. | Keeps the rigorous evaluation honest. One-class selectors are currently supervised by y_oc (`unified_bayesian.py:1339-1342`). | Pomerantsev, Kucheryavskiy & Rodionova 2025 *Anal Chim Acta* 1368:344302. | Medium | S-M | F4 |
| CS5 | Rigorous DD-SIMCA, OOF p-values, acceptance and extreme plots, Data Quality rebuild | Covered under F4. | n/a | High | n/a | n/a |

### Other method additions (modelling workflow)

| ID | What | Why | Evidence | Impact | Effort |
|---|---|---|---|---|---|
| MW1 | **Stability selection** wrapper around any score-array selector: per-band frequency over subsamples and seeds, pooled within a ±k nm tolerance, exposed as a selector ("stability(CARS)") plus a frequency plot. | Six CARS seeds gave a top-30 Jaccard of 0.43 with only 3 common bands, yet the frequencies concentrate in about 6 regions. Interpretation should rest on regions, not on one run's pixels. (This is the queued MC-PLS candidate, done generally.) | Meinshausen & Bühlmann 2010 *JRSS B* 72(4):417-473. 50 runs take about 30 s. | High | M |
| MW2 | **Scatter-correction family as search axes**: MSC, EMSC (CS1), SNV+detrend and Norris gap derivatives crossed with derivatives; the baseline method as a list. | Lets the search find out whether MSC or EMSC beats SNV, which matters most for ATR-FTIR bone. MSC exists today only as a global toggle (`preprocess.py:370-381`). | Barnes et al. 1989 *Appl Spectrosc* 43(5):772-777; Norris & Williams 1984 *Cereal Chem* 61(2):158-165. | Medium | M |
| MW3 | **Per-sample prediction intervals** ŷ ± t·SEP·√(1+hᵢ+1/n) for PLS/PCR and split-conformal for the rest; one status column per sample. Wire in or delete the unused `diagnostics.jackknife_prediction_intervals` (`:143`). | Collaborators get a sample-specific error bar instead of one global RMSECV. | `model_io.py:858-866`. Faber & Kowalski 1997 *J Chemom* 11(3):181-238. | Medium | M |
| MW4 | **Interpretability panel**: signed b-coefficients with jack-knife CIs from the CV sub-models, VIP with its fold range, selected bands over the mean spectrum, click-to-annotate band assignments. | This is the discussion section of every NIR paper. | GUI `:37826-37960` shows VIP only, from the training fit. | Medium | M |
| MW5 | **PCR, PCA-LDA and PLS-LDA** as baseline models. | The baseline reviewers ask for. Also the first test of the ST6 registry. | None in `model_registry.py`. | Low-Med | S |

---

## (d) Speed

QW1 (thread budget) and QW11 (Bayesian single pass) are in the quick wins. The rest:

| ID | What | Why | Evidence | Impact | Effort | Depends on |
|---|---|---|---|---|---|---|
| SP1 | **PLS CV kernel that covers every LV count in one fit**: one fit per fold at max LVs gives predictions for every k. Used in the grid's PLS branch and the selectors; sklearn `PLSRegression` is kept for saved and exported models, so `.dasp` files are unchanged. | PLS is the core NIR/FTIR model. | 50 sklearn fits 133 ms vs 7.2 ms, max difference 3e-13. About 90 of 105 s in a PLS subset grid was overhead. Dayal & MacGregor 1997 *J Chemom* 11:73-85. | High | M | none |
| SP2 | **Vectorised SPA** (all 2151 chains at once from a precomputed correlation² matrix) and interval selectors on the SP1 kernel. | SPA and mwPLS become usable choices in a grid instead of multi-minute stalls. | SPA 58 s → about 2-3 s with the same selection. mwPLS 38 s (9,070 PLS fits). | High | S | SP1 |
| SP3 | **Trim per-config bookkeeping**: reuse the fitted pipeline for importances, compute metrics in numpy, build the DataFrame once, send the DIAGNOSTIC dump to `logger.debug`. | 30-45% off PLS, Ridge and PLS-DA grids; no more 1.2 MB of stdout per run. | `add_result` `pd.concat` is O(n²): 15 s. Metric validation 18 s. `search.py:5242-5290`. | Medium | S | none |
| SP4 | **One-class**: numpy confusion-count metrics, closed-form χ² moments (instead of Nelder-Mead in `_fit_chi2`), parallel folds. | Glyptal screening becomes interactive. | Comprehensive tier 60 s, of which metrics 11.9 s and `chi2.fit` 5.0 s. | Medium | S | QW1 |
| SP5 | **Honest ETA and early results**: time one config per model before the run and show "≈ 3-6 min"; run cheap models first; push rows to Results as they finish; check Pause between folds; add a determinate progress bar, a stage line, and a single sticky Run bar with a one-line summary, replacing the 5 Run buttons. | You see where the time will go, get a usable leaderboard within seconds, and can stop once the answer is clear. | Configs differ in cost by about 100x. The ETA is a uniform average (GUI `:31299-31303`). Results appear only at the end (`:30012`, `:30088`). | Medium | M | QW1 |
| SP6 | **Task-level parallelism**: send whole configs (with their subsets) to a persistent pool with single-threaded models. | Every grid, not only boosters, scales with cores. | PLS, Ridge and SVM are serial by design (`search.py:174`), so most of a grid runs on one core. | High | L | QW1, ideally ST5 |
| SP7 | **NSGA-II**: evaluate the population in parallel and cache preprocessed matrices. | A default NSGA-II run drops from about 30 min to a few minutes. | 255 ms per evaluation on one core × 7,200 evaluations. | Medium | M | QW1 |
| SP8 | **Bayesian**: median pruning for slow model families, and concurrent per-model studies. | Multi-model runs take about as long as the slowest model, not the sum. | GUI `:30421-30456` runs studies sequentially. The pruning saving is a literature estimate, not measured. | Medium | M | QW1 |

---

## (e) Look and feel

QW7 (DPI, fonts), QW8 (cards, width) and QW9 (help) are in the quick wins. The rest:

| ID | What | Why | Evidence | Impact | Effort | Depends on |
|---|---|---|---|---|---|---|
| LF1 | **Readable leaderboard**: per-column number formats (no `8.00268e-05`), numbers right-aligned, CV metrics before calibration metrics, constant columns hidden, a Q1-Q4 chip instead of full-row tints, and the empty Ensemble card collapsed. | Compare candidates at a glance and spot overfit rows. | GUI `:32672-32680` formats every float with `.6g`. | Medium | M | fits with F5 |
| LF2 | **Metric tiles and a proper parity plot** in Model Development: square axes, grey 1:1 line, colour by fold or group rather than Y, axis labels from the target column ("%Collagen, reference"). | The headline answer is readable in one second, and the plot can go straight into a slide. | GUI `:37440-37530`; colour-by defaults to "Y Value" (`:3026-3029`). | Medium | S | house style (F5) |
| LF3 | **Row → saved model in one step**: a right-click menu on Results rows (refit and save `.dasp`, export code, send to Prediction) and a "Use this model for prediction →" link after saving. | About 8 actions and a round trip through the file system become 2 clicks. | GUI `:43974` (models load only from disk). No Button-3 binding on Results. | High | M | none |
| LF4 | **Recipes and a File menu**: Save/Open Recipe (reusing `capture_gui_settings`), Open Results (reload the CSV with its training_config), recent files, restore last settings, and outputs written to an absolute Documents folder with an "Open folder" button. | Share an exact analysis recipe across the group; leaderboards survive closing the app. | The menubar has only About and Help (`:61754-61768`). Outputs go to relative `outputs/` folders (`:31052`). | High | M | ST2 makes it robust |
| LF5 | **Guided "New Analysis" path**: Data → Target and task (auto-detected, with its reason shown) → Recipe preset → Review, with n-aware CV defaults (repeated k-fold for n = 30-200) and "Use the example data". Also rename the "Advanced Configuration (Optional)" header that currently hides Target. | CSV to a defensible leaderboard in 4 decisions. The default for n = 49 should match the app's own on-screen advice. | GUI `:6534`, `:2969-2970`, `:11594`. | High | L | LF4, SP5 |
| LF6 | **One preprocessing choice**: "I'll pick" vs "Let DASP search" with a single engine dropdown, disabled when Bayesian search already covers preprocessing; SG3/SG4 and deriv_snv behind "advanced". | Removes a class of duplicate or contradictory runs. | Four overlapping mechanisms (`:11824-12153`). | Medium | M | none |
| LF7 | **Actionable errors and a visible log drawer**: `_show_error(title, exc, hint)` mapping common failures to a fix, with Copy details and Open log buttons; `print()` routed to logging. | Collaborators can fix a data problem themselves, and bug reports arrive with the log. | 313 `showerror` calls, about 76 bare "{e}"; 753 `print()` calls invisible in the bundle. | Medium | M | none |
| LF8 | **One Predict workspace** (models → spectra → optional transfer chain → predict and export), replacing Model Prediction, Multi-Model and CT Mode A. | One obvious place to predict, and no drifting duplicate code paths. | GUI `:43601`, `:52398`, `:55106`. | Medium | L | F2, ST4 |
| LF9 | **Dataset context bar** in the header in place of the 6 theme buttons ("49 samples × 2151 λ · %Collagen · regression · CV 5-fold"); keep one light theme; drop emoji icons; expand "Advanced" by default or rename it "Instruments & contaminants". | Answers "what am I working on?" on every page and makes the science modules discoverable. | Theme switching wipes status colours (`:5000-5040`). Advanced is collapsed by default (`:5849`). | Medium | M | none |
| LF10 | **One heading and stepper style**: numbered steps; Calibration Transfer's A/B/C/A1/A2 lettering renumbered 1-4. | You always know where you are in a workflow. | GUI `:54663-55430`. | Low-Med | S | with F2 |

---

## (f) Enabling and structural work

| ID | What | Unblocks | Evidence | Impact | Effort |
|---|---|---|---|---|---|
| ST1 | **RunRequest**: freeze all engine inputs at the click into a typed object; the worker thread never reads Tk. Migrate one engine per PR, grid first. | Removes the bug class of worker threads reading live Tk state (most of PR #79's 13 review rounds). Enables recipes (LF4), batch or queued runs, and a reproducibility record next to results; gives agent scripts the same object. | The worker reaches 518 live `.get()` reads of 390 vars; only 155 are captured. | High | L |
| ST1a | **Main-thread guard**: a test fixture that records, then raises on, Tk access off the main thread, plus a CI ratchet (ceiling 518, may only go down). | Catches the recurring regression at commit time. | SESSION_LOG_ARCHIVE:2474 ("cycle 4" of the same anti-pattern). | High | S |
| ST2 | **Settings registry** that generates the Tk vars and the capture, required and legacy lists (the pattern T-51 PR D proved). | ST1, LF4; dead toggles such as `use_msc` get caught automatically. | `run_gui_settings.py:86-266`. | High | M |
| ST3 | **CVPlan** | This is F1's core; also stability selection and in-fold correction. | See F1. | High | M |
| ST4 | **Extract tabs**: Contaminant first (40 methods, 1 external entry point), then Interference and Spectral Library, then Calibration Transfer behind a non-Tk `TransferChain` service; build tabs lazily. | F2, F3 and LF8 get done in a 2-4k-line module with headless tests; start-up time drops. | `coupling2.py` output. | High | M |
| ST5 | **Shared `evaluate_candidate()`**. Step 1 (S) moves the four copied helpers (`_needs_resampling_pipeline` etc.) into one module; later steps make grid, Bayesian, NSGA-II and one-class call one kernel. | A fix lands once. Out-of-fold predictions are uniform for F5 statistics and SP6 parallelism. | About 40 "sister-site" bug mentions in the logs. | High | XL (step 1 is S) |
| ST6 | **ModelSpec and PreprocessSpec registries**, proven first on a new model (PCR). | MW2, MW5, and F3's in-fold correction as an ordinary step. | "catboost" appears in 14 files. | Medium | L |
| ST7 | **`rebuild.py`** (results row → pipeline) and the GUI's wrapper classes moved into the package, with a pickle `find_class` shim and a fixture `.dasp` test done **first**. | LF3; ends the "refit doesn't match the search" bug class. | GUI `:2167-2540`; `model_io.py:216` (joblib pickle). | Medium | M |
| ST8 | **SpectraInput component** replacing 7 copies of load-and-convert. | A reader fix (such as relaxing the 100-band minimum for CT5) is made once. | 18 per-tab data-type attributes; 44 loader methods. | Medium | M |

---

## Dropped or deferred

- **Deferred to a later wave:** the full Screen & Clean merge and the one Predict workspace. Both are L, so do them after ST4 lets the tabs be worked on separately.
- **Deferred indefinitely:** a dark theme, per-monitor DPI v2, "notify me when done", pre-warming the executor, and weekly random-order test runs. Each has low evidence of user pain.
- **Not proposed:** multi-target/PLS2 (you are leaning against PR #63) and anything CLI-shaped. RunRequest and `evaluate_candidate` are Python primitives, consistent with AGENT_COMPOSITION.

---

## Recommended order for the first 8 PRs

1. **QW1: one thread-budget rule, plus the QW10 test split.** This is the largest speed gain available (60-70x per booster config, about 50x for LOF), and every later PR benefits from a faster test suite. S.
2. **QW2 + QW6: stop the app misleading.** CT relabel, guide, default TSR and the resubstitution R² label; interference controls greyed out with `apply_to_analysis` off; export stub fixed; dead `use_msc` removed. S, mostly text and defaults.
3. **QW3: contaminant EPO maths fix with behavioural tests and the "analyte retained" figure.** Without it, the Glyptal workflow's default correction does nothing useful. S.
4. **QW4 + QW5: holdout direction fix and the figures-of-merit function.** Both are small, and together they make external validation both correct and complete. S.
5. **QW7 (+ QW8 if it stays small): DPI awareness and fonts.** The cheapest visible upgrade, about 5 lines for DPI. S.
6. **F2 part 1 (backend): `calibration_transfer.evaluate_transfer()` (leave-one-standard-out), CT1 low-rank PDS, CT2 dual-form DS, CT3 slope/bias adapter.** This gives the transfer methods that actually worked and a way to prove it, usable from scripts straight away. M.
7. **F2 part 2 (GUI): Validate and Compare-methods panel, pairing by sample ID, "Plan transfer standards".** This is the calibration-section improvement you asked for, delivered as a leaderboard with a no-correction baseline. M.
8. **F1 part 1: `CVPlan` plus a Specimen/Group ID column, driving the grid search and the holdout.** Starts closing the largest scientific gap for replicate- and site-structured data. M.

**Next after these:** F4 part 1 (real DD-SIMCA), CS1 (EMSC with interferents), ST4 (extract the Contaminant tab), then F3 (in-fold, saved contaminant correction), SP1 and SP2 (PLS kernel, SPA), and QW11 (Bayesian single pass). QW9 (help and docs) should follow shortly after PR 7, so Chapter 4 describes the new transfer workflow.
---

# Appendix: specialist lens reports (raw ideas before synthesis)

## Lens: calibration-transfer

**Current state.** The DS, PDS and TSR estimators work. Beyond that, dasp handles calibration transfer poorly: the tab looks thorough but misleads the user. I ran a benchmark with scratch scripts in C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\calibration_transfer\. The primary data were the shipped 49 bone-collagen ASD spectra, trimmed to 400-2450 nm and taken every 2 nm (1026 bands), in absorbance. The satellite instrument was simulated from them: +6 nm shift, 8 nm blur, a 20% gain tilt, a curved baseline and noise. A PLS model was fitted on the primary calibration set. Transfer standards were picked by Kennard-Stone from that calibration set. RMSEP was measured on 15 held-out satellite spectra, averaged over 20 random splits. Figures below are %Collagen.
- **Baselines:** no correction scored 3.91; the primary-instrument ideal was 2.62.
- **Methods that already work:** TSR scored 2.62-2.64 and a prediction slope/bias fit 2.62-2.71, with only 3-12 standards.
- **DS at the GUI default (lambda 1e-3):** 4.94 with 3 standards, worse than no correction. It only beat no correction from 8 standards (3.16).
- **dasp PDS:** 3.46-5.08, and it got worse as standards were added (3.46 at 5, 4.12 at 8, 5.08 at 12). The same PDS with centring and a truncated SVD held at 2.58-2.61 across 3-12 standards.
- **dasp CTAI:** 3.73-4.49, worse than no correction at every n.
- **dasp JYPLS-inv:** 9.4-11.7, about three times worse than no correction.
- **dasp NS-PFCE:** 4.26 with 3 standards, falling to 2.59 only at 12. It is the GUI default method.

The three "advanced" methods have the published names but not the published algorithms:
- **CTAI** (calibration_transfer.py:696-964) is a paired, PCA-truncated DS. It raises an error on unpaired data, yet its docstring and UserGuide s4.5 say "No transfer samples required". Published CTAI (Zhao et al. 2019) needs no standards and works on the PLS score and prediction space.
- **NS-PFCE** (:1023-1328) is a damped, ridge-regularised DS iteration. It compares rows by position even though its docstring says "unpaired", and it cites "Literature reference needed" at :1119.
- **JYPLS-inv** (:1428-1663) stacks both instruments into one PLS model with shared loadings, so it cannot represent the difference between instruments.

The GUI adds three incompatible expansions of CTAI:
- Guide at :54646: "Cross-Transfer Adaptive Interpolation", recommended for different wavelength ranges.
- Tooltip at :1037: "Adaptive Integration".
- Code: "Affine Invariance".

The NS-PFCE tooltip (:1045) calls it "Null-Space Projection...". The guide recommends DS when there are fewer than 10 standards (:54645), which is backwards, and offers "Feature-based matching" (:54647), which does not exist.

How success is shown: `_plot_transfer_quality` plots mean ± std spectra plus a flattened scatter with an R² computed on the same standards used to fit (:48344). For DS that R² was 1.00 in every run, even when the transfer made predictions worse.
- The tab never reports RMSEP before and after transfer, although UserGuide s4.6 describes that workflow.
- There is no held-out check and no per-standard residual plot.
- Pairing is by row order for every method except JYPLS-inv; sample IDs are dropped (:48954-49040).
- The live TSR uses the first n rows (:49350), although the tooltip promises Kennard-Stone. Kennard-Stone lives only in the dead `_build_ct_transfer_model` (:46956, no callers).
- A satellite on a different grid is interpolated onto the primary grid with linear extrapolation (:49299).
- The transfer model is saved separately from the .dasp prediction model.
- Transfer is limited to regression in "predict" mode. Classification and one-class users only get the export-spectra mode.

Speed is not the problem: on the full 2151-band grid, DS takes 1.5 s, NS-PFCE 5.9 s, and PDS, TSR and CTAI take 0.1-0.3 s. The problem is choosing a method and knowing whether it worked. For this group's usual case (a handful of rescannable specimens and a small-n model), the methods that performed best are all simple (TSR, prediction slope/bias, well-regularised PDS). The tab hides this: its default method is NS-PFCE, and its diagnostic plot reports a near-perfect fit even when transfer hurts.

### Replace the resubstitution R² with RMSEP before and after transfer, checked on held-out standards (reporting, impact high, effort M)

- **Problem:** The transfer quality display (_plot_transfer_quality, spectral_predict_gui_optimized.py:48165-48428) shows mean ± std spectra plus a flattened primary-vs-transferred scatter. Its R² (:48344) is computed on the same standards used to fit the transfer. For DS with n<p this R² is 1.00 by construction, even when transfer makes predictions worse. The tab never reports prediction error. UserGuide.md s4.6 (lines ~8403-8425) describes a before/after RMSEP workflow with an improvement ratio that the code does not implement. Mode A's result plot (:51013) shows only a histogram of predictions and mean spectra.
- **Proposal:** Add a 'Validate transfer' panel that needs the primary .dasp model (already loaded in C1), plus reference y for the standards if the user has it.
(1) Leave one standard out: refit the transfer on the other standards and transform the left-out satellite spectrum. Report, in y units: RMSEP with no correction, RMSEP after transfer, bias, slope, and SEP.
(2) With no y: report the left-out spectral RMSE per wavelength, before vs after.
(3) Plot each standard's residual spectrum (primary minus transferred) with the model's regression-coefficient vector overlaid, so users can see whether the residuals sit on bands that matter for the model.
(4) Show Hotelling T² and Q of the transferred spectra in the primary model's PCA space, reusing the applicability-domain payload that predict_with_uncertainty already computes.
(5) For classification and one-class models, report class agreement or inlier rate instead of RMSEP.
Use the same leave-one-out loop to rank every eligible method; see the 'which method?' idea.
- **Benefit:** Users can see whether transfer helped, in the units they report (%Collagen). They no longer trust a plot that reads R²=1.00 when DS with 3 standards raised RMSEP from 3.91 to 4.94.
- **Evidence:** Benchmark: 'resub R2 flattened (GUI plot, DS)' = 1.00 in all 80 runs, while DS with lambda=1e-3 gave RMSEP 4.94 with 3 standards against 3.91 uncorrected (bench_severe_out.txt in scratchpad\improve\calibration_transfer\). Code refs: GUI :48344, :51013-51065; UserGuide s4.6. Practice: Workman JJ (2018) A review of calibration transfer practices and instrument differences in spectroscopy. Applied Spectroscopy 72(3):340-365, doi:10.1177/0003702817736064 (success is judged by prediction error on the child instrument, not by spectral fit).

### Relabel or remove the misnamed CTAI, NS-PFCE and JYPLS-inv methods, and correct the guide (method, impact high, effort S)

- **Problem:** The three advanced methods carry the names of published algorithms they do not implement, and the GUI describes them in contradictory ways.
- CTAI (calibration_transfer.py:696-964) is a paired, PCA-truncated DS. It raises an error on unpaired data (:859-864), but its docstring and UserGuide s4.5 say 'No transfer samples required!'. It cites 'Fan, W., et al. (2019) Analytical Methods 11(7) 864-872' (:781). The real CTAI is Zhao et al. 2019 (Molecules), which needs no standards and corrects predictions in PLS score space.
- NS-PFCE (:1023-1328) is a damped ridge-DS loop that compares rows by index even though it claims to handle 'unpaired' data. It cites 'Literature reference needed' (:1119). Real PFCE constrains the model's regression coefficients.
- JYPLS-inv (:1428-1663) fits one PLS model with loadings shared by both instruments. It is disabled in the GUI (:54887) but is still a public backend primitive, and falls back to y = zeros (:49433).
- Tooltips give CTAI as 'Adaptive Integration' (:1037), the guide gives it as 'Cross-Transfer Adaptive Interpolation' (:54646), and the NS-PFCE tooltip reads 'Null-Space Projection...' (:1045).
- The guide recommends DS for fewer than 10 standards (:54645) and 'Feature-based' methods (:54647) that do not exist.
- NS-PFCE is the GUI default (:54854).
- **Proposal:** Short term (S):
- Rename CTAI to 'PC-DS (truncated-SVD DS)' and NS-PFCE to 'Iterative ridge DS'.
- Remove JYPLS-inv from the public surface, or mark it experimental with a hard error when y is missing.
- Fix all tooltips, the quick guide and UserGuide Ch.4 (method table, s4.5, and the TSR reference; see evidence).
- Make TSR or regularised PDS the default.
Longer term (M): if standard-free or coefficient-space methods are wanted, implement them from their papers and validate them against the authors' reference code. PFCE has an official implementation at github.com/JinZhangLab/PFCE.
- **Benefit:** Users stop picking methods whose names promise things the code does not do, such as 'no standards needed' or 'different wavelength ranges'. Methods and references also become citable in papers: a reviewer who checks the CTAI or PFCE reference will currently find a different algorithm.
- **Evidence:** Benchmark RMSEP (no correction 3.91): dasp-CTAI 3.73-4.49; dasp-JYPLS-inv 9.41-11.68; dasp-NS-PFCE 4.26 with 3 standards. References:
- Zhao Y, Zhao Z, Shan P, Peng S, Yu J, Gao S (2019) Calibration transfer based on affine invariance for NIR without transfer standards. Molecules 24(9):1802, doi:10.3390/molecules24091802 (read from Matt's Paperpile copy).
- Zhang J, Li B, Hu Y, Zhou L, Wang G, Guo G, Zhang Q, Lei S, Zhang A (2021) A parameter-free framework for calibration enhancement of near-infrared spectroscopy based on correlation constraint. Analytica Chimica Acta 1142:169-178.
- Folch-Fortuny A, Vitale R, de Noord OE, Ferrer A (2017) Calibration transfer between NIR spectrometers: new proposals and a comparative study. Journal of Chemometrics 31(3):e2874, doi:10.1002/cem.2874. This is where JYPLS-inv was proposed. I found the full author list from a search snippet; please verify it.
- TSR is cited as Shenk & Westerhaus 1991 Crop Sci 31:469-474 (:527), which is their sample-selection paper. The standardization paper is Shenk JS, Westerhaus MO (1991) New standardization and calibration procedures for NIRS analytical systems. Crop Science 31(6):1694-1696, doi:10.2135/cropsci1991.0011183X003100060064x.
- The '12-13 samples ... statistically indistinguishable' claim (:463-465, :534) has no source.

### Fix PDS and DS for few standards: add an intercept, low-rank windows, and automatic regularisation (method, impact high, effort S)

- **Problem:** estimate_pds (calibration_transfer.py:173-245) fits each window by minimum-norm lstsq, with no centring or intercept and no rank control. Once the number of standards approaches the window width it overfits noise, so it gets worse as standards are added. estimate_ds (:117-158) solves a p×p system with a fixed lambda. The GUI default is 1e-3 (:54906), which is too small when n is much smaller than p. DS also stores a p×p matrix: about 37 MB at 2151 bands, and 1.5 s to fit.
- **Proposal:** PDS changes:
- Centre each window, fit it by truncated SVD or local PLS/PCR with the rank capped at n-1 (as in Wang et al. 1991), and store an offset vector.
- Choose the window width (for example 5-31) and rank by leave-one-standard-out.
- Offer double-window PDS as an option.
DS changes:
- Use the dual form A = Xsᵀ(XsXsᵀ+λI)⁻¹Xp, which solves an n×n system (milliseconds, small storage).
- Pick λ by leave-one-standard-out, or use truncated SVD.
Add both as options to the existing functions, keeping backward compatibility, since `estimate_pds` is on the agent-composition surface.
- **Benefit:** PDS becomes as good with 3 standards as it is with 12, which is the realistic number of rescannable bone specimens. DS stops making things worse and becomes fast enough to cross-validate.
- **Evidence:** Benchmark (severe case, RMSEP; no correction 3.91, ideal 2.62):
- dasp PDS w=11: 3.61 / 3.46 / 4.12 / 5.08 at 3 / 5 / 8 / 12 standards.
- Centred + truncated-SVD PDS (about 25 lines, pds_centered_svd in ct_bench.py): 2.61 / 2.59 / 2.58 / 2.58.
- DS with lambda=1e-3: 4.94 / 3.84 / 3.16 / 2.54.
- Timing: estimate_ds 1.52 s at 2151 bands.
References:
- Wang Y, Veltkamp DJ, Kowalski BR (1991) Multivariate instrument standardization. Analytical Chemistry 63(23):2750-2756.
- Bouveresse E, Massart DL (1996) Improvement of the piecewise direct standardisation procedure for the transfer of NIR spectra for multivariate calibration. Chemometrics and Intelligent Laboratory Systems 32(2):201-213 (cited in dasp at :1484).

### Pair standards by sample ID and help choose which specimens to rescan (workflow, impact high, effort S)

- **Problem:** _load_primary_spectra and _load_satellite_spectra (GUI :48954-49100) keep only (wavelengths, X) and drop IDs, so DS, PDS, TSR and CTAI pair spectra by row order. A folder sorted differently on the two instruments pairs the wrong specimens without any warning. The live TSR uses the first n rows (`transfer_indices = np.arange(n_samples)`, :49350), although its tooltip (:1028-1034) promises Kennard-Stone. Kennard-Stone and ID matching exist only in the JYPLS branch (:49376-49400) and in the dead `_build_ct_transfer_model` (:46956; also `_load_ct_paired_spectra` :46640, neither has callers). sample_selection.py already provides kennard_stone, duplex and spxy (:31, :138, :243). Nothing helps the user decide which specimens to take to the second instrument.
- **Proposal:** (1) Load spectra as DataFrames with an ID index; the loader `_load_spectra_from_directory_as_df` already exists at :48656. Pair by ID using io.align_xy-style fuzzy matching, and show matched and unmatched lists.
(2) Add a 'Plan transfer standards' step. From the primary calibration set (or the loaded model's training spectra), suggest k specimens by Kennard-Stone, or by SPXY when y is known, plus the highest-leverage samples. Show them on a PCA score plot and export a CSV checklist to take to the other instrument.
(3) Show a rule of thumb: with the leave-one-out panel, adding standards until RMSEP stops improving.
(4) Delete the dead duplicate build path, about 400 lines.
- **Benefit:** The step users actually find hard (which 5-10 bones to rescan, and making sure the pairs match) is guided and checked. Silent mis-pairing becomes impossible.
- **Evidence:** GUI :49350 (TSR takes the first n rows); :46956 and :46640 have no call sites (grep); :48656 is the ID-preserving loader. Reference: Kennard RW, Stone LA (1969) Computer aided design of experiments. Technometrics 11(1):137-148.

### Offer three kinds of fix: correct the spectra, correct the predictions, or make the model robust (method, impact high, effort M)

- **Problem:** The tab only offers spectrum-to-spectrum mapping. With few standards, the cheapest and most reliable option is often a slope/bias correction of the predictions, fitted on satellite standards with known y. It was the best or equal-best method in the benchmark. dasp already stores a slope/bias correction in the .dasp file (model_io.py:97, :332-359) and applies it at prediction (:821-824). However, it is only computed from the model's own cross-validation predictions in Model Development (GUI :37596-37620), never from satellite standards. Two other approaches are also missing:
- Model updating: adding the few satellite standards to the calibration set.
- Transfer by orthogonal projection (TOP), also called EPO on the primary-satellite difference spectra: removing between-instrument directions before refitting. EPO already exists in interference.py:820 (an earlier stub at :565 has the same class name and is shadowed by it).
- **Proposal:** Add a first choice to the tab: 'What do you want to fix?'
(a) Spectra: DS, PDS or TSR.
(b) Predictions: slope/bias from satellite standards. This writes a copy of the .dasp model with bias_correction plus the instrument ID. Warn when there are fewer than 3 standards or their y range is narrow.
(c) Model: TOP/EPO on the difference spectra, or augmentation with the satellite standards, then refit through the existing Model Development refit path.
Run all eligible options through the same leave-one-standard-out panel and show them as a small leaderboard.
- **Benefit:** Users with 3-5 standards get the method that performed best here (slope/bias 2.62-2.71 against 3.91 uncorrected) in one click, stored in the model file they already use, without a p×p spectral mapping.
- **Evidence:** Benchmark severe case: y slope/bias 2.71 / 2.62 / 2.66 / 2.68 at 3 / 5 / 8 / 12 standards; model update (augment x3) 2.84-2.63. In the mild case, TOP/EPO k=2 + refit was the best method (2.42-2.59); in the severe case it was worse (3.2-3.4). It depends on the situation, which is why a leave-one-out comparison is needed. References:
- Bouveresse E, Hartmann C, Massart DL, Last IR, Prebble KA (1996) Standardization of near-infrared spectrometric instruments. Analytical Chemistry 68(6):982-990 (slope/bias correction).
- Andrew A, Fearn T (2004) Transfer by orthogonal projection: making near-infrared calibrations robust to between-instrument variation. Chemometrics and Intelligent Laboratory Systems 72(1):51-56.
- Roger JM, Chauchard F, Bellon-Maurel V (2003) EPO-PLS external parameter orthogonalisation of PLS application to temperature-independent measurement of sugar content of intact fruits. Chemometrics and Intelligent Laboratory Systems 66(2):191-204.

### Add a 'Which method?' button that tries every eligible method and ranks them by held-out RMSEP (usability, impact high, effort M)

- **Problem:** Users face 6 radio buttons (DS, PDS, TSR, CTAI, NS-PFCE, JYPLS-inv; GUI :54857-54890) with parameters for all methods shown at once (:54893-54995) and a static guide that is wrong. The default is NS-PFCE (:54854). Nothing tells the user what the data support. This is the opposite of dasp's main pitch, which is to try the combinations and rank them.
- **Proposal:** Replace the static guide with a 3-question situation picker:
- Do you have specimens scanned on both instruments?
- How many?
- Do you have reference values for them?
Then one 'Compare methods' button fits every eligible option (spectral: TSR, regularised PDS over a small window grid, dual-form DS; prediction: slope/bias, bias-only; model: TOP, augmentation). It scores them by leave-one-standard-out RMSEP (or spectral RMSE when there is no y) and shows a leaderboard with the 'no correction' row always included as a baseline. Double-clicking a row builds and saves that transfer. Show each method's parameters only when it is selected. All fits take 0.1-1.5 s at 2151 bands, and less with the dual DS form, so a full comparison with 12 standards takes seconds.
- **Benefit:** Choosing a method becomes like the main leaderboard: evidence first, with a visible 'did nothing help?' baseline. Fewer settings to understand, and it is faster than trial and error.
- **Evidence:** GUI :54854 (default nspfce), :54643-54647 (guide). Timings measured on 2151 bands with 12 standards: DS 1.52 s, PDS 0.11 s, TSR 0.10 s, CTAI 0.26 s, NS-PFCE 5.90 s, JYPLS-inv 0.47 s (bench_out.txt).

### Store the transfer inside the .dasp model and apply it automatically by instrument (workflow, impact medium, effort M)

- **Problem:** Transfer models are saved as separate JSON+NPZ pairs (calibration_transfer.py:316-384) with primary_id and satellite_id hard-coded to 'primary' and 'satellite' (GUI :49480-49481). Users must pair the right transfer file with the right model in Mode A, in Model Prediction, or in the Multi-Model transfer chain (:53771-53922). Nothing links the transfer to the model it was validated against, to the preprocessing it assumes, or to the reflectance/absorbance state beyond a meta flag. There is also no way to track drift on the same instrument over time.
- **Proposal:** Allow a .dasp model to hold a dict of per-instrument adapters: {instrument_id: transfer model and/or slope-bias, validation RMSEP, n standards, date}. Model Prediction then gets an 'Instrument' dropdown and applies the matching adapter. For drift, temperature or presentation changes, add a 'check standard' log: predict a stable reference specimen each session and show a control chart with ±2·SEP limits. When the chart goes out of limits, offer to rebuild the slope/bias correction from a few standards.
- **Benefit:** There is one file to share with collaborators and no risk of mismatched transfer files. It also covers the common real case of the same instrument drifting, a lamp change, or a new sample cup, not only a second instrument.
- **Evidence:** calibration_transfer.py:316-437; GUI :49480-49481 (hard-coded IDs), :53771-53922 (multi-model transfer chain); model_io.py:97 and :821 (bias_correction already persisted and applied). Practice: Workman JJ (2018) Applied Spectroscopy 72(3):340-365 (monitoring with check samples).

### Handle different wavelength grids properly: intersect ranges, match resolution, allow low-band instruments (method, impact medium, effort M)

- **Problem:** When grids differ, _build_transfer_model_new interpolates the satellite onto the primary grid with fill_value='extrapolate' (GUI :49299). A handheld instrument covering 950-1650 nm would be linearly extrapolated across 350-2500 nm, producing fabricated data. It shows only an 'Info' popup. instrument_profiles.estimate_smoothing_between_instruments (:412-455), which finds the Gaussian blur that makes a high-resolution spectrum look like the low-resolution one, is never used in the build path. equalization.build_equalization_mapping_for_instrument (:53-93) handles only DS and PDS and has a no-op smoothing step. Every reader rejects files with fewer than 100 wavelengths (io.py:140, 316, 773, 1109, 1486), which blocks some MEMS and handheld units.
- **Proposal:** (1) Default to the overlap of the two ranges on the coarser grid, show a range-bar plot (already drawn for equalization at :48430-48480), and refuse to extrapolate.
(2) Add a 'Match resolution' option that blurs the higher-resolution instrument's spectra with the estimated sigma before fitting the transfer. The same blur must be applied at prediction.
(3) Let the user retrain the primary model on the overlap region, a one-click handoff to Analysis Configuration with a wavelength restriction, when the satellite covers only part of the model's range.
(4) Let the 100-band floor be overridden for satellite data.
- **Benefit:** Moving a benchtop ASD model to a cheaper or portable instrument, a realistic next step for museum and field archaeology, becomes possible without silent garbage.
- **Evidence:** GUI :49299; instrument_profiles.py:412; equalization.py:53-93; io.py band floors as cited. A relevant example of two compact NIR instruments with different ranges on bone: Riu J, Giussani B, Monti M, Baruffaldi L, Campeny M, Quesada J. Shedding light on the past: temporal classification of zoological specimens from museum collections with portable NIR sensors and multivariate error modeling. Analytical Chemistry, doi:10.1021/acs.analchem.5c06767 (Matt's Paperpile copy; the year, volume and pages were not on page 1).

### Add real standard-free options (di-PLS) for cases where specimens cannot be rescanned (method, impact medium, effort L)

- **Problem:** Archaeological and museum material often cannot be rescanned: it was destroyed in sampling, is held by another lab, or a dataset is pooled from other studies, as in the multi-site ATR-FTIR bone collagen paper. dasp advertises standard-free transfer (CTAI docstring, 'Feature-based' in the guide), but every implemented method needs paired standards. The simplest standard-free baseline, centring each instrument's data on its own mean, did not help here (3.77 against 3.91 uncorrected), so a naive fix is not enough.
- **Proposal:** Implement domain-invariant PLS (di-PLS), which aligns the source and target distributions in latent-variable space using only unlabeled target spectra. It fits as a new model type (for example `di-PLS`) that takes an extra 'target-domain spectra' input, so it can be cross-validated by the existing search. Implement it natively (the core is about 150 lines of NIPALS with a regulariser), or depend on diPLSlib after checking its license and adding it to pyproject.toml. Optionally add the published CTAI later (see the renaming idea). Always show the no-correction baseline and warn that standard-free methods assume similar y distributions in both domains.
- **Benefit:** Pooled multi-lab datasets and instruments with no shared specimens get a principled option instead of a misnamed one.
- **Evidence:** Benchmark: 'domain mean-centre (no std)' 3.77 (severe case) and 3.54 (mild case, where no correction scored 2.83), so it made the mild case worse. References:
- Nikzad-Langerodi R, Zellinger W, Lughofer E, Saminger-Platz S (2018) Domain-invariant partial-least-squares regression. Analytical Chemistry 90(11):6693-6701, doi:10.1021/acs.analchem.8b00498.
- Mikulasek B, et al. (2023) Partial least squares regression with multiple domains. Journal of Chemometrics, doi:10.1002/cem.3477. The first author's co-authors, volume and pages were not in the search result; please verify.
- Python implementations: github.com/B-Analytics/diPLSlib and github.com/dario-passos/di-PLS.

### Add a leaderboard column scoring how well each candidate model survives a change of instrument (workflow, impact medium, effort M)

- **Problem:** The main search ranks candidates only on primary-instrument cross-validation. How well a model transfers depends heavily on its preprocessing and bands (derivatives and SNV remove offset and tilt; narrow bands suffer from shifts). That is visible here: an uncorrected 2-nm shift, blur and gain change only moved RMSEP from 2.62 to 2.83 in the mild case. Users cannot ask the search to prefer models that transfer. Transfer lives in a separate tab, after model selection.
- **Proposal:** In Analysis Configuration, allow an optional 'robustness set': satellite spectra of standards with y, or paired spectra without y. Each Results row then gets 'RMSEP_satellite' (predict the satellite standards without transfer, and optionally after slope/bias) as an extra column. It can also serve as an optional tie-break or composite-score term. This needs only predictions from already-fitted models, so the cost is negligible. Later, TOP/EPO built from the difference spectra could become a within-fold preprocessing option.
- **Benefit:** Users can pick a model that does not need transfer at all, or needs only a slope/bias fix, which is the most defensible option with few standards.
- **Evidence:** Benchmark mild case: no correction 2.83 against ideal 2.62 for a CV-tuned PLS model on absorbance, which shows that the sensitivity is model-dependent. References:
- Zeaiter M, Roger JM, Bellon-Maurel V (2005) Robustness of models developed by multivariate calibration. Part II: The influence of pre-processing methods. TrAC Trends in Analytical Chemistry 24(5):437-445. I cited this from memory and did not verify it in Paperpile or on the web.
- Andrew & Fearn 2004 (cited above).

### Rewrite UserGuide Chapter 4 as a small-n transfer cookbook (docs, impact medium, effort S)

- **Problem:** UserGuide.md Ch.4 (lines ~8263-8430) repeats the wrong method table ('Feature-based', DS for fewer than 10 standards), says CTAI needs 'No transfer samples required!' while listing 'Paired samples' as a requirement in the same section, and documents a validation workflow (s4.6) that does not exist. It gives no advice on how many standards, which specimens, or what counts as success. Section numbers are also ambiguous, because the guide has two 'Chapter 4's (known issue).
- **Proposal:** Organise the chapter by scenario, each with a recipe and the diagnostic to check:
- A second instrument with 3-15 rescannable specimens: slope/bias or TSR, then regularised PDS; choose by leave-one-out RMSEP.
- The same instrument after drift or a lamp change: check standard, then slope/bias.
- Different wavelength range or resolution: take the overlap, match resolution, retrain.
- No shared specimens: di-PLS, with the no-correction baseline.
- Classification or one-class models: judge by class agreement.
Add a worked example that reproduces this benchmark on the shipped data. Cite the actual references; see the relabelling idea.
- **Benefit:** Collaborators can follow a defensible procedure and cite it in a paper.
- **Evidence:** UserGuide.md:8276-8282 (method table), :8373-8400 (s4.5 contradiction), :8403-8425 (s4.6). Benchmark script and results: C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\calibration_transfer\ct_bench.py, bench_out.txt, bench_severe_out.txt.

## Lens: contamination

**Current state.** dasp flags contamination reasonably well, but its contaminant-removal path is mostly not working. Scratch scripts are in C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\contamination\.

What is good: the five one-class engines, CV that trains on inliers only with pooled metrics (contamination.py:552-896), external validation, and multi-class SIMCA. simca.py already contains cross-fit empirical p-values (_cross_fit_null :946, _empirical_p :1053), Wilson CIs (:1197) and Wold modeling power (:1324). Those are the right building blocks, but they are only wired into the multi-class path.

Problems with FLAGGING:
(a) "DD-SIMCA" (contamination.py:62-325) is not Pomerantsev's DD-SIMCA. It combines T2 and Q with Fisher's method, fits the chi2 limits on in-sample statistics, and has no extreme/outlier (gamma) split and no robust estimator. Its type-I error is badly off at dasp's sample sizes: on the 49 SNV bone spectra, held-out clean rejection at alpha=0.05 is 14% (k=3), 22% (k=5) and 36% (k=8). In simulation at n=20-40, actual false rejection is 10-100% when k is too large (oc_typeI.py).
(b) The decision threshold is tuned on the contaminated samples. The Bayesian search tunes SIMCA alpha over 0.01-0.20 and contamination/nu (unified_bayesian.py:964-990), and the grid does the same (contamination.py:421-464), to maximise balanced accuracy on the same outliers that are then reported. In simulation, CV sensitivity was 0.744 against 0.683 on fresh contaminants (oc_bias.py).
(c) Outputs are hard to communicate. There is no acceptance plot and no extreme plot anywhere. Prediction-time "confidence" is a sigmoid of the decision score, and the AD status comes from in-sample q10/q25 of training scores (model_io.py:979-1011), so 10% of training inliers are labelled "extrapolation" by construction.
(d) Data Quality counts T2 and Mahalanobis as two independent votes, but they are the same statistic (corr(T2, MD^2) = 1.0000, ratio exactly 1). The Q flag is an in-sample 95th percentile, so it always flags about 5% of samples (outlier_detection.py:221). On pure Gaussian null data, 3 of 49 samples came out as "2+ methods" outliers (dq_check.py).

Problems with REMOVAL:
(e) The GUI's EPO correction (_apply_epo_projection :60158) and EPO influence detection (contaminant_analysis.py:1772) use EstimatedEPO's default estimation_method='pca_diff'. That method builds noisy copies of the mean difference and then mean-centres them (:675-678), which deletes the contaminant direction. The removed subspace contains 4.5% of the true contaminant direction and leaves 3.879 of 3.886 of the group difference in place (epo_check.py). Its wavelength-influence map correlates 0.04 with the true band (infl.py).
(f) "OPLS-DA Filter" (:1158-1164) keeps the clean-vs-contaminated direction and strips the within-group variation, which carries the analyte. That is the opposite of removal.
(g) The correction overwrites self.X in place (:60088-60104). It is not fitted inside CV folds, not saved in the .dasp model, and not applied at prediction. "Export Corrected Spectra" is a placeholder that still shows a success label (:60250-60262), and EstimatedEPO.transform returns mean-centred spectra.
(h) The per-fold interference pipeline (preprocess.py:345-475) exists, but it is disabled in the GUI (:30853 "DISABLED: Code stashed", and :41437). Meanwhile the Interference tab's "apply_to_analysis" defaults to True (:2901).
(i) The only removal method that works end to end today is "Exclude Regions", which writes a custom-region string into Analysis Configuration.
(j) The tests check only array shapes, never that a contaminant is actually removed (tests/test_contaminant_analysis.py:225-270).

Speed: the "LOF is O(n^2)" known issue is misdiagnosed. The real cost is OpenMP thread spin-up on a 24-core machine: the 9-config LOF grid takes 47.6 s by default and 0.96 s under threadpool_limits(openmp=1).

Discoverability: the clean and contaminated groups are loaded from separate files inside Contaminant Analysis, not taken from the main dataset. Work is split across 5 places: Data Quality, one-class in Analysis Config, Contaminant Analysis, Interference Removal and Spectral Library.

### Fix the contaminant-removal maths and test that it removes the contaminant (method, impact high, effort S)

- **Problem:** Default EstimatedEPO 'pca_diff' mean-centres a library of noisy copies of the group-mean difference (contaminant_analysis.py:621-645, then _build_projection_matrix :675-678). Centring subtracts the contaminant direction and leaves only random noise to project out. The GUI's 'EPO Projection' (spectral_predict_gui_optimized.py:60158-60172) and the 'Estimated EPO' detection influence map (contaminant_analysis.py:1772) both use this default. 'bootstrap' has the same centring flaw: it projects out sampling variation of the difference, not the difference itself. 'OPLS-DA Filter' (:1158-1164; GUI :60190-60204) removes orthogonal, within-group variation, where the analyte lives, and keeps the contaminant. EstimatedEPO.transform returns X - mean (:763-766), so corrected spectra are centred, which changes what SNV or derivatives do downstream. The unit tests assert only shapes (tests/test_contaminant_analysis.py:225-270).
- **Proposal:** (1) Do not centre the interference library. Build D from the uncentred group-mean difference plus deviations of contaminated spectra from the clean-class PCA reconstruction, then take the SVD of D and use P = I - VV'. Return X @ P without centring, so the output stays a spectrum. (2) Make 'mean_diff' the default and delete 'pca_diff' or rewrite it. (3) Rewrite the OPLS-DA correction to remove the predictive (group) component, or drop it from 'Apply Correction'. (4) Report an 'analyte retained' figure next to every correction: the fraction of the clean-class PCA subspace, or of a PLS regression vector fitted on clean samples, that survives P. This is the net-analyte-signal idea, and it warns when the contaminant band overlaps the analyte. (5) Add behavioural tests: on a synthetic contaminant band, after correction the group difference along the band is under 5% and the retained analyte signal is over 90%.
- **Benefit:** 'Apply EPO correction' for Glyptal-treated bone would actually remove the Glyptal signal, and users would see how much collagen signal each correction costs.
- **Evidence:** epo_check.py: fraction of the contaminant direction in the removed subspace is mean_diff 0.997, pca_diff 0.045, bootstrap 0.999 (this case has variable contaminant amount). Residual group difference along the band: pca_diff 3.879 against raw 3.886. infl.py: corr(EPO influence, true band) = 0.039. opls_check.py: pooled PLS CV R2 goes from 0.997 to 0.928 after the GUI OPLS-DA filter. References: Roger J-M, Chauchard F, Bellon-Maurel V (2003) EPO-PLS external parameter orthogonalisation of PLS: application to temperature-independent measurement of sugar content of intact fruits. Chemometr Intell Lab Syst 66(2):191-204, doi:10.1016/S0169-7439(03)00051-0. Lorber A, Faber K, Kowalski BR (1997) Net analyte signal calculation in multivariate calibration. Anal Chem 69(8):1620-1626, doi:10.1021/ac960862b.

### Fit contaminant correction inside CV folds, save it with the model, and compare with and without (workflow, impact high, effort L)

- **Problem:** 'Apply Correction' fits on the whole of both groups and overwrites self.X in place (GUI :60088-60104). The fitted transformer lives only as self.contam_epo_transformer, is never written to the .dasp file (model_io.py has no interference or contaminant hooks), and is not applied in Model Prediction. 'Export Corrected Spectra' is a stub that shows a success label (:60250-60262). The backend already has per-fold EPO/DOSC/GLSW steps (preprocess.py:345-475; search.py:1882-1917), but the GUI disables them (:30853 'DISABLED: Code stashed (broke R² reproducibility)'; :41437 interference=None). The Interference tab's 'apply_to_analysis' checkbox defaults to True (:2901) and silently does nothing. Correcting the whole dataset and then cross-validating leaks the group information and gives optimistic RMSECV, and nothing in the app shows whether the correction helped.
- **Proposal:** Add a ContaminantCorrector transformer (EPO, true GLSW, or the EMSC-interferent method below). It is fitted on fold-train rows using a group column from the main dataset, runs inside the preprocessing pipeline, and is serialised into the .dasp and replayed by predict_with_model. Fix the reproducibility bug that caused the stash; don't leave the feature switched off. In Results, add one 'Correction A/B' check: the same CV splits with and without correction, RMSECV/R2 split by group (clean vs each contaminant), and a paired per-specimen difference. Remove or grey out 'apply_to_analysis' until it is wired up.
- **Benefit:** A user could say 'Glyptal correction lowers RMSECV on consolidated specimens from X to Y with no cost on clean bone', defend that claim, and have new consolidated specimens corrected automatically at prediction.
- **Evidence:** GUI :60025-60110 (apply), :60250-60262 (export stub), :30853 and :41437 (interference disabled), :2901 (apply_to_analysis=True). preprocess.py:345-475 (pipeline steps exist). No 'interference' anywhere in model_io.py (grep).

### EMSC with a contaminant spectrum: estimate how much contaminant each sample carries, then subtract it (method, impact high, effort M)

- **Problem:** None of dasp's removal methods gives a per-sample amount of contaminant. EPO and GLSW project out or down-weight a whole subspace, and region exclusion throws bands away. For consolidants such as Glyptal, the useful questions are how much is on each specimen and whether it can be subtracted. EMSC is absent from src (confirmed by grep), although EMSC with interferent spectra is the classic tool for exactly this.
- **Proposal:** Implement an EMSC transformer (Martens & Stark 1991). Each spectrum is fitted as a + b*m(lambda) + polynomial baseline + sum_j c_j*k_j(lambda), where m is the clean mean and k_j is the interferent. k_j can come from the Spectral Library (a reference film or ATR spectrum of Glyptal), from the uncentred group-mean difference, or from the first PCs of contaminated-minus-clean residuals. Output the corrected spectrum (x - sum c_j k_j)/b and the per-sample c_j with standard errors. Show c_j as a 'contaminant load' column in the Data Viewer and in prediction output, and offer it as a screening statistic next to the one-class models.
- **Benefit:** A single tool that both flags and removes consolidant, gives an interpretable per-specimen amount (useful in the archaeology write-up), and doubles as the missing EMSC preprocessing.
- **Evidence:** No EMSC in src/spectral_predict (grep). Contaminant use case: UserGuide.md:8652-8655. References: Martens H, Stark E (1991) Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy. J Pharm Biomed Anal 9(8):625-635, doi:10.1016/0731-7085(91)80188-F. Martens H, Nielsen JP, Engelsen SB (2003) Light scattering and light absorbance separated by extended multiplicative signal correction. Anal Chem 75(3):394-404, doi:10.1021/ac020194w.

### Make PCA-SIMCA real DD-SIMCA with a false-rejection rate that matches alpha (method, impact high, effort M)

- **Problem:** PCASIMCA (contamination.py:62-325) combines T2 and Q p-values with Fisher's method, using chi2 fits to in-sample training statistics. At dasp's sample sizes, training Q and T2 underestimate what new samples produce, so the real false-rejection rate is far above alpha. It also lacks the DD-SIMCA full distance f = Nh*h/h0 + Nq*q/q0 ~ chi2(Nh+Nq), the extreme (alpha) versus outlier (gamma, family-wise) split, and robust estimators of h0 and Nh. The docstring cites DD-SIMCA, but the implementation differs from it.
- **Proposal:** Implement standard DD-SIMCA: the full distance, extreme and outlier limits, and 'classic' and 'robust' parameter estimation. Add an option to estimate h0/Nh/q0/Nq from out-of-fold distances, reusing the cross-fit pattern in simca.py:946-1072. Cross-check the numbers against the MIT reference implementation (pip install ddsimca, github.com/svkucheryavski/ddsimca-py) in a parity test. Keep the Fisher variant as a legacy option so saved models still load.
- **Benefit:** 'alpha = 0.05' would mean about 5% of genuine clean specimens rejected, not 14-36%, so a screening decision can be quoted in a paper.
- **Evidence:** oc_typeI.py on example bone ASD (SNV, repeated 5-fold, alpha=0.05): held-out clean rejection 0.143 (k=3), 0.220 (k=5), 0.355 (k=8). Synthetic 4-rank class with p=200: n=20, k=5 gives actual FPR 0.908; n=40, k=5 gives 0.312; an out-of-fold-calibrated threshold gives 0.015-0.028. References: Pomerantsev AL (2008) Acceptance areas for multivariate classification derived by projection methods. J Chemometrics 22(11-12):601-609, doi:10.1002/cem.1147. Pomerantsev AL, Rodionova OYe (2014) Concept and role of extreme objects in PCA/SIMCA. J Chemometrics 28:429-438, doi:10.1002/cem.2506. Kucheryavskiy S, Rodionova O, Pomerantsev A (2024) A comprehensive tutorial on Data-Driven SIMCA: theory and implementation in web. J Chemometrics 38(7):e3556, doi:10.1002/cem.3556.

### Add a 'rigorous' mode: fix alpha in advance and stop tuning thresholds on the contaminated samples (method, impact high, effort M)

- **Problem:** One-class search tunes the decision threshold (SIMCA alpha 0.01-0.20, IF/LOF/EE contamination 0.001-0.3, OCSVM nu) to maximise balanced accuracy on the outliers (unified_bayesian.py:964-990; grid contamination.py:421-464). Variable selection is also supervised by y_oc (unified_bayesian.py:1339-1342). Every outlier is in every test fold (contamination.py:664-668). The one-class model therefore becomes a two-class model tuned to the contaminants on hand, and the reported sensitivity is optimistic for new kinds of contaminant, which is the reason to use a one-class model at all. Per-contaminant sensitivity is computed only in calibration, not CV (contamination.py:804-812).
- **Proposal:** Add a 'Rigorous (target class only)' option and make it the default. The user fixes alpha. Complexity (k, nu, n_neighbors) is chosen only by how close CV specificity on inliers is to 1-alpha, plus an extreme plot. Contaminated samples are held back for one final sensitivity estimate. Also report leave-one-contaminant-group-out sensitivity using y_original, which tests unseen contaminant types. Keep the current behaviour as 'Compliant (uses alternative class)', labelled as optimistic.
- **Benefit:** Screening claims such as 'catches 90% of consolidated specimens' would hold for new consolidants and new sites, not only the ones the model was tuned on.
- **Evidence:** oc_bias.py (15 reps, 40 inliers and 10 weak-contaminant outliers, 36-config SIMCA grid chosen by CV balanced accuracy): CV sensitivity 0.744 against 0.683 on 500 fresh contaminants. CV specificity 0.718 against a nominal 0.99, because alpha=0.01 was chosen in 14 of 15 reps. References: Rodionova OYe, Oliveri P, Pomerantsev AL (2016) Rigorous and compliant approaches to one-class classification. Chemometr Intell Lab Syst 159:89-96, doi:10.1016/j.chemolab.2016.10.002. Rodionova OYe, Titova AV, Pomerantsev AL (2016) Discriminant analysis is an inappropriate method of authentication. TrAC Trends Anal Chem 78:17-22.

### Make LOF and the one-class grid about 4x faster by capping OpenMP threads (speed, impact high, effort S)

- **Problem:** The known issue blames one-class slowness on LOF's O(n^2) cost (PROJECT_STATUS.md:524). At n=61 that cost is negligible. The real cost is scikit-learn's OpenMP pairwise-distance code starting 24 threads for a microsecond job on every fit and score. A single LOF fit+score takes 0.80-1.02 s by default and 0.001 s with OpenMP limited to one thread. Capping BLAS instead does not help (0.28 s). Nothing in src or the GUI uses threadpool_limits (grep).
- **Proposal:** Wrap run_one_class_cv, the one-class Bayesian objective and simca._cross_fit_null in threadpoolctl.threadpool_limits(1, user_api='openmp'). Better, apply the cap once per search worker, since dasp parallelises at the config level. Separately, stop refitting PCA for every alpha: alpha only moves the threshold, so fit once per (fold, max k) and score every k and alpha from that one SVD.
- **Benefit:** A default one-class grid on the example data would run in about 16 s instead of 62 s, and the Bayesian one-class path would speed up by a similar factor.
- **Evidence:** oc_time.py on 49 SNV bone spectra plus 12 synthetic contaminated spectra (5-fold, full default grid). Default: LOF 47.56 s (5.3 s/config), IF 10.2 s, SIMCA 1.95 s, total 61.8 s. With threadpool_limits(1): LOF 0.96 s, total 16.3 s. lof3.py: openmp cap gives 0.0009 s per fit, blas cap 0.28 s, default 0.80 s (24-core machine).

### Add an acceptance plot, an extreme plot, and p-values in one-class results and predictions (visual, impact high, effort M)

- **Problem:** One-class diagnostics are generic: a metrics bar chart, count bars, a confusion matrix and a sorted decision-score histogram (GUI :37286-37360, :38269ff). There is no acceptance plot or extreme plot anywhere (grep for 'acceptance' and 'extreme' finds none). Predictions show Inlier/Outlier plus a 'confidence' that is a logistic function of the raw score. The AD status uses in-sample q10/q25 of training scores (model_io.py:979-1011), so it can disagree with the predicted label and marks 10% of training inliers as 'extrapolation'. PCASIMCA.p_joint exists (contamination.py:286) but is used only by multi-class SIMCA (simca.py:567).
- **Proposal:** In Model Development and Model Prediction, show: (1) a DD-SIMCA acceptance plot, log(1+h/h0) against log(1+q/q0), with the extreme and outlier boundary curves and points coloured by clean, each contaminant group and new samples; (2) an extreme plot of observed against expected extreme counts over alpha with a binomial tolerance band, used to choose k; (3) sensitivity per contaminant group and specificity with Wilson 95% CIs, reusing simca.wilson_ci (:1197). Prediction output becomes: p-value, regular/extreme/outlier, and the alpha and gamma used. Include these figures in an exported one-class report.
- **Benefit:** Results a reviewer recognises, with honest uncertainty (sensitivity from 12 contaminated specimens has a CI roughly ±0.25 wide), and a per-sample verdict that matches the plot.
- **Evidence:** GUI :37286-37360 (current plots). model_io.py:979-1011 (percentile AD and sigmoid confidence). contamination.py:286 (p_joint unused in the one-class path). simca.py:1197 (wilson_ci exists). The acceptance and extreme plots are standard DD-SIMCA outputs: Pomerantsev & Rodionova 2014, J Chemometrics 28:429-438, doi:10.1002/cem.2506. Kucheryavskiy, Rodionova & Pomerantsev 2024, J Chemometrics 38(7):e3556, doi:10.1002/cem.3556.

### Rebuild Data Quality outlier screening: remove the double vote, control false flags, use the modelling preprocessing, add a robust option (method, impact high, effort M)

- **Problem:** generate_outlier_report adds T2, Q, Mahalanobis and Y flags into 'Total_Flags' and treats 2 or more as moderate or high confidence (outlier_detection.py:589-600). Mahalanobis distance in PCA-score space is the same quantity as T2 (verified: correlation 1.0, MD^2/T2 ratio 1.0000-1.0000), so every T2 flag counts twice. The Q limit is the in-sample 95th percentile (:221), which flags about 5% of any dataset. The T2 limit is a per-sample 95% F limit with no family-wise correction. PCA is classical, so gross outliers can mask themselves. Detection runs on raw or absorbance spectra (GUI :21648), not on the SNV/derivative data the models use.
- **Proposal:** Replace the voting table with DD-SIMCA extreme/outlier classification. Outliers use a family-wise gamma limit, (1-gamma)^(1/n), so a clean dataset of 49 rarely produces any flag. Run it on the user's chosen preprocessing. Offer robust estimation: the robust DD-SIMCA parameters, or ROBPCA. Show the same acceptance plot as the one-class idea, and keep a separate Y-range check.
- **Benefit:** Fewer clean specimens wrongly dropped from small datasets, and outlier removal that can be justified in a methods section.
- **Evidence:** dq_check.py: on the example bone data corr(T2, MD^2) = 1.0. On pure Gaussian null data (49x300, 5 PCs, no outliers): T2 flags 3, Q 3, Maha 3, and 3 samples reach '2+ flags'. outlier_detection.py:130-136 (F limit), :221 (percentile Q), :589-600 (voting). References: Hubert M, Rousseeuw PJ, Vanden Branden K (2005) ROBPCA: a new approach to robust principal component analysis. Technometrics 47(1):64-79, doi:10.1198/004017004000000563. Pomerantsev & Rodionova 2014 (above).

### Give every one-class engine calibrated out-of-fold p-values instead of a tuned threshold (method, impact medium, effort M)

- **Problem:** For OCSVM, IsolationForest, LOF and EllipticEnvelope, the Inlier/Outlier cut is set by nu or contamination as an in-sample quantile (contamination.py:327-379). Scores are not comparable across engines or across folds; run_one_class_cv drops pooled scores under repeated CV for this reason (:716-727). simca.MultiClassClassModel already turns these engines into add-one-smoothed empirical p-values using a cross-fit null (simca.py:946-1072), but the one-class path does not use it.
- **Proposal:** Factor _cross_fit_null and _empirical_p into a shared helper and apply it in the one-class path. Every engine then outputs p = (1 + #{OOF null <= s})/(m+1) and uses one user-set alpha. Save the null array in the .dasp so prediction returns p-values. With this, engines can be compared at equal nominal specificity, and pooled AUC across repeats becomes valid.
- **Benefit:** All five engines share one meaning of alpha, so choosing between OCSVM and SIMCA is a fair comparison and prediction outputs are consistent.
- **Evidence:** simca.py:946-1072 (existing cross-fit null). contamination.py:716-727 (scores dropped under repeated CV). oc_typeI.py: an out-of-fold-calibrated threshold on PCASIMCA holds FPR at 0.004-0.034 against nominal 0.05, compared with 0.10-1.00 in-sample. Reference: Vovk V, Gammerman A, Shafer G (2005) Algorithmic Learning in a Random World. Springer, New York, ISBN 978-0-387-00152-4 (conformal p-values).

### Use the fitted PLS model's own total distance for the regression applicability domain, and attach an optional contamination screen (method, impact medium, effort M)

- **Problem:** Regression AD fits a separate PCA. It uses an F(0.99) T2 limit and an in-sample 99th-percentile Q limit (model_io.py:1190-1215), plus nearest-neighbour percentile zones (:1160-1175), which gives several overlapping status vocabularies ('good/caution/extrapolation' and 'within_domain/influential/new_features/outside_domain'). None of this uses the calibration model's latent space. At prediction there is also no way to ask whether a new specimen looks consolidated before trusting its %collagen.
- **Proposal:** Compute AD from the PLS X-scores and X-residuals of the saved model, using the data-driven total distance with regular/extreme/outlier categories (Rodionova & Pomerantsev 2020). Allow a saved one-class 'clean bone' DD-SIMCA model to be attached to a regression model, so Model Prediction shows '%Collagen = 12.3, clean-class p = 0.002 (outlier; possible consolidant)' and optionally the EMSC contaminant load.
- **Benefit:** One clear rule for trusting a prediction, and consolidated or out-of-population specimens are caught before a wrong %collagen value is reported.
- **Evidence:** model_io.py:1136-1235 (current regression AD) and :979-1011 (one-class AD). Reference: Rodionova OYe, Pomerantsev AL (2020) Detection of outliers in projection-based modeling. Anal Chem 92(3):2656-2664, doi:10.1021/acs.analchem.9b04611.

### Merge contamination work into one Screen & Clean workspace driven by a group column in the main dataset (usability, impact medium, effort L)

- **Problem:** Screening and cleanup are spread over Data Quality (:11178), one-class settings in Analysis Configuration, Contaminant Analysis (:57604, 4 subtabs), Interference Removal (:55487, 4 subtabs) and Spectral Library (:56339). Contaminant Analysis makes the user load clean and contaminated data again from separate files or folders (_contam_load_clean_data :58606) instead of using a group column in the already-loaded dataset. The interferent libraries in Interference Removal (self.interferent_libraries, :2877) and the estimated contaminant spectra in Contaminant Analysis never meet. Its 'Apply to' radio defaults to 'Contaminant Groups', not the main dataset (:58300).
- **Proposal:** One workspace with four steps, all on the main dataset: 1 Groups (pick a group column: clean, Glyptal, PVA, unknown). 2 Screen (DD-SIMCA acceptance plot, EMSC load, difference spectra). 3 Correct (EMSC-interferent, fixed EPO, GLSW; interferent from the Spectral Library or estimated from groups; before-and-after acceptance plot and analyte-retained figure). 4 Use (push the correction to the pipeline as a CV-fitted step, or push excluded regions). Then retire the duplicate loaders and the Interference 'Application' subtab.
- **Benefit:** The Glyptal workflow becomes a single linear path instead of five tabs, with no reloading of files.
- **Evidence:** Tab anchors in the brief. GUI :58606 (separate loader), :2877 (interferent_libraries), :58300 (default 'Contaminant Groups'), :59909-59984 (apply clean regions, currently the only step that reaches the analysis).

### Replace the diagonal 'GLSW' with real GLSW, and offer one-class variable selection (Wold modeling power / LOVE) (method, impact medium, effort M)

- **Problem:** interference.GLSW 'covariance' mode is inverse per-wavelength variance, which amounts to autoscaling without centring (interference.py:704-718). ContaminantGLSW is a heuristic per-wavelength weight built from a variance ratio plus the mean difference (contaminant_analysis.py:1271-1303). Neither is the clutter-covariance filter of Martens et al. (2003), so neither can down-weight a contaminant direction that overlaps analyte bands. For one-class models, variable selection is effectively supervised by y_oc (unified_bayesian.py:1339-1342), even though simca.wold_variable_selection (simca.py:1627) and wold_modeling_power (:1324) exist and are wired only into multi-class SIMCA.
- **Proposal:** Implement GLSW as G = V (S^2/a + I)^(-1/2) V', where the clutter covariance comes from contaminated-minus-clean-mean residuals, or from paired or replicate differences. Expose 'a' as the single parameter. For one-class variable selection, expose Wold modeling power computed on the target class only. Optionally add the LOVE wrapper, which removes variables by their effect on DD-SIMCA performance, instead of selectors driven by y_oc.
- **Benefit:** A GLSW that actually targets the interferent direction, and variable selection that stays one-class, which in turn keeps the rigorous evaluation honest.
- **Evidence:** interference.py:686-718, contaminant_analysis.py:1271-1303, unified_bayesian.py:1339-1342, simca.py:1324 and :1627 (no callers outside simca.py). References: Martens H, Høy M, Wise BM, Bro R, Brockhoff PB (2003) Pre-whitening of data by covariance-weighted pre-processing. J Chemometrics 17:153-165, doi:10.1002/cem.780. Pomerantsev AL, Kucheryavskiy S, Rodionova OYe (2025) Variable selection for one class classifiers. Introduction of LOVE. Anal Chim Acta 1368:344302, doi:10.1016/j.aca.2025.344302.

## Lens: modelling-workflow

**Current state.** For a working NIR/MIR chemometrician, dasp's search engine is stronger than most commercial tools. It pools CV predictions correctly (search.py:4966-4993 computes RMSECV from pooled predictions, not by averaging folds), reports RPD, RER, Bias, MAE, CCC and CV-ANOVA (search.py:4998-5010, 5460-5476), offers about 19 selectors, and draws a VIP plot with the VIP>1 line (GUI :37826-37960). The weak part is the scientific workflow around the search.

1) Validation design ignores replicate and site structure. There is no group concept anywhere in the main flow: build_cv_splitter offers only kfold, repeated and LOO (cv_utils.py:275-340), and group splitters raise NotImplementedError (cv_utils.py:220-225).

2) The holdout partitioner runs Kennard-Stone/SPXY backwards. It assigns the samples KS picks to the VALIDATION set, and it does so on raw spectra (GUI :20589-20632, :20859; X_available = self.X at :20795). On the bundled 49-sample bone data this puts the most extreme %Collagen samples (0.9, 1.1, 1.7, 21.7, 22.1) into validation. Two validation samples then fall outside the calibration y-range, and RMSEP is 3.47 against a random-split mean of 2.34.

3) The statistics are incomplete for a paper or an ISO 12099 check. There is no SEP/SECV, RPIQ, slope/intercept or bias significance test. External validation stores only RMSEP and R2pred (search.py:1186-1189), and the pred-vs-obs box shows only R², RMSE, MAE and n (GUI ~:37517).

4) Nothing puts an uncertainty on a single split or on the gap between two models. With n=49 and 10 held-out samples, 200 random splits gave RMSEP from 1.43 to 3.43 (5th-95th percentile).

5) Preprocessing coverage is thin as a searchable axis. MSC and OSC exist (interference.py:217, 356), but only as global toggles applied to every config (preprocess.py:370-381, search.py:1889). EMSC, SNV-detrend and Norris gap derivatives are absent.

6) Variable selection gives one run and one answer. Six CARS seeds on the example data gave a mean pairwise Jaccard of 0.43 on the top-30 bands, and only 3 bands appeared in all six runs. Selection frequency still concentrated on about 6 contiguous regions (near 1195, 1242, 1656, 1702, 1726 and 2268 nm), so a frequency ranking would be informative.

7) Reporting is a 156-line top-5 Markdown text dump (report.py:10-156): no methods text, no figures, no figures-of-merit table. Plot export is PNG/PDF/SVG at 300 dpi but keeps the GUI theme background (GUI :16901).

8) There are no per-sample prediction intervals for regression. predict_with_uncertainty returns only the global RMSECV plus RandomForest tree variance (model_io.py:864-866), and diagnostics.jackknife_prediction_intervals (diagnostics.py:143) has no callers.

Probe scripts and outputs are in C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\modelling\probe.py and probe2.py.

### First-class group/specimen column driving CV, holdout, replicate averaging and selector inner CV (method, impact high, effort L)

- **Problem:** Every real dataset named in the logs has replicate scans per specimen or site/batch structure: Border Cave, the multi-site FTIR bone paper, and leaf_phys_nir. Yet build_cv_splitter only knows kfold, repeated_kfold and loo (cv_utils.py:275-340), and compute_min_train_fold_size raises NotImplementedError for group_kfold and leave_one_group_out (cv_utils.py:220-225). The GUI has no group column outside Contaminant Analysis (the contam_combined_group_col at GUI :2936 is the only one). Replicates of one specimen can therefore fall in both train and test folds, which inflates R2cv. The holdout split (GUI :20859) and the selector inner loops, which are hard-coded to 5-fold (cv_folds=5 defaults in variable_selection.py:62, 373, 1250), are not group-aware either. GAP_ANALYSIS.md #1 and #12 call this the largest single gap.
- **Proposal:** (a) Add a 'Specimen/Group ID' picker at Import & Preview, stored as self.groups aligned to X. (b) Extend build_cv_splitter(strategy, ..., groups=None) with 'group_kfold' (GroupKFold / StratifiedGroupKFold), 'logo' (LeaveOneGroupOut), 'venetian_blinds' and 'contiguous_blocks' (the PLS_Toolbox staples for sorted or time-ordered data). Thread groups through run_search, unified_bayesian and the selector cv_folds arguments; the selectors already take a folds argument, so pass a splitter instead of an int. (c) Add an optional 'average replicates by group before modelling' toggle, reporting per-specimen instead of per-scan metrics. (d) Make the holdout partitioners select whole groups (KS on group-mean spectra).
- **Benefit:** Honest CV numbers on replicate- and site-structured data without dropping to a hand-written Python loop (AGENT_COMPOSITION.md s4). A reviewer's first objection to a pooled multi-site paper is answered inside the app.
- **Evidence:** cv_utils.py:177-225, 275-340; GUI :20780-20870 (holdout has no group handling); docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md #1, #12, #13; PROJECT_STATUS 'CV Strategy Phase 2' (inner loops hard-coded to 5-fold).

### Fix the holdout partitioner direction (KS/SPXY choose the calibration set), run it on preprocessed/PCA space, add stratified KS and DUPLEX (method, impact high, effort S)

- **Problem:** _validation_kennard_stone and _validation_spxy (GUI :20589-20700) return the KS-selected samples as the VALIDATION set (GUI :20859-20861, :20868). KS starts from the two most distant spectra and then adds boundary points, so validation gets the extremes and calibration loses its edges. The partitioner also runs on raw self.X (GUI :20795), where baseline and scatter dominate Euclidean distance. The measured effect on example/ (49 bone spectra, 10 held out), using PLS-5 on SNV+SG1: with KS on raw X assigned to validation, the validation y values are 0.9, 1.1, 1.7, 5.9, 9.7, 18.4, 19.2, 19.6, 21.7 and 22.1. Two of them fall outside the calibration range (0.9-21.5), and RMSEP is 3.47. The textbook direction (KS picks calibration, the remainder validates) gives 0 samples outside the range and RMSEP 2.14. 200 random splits gave a mean of 2.34. The GUI also duplicates sample_selection.py instead of reusing it, and it does not offer the DUPLEX implementation that already exists at sample_selection.py:138.
- **Proposal:** Route the holdout through sample_selection.kennard_stone/spxy/duplex and assign the selected samples to CALIBRATION, with the remainder as validation, following Kennard & Stone 1969 and Galvão et al. 2005. Compute distances on the preprocessed spectra of the chosen pipeline, or on Mahalanobis distance in the first k PCA scores, and add a dropdown for this space. Add 'Stratified KS' (KS within each class or y-quantile bin, per GAP_ANALYSIS #11) and 'DUPLEX' (Snee 1977). Show a small plot of the cal/val y-distributions and the PCA score coverage before the split is confirmed.
- **Benefit:** External validation stops being systematically pessimistic, because it no longer forces extrapolation. The split follows the convention reviewers expect, and the user can see that calibration spans validation.
- **Evidence:** GUI :20589-20632 (KS), :20634-20700 (SPXY), :20795 (raw X), :20859-20868; sample_selection.py:31, 138, 243. Probe: scratchpad/improve/modelling/probe.py and probe2.py (RMSEP 3.47 or 3.38 when KS picks validation, vs 2.14 when KS picks calibration). Refs: Kennard RW, Stone LA (1969) Computer aided design of experiments. Technometrics 11(1):137-148, doi:10.1080/00401706.1969.10490666. Galvão RKH, Araujo MCU, José GE, Pontes MJC, Silva EC, Saldanha TCB (2005) A method for calibration and validation subset partitioning. Talanta 67(4):736-740, doi:10.1016/j.talanta.2005.03.025. Snee RD (1977) Validation of regression models: methods and examples. Technometrics 19(4):415-428, doi:10.1080/00401706.1977.10489581.

### Complete figures-of-merit table (ISO 12099 / ASTM E1655) for both CV and external validation (reporting, impact high, effort S)

- **Problem:** The CV rows lack SECV/SEP (bias-corrected SD of residuals), RPIQ, slope and intercept of predicted vs reference, and a bias significance test (search.py:4998-5010, 5460-5476). External validation is poorer still, storing only RMSEP and R2pred (search.py:1186-1189): no bias, SEP, RPD or slope on the test set. The Model Development pred-vs-obs box shows only R², RMSE, MAE and n (GUI ~:37517). Bias and slope appear only in the separate bias-correction panel (GUI :15936, :37598). On the example data (PLS-5, 5-fold) the missing statistics are SECV 2.40, RPIQ 4.88 and slope 0.949, next to the existing RPD of 2.85.
- **Proposal:** Add a single metrics function, e.g. scoring.regression_figures_of_merit(y, yhat, y_cal=None), returning n, RMSE, bias, SEP (sd of residuals, ddof=1), bias t-test p and ISO 12099 bias control limit, slope, intercept, slope-vs-1 t-test p, R², RPD, RPIQ, RER and CCC. Call it for calibration, CV and test, and put the three sets side by side in Results, Model Development and the report. Add RPIQ and SEP as leaderboard columns and as filter/ranking options.
- **Benefit:** The numbers a reviewer or a feed/food lab expects are on screen and in the export, so nobody has to recompute them in Excel. Bias and slope tests also tell the user when a slope/bias correction is justified.
- **Evidence:** search.py:1186-1189, 4998-5010, 5460-5476; GUI :37517 (stats_text), :15936. Probe output: RMSECV 2.395, bias -0.284, SECV 2.403, RPD 2.85, RPIQ 4.88, slope 0.949. Refs: Bellon-Maurel V, Fernandez-Ahumada E, Palagos B, Roger J-M, McBratney A (2010) Critical review of chemometric indicators commonly used for assessing the quality of the prediction of soil attributes by NIR spectroscopy. TrAC Trends Anal Chem 29(9):1073-1081, doi:10.1016/j.trac.2010.05.006 (in Matt's Paperpile). ISO 12099:2017 Animal feeding stuffs, cereals and milled cereal products - Guidelines for the application of near infrared spectrometry. ASTM E1655-17 Standard Practices for Infrared Multivariate Quantitative Analysis.

### Publication report: auto-written methods paragraph, figures-of-merit table and a standard figure set, exported as HTML/PDF (reporting, impact high, effort M)

- **Problem:** report.py (156 lines) writes the top-5 rows as Markdown text (report.py:10-156). It has no data description (n, spectral range, instrument, data type), no CV or holdout design, no preprocessing or selection narrative, and no figures. The 38 plot-export buttons save one figure at a time with the GUI theme background (fig.patch.set_facecolor(self.colors['bg']), GUI :37926 and :16901). A paper therefore needs about 10 manual exports and hand-written methods text. MASTER_ANALYSIS.md lists reporting as blocker 1.
- **Proposal:** Add a 'Publication report' button on Results and Model Development that produces one self-contained HTML file (optional PDF via matplotlib PdfPages) plus a figures/ folder of SVG and PDF at journal widths of 85 and 175 mm. It should contain: (1) a methods paragraph generated from the run config (n specimens and scans, wavelength range and step, preprocessing chain with SG window/order, selector and n bands, model and LVs, CV splitter with folds/repeats/groups, holdout method and size, number of candidate configurations evaluated, dasp version and seed). (2) The figures-of-merit table from the idea above for cal, CV and test. (3) Standard figures: pred vs reference with 1:1 line and CV/test markers, RMSECV vs LV, residuals vs predicted, signed coefficients plus VIP over the mean spectrum with selected bands shaded, and a PCA score plot with the cal/val split. (4) A sample-level CSV (ID, group, y, ŷ_cv, residual, leverage, T², Q). Use a 'publication' matplotlib style: white background, 8-9 pt fonts, a colour-blind-safe palette.
- **Benefit:** Moves the output from 'screenshot the app' to a methods section and supplementary figures the user can drop into a manuscript. This is also the clearest visual upgrade toward commercial readiness.
- **Evidence:** report.py:10-156; GUI :16861-16905 (_add_plot_export_button, dpi=300, themed bg), :31062 (only caller of write_markdown_report); docs/analysis_vs_unscrambler/MASTER_ANALYSIS.md ('Plot quality: functional matplotlib').

### Validation uncertainty: repeated-split / rdCV robustness run, bootstrap CIs, and a paired randomization test between two leaderboard rows (method, impact high, effort M)

- **Problem:** All reported numbers are point estimates. On n=49 with a 10-sample holdout, RMSEP ranges from 1.43 to 3.43 (5th-95th percentile) across 200 random splits, so a single holdout cannot separate a good model from a lucky split. The leaderboard also ranks hundreds of configurations by R2cv (scoring.py:15-60), and the top row's R2cv is the maximum of many noisy estimates (winner's curse). Nothing reports how many configurations were compared, and there is no test of whether rank 1 beats rank 2 (GAP_ANALYSIS #2, #3, #5). The user's stance, recorded in SESSION_LOG_ARCHIVE.md:4018, is that selection bias is handled by external test or double CV, not by per-fold selection. rdCV is exactly that double-CV route, and it is not available.
- **Proposal:** Add a 'Robustness' action on a selected row (or two rows) that reruns the frozen pipeline: preprocessing, the locked band set, model and hyperparameters. Option A: N repeated random or stratified (and group-aware) splits, reporting the mean ± SD and 5-95% range of RMSEP, R² and bias. Option B: repeated double CV (Filzmoser et al. 2009), where the outer loop re-runs the inner LV choice and optionally the selector, giving an honest prediction-error distribution. For two rows, add a paired randomization t-test on pooled CV squared errors (van der Voet 1994), with a specimen or group block bootstrap CI on ΔRMSECV. Show 'N configurations compared' next to the top score, with a tooltip explaining optimism.
- **Benefit:** Lets the user make defensible claims such as 'model A beats B' or 'this holdout is not a lucky split' from inside the app, which is what paper reviewers ask for. Collaborators get a realistic error range instead of the best-of-500 R2cv.
- **Evidence:** Probe: 200 random 39/10 splits give RMSEP mean 2.343, SD 0.624, 5-95% 1.426-3.431; scoring.py:15-60; search.py:1186-1189 (single-shot holdout); GAP_ANALYSIS.md #2, #3, #5, #15. Refs: Filzmoser P, Liebmann B, Varmuza K (2009) Repeated double cross validation. J Chemometrics 23(4):160-171, doi:10.1002/cem.1225. van der Voet H (1994) Comparing the predictive accuracy of models using a simple randomization test. Chemom Intell Lab Syst 25(2):313-323, doi:10.1016/0169-7439(94)85050-X. Cawley GC, Talbot NLC (2010) On over-fitting in model selection and subsequent selection bias in performance evaluation. J Mach Learn Res 11:2079-2107.

### Monte-Carlo stability selection: per-band selection frequency across resamples and seeds, with band-tolerance consensus (method, impact high, effort M)

- **Problem:** Each selector returns one run's scores. CARS in particular is seed-sensitive. On example/ (SNV+SG1, PLS-5, 50 iterations), six seeds gave top-30 sets with mean pairwise Jaccard 0.43 (range 0.33-0.50): 57 distinct bands in the union and only 3 in the intersection. Frequency still concentrated: 26 bands appeared in ≥4 of 6 runs, falling into about 6 contiguous regions (1194-1198, 1241-1243, 1654-1659, 1701-1703, 1724-1728 and 2266-2270 nm). A single CARS run therefore reports noise in its exact band choice, while the frequency ranking carries a stable chemical signal. GAP_ANALYSIS #4 (multi-source consensus with ±tolerance) names the same need. At 1 nm sampling, neighbouring bands must be treated as the same feature.
- **Proposal:** Add variable_selection.stability_selection(selector, X, y, n_runs=50, subsample=0.8, seed_base, groups=None). It reruns any existing importance-array selector (cars, uve, spa, vip-threshold, mc-uve) on random or group subsamples and returns a per-band selection frequency. It also returns a smoothed frequency that pools bands within a user tolerance (±k nm), per Meinshausen & Bühlmann 2010. Expose it as a selector ('stability(CARS)' etc.) that keeps bands above a frequency threshold π, and as a plot of frequency vs wavelength over the mean spectrum with consensus regions shaded. Add the frequency vector to the saved model and the report. Rank it next to CARS/UVE/SPA in the leaderboard so the user sees whether the stable set costs RMSECV.
- **Benefit:** Band choices that survive resampling are defensible in a paper, and the band-assignment discussion rests on regions rather than one run's pixels. Rerunning CARS also stops reshuffling the user's interpretation.
- **Evidence:** variable_selection.py:1250 (cars_selection, single random_state), 62 (uve_selection); probe.py CARS seed test (Jaccard 0.43 mean, union 57, intersection 3, 6 runs in 3.8 s, so 50 runs take about 30 s on this data); GAP_ANALYSIS.md #4. Refs: Meinshausen N, Bühlmann P (2010) Stability selection. J R Stat Soc B 72(4):417-473, doi:10.1111/j.1467-9868.2010.00740.x. Cai W, Li Y, Shao X (2008) A variable selection method based on uninformative variable elimination for multivariate calibration of near-infrared spectra. Chemom Intell Lab Syst 90(2):188-194, doi:10.1016/j.chemolab.2007.10.001. Li H, Liang Y, Xu Q, Cao D (2009) Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration. Anal Chim Acta 648(1):77-84, doi:10.1016/j.aca.2009.06.046.

### Scatter-correction family as real search axes: MSC, EMSC, SNV+detrend, Norris gap derivatives, and baseline choice (method, impact medium, effort M)

- **Problem:** The grid's preprocessing axis is raw, snv, sg1-sg4 and deriv_snv (search.py:2457-2700; preprocess.py:286 docstring lists 'raw','snv','deriv','snv_deriv','deriv_snv'). MSC and OSC exist (interference.py:217, 356), but only as global toggles applied to every config (preprocess.py:370-381; search.py:1889), so the search cannot compare SNV with MSC. Baseline is one global method (search.py:593) and only an on/off in Bayesian search (unified_bayesian.py:643-647). EMSC, the standard for FTIR/ATR bone and tissue work, is absent, and so are SNV-detrend (Barnes 1989) and Norris gap-segment derivatives (the default in many NIR instrument packages). A grep of src finds none of them.
- **Proposal:** Add preprocess.EMSC(reference='mean', poly_order=2, interferents=None, constituents=None) as a sklearn transformer. It fits a reference and polynomial on the training fold and optionally takes a good or bad spectra list, which could come from the Contaminant/Interference libraries. Also add Detrend(order=2) and NorrisGapDerivative(gap, segment, order). Make msc, emsc and snv_detrend members of preprocessing_methods, crossed with the derivative options, and add the same categorical to unified_bayesian.suggest_preprocessing. Offer baseline_method as a list crossed in the grid rather than a single choice.
- **Benefit:** The search can find whether MSC or EMSC beats SNV for a dataset. That matters most for ATR-FTIR bone (particle size, contact pressure) and for leaf NIR. EMSC with an interferent spectrum also links the Contaminant tab (e.g. Glyptal) to per-fold modelling.
- **Evidence:** preprocess.py:286-381; search.py:1882-1900, 2457-2700; interference.py:217, 356; unified_bayesian.py:643-647. Refs: Martens H, Stark E (1991) Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy. J Pharm Biomed Anal 9(8):625-635, doi:10.1016/0731-7085(91)80188-F. Afseth NK, Kohler A (2012) Extended multiplicative signal correction in vibrational spectroscopy, a tutorial. Chemom Intell Lab Syst 117:92-99, doi:10.1016/j.chemolab.2012.03.004. Barnes RJ, Dhanoa MS, Lister SJ (1989) Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. Appl Spectrosc 43(5):772-777, doi:10.1366/0003702894202201. Fearn T (2008) The interaction between standard normal variate and derivatives. NIR news 19(7):16 (in Matt's Paperpile). Norris KH, Williams PC (1984) Optimization of mathematical treatments of raw near-infrared signal in the measurement of protein in hard red spring wheat. I. Influence of particle size. Cereal Chem 61(2):158-165.

### Interpretability panel: signed regression coefficients with jack-knife CIs, VIP from CV folds, selected bands over the mean spectrum (visual, impact medium, effort M)

- **Problem:** Model Development shows only VIP stems (PLS) or tree importances, computed once on the full training fit (GUI :37826-37960; models.compute_vip at models.py:1948). There is no signed regression-coefficient (b-vector) plot, which is how chemometricians read which absorption bands push y up or down. There is also no uncertainty on VIP or b, no overlay on the mean or derivative spectrum, and no view of which bands the selector kept against the full spectrum. compute_pls_complexity_curve exists (diagnostics.py:237) but is separate from the band view.
- **Proposal:** Add a three-row linked figure: (1) the mean preprocessed spectrum ± SD with selected bands shaded; (2) the signed PLS b-coefficients with Martens & Martens (2000) jack-knife 95% intervals from the CV sub-models, greying bands whose interval crosses 0; (3) VIP with the fold-to-fold range and the VIP=1 line. Add click-to-annotate with the peak_calculator band assignment and 'export table' (wavelength, b, CI, VIP, VIP range, selection frequency). The CV sub-models are already fitted during CV, so keep their coefficients instead of refitting.
- **Benefit:** The user can make statements such as 'the model relies on the 1730 and 2270 nm C-H/N-H combination bands, with stable positive weights', which is the discussion section of every NIR paper, and can spot models that lean on noise or water bands.
- **Evidence:** GUI :37826-37960 (_plot_wavelength_importance: VIP only, training fit); models.py:1948 (compute_vip), 2021; diagnostics.py:237. Refs: Martens H, Martens M (2000) Modified Jack-knife estimation of parameter uncertainty in bilinear modelling by partial least squares regression (PLSR). Food Qual Prefer 11(1-2):5-16, doi:10.1016/S0950-3293(99)00039-7. Mehmood T, Liland KH, Snipen L, Sæbø S (2012) A review of variable selection methods in partial least squares regression. Chemom Intell Lab Syst 118:62-69, doi:10.1016/j.chemolab.2012.07.010.

### Per-sample prediction intervals and a unified regression AD verdict at prediction time (method, impact medium, effort M)

- **Problem:** For regression, predict_with_uncertainty returns only a model-level RMSECV and RandomForest tree variance (model_io.py:858-866, ~:1100-1108). Every new sample therefore gets the same ± regardless of leverage or spectral residual. The AD payload has Hotelling T² and a nearest-neighbour distance (model_io.py ~:1180-1201) but does not feed into the interval. diagnostics.jackknife_prediction_intervals (diagnostics.py:143) exists but has no callers in the GUI or backend.
- **Proposal:** For PLS/PCR, store the calibration score covariance and SEC/SECV at save time. At prediction, report ŷ ± t·SEP·sqrt(1 + h_i + 1/n), the leverage-based interval of ASTM E1655 (Faber & Kowalski 1997 give the fuller error-propagation form). For non-linear models, offer split-conformal intervals from the pooled CV residuals. Combine T², Q (spectral residual) and the interval width into one per-sample status column (OK / caution / outside) in the Model Prediction table and CSV export. Either wire in jackknife_prediction_intervals or delete it.
- **Benefit:** When collaborators apply a saved collagen model to new bones, each prediction comes with an honest, sample-specific error bar and an explicit flag, instead of one global RMSECV.
- **Evidence:** model_io.py:829-880 (docstring: regression uncertainty = rmsecv + tree_variance only), ~:1100-1108, ~:1180-1201; diagnostics.py:143 (no callers found by grep). Refs: Faber K, Kowalski BR (1997) Propagation of measurement errors for the validation of predictions obtained by principal component regression and partial least squares. J Chemometrics 11(3):181-238, doi:10.1002/(SICI)1099-128X(199705)11:3<181::AID-CEM459>3.0.CO;2-7. ASTM E1655-17. Angelopoulos AN, Bates S (2023) Conformal prediction: a gentle introduction. Found Trends Mach Learn 16(4):494-591, doi:10.1561/2200000101.

### Parsimonious LV choice: one-SE or randomization rule on the RMSECV-vs-LV curve, shown in Results (method, impact medium, effort S)

- **Problem:** PLS LVs are chosen by the search as just another hyperparameter, ranked by -R2cv plus optional 0-10 penalties (scoring.py:15-60). On n≈50 this tends to favour the LV count at the noisy minimum. The chemometric convention is the first LV count whose RMSECV is within one SE of the minimum, or not significantly worse by a randomization test. The RMSECV-vs-LV curve exists only on demand in Model Development (diagnostics.py:237, GUI :39318). CV-ANOVA is computed per row (search.py:5470) but not used for LV choice.
- **Proposal:** For PLS/PLS-DA rows, keep the per-LV pooled CV errors that the grid already computes. Mark the minimum-RMSECV LV and the one-SE LV (Filzmoser et al. 2009 'standard error method'), or the van der Voet randomization-test LV. Add a leaderboard option 'LV rule: min / 1-SE / randomization' and show a mini RMSECV-vs-LV sparkline in the row detail. Record the rule in the report's methods text.
- **Benefit:** Simpler, more robust models by default, less over-fitting on small bone datasets, and an LV choice the user can justify in one sentence.
- **Evidence:** scoring.py:15-60; diagnostics.py:237; GUI :39318-39330; search.py:5468-5474 (cv_anova_pvalue). Refs: Filzmoser P, Liebmann B, Varmuza K (2009) J Chemometrics 23:160-171, doi:10.1002/cem.1225 (standard error method); van der Voet H (1994) Chemom Intell Lab Syst 25:313-323, doi:10.1016/0169-7439(94)85050-X.

### Add PCR (and PCA-LDA) as baseline models (method, impact low, effort S)

- **Problem:** A grep of src finds no PCR, which is the standard baseline reviewers expect beside PLS. The model list is PLS, Ridge, Lasso, ElasticNet and the ML models. For classification, LDA on PLS or PCA scores is also missing (GAP_ANALYSIS #6). Without these, 'PLS vs a simpler latent-variable model' cannot be shown inside the app.
- **Proposal:** Register PCR as Pipeline(PCA(n_components=k), LinearRegression) with the same LV grid, VIP-equivalent (loadings·coef) importances, the leverage-based AD from the prediction-interval idea above, and code export. Register PCA-LDA and PLS-LDA for classification. They should reuse the existing PLS component grid and the Tier system.
- **Benefit:** A cheap, recognised baseline row in every leaderboard, which reviewers ask for.
- **Evidence:** Task context grep (no PCR or EMSC in src); GAP_ANALYSIS.md #6; model_registry.py (no PCR entry).

## Lens: speed

**Current state.** Speed is the weakest part of dasp as a workbench. The trouble is not the numerical work. It is how that work is scheduled and wrapped.

**Measurement setup.** All timings are on the shipped example: 49 x 2151 ASD bone spectra, %Collagen target, 5-fold CV, .venv314, on a 24-thread Ultra 9 285K. The machine was shared with other agents' jobs, so absolute times are noisy. Every speed-up quoted below compares two variants run back to back in the same process or time window. Scratch scripts and profiles are in C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\speed\ (t1..t16 *.py, *.txt).

**What I measured.**
- **Grid search:**
  - Preprocessing is correctly computed once per preprocessing config, at search.py:2872-2895.
  - A PLS-only grid with 4 preprocessing flags (8 configs) and default variable subsets gave 1590 rows in 46-105 s.
  - Under the profiler, only about 10 of those 105 s were PLS arithmetic. The rest breaks down as:
    - about 36 s of sklearn parameter and array validation;
    - 18 s of sklearn metric validation;
    - about 8 s of clone/get_params introspection;
    - 15 s of quadratic pd.concat in add_result;
    - 1590 x 20-line DIAGNOSTIC prints, which came to 31k lines and 1.2 MB of stdout.
  - PLS, Ridge, ElasticNet and SVM run on one core, because they are in MODELS_PREFER_SERIAL_CV (search.py:174).
- **Tree and boosting models (LightGBM, RandomForest; XGBoost and CatBoost are built the same way):**
  - The models are built with n_jobs=-1 (CatBoost's thread_count is left at its all-cores default).
  - They then run inside a loky Parallel(n_jobs=-1) over folds (search.py:1580, 4935; models.py n_jobs=-1).
  - A single LightGBM config in the real run_search took about 25-28 s at steady state. The same config took 0.35-0.47 s with the model's n_jobs set to 1.
  - A LightGBM-only standard-tier grid (96 configs x 8 preprocessing configs = 768 configs) was still unfinished after more than 38 min. That process is orphaned: PID 21268 plus 8 loky workers started 00:17. My TaskStop only killed the shell, and I was not permitted to kill the python processes.
- **Variable selection:**
  - SPA takes 58 s per preprocessing config.
  - mwPLS takes 38 s and mc-siPLS 13 s.
  - ipls_selection takes 6 s on its first call, almost all of it spent spawning loky workers. Its actual work takes 0.12 s.
- **Bayesian search:** per-trial cost is fine for PLS (74 ms/trial). Two things waste time:
  - classification trials run CV twice;
  - every trial does an extra full refit.
  - There is also no pruning, and the models are optimized one after another.
- **NSGA-II:** a single core evaluates about 255 ms per chromosome. The default 60 x 120 = 7200 evaluations therefore take about 30 min, with 23 cores idle.
- **One-class search:** the comprehensive tier took 60 s. About 20% of that is sklearn metric validation on 10-sample arrays, plus 5 s of scipy Nelder-Mead in the chi-square fit.
- **Things that are not problems:**
  - GUI Treeview population: 5000 rows in 0.44 s.
  - Tee logging of stdout: 0.35 s for 31k lines.
  - GUI module import: 1.6 s.

**What users see.** The ETA is a uniform average over configs, and configs differ in cost by about 100x across models. Results appear only when the whole run ends.

**Housekeeping.** One scratch script accidentally wrote stdout_capture.txt into the repo root through os.chdir. I moved it to the scratch directory, and git status is clean.

### Stop nested thread oversubscription: single-threaded models inside parallel CV folds (speed, impact high, effort S)

- **Problem:** Tree and boosting models are built with n_jobs=-1: models.py get_model_grids (e.g. RF at :1218, XGB at :1473 and :1518, LightGBM likewise) and build_model (models.py:634-744). CatBoost's thread_count is left at its all-cores default. These models are then cross-validated inside a loky Parallel(n_jobs=-1) over folds: search.py:1580, :3184 and :4935, and unified_bayesian.py:1796 with cross_val_predict n_jobs=-1 at :1854 and :1920. On a 24-thread machine that means up to 24 workers x 24 OpenMP threads. NSGA-II already does this correctly (nsga2_search.py:609-1009 pass n_jobs=1).
- **Proposal:** 1. In every path where the fold loop is parallel, force the model to one thread: n_jobs=1 for RF, XGB and LGBM, thread_count=1 for CatBoost.
2. Size the outer pool as n_jobs=min(n_splits, physical_cores) instead of -1.
3. Wrap the fold loop in joblib.parallel_config(inner_max_num_threads=1) or threadpoolctl so BLAS and OpenMP inherit one thread.
4. For the single full-data refit, allow min(4, cores) threads.
5. Centralise the rule in one helper, e.g. cv_utils.resolve_threads(model_name, outer_jobs), so the grid, Bayesian, one-class and validation paths cannot drift apart again.
- **Benefit:** A standard-tier grid (PLS, Ridge, ElasticNet, RF, LightGBM) goes from 'leave it overnight' to minutes. This probably also explains the 'CatBoost trial runs 20+ minutes' complaint, and it makes Pause responsive.
- **Evidence:** **Real run_search, one LightGBM config, raw spectra** (t13c.py):
- As shipped: 8.4 s, 3.6 s, 2.5 s, 2.5 s, 2.9 s, then 28.0, 28.8, 28.3, 24.9 and 27.7 s once all workers were warm.
- Model n_jobs=1: 2.7, 2.1, 2.2, 2.0 and 2.0 s during worker warm-up, then 0.46, 0.35, 0.41, 0.47 and 0.42 s. That is 60-70x faster at steady state.

**Microbenchmark, one 5-fold CV** (t7_one.py, t5_threads.py):
- LightGBM: outer -1 / inner -1 took 3.68 s (24.6 s under contention). Outer 5 / inner 1 took 0.22 s. Serial outer / inner -1 took 0.40-158 s.
- RF200: outer -1 / inner -1 took 1.38 s; outer 5 / inner 1 took 0.13 s.

**Full LightGBM-only standard grid** (96 configs x 8 preprocessing configs = 768 configs): still running after more than 38 min (g_LightGBM.txt has no ELAPSED line). At the measured 0.4 s per config it would take about 5 min.

**Reference:** joblib docs, 'Avoiding over-subscription of CPU resources', https://joblib.readthedocs.io/en/stable/parallel.html

### All-LV PLS CV kernel: one fit per fold yields every component count (speed, impact high, effort M)

- **Problem:** The PLS grid makes each n_components value in 1..max_n_components (models.py:1078-1110) a separate config. Each config runs 5 sklearn fold fits plus a full refit (search.py:4912-4945, :5144). The fits are also wrapped in heavy sklearn validation. In the PLS-with-subsets profile (pls_sub.txt), check_array took 24 s cumulative, metric target checks 18 s, and clone/get_params/inspect.signature about 8 s. Actual PLS numerics were about 10 s of the 105 s run.
- **Proposal:** 1. Add a small numpy PLS1 (NIPALS or SIMPLS) kernel. Per fold, it fits once at max LVs and returns predictions for every k via the cumulative regression vectors B_k = W_k (P_k'W_k)^-1 q_k.
2. Use it in the grid's PLS branch to emit all LV rows from a single pass.
3. Compute calibration metrics from the same kernel on the full data.
4. Keep sklearn PLSRegression for the saved and exported model, so .dasp files are unchanged.
5. Expose the kernel as the shared scorer for the selectors (next idea), and optionally for PLS-DA's PLS stage.
- **Benefit:** PLS, which is the core model for NIR and FTIR, returns results in seconds instead of minutes, with identical numbers. The largest gain is in variable-subset runs, where PLS is refit thousands of times.
- **Evidence:** **Benchmark, t2_plskernel.py (5-fold, 10 LVs, example data):**
- 50 sklearn fits took 133 ms. The all-LV kernel took 7.2 ms, which is 18.5x faster.
- Maximum prediction difference was 3.2e-13, and RMSECV per LV was identical to 4 decimals (3.9837 ... 2.6404).

**End-to-end:** a PLS-only grid (8 preprocessing configs, subsets on) took 46-105 s for 1590 rows. Overhead dominates today at about 55 ms per config, against about 8 ms of numerics. My estimate is 5-10x end to end.

**References:**
- Dayal, B.S. & MacGregor, J.F. (1997) Improved PLS algorithms. Journal of Chemometrics 11(1):73-85, doi:10.1002/(SICI)1099-128X(199701)11:1<73::AID-CEM435>3.0.CO;2-#.
- The same 'all ncomp from one fit' CV is what the R pls package does: Mevik, B.-H. & Wehrens, R. (2007) The pls Package: Principal Component and Partial Least Squares Regression in R. Journal of Statistical Software 18(2):1-23, doi:10.18637/jss.v018.i02.

### Vectorise canonical SPA and put the interval selectors on the fast PLS kernel (speed, impact high, effort S)

- **Problem:** **SPA** (variable_selection.py:335-512) runs one Python chain per seed, for all J=2151 seeds.
- Each chain step rebuilds sorted(available) and an (avail x selected) correlation product.
- Each seed is then scored with sklearn cross_val_score of a 10-LV PLS.
- The threading pool is capped at 8 workers.
- SPA results are cached per preprocessing config (search.py:2836), so a grid with 8 preprocessing configs pays this cost 8 times.

**Interval selectors:** mwpls (:2253) and mc_sipls (:2130) make thousands of sklearn PLS fits, each carrying full validation overhead.
- **Proposal:** 1. Precompute C2 = (corr(X))^2 once, J x J (about 37 MB at J=2151; use float32 or blocks for larger J).
2. Advance all J chains at once with a masked argmin, then score += C2[next]. This reproduces the current criterion exactly.
3. Score every chain with the numpy PLS kernel instead of cross_val_score.
4. Route mwPLS, mc-siPLS, iPLS forward/backward and UVE through the same kernel, which gives all LVs per fit.
- **Benefit:** SPA and interval PLS become practical choices inside a grid instead of multi-minute stalls. Border Cave and leaf NIR users who script these selectors also benefit directly.
- **Evidence:** **Current cost, t3_varsel.py (SG first derivative, w=11):**
- SPA(n=30) took 57.8 s. The profile is dominated by sklearn PLS power iterations and 23 s of threading sleep.
- mwpls took 37.9 s (9070 PLS fits), mc_sipls 13.0 s, ipls_forward 2.4 s and CARS 1.0 s.

**Prototype, t4_spa.py:**
- The vectorised chains took 0.9-2.4 s for all 2151 seeds, and the chains were identical to the current code.
- Numpy PLS scoring of all 2151 chains took 1.14 s, and the seed-0 CV R2 matched sklearn to 8 decimals (0.62037103).
- SPA therefore drops from about 58 s to about 2-3 s, 20-25x faster, with the same selection.

**Reference:** Araujo, M.C.U. et al. (2001) The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. Chemometrics and Intelligent Laboratory Systems 57(2):65-73, doi:10.1016/S0169-7439(01)00119-8.

### Task-level parallelism in the grid and one-class searches: use all cores for fast models (speed, impact high, effort L)

- **Problem:** The grid loop over preprocessing x model x config is strictly serial (search.py:2912-3190). Fold parallelism can use at most n_splits cores (5 of 24). For PLS, PLS-DA, Ridge, Lasso, ElasticNet and SVM it is switched off entirely (MODELS_PREFER_SERIAL_CV, search.py:174), because joblib's per-call overhead outweighs a 5-fold job. One-class CV is fully serial as well (contamination.py:552 run_one_class_cv; models built with n_jobs=1 at :351-406). The result is that most of the grid runs on one core.
- **Proposal:** 1. Make the unit of work 'one full config plus its variable subsets'. It is self-contained because the subsets depend only on that config's importances.
2. Dispatch these units, not folds, to a persistent loky pool with n_jobs set to the physical core count, and make every model single-threaded.
3. Batch small units (for example all PLS LVs for one preprocessing config) so dispatch overhead is amortised.
4. Merge the results in submission order so rankings stay deterministic.
5. Apply the same pattern to run_one_class_search and compute_validation_metrics_for_top_models.
- **Benefit:** On a typical 8-16-thread laptop, the whole grid (not only boosters) scales with cores. Researchers can afford Comprehensive tiers and more preprocessing variants in one sitting.
- **Evidence:** **Serial by design:** the PLS subset run spent about 59 s of its 105 s in serial fold fits. Ridge and ElasticNet (140 configs) took 42.9 s serial, 26 s of it in enet_path.

**What parallelism buys:** outer loky 5 / inner 1 was 3.4x faster than serial for LightGBM (0.22 vs 0.73 s) and for RF (0.13 vs 0.45 s) (t7_one.py). Configs are independent, so with 8 physical cores I expect 4-7x on fast-model grids.

**One-class comprehensive:** 60.4 s for 132 rows, all serial (t9_oc.py).

### Bayesian trials: one CV pass for labels and probabilities; defer calibration refits (speed, impact medium, effort S)

- **Problem:** **Two CV passes per classification trial.** Each trial calls cross_val_predict_pooled once for labels (unified_bayesian.py:1919) and again with method='predict_proba' (:1936). That refits every fold twice, although the comment at :1911 claims the double pass was removed. The early-stopping branch has the same duplication (:1913 and :1927).

**A full refit in every trial.** Each trial also refits on all data (:2056) only to store calibration metrics, which matter only for the few trials a user will open.
- **Proposal:** 1. Replace the two calls with one fold loop that fits once and calls both predict and predict_proba on the fitted fold model. With repeated CV, pool the labels and probabilities together.
2. Move the calibration refit out of the objective. Compute it after optimize() for the top-K trials that are written to results_df, or lazily when a row is opened.
- **Benefit:** Classification Bayesian runs (PLS-DA, RF, XGB, LGBM, CatBoost on CollagenCat-type problems) finish in about half the time, and regression runs about 15% sooner, with the same TPE trajectory.
- **Evidence:** **Measured, t15_bcls.py (PLS-DA, 30 trials):** 58 cross_val_predict calls for 29 non-duplicate trials, 2 per trial, and 319 pipeline fits (290 CV + 29 refits).

**Saving:** fits per classification trial drop from 11 to 6 (-45%), and per regression trial from 6 to 5 (-17%).

**For comparison:** Bayesian LightGBM regression cost 765 ms/trial (t8_bayes.py).

### Trim per-config bookkeeping in the grid: duplicate refits, sklearn metric validation, row concat, diagnostic prints (speed, impact medium, effort S)

- **Problem:** Each grid config pays fixed costs that swamp fast models:
- **Refits:** _run_single_config refits the full pipeline for params and calibration (search.py:5144). The subset branch then fits the same model again for importances (search.py:3242-3247).
- **Metric validation:** sklearn metric wrappers validate 10-sample arrays. mean_squared_error, r2_score and friends cost 18 s in the PLS profile.
- **Row accumulation:** add_result appends through pd.concat (scoring.py:672-690), which is O(n^2).
- **Prints:** a 20-line DIAGNOSTIC parameter dump is printed per config (search.py:5242-5290). In the GUI it is tee'd into the rotating run log.
- **Proposal:** 1. Return the fitted full-data pipeline from _run_single_config and reuse it for importances.
2. Compute the regression and classification metrics with plain numpy on the already pooled arrays (RMSE, R2, MAE, bias, RPD, confusion counts).
3. Accumulate result dicts in a list and build the DataFrame once, at the end or per preprocessing block for progress.
4. Demote the DIAGNOSTIC dump to logger.debug.
- **Benefit:** About 30-45% less wall time on PLS, Ridge and PLS-DA grids, and smaller log files, with no change to results.
- **Evidence:** **PLS-with-subsets profile** (pls_sub.txt, 105 s): mean_squared_error and r2_score cost 15.5 + 10.4 s cumulative, add_result 15.2 s, and 9640 pipeline fits went to 7950 folds plus about 1690 refits.

**Concat scaling** (t16_concat.py):
- 1000 rows: 0.5 s
- 3000 rows: 1.7 s
- 6000 rows: 4.5 s
- Building from a list: 2-11 ms.

**Print volume:** 31,109 lines and 1.2 MB of stdout for one PLS run (stdout_capture.txt), with 1590 DIAGNOSTIC blocks.

### NSGA-II: evaluate the population in parallel and cache preprocessed matrices (speed, impact medium, effort M)

- **Problem:** NSGA2 _evaluate loops over the population serially (nsga2_search.py:1242-1320), and every model is single-threaded (n_jobs=1). _compute_prediction_error re-runs SNV and Savitzky-Golay on the full matrix for every chromosome (nsga2_search.py:1341-1345), although the preprocessing gene space is only PREPROC_TYPES x 14 WINDOW_SIZES (nsga2_search.py:84-101). The defaults are population_size=60 and n_generations=120 (:1779-1780).
- **Proposal:** 1. Precompute and cache a dict of transformed matrices keyed by (preproc_idx, window_idx), lazily on first use.
2. Evaluate the uncached chromosomes of each generation with joblib (loky, n_jobs set to physical cores, models single-threaded), or use pymoo's parallel elementwise runner.
3. Keep the fitness cache and assemble F and G in index order, so runs stay reproducible.
- **Benefit:** A default NSGA-II run drops from about half an hour to a few minutes on a multicore desktop, which makes the multi-objective parsimony search usable in routine work.
- **Evidence:** **Measured, t11_nsga.py (PLS, Ridge and RF; pop=30, gen=6):** 58.2 s total. 180 evaluations took 45.9 s in _evaluate (255 ms per evaluation) on one core, plus 12 s of CARS or LightGBM guidance.

**Default size:** 7200 evaluations x 0.25 s is about 30 min serial.

**Estimate:** a population of 60 is embarrassingly parallel, so 6-12x on 8-16 cores.

### A joblib policy for tiny jobs: no process pools for millisecond work, lighter workers (speed, impact medium, effort S)

- **Problem:** **Tiny jobs on process pools.** Several call sites use loky with n_jobs=-1 for sub-millisecond PLS jobs. ipls_selection calls cross_val_score(n_jobs=_get_cv_n_jobs()), which is -1, once per interval (variable_selection.py:43-47 and :657-662). The grid fold loop likewise opens up to 24 workers for 5 folds.

**Heavy workers.** Each new loky worker must import spectral_predict.search to unpickle _run_single_fold. That import takes 2.3 s because it pulls in sklearn, imblearn and pandas (python -X importtime). The first 5 or so configs of every run and the first iPLS call of every session pay this cost.
- **Proposal:** 1. Add one helper, e.g. parallel_policy(n_tasks, est_task_seconds), that picks serial when n_tasks x est_task_seconds is below about 0.5 s, and otherwise uses min(n_tasks, physical cores) workers.
2. Move _run_single_fold and its metric helpers into a lean module (for example spectral_predict/_fold_worker.py, imported without search.py's heavy dependencies) so worker start-up is cheap.
3. Optionally pre-warm the reusable executor when the GUI opens the Analysis tab.
- **Benefit:** The first seconds of every run stop being dead time, and selectors such as iPLS feel instant.
- **Evidence:** **ipls_selection** (t14_ipls.py):
- As shipped: 6.04 s on the first call, then 0.25 s.
- With n_jobs=1: 0.12 s and 0.11 s.

**Grid warm-up** (t13c.py): the first 5 LightGBM configs took about 2.0-2.7 s each, then about 0.4 s once the workers were warm.

**Import cost:** spectral_predict.search takes 2.34 s to import (imp.txt).

### One-class search: closed-form chi-square fit, numpy metrics, parallel folds (speed, impact medium, effort S)

- **Problem:** **Metrics:** one_class_metrics (contamination.py:473-549) calls 6 sklearn metrics per fold on tiny arrays. That was 924 calls costing 11.9 s of a 60 s comprehensive run.

**Chi-square fit:** PCA-SIMCA's _fit_chi2 (contamination.py:187-215) uses scipy chi2.fit(method='mm'). Despite the name, this runs a Nelder-Mead optimiser (240 fmin calls, 5.0 s), although a closed-form method-of-moments fallback already exists in the same function.

**IsolationForest:** it dominates at 29 s for 216 fits. Folds and configs run serially (run_one_class_cv, contamination.py:552).

**Known complaint:** 'one-class grid 20-100x slower than classification' (SESSION_LOG_ARCHIVE.md:541).
- **Proposal:** 1. Compute sensitivity, specificity, precision, F1, accuracy and balanced accuracy from four confusion counts in numpy. Keep roc_auc_score only for AUC.
2. Use the closed-form scaled chi-square moments first (dof = 2*mean^2/var, scale = var/(2*mean)), the same estimator DD-SIMCA uses.
3. Run one-class folds and configs through the task-level pool from the grid idea, with single-threaded estimators.
- **Benefit:** Contamination and membership screening (the Glyptal-consolidant use case) becomes interactive rather than a coffee break, and the Bayesian one-class path gets faster through the same helpers.
- **Evidence:** **Measured, t9_oc.py (comprehensive tier, raw/SNV/first derivative, 132 rows):** 60.4 s total, made up of:
- IsolationForest.fit: 28.8 s
- one_class_metrics: 11.9 s
- PCASIMCA._fit_chi2 via scipy fmin: 5.0 s

**Estimate:** about 60 s drops to about 40 s serial with the first two changes, and to roughly 8-12 s with 5-way fold parallelism.

**Reference:** Pomerantsev, A.L. (2008) Acceptance areas for multivariate classification derived by projection methods. Journal of Chemometrics 22(11-12):601-609, doi:10.1002/cem.1147.

### Bayesian wall-clock: fold-level pruning and concurrent per-model studies (speed, impact medium, effort M)

- **Problem:** **No pruner.** No pruner is configured, and nothing reports intermediate values: unified_bayesian.py:2193-2197 notes that no code path raises TrialPruned. Every trial therefore pays full k-fold CV, even when its first two folds are already far worse than the median.

**Models run one after another.** The GUI runs each model's study sequentially (GUI :30421-30456), even though the studies are independent and have separate study names and storage.
- **Proposal:** 1. For slow families (RF, XGB, LGBM, CatBoost, SVM, MLP), report the running pooled RMSE or balanced accuracy after each fold via trial.report and trial.should_prune.
2. Use MedianPruner(n_startup_trials=10, n_warmup_steps=2), and do not prune under LOO.
3. Record pruned trials so the fingerprint-replay cache stores their value.
4. Separately, run the selected models' studies concurrently in a small process pool (min(n_models, cores // 2)) with single-threaded estimators. Forward progress to the GUI through a queue.
- **Benefit:** Multi-model Bayesian runs finish in roughly the time of their slowest model instead of the sum of all of them, and bad booster trials stop early.
- **Evidence:** **Current per-trial cost** (t8_bayes.py): PLS 74 ms/trial and LightGBM 765 ms/trial, with 15 LightGBM fits per trial.

**Concurrency estimate:** a standard tier has 5 models run sequentially. Concurrent studies should cut wall time by about 2-4x.

**Pruning estimate:** median pruning typically saves 30-50% of fold fits on the losing trials. This is a literature-based estimate, not measured here.

**Reference:** Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019) Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2623-2631, doi:10.1145/3292500.3330701.

### Honest ETA, cheap-models-first ordering and a live partial leaderboard (workflow, impact medium, effort M)

- **Problem:** **ETA:** the GUI computes remaining time as (elapsed / current) x remaining configs (GUI :31299-31303). Grid configs differ in cost by about 100x: PLS takes 5-60 ms per config, LightGBM 0.4-28 s. The ETA is therefore wrong for most of the run, and 'total' counts only full-spectrum configs, not subsets (search.py:2811-2813).

**Results timing:** results reach the Results tab only after the whole search returns (GUI :30012, :30088).

**Pause:** Pause is checked only between configs (search.py:2915, :2979).
- **Proposal:** 1. Before the run, time one config per selected model (and one call per selected selector) and multiply by the grid sizes. Show 'estimated 3-6 min' before the user presses Run, and warn when a tier or grid choice implies hours.
2. Order the loop cheapest-first (PLS, Ridge, PLS-DA before the boosters).
3. Push completed rows to the Results tab every N seconds through root.after, so users can inspect and refine early winners while slow models finish.
4. Check the controller between folds as well as between configs.
- **Benefit:** Users see where the time will go, get a usable leaderboard within seconds, and can stop a run once the answer is clear.
- **Evidence:** **Per-config cost by model:** PLS about 55 ms (pls_sub.txt: 88 s over 1600 configs), Ridge and ElasticNet about 300 ms (42.9 s over 140), RF about 2.4 s (50.9 s over 20), LightGBM 0.4-28 s (t13c.py).

**Measurement noise:** the same LightGBM config varied from 0.35 to 28 s depending on the thread state of the loky workers. A uniform ETA cannot track that.

**Code references:** GUI progress callback at spectral_predict_gui_optimized.py:31289-31330, and results populated only at the end at :30012 and :30088.

## Lens: gui-usability

**Current state.** DASP's engine is strong, but a new user with one CSV and a reference column gets little guidance through the GUI. Everything below was checked in the code, and I rendered each page of the real app (own window, screenshots in scratchpad/improve/gui-usability/shots/).

**Structure.** Navigation is a sidebar with 4 sections (spectral_predict_gui_optimized.py:5825-5848). It sits over a hidden notebook of 15 pages. Calibration Transfer, Interference, Contaminant, Spectral Library and Data Management sit under a collapsed "Advanced" section (:5842-5848), so the two science workflows the user cares most about are hidden by default. Startup builds every page eagerly: 3,530 widgets and 771 Tk variables, in about 2 s of import plus 5-6 s of build (measured).

**Steps from CSV to a saved model.**
- Import & Preview: the required Target Variable and Analysis Type controls sit under a header reading "2. Advanced Configuration (Optional)" (:6534), below the fold.
- Analysis Configuration: CV and preprocessing are on Basic Settings, and the search engine and tier are on the third subtab, Model Config (:12589-12614). There are 5 separate Run Analysis buttons.
- Default CV is plain 5-fold for any n (:2969-2970), although the on-screen tip recommends repeated k-fold for n=30-200.
- A Quick-tier grid on the 49-sample example took 107-151 s and produced 1,915 ranked rows with 53 columns. The top 8 rows sit within 2.5% of the best RMSEcv, and 41 rows sit within 10%. The ranking penalties default to 0 (:2977-2978), so the #1 row is a 325-variable region subset rather than the near-equal full-spectrum model.
- To save a model: double-click a row → Model Development/Selection → Run Model (refit) → Results subtab → Save Model → file dialog. The Prediction tab then needs the file loaded again, because `loaded_models` is filled only from disk (:43974).

**No persistence.** Nothing persists between sessions except peak presets and the Bayesian resume snapshot. The menubar has only About and Help (:61754-61768), with no File, Open or Recent. Results go to relative "outputs/" and "reports/" folders (:31052, :31060), and past results cannot be reloaded into the Results tab.

**Help text is out of date.**
- The Quick Start dialog (_show_help) and UserGuide chapter 2 describe actions that do not exist: "click any row to view Predicted vs Actual", and "Export Model → .pkl / PDF report".
- Offline help opens a 380 KB .md file with os.startfile.
- About 21 on-screen strings still cite old tab numbers ("Tab 11A", "on 13A", "Main Dataset (Tab 1)").

**Error handling.** Of 313 showerror calls, about 76 are "Error: {e}" with nothing on what to do next. There are 753 print() calls, which are invisible in the windowed bundle.

**Calibration Transfer and Contaminant Analysis are the weakest pages.**
- Calibration Transfer: the method guide names a "Feature-based" method that does not exist and omits TSR, NS-PFCE and JYPLS-inv. Every method's parameters are shown at once. A build reports only the method, the sample count and the wavelength range, with no measure of how well the transfer worked.
- Contaminant Analysis: "Apply to Main Dataset" overwrites self.X in place, with no undo. Nothing about the correction is saved in the .dasp model, so predicting on new raw spectra silently skips it.
- Interference Removal: the Method Configuration EPO/DOSC checkboxes are ignored by the analysis. The plumbing is commented out at :30853.

**What works well.** Model Prediction has clear numbered steps. Tooltips are thorough on the Results columns, CV and Calibration Transfer. The live ETA exists, but it is a text label with no progress bar.

### Guided 'New Analysis' path with recipe presets and n-aware defaults (usability, impact high, effort L)

- **Problem:** Starting an analysis means visiting 3 pages and 4-5 subtabs, and the required settings are labelled optional or buried.
- Target Variable and Analysis Type sit under '2. Advanced Configuration (Optional)' (spectral_predict_gui_optimized.py:6534, controls at :6563-6580).
- CV is on Config/Basic Settings (:11570); the search engine and tier are on Config/Model Config (:12589-12614).
- Three preprocessing-discovery engines are separate cards on Basic Settings (:12029, :12088, :12153).
- Default CV is 5-fold regardless of n (:2969-2970), while the on-screen tip says 'n<30: LOO or repeated 5-fold | 30-200: Repeated K-Fold' (:11594). Nothing acts on that tip, so the 49-sample example runs on the setting the app itself advises against.
- No 'try the example data' entry point exists.
- **Proposal:** Add a 'New Analysis' wizard, as a modal or a pinned first sidebar item, with 4 screens.
1. Data: one file picker that detects directory, reference or combined input, plus a 'Use example bone-collagen data' button.
2. Target and task: target dropdown, task auto-detect shown as an editable suggestion with its reason, and a class histogram for classification.
3. Recipe: 4-5 named presets, each writing a dict onto the existing Tk vars through run_gui_settings.restore_gui_settings:
   - 'Fast PLS baseline (1-2 min)'
   - 'Standard calibration (Bayesian, 300 trials)'
   - 'Small-n careful (repeated 5x5 CV, full-spectrum only, variable penalty 3)'
   - 'Classification'
   - 'One-class screening'
   Set CV from n: repeated k-fold for 30-200 samples.
4. Review and Run: a plain-language summary and a time estimate (see the cost-preview idea).
The wizard ends on the Results page. The existing pages stay as 'Expert configuration'. Rename the Import header so Target and Task are not labelled optional.
- **Benefit:** A collaborator or a future commercial user gets from CSV to a defensible leaderboard in 4 decisions instead of about 20 scattered controls. The small-n defaults then match the app's own advice.
- **Evidence:** spectral_predict_gui_optimized.py:6534 ('2. Advanced Configuration (Optional)' holds Target and Analysis Type); :2969-2970 (folds=5, cv_strategy='kfold'); :11594 (n-based CV tip); :12612-12614 (engine radios on the 3rd config subtab); src/spectral_predict/run_gui_settings.py:351,377 (capture/restore over 155 whitelisted settings, reusable for presets). Screenshots: scratchpad/improve/gui-usability/shots/01_Import_and_Preview.png and cfg_2.png.

### Results comparison view: best-per-family grouping, near-tie flags and side-by-side compare (reporting, impact high, effort L)

- **Problem:** The leaderboard is one flat table with too many near-identical rows to choose from.
- Measured: the Quick tier on the 49-sample example gave 1,915 rows by 53 columns.
- The top 8 rows span only 1.621-1.661 RMSEcv (2.5%), and 41 rows sit within 10% of the best.
- Rank 1 is PLS on 325 variables ('top10regions'). The first full-spectrum model, ElasticNet, is at rank 8 and only 2.5% worse.
- Ranking is pure R2cv because both penalties default to 0 (:2977-2978). run_search itself warns 'Subset models may rank higher due to lower variable counts' (AGENT_COMPOSITION.md s7), but the GUI does not act on it.
- The only row interactions are header sort, a 'Select' checkbox used for ensembles (:32753 onward) and double-click to refine (:33966).
- There is no filter by model or preprocessing, no grouping, no uncertainty on the metric, and no way to compare 2-3 candidates side by side.
- **Proposal:** Add a 'Summary' view above the full table.
- (a) Best row per Model x Preprocess x SubsetTag family, collapsible to all members.
- (b) A 'statistically tied with #1' badge using the one-standard-error rule. Use the per-fold RMSE spread already computed under repeated k-fold; under plain k-fold, show 'tie unknown, use repeated CV'.
- (c) A 'simplest in the tied set' star: fewest variables or latent variables, full spectrum preferred.
- (d) Quick filter chips for model, preprocessing and full vs subset.
- (e) A 'Compare checked' button that opens a panel with predicted-vs-observed overlays, residuals and metric bars for 2-4 rows. Rows would need their CV predictions stored, or a fast refit on demand.
- **Benefit:** The user can see that the top 8 models are effectively tied and can pick the simplest, most transferable one. A near-tie is no longer taken as a real ranking, which matters for n≈50 bone-collagen and FTIR screening papers.
- **Evidence:** Measured in scratchpad/improve/gui-usability/time_quick.py (results in quick_df.pkl): 1,915 rows; the top 12 rows are printed in the session log (rank 1 PLS 325 vars RMSEcv 1.621; rank 8 ElasticNet full 2135 vars RMSEcv 1.661); 16 rows within 5% and 41 within 10% of the best. Reference: Hastie T, Tibshirani R, Friedman J (2009) The Elements of Statistical Learning, 2nd ed., Springer, s7.10 (one-standard-error rule).

### One-step path from a Results row to a saved model and to Prediction (workflow, impact high, effort M)

- **Problem:** Saving a leaderboard row as a model takes 5-6 actions across 2 pages.
1. Double-click the row, which jumps to Model Development/Selection (:34044-34050).
2. Click Run Model (:15172).
3. Switch to the Results subtab.
4. Click Save Model (:15900).
5. Answer the file dialog.
Then, in Model Prediction, the user must browse for the same file again, because `loaded_models` is filled only from disk (:43974). The Results tree has no right-click menu; the only tree context menus are on data sheets (:6419, :11159). Only multiclass rows offer a direct 'Save Model (.dasp)' (:32042).
- **Proposal:** Add a right-click menu on Results rows with these actions:
- 'Open in Development' (the current double-click)
- 'Refit and save as .dasp...' (a background refit reusing the Development refit path, then a save dialog)
- 'Export code...'
- 'Refit and send to Prediction'
After any successful save in Development, show an inline 'Use this model for prediction →' link that appends the in-memory model dict to loaded_models and switches to Prediction.
- **Benefit:** Going from choosing a model to predicting new samples drops from about 8 actions and a round trip through the file system to 2 clicks.
- **Evidence:** spectral_predict_gui_optimized.py:33966-34050 (_on_result_double_click only loads for refinement); :15172, :15900 (Run Model and Save Model on different subtabs); :43974 (loaded_models.append only in the file-load path); grep for 'Button-3' finds no Results binding.

### Save and load analysis recipes, remember the last session, and add a File menu (usability, impact high, effort M)

- **Problem:** Nothing the user configures survives a restart.
- The 155 analysis-defining settings are captured only for crash-resume of Bayesian runs (run_gui_settings.CAPTURABLE_SETTINGS; used at :26951 and :27608).
- There is no Save or Load settings action for normal use.
- last_directory exists only in memory, in the Multi-Model code (:53022).
- The menubar has only About and Help (:61754-61768).
- Results are auto-written to relative 'outputs/' and 'reports/' folders (:31052, :31060), which depend on the working directory in the installed bundle. A past results CSV cannot be reopened in the Results tab, so the leaderboard is lost when the app closes.
- resource_paths.get_user_data_dir() (%LOCALAPPDATA%\dasp) already exists but is used only for logs and Optuna.
- **Proposal:** Add a File menu with these entries:
- New Analysis
- Open Recipe..., Save Recipe as... (JSON of capture_gui_settings plus the target and task; data paths shown for confirmation, as the resume flow already does)
- Open Results... (reload results_*.csv together with its training_config so that double-click refine works)
- Recent datasets and recipes
Auto-save the last recipe and the directories used to get_user_data_dir()/settings.json, and offer 'Restore last settings' at startup. Write outputs to an absolute, user-visible folder (default Documents\SpectralPredict\<dataset>_<timestamp>), and show the path with an 'Open folder' button when a run finishes.
- **Benefit:** Group members can share an exact analysis recipe, for example for Border Cave or leaf NIR. Re-running on new data takes one click, and finished leaderboards can be reopened later.
- **Evidence:** src/spectral_predict/run_gui_settings.py:1-40 docstring, :351 capture, :377 restore; src/spectral_predict/resource_paths.py:84-120; spectral_predict_gui_optimized.py:2968 (output_dir='outputs'), :31052-31062 (relative outputs and reports), :61754-61768 (menubar).

### Calibration Transfer: correct the method guide, show only relevant parameters, and add a leave-one-standard-out bake-off (method, impact high, effort M)

- **Problem:** The in-tab Quick Method Selection Guide (:54643-54648) needs correcting:
- It names 'Feature-based matching methods', which DASP does not implement.
- It omits TSR, NS-PFCE and JYPLS-inv, all implemented (calibration_transfer.py:441, :1023, :1428).
- It recommends DS for fewer than 10 standards. That runs against the reason PDS was introduced: DS becomes ill-conditioned when the number of transfer samples is much smaller than the number of wavelengths.

The page layout adds to the confusion:
- All six methods' parameters (DS lambda, PDS window, TSR and JYPLS samples, NS-PFCE max iterations and selector) show at once, whichever method is selected (:54893-54994).
- Step labels are tangled: 'A)', then 'C) Build New' containing 'A1' and 'A2', then 'STEP 2 / B)', then 'STEP 3A / C)', then 'D)'.

A build reports only the method, the number of standards and the wavelength range (_build_transfer_model_new, around :49483-49490). There is no transfer-quality number, and the backend has no evaluation function (calibration_transfer.py lists only estimate_* and apply_*), so users choose a method blind.
- **Proposal:** 1. Rewrite the guide to cover all six implemented methods, and drop 'Feature-based'.
2. Show only the selected method's parameters.
3. Renumber into 3 linear steps: Load standards → Build and evaluate → Use (predict or export).
4. Add a 'Compare methods' button that runs leave-one-standard-out for every applicable method (and a small PDS window grid), then reports:
   - (a) RMSE between the transferred satellite spectrum and the primary spectrum of each held-out standard;
   - (b) when a primary .dasp model and reference values for the standards are loaded, prediction RMSEP, bias and slope on the held-out standards.
   Show it as a sortable table plus an overlay plot, with 'Use best' filling in the build.
5. Add an evaluate_transfer() primitive in calibration_transfer.py so scripted users get the same check.
- **Benefit:** Users get a defensible, data-driven choice of transfer method and parameters instead of trial and error, plus a number to report in a paper. This is the main gap the user raised for this section.
- **Evidence:** spectral_predict_gui_optimized.py:54643-54648 (guide text), :54663-55120 (layout labels A/C/A1/A2/STEP 2/B/STEP 3A/C/D), :49483-49490 (build info text); src/spectral_predict/calibration_transfer.py:117-1702 (estimate/apply only). Screenshot: scratchpad/improve/gui-usability/shots/11_Calibration_Transfer.png. References: van den Berg F, Rinnan Å (2009) Calibration transfer methods. In: Sun D-W (ed.) Infrared Spectroscopy for Food Quality Analysis and Control, Academic Press/Elsevier, ISBN 978-0-12-374136-3, ch.5 pp.105-118 (p.113: DS has problems 'if the number of transfer samples m is much smaller than the number of variables n ... This observation led to the alternative' PDS); Wang Y, Veltkamp DJ, Kowalski BR (1991) Multivariate instrument standardization, Analytical Chemistry 63(23):2750-2756.

### Contaminant Analysis: make the correction a saved, undoable pipeline step instead of an in-place edit of the data (workflow, impact high, effort L)

- **Problem:** 'Apply Correction' with 'Main Dataset (Tab 1)' overwrites self.X with the EPO-, GLSW-, OPLS- or region-corrected matrix (_contam_apply_correction, :60025, assignments at :60099-60102).
- A backup, X_before_contam_correction, is stored (:60092), but nothing reads it, so there is no Undo.
- The correction is not recorded anywhere a later run or a saved model can see: model_io.py has no contam, EPO or GLSW references.
- A model trained on corrected data and applied in Model Prediction to new raw spectra therefore silently skips the correction.
- Nothing in Configuration or Results shows that the data was modified.
- The only other routes are export-and-reimport, or 'Apply Clean Regions to Analysis' (:57989, :58231).
- The page also mixes in one-class wording ('Reset All One-Class Data', :57814) and stale tab numbers ('Load data on 13A').
- **Proposal:** Replace 'apply to main dataset' with 'Add to analysis pipeline'. It stores the fitted projection (EPO P matrix, GLSW weights, OPLS filter or excluded regions) as a named preprocessing step. The search applies it, and the .dasp model saves it and applies it again at prediction time, just as the calibration-transfer model is applied (GUI :46505).

Add these supporting pieces:
- A persistent banner on Configuration and Results: 'Contaminant correction active: EPO k=2 (Glyptal)', with Remove and Undo buttons.
- A 'Validate' button that runs a quick PLS CV with and without the step and reports the change in RMSECV.
- Replace the stale tab-number and one-class wording.
- **Benefit:** Glyptal-consolidated bone can be corrected once and predicted correctly from then on. New specimens get the same correction automatically, and the before/after effect is quantified inside the app, which is the J6 workflow the user named.
- **Evidence:** spectral_predict_gui_optimized.py:60025-60120 (_contam_apply_correction overwrites self.X), :60091-60092 (unused backup), :58255-58350 (Apply & Validate subtab), :57814 ('Reset All One-Class Data'); src/spectral_predict/model_io.py (grep for contam/epo/glsw: 0 hits).

### Pre-run cost preview, run summary and a real progress bar (speed, impact medium, effort M)

- **Problem:** Before clicking Run, the user sees no estimate of how many configurations will be tested or how long it will take.
- The only cost check is a warning dialog that fires only for non-kfold CV (_run_analysis :24425-24475).
- Measured: Quick tier on 49x2151 = 1,915 configurations, 107-151 s. Comprehensive tier or Bayesian with 300 trials can run for hours, and pause takes effect only between trials.
- The Progress page (:14763-14820) has a text label ('Progress: i/N configurations', :31331) and a scrolling log. The app's only ttk.Progressbar is in Model Prediction (:43753).
- There are 5 separate 'Run Analysis' buttons, one per config subtab (:11489, :12234, :12570, :14473, :14627), but none shows what will actually run.
- **Proposal:** Replace the duplicated Run buttons with a single sticky run bar at the bottom of Analysis Configuration. It shows a live one-line summary, for example 'Regression · repeated 5x5 CV · SNV, SG1, SG2 (w=17) · 3 models · Quick grid ≈ 1,900 configs ≈ 2-3 min'. Build it from estimate_total_cv_fits and a per-model time-per-fit table calibrated from past runs stored in the user data directory. Warn in amber when above 30 min, and suggest cheaper options (fewer windows, Bayesian instead of grid, drop CatBoost).

On the Progress page:
- Add a determinate ttk.Progressbar.
- Show a stage line (preprocessing → variable selection → models → ensembles).
- Keep 'best so far' next to the ETA.
- Add a 'Notify me when done' option (a Windows toast or window flash).
- **Benefit:** Users choose settings knowing the cost, avoid starting a run that goes overnight by accident, and can see at a glance how far along a run is.
- **Evidence:** spectral_predict_gui_optimized.py:24425-24475 (cost warning only when cv_strategy != 'kfold'), :31290-31332 (text-only ETA), :43753 (the only Progressbar), 5 Run buttons at the lines listed above; measured timing in scratchpad/improve/gui-usability/time_quick.py (107.2 s and 151.4 s, 1,915 rows).

### Fold the three preprocessing-discovery engines into one choice (usability, impact medium, effort M)

- **Problem:** Basic Settings offers four overlapping ways to decide preprocessing:
- the manual checkboxes: Raw, SNV, SG1-4 and deriv_snv, with 5 window checkboxes (:11824-11903);
- 'Basic Preprocessing Discovery', subtitled 'NSGA-II-style' (:12029);
- 'TPE Preprocessing Discovery' (:12088);
- 'Exhaustive Preprocessing', 238 combinations (:12153).
Independently of those, the third subtab offers Grid, Bayesian (which already optimises preprocessing jointly, :12613) and NSGA-II. A novice cannot tell which combinations make sense. For example, TPE preprocessing discovery combined with Bayesian search duplicates work.

The jargon is unexplained on the controls themselves: 'deriv_snv (advanced)', 'SG3/SG4', and 'Window=17 ⭐' with no statement of what the star means.
- **Proposal:** Use one radio group: 'Preprocessing: (•) I'll pick (checkbox list) / ( ) Let DASP search'. The search option offers a single engine dropdown (Exhaustive, TPE or Basic) that is disabled, with an explanation, when Bayesian search is selected, since Bayesian already searches preprocessing. Keep SG3/SG4 and deriv_snv behind 'Show advanced transforms'. Replace the star with 'recommended for 1 nm data'.
- **Benefit:** There are fewer contradictory settings, which removes a whole class of wasted or duplicate runs, and it is clearer what the search actually varied.
- **Evidence:** spectral_predict_gui_optimized.py:11824-11903 (manual checkboxes and windows), :12029, :12088, :12153 (three discovery cards), :12612-12614 (engine choice on a different subtab); window_17 default=True at :3362.

### One Predict workspace instead of three separate prediction pages (workflow, impact medium, effort L)

- **Problem:** Predicting on new spectra is built three times, each with its own model loader, data loader and reflectance/absorbance conversion:
- Model Prediction: Steps 1-3 (:43638-43753);
- Multi-Model: primary and auxiliary models, 'Step 1.5' transfer models, then load spectra, including live folder monitoring (:52438-52600);
- Calibration Transfer Mode A: C1 load primary model, C2 load satellite data, C3 run (:55106-55240).
A user with a satellite instrument must find prediction in the Advanced > Cal Transfer page instead of Prediction. Every page is also built at startup (3,530 widgets, 5-6 s measured build time).
- **Proposal:** Merge the three into one Predict page:
1. Models: one primary model plus optional auxiliary models.
2. Spectra: file, folder, validation set or live folder.
3. Optional 'Instrument transfer' dropdown listing saved transfer models (a chain, as Multi-Model already supports).
4. Predict and export, with applicability-domain flags.
The Calibration Transfer page then covers only building and evaluating transfer models, with a 'Use in Predict' button. Build the Advanced pages lazily on first open to cut startup time.
- **Benefit:** There is one obvious place to predict, and transfer, auxiliary models and applicability checks become options there. This removes duplicated code paths that drift apart, and the app starts faster.
- **Evidence:** spectral_predict_gui_optimized.py:43601-43770 (Model Prediction), :52398-52600 (Multi-Model incl. 'Step 1.5: Load Transfer Models'), :55096-55240 (CT Mode A C1-C3); startup timing and widget count from scratchpad/improve/gui-usability/shots.py (import 1.9-2.4 s, build 4.9-5.9 s, 3,530 widgets, 771 Tk vars).

### Actionable error messages and a visible log panel (usability, impact medium, effort M)

- **Problem:** The app's errors often do not tell the user what to do, and some are never seen.
- About 76 of 313 showerror dialogs are 'Error' with the raw exception text, for example 'Failed to load data:\n{e}' (:19508) and 'Failed to build transfer model' (around :49511).
- Reader errors state the rule but not the fix: 'Expected at least 100 wavelengths, got N' (io.py:140).
- 753 print() calls report details, such as the '> Attached training configuration' messages in _on_result_double_click, that never reach the user in the windowed PyInstaller bundle.
- run_logging already writes to %LOCALAPPDATA%\dasp\logs, but the GUI never points users there.
- **Proposal:** Add a _show_error(title, exc, hint=None) helper. It maps common failures to a next step:
- ID mismatch: show the first 5 unmatched IDs on each side and suggest the ID column.
- Fewer than 100 wavelengths: explain the limit and name the instrument class.
- Wavelength-grid mismatch: offer to resample.
- Reflectance vs absorbance mismatch: offer to convert.
The dialog has 'Copy details' and 'Open log' buttons. Route print() to logging and add a collapsible 'Log' drawer at the bottom of the window that shows the last 200 log lines.
- **Benefit:** Collaborators can fix a data problem themselves instead of sending Matt a screenshot, and bug reports arrive with the log attached.
- **Evidence:** grep counts in spectral_predict_gui_optimized.py: 313 'messagebox.showerror', about 76 with only '{e}'/'{str(e)}', 753 'print('; src/spectral_predict/io.py:140; src/spectral_predict/run_logging.py:7 (log location); src/spectral_predict/resource_paths.py:113.

### Remove or label controls that do nothing (usability, impact medium, effort S)

- **Problem:** Some controls change nothing, which undermines trust in every other control.
- Interference Removal > Method Configuration offers 'Enable EPO' and DOSC toggles and settings (:55685-55830, stored in advanced_interference_settings, :2882). The analysis call has them commented out: '# interference_settings=interference_settings,  # DISABLED: Code stashed (broke R² reproducibility)' (:30853). Users can enable EPO, run, and get results computed without it.
- For one-class runs, the preprocessing-importance dropdown has no effect (PROJECT_STATUS known issues).
- The Explore, Data Viewer and Data Quality pages overlap: PCA appears in both Explore (:6956) and Data Quality (:11240), and predictor screening appears in both (:6968, :11256).
- **Proposal:** Until the interference plumbing is restored, hide the Interference 'Method Configuration' subtab or show an amber 'Not yet applied during analysis — use Application → Export' banner, and disable the one-class dropdown with a tooltip. Longer term, merge Interference Removal, Contaminant Analysis and Spectral Library into one 'Interferents & Contaminants' area with a shared spectrum library. Keep one PCA view and one predictor-screening view: Data Quality gets outlier PCA, and Explore links to it.
- **Benefit:** Every visible control does what it says, and related functions live in one place instead of three.
- **Evidence:** spectral_predict_gui_optimized.py:30853 (interference_settings disabled), :2882 and :55685-55830 (live-looking EPO/DOSC controls), :6956 and :11240 (two PCA views), :6968 and :11256 (two predictor-screening views).

### Help that matches the app: correct Quick Start, per-page help links, no stale tab numbers (docs, impact medium, effort S)

- **Problem:** The built-in help describes a different app.
- The Help > Quick Start dialog (_show_help) says to choose the tier in the 'Configuration Tab' (it is on the Model Config subtab) and lists 'Cal Transfer' and others without saying they are under the collapsed Advanced section.
- UserGuide.md chapter 2 (lines 181-280) tells users to 'Click on any model row to view Predicted vs Actual plot' (a single click only toggles Select or sorts, :32753) and to 'Click Export Model ... .pkl ... Model report (PDF)'. Neither exists: models save as .dasp from Model Development, and there is no PDF.
- The guide has two 'Chapter 4' and two 'Chapter 5' headings (UserGuide.md:660, :958, :1017, :1439).
- Help > User Guide (Offline) opens a 380 KB .md file with os.startfile, usually in Notepad.
- About 21 on-screen strings cite old tab numbers ('Tab 11A', 'on 13A', 'Main Dataset (Tab 1)').
- The window says 'ASP - Advanced Spectral Prediction' (:2672) while the docs say 'Spectral Predict'.
- **Proposal:** Rewrite the Quick Start from the real click path (or the wizard, once built) and render UserGuide.md to HTML at build time. Add a small '?' button in each page header that opens the matching guide anchor. Replace tab-number references with page names, using a constant map so they cannot drift again. Pick one product name. Add a CI docs check that fails if the guide mentions a button label that is not found in the GUI source.
- **Benefit:** New users can trust the help, and each page links straight to its own explanation.
- **Evidence:** spectral_predict_gui_optimized.py _show_help (Quick Start text), _open_user_guide_offline (os.startfile of docs/UserGuide.md), :2672 (window title); docs/UserGuide.md:181-280 (Quick Start actions that do not exist), :660/:1017 (duplicate Chapter 4); grep for 'Tab [0-9]+[A-D]?' in UI strings: 21 hits (e.g. :55695 'Requires interferent library from Tab 11A').

## Lens: gui-visual

**Current state.** I launched the GUI, loaded the bundled 49-sample BoneCollagen example, took screenshots of all 15 tabs and their subtabs, then ran a small PLS/Ridge search so the Results table, parity plot and learning-curve views had real content. The screenshots are in C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\gui-visual\ (loaded_*.png, r_*.png, and dpi_*.png for the DPI-aware comparison). Every driver process exited through os._exit, so no GUI process was left running. One capture of the Results tab caught the desktop instead of the app window, so I deleted it and replaced it with r_results.png.

The app looks competent but home-made, and the problems are fixable. The dark sidebar, the blue accent and Segoe UI headings give it a coherent shell. Below that shell the visual system is inconsistent:

- **Blurry on this machine.** The process is not DPI-aware on a 125% display, so Windows stretches the whole UI as a bitmap and it is soft everywhere.
- **Fonts silently wrong.** A font-fallback bug makes every ttk widget render in Arial while the headings render in Segoe UI.
- **Cards look cluttered.** Each card has a 6 px grey frame plus a 3 px blue frame, and ttk labels draw grey boxes on the card background.
- **Space is wasted.** Scrollable pages do not stretch to the window, so on a maximized screen the forms fill only about 45% of the width.
- **Headings differ by tab.** Each tab has its own heading style, and Calibration Transfer letters its sections A, B, C, A1, A2, then B and C again.
- **No shared plot style.** There is no matplotlib style or rcParams anywhere; font sizes, bold titles, 'b-' and 'r--' lines and wheat-coloured stats boxes are set inline in each of 57 figures, and some plots have overlapping labels.
- **Themes are lossy and emoji look broken.** Switching theme removes the card borders and the colours that carry meaning, and about 300 emoji render as small monochrome glyphs at mixed sizes.

The two areas the user named, Calibration Transfer and Contaminant Analysis, are the least visually developed. Their diagnostic plots are side-by-side mean spectra on separate axes, plus a pooled-intensity R² scatter for transfer quality. That makes them weak tools for judging whether a transfer or a correction worked.

Speed is not a problem for this lens. The module imports in 2.1 s, the UI builds in 2.0 s (3,530 widgets), and loading and plotting the example takes 2.3 s. The most useful improvements are cheap, low-risk fixes (DPI, fonts, card styling, content width), followed by a shared plot style and purpose-built diagnostic views for transfer, contaminant correction and model results.

### Declare DPI awareness so the UI stops being bitmap-stretched on scaled Windows displays (visual, impact high, effort S)

- **Problem:** The display here is physically 2560x1440 at 120 dpi (125% scaling). The GUI process reports DPI awareness 0 before and after tk.Tk(), and Tk sees a virtual 2048x1152 screen at 96 dpi. Windows therefore upscales the whole window as a bitmap, so all text and lines are soft. main() (spectral_predict_gui_optimized.py:61671-61717) never calls SetProcessDpiAwareness. The PyInstaller spec's EXE block (spectral_predict_py312.spec:296-306) sets no manifest or DPI option, and I did not check PyInstaller's default manifest, so the bundle is probably also unaware. CPython's own Tk app, IDLE, opts in explicitly (Lib/idlelib/util.py:44).
- **Proposal:** In main(), before tk.Tk(), call ctypes.windll.shcore.SetProcessDpiAwareness(1) (system-aware, which is what IDLE does) inside try/except. Then derive a scale factor s = root.winfo_fpixels('1i')/96 and multiply the pixel constants by it: SIDEBAR_CONFIG widths, SPACING, the card padding, and the 70 px top bar. Per-monitor awareness (2) can come later. Add a DPI-aware manifest to the spec so the installed app gets the same behaviour.
- **Benefit:** Every screen becomes crisp immediately. That is the single most visible 'looks professional' change, and it costs about 5 lines.
- **Evidence:** Probe: awareness before/after Tk = 0/0; tk scaling 1.333; winfo_screen 2048x1152; with awareness on, system DPI is 120 and the screen is 2560x1440. Side-by-side crop: gui-visual/hidpi_before_after.png (top is current and blurry; bottom is DPI-aware and sharp). Full DPI-aware captures dpi_loaded_01_import_preview.png, dpi_loaded_02_explore.png and dpi_loaded_11_calibration.png show the layout survives unchanged apart from pixel-sized elements (sidebar, padding) being about 20% smaller. Reference: Microsoft, 'High DPI Desktop Application Development on Windows', learn.microsoft.com/windows/win32/hidpi/high-dpi-desktop-application-development-on-windows.

### Fix the silent Arial fallback and move to 5 named fonts (visual, impact high, effort S)

- **Problem:** _apply_theme sets heading_font = body_font = ('Segoe UI', 'Arial') (:4665-4666) and passes font=(body_font, 10) to every ttk style (TButton, TLabel, TCheckbutton, TRadiobutton, TNotebook.Tab, TLabelframe.Label; :4680-4825). Tk does not treat a tuple of families as a fallback list, and tkfont.Font(font=(('Segoe UI','Arial'),10)).actual() returns family 'Arial'. As a result every ttk label, button, tab and checkbox renders in Arial, while tk.Label headings and cards render in Segoe UI: two typefaces mixed on every screen. On top of that there are 119 literal font=(...) tuples spread over Segoe UI 8/9/10/11/12/13/15/16, Arial 8/10/12/16, Consolas, Courier, Tahoma and TkDefaultFont.
- **Proposal:** Configure the Tk named fonts once: TkDefaultFont, TkTextFont, TkMenuFont and TkHeadingFont set to Segoe UI 10 (platform switch retained), and TkFixedFont set to Consolas 9. Then create 5 app fonts: Body 10, Small 9, H2 12 bold, H1 16 bold, Mono 9. Point every ttk style at those named fonts. Replace the literal tuples in a mechanical sweep, which can be done incrementally because named fonts also restyle live.
- **Benefit:** One consistent typeface and a clear 4-5 step type scale, so the app stops looking assembled from parts. It also makes a later global size change (accessibility, HiDPI) a one-line edit.
- **Evidence:** Measured: tkfont.Font(font=(('Segoe UI','Arial'),10)).actual() gives {'family': 'Arial'}, while ('Segoe UI',10) gives Segoe UI. Font literal census by grep: 18x Segoe 9, 18x Consolas 9, 14x Segoe 8, 9x Segoe 10, 7x Arial 10, 4x TkDefaultFont 9, and others. Screenshot loaded_01_import_preview.png: the 'Spectral File Directory:' label (Arial) sits beside the 'Data Files' card title (Segoe UI).

### Flatten cards and remove the grey 'label chips' (visual, impact high, effort S)

- **Problem:** _create_card (:5339-5374) nests a 6 px grey 'shadow' frame, a 3 px accent-blue frame and an inner card with 20 px padding, so each card reads as a box inside a box (visible on Import, Configuration and Results). Most controls inside cards are plain ttk.Label/Checkbutton widgets whose style background is colors['bg'] (#F0F0F0; TLabel at :4736). The card background is #F8F8F8, so each label paints a visible grey rectangle. Only 53 of 1,031 ttk.Label calls use CardLabel.TLabel. Section headers (_create_section_header, :5376) are also often followed by a card with the same title, for example 'Analysis Subset' twice in a row on Configuration > Basic Settings.
- **Proposal:** Redraw cards as one frame: a 1 px border_light outline on a white card_bg with 16-20 px padding, and no shadow or accent frame. An accent colour, if kept, becomes a 3 px left rule on the section header only. Add a Card.* ttk style family (Card.TLabel, Card.TCheckbutton, Card.TRadiobutton, Card.TFrame). Have _create_card return a helper that applies those styles to the card's descendants after construction, so the 1,000-plus call sites do not need editing. Drop section headers that duplicate the card title.
- **Benefit:** Screens look calmer and more modern, with no background 'noise' behind every label, and vertical space is recovered on long forms.
- **Evidence:** Code: :5339-5374 (three nested frames), :4736-4739 (TLabel background = colors['bg']), :4761 (CardLabel.TLabel, used 53 times). Screenshots: loaded_01_import_preview.png, loaded_05_config.png (duplicate 'Analysis Subset' heading) and crop_import_100pct.png (grey chips behind 'Spectral File Directory:' and the hint text).

### Let scrollable pages use the window width and give side modules a two-pane layout (visual, impact high, effort M)

- **Problem:** There are 30 scrollable pages built as canvas.create_window((0,0), window=frame, anchor='nw'), and only one of them (Multi-Model, :52432) binds <Configure> to keep the inner frame as wide as the canvas. Everywhere else the content shrink-wraps to its natural width. On a maximized window, Import, Quality Check, Prediction, Calibration Transfer, Interference and Contaminant Analysis use roughly the left 40-50% and leave the right half empty. In the reverse case, Model Development > Results is wider than the viewport and shows a horizontal scrollbar.
- **Proposal:** Extract one ScrollableFrame class that keeps the width in sync, binds the mouse wheel, and caps form width at about 1400 px, and use it on all 30 pages. On side modules, fill the freed space with a live preview pane: a config column on the left and a plot on the right. On Calibration Transfer, the primary and satellite overlay would update as files load. On Contaminant Analysis, the clean vs contaminated group overlay would update as groups are added. On Quality Check, the plots would sit beside the parameters instead of below them.
- **Benefit:** Users see the effect of each choice without scrolling or opening a popup, and the empty half-screen that makes the app look unfinished goes away.
- **Evidence:** grep: 30 create_window((0, 0) calls versus 1 itemconfig(width=e.width) at :52432. Example page: :6440-6448 (Import). Screenshots: loaded_11_calibration.png, loaded_13_contaminant_analysis.png, loaded_04_quality_check.png, loaded_09_prediction.png (right half empty) and r_dev_results.png (horizontal scrollbar).

### Result-first diagnostics for Calibration Transfer (method, impact high, effort M)

- **Problem:** After a build, _plot_transfer_quality (:48165) draws three separate subplots: primary mean±SD, satellite-before and satellite-after. Each has its own y-axis, so the before/after difference can only be judged by eye across panels. It then adds a 'Transfer Quality Scatter Plot' of every pixel of the primary spectra against every pixel of the transferred spectra, and reports an R² over the pooled intensities (:48334-48357). That R² is dominated by the shared spectral shape and is close to 1 even for a poor transfer. It is also computed on the same standards used to fit the transfer. The in-tab guide still lists 'Feature-based matching methods' (:54647), which do not exist, in Consolas (:54652). Section labels run A, B, C, A1, A2, then B, C, C1-C3, D1-D3 (:54663-55430).
- **Proposal:** Replace these plots with a single 'Transfer diagnostics' pane showing:
(a) primary, satellite-before and satellite-after for the standards overlaid on shared axes.
(b) the difference spectrum (satellite minus primary) before and after, as mean ± SD with RMS per wavelength, where success is a flat line near zero.
(c) PCA scores of both instruments, before and after, on one plot.
(d) when a primary model is loaded, predicted vs reference for the satellite samples without and with transfer, reporting RMSEP and bias for each.
Compute the after-transfer curves leave-one-standard-out so the view is not in-sample. Drop the pooled-pixel R², turn the guide into a small table of the six methods that actually exist, and renumber the steps 1-4.
- **Benefit:** A user can tell whether a transfer worked, and which method to choose, from one view rather than from an inflated R². This directly answers the request to do a better job on the calibration section.
- **Evidence:** Code: :48165-48260 (three panels with independent axes), :48334-48357 (pooled ravel() R²), :54640-54652 (guide), :54663-55430 (letter scheme). Screenshot loaded_11_calibration.png. These are the diagnostics van den Berg & Rinnan use in their worked example: predicted vs reference without transfer (Fig 5.3) and after preprocessing (Fig 5.4), PCA scores of both systems (Fig 5.5), and transferred vs master spectra (Fig 5.6). Reference: van den Berg, F. & Rinnan, Å. (2009) 'Calibration Transfer Methods', ch. 5 in Infrared Spectroscopy for Food Quality Analysis and Control (Elsevier/Academic Press, ISBN 978-0-12-374136-3), pp. 105-118. The PDF gives the ISBN and page range; I named the publisher from memory and did not check the editor (I believe it is Da-Wen Sun).

### Before/after view for Contaminant correction that shows whether groups were actually merged (method, impact high, effort M)

- **Problem:** _show_correction_comparison (:60206-60250) plots only the mean of all corrected spectra: 'Before' in blue and 'After' in green, on two separate axes with independent y-scales and a generic 'Intensity' label. It is drawn with pyplot plt.subplots in a Toplevel popup. It cannot show the thing the user needs to know, which is whether clean and contaminated specimens (for example bone with and without Glyptal) still differ after EPO, OPLS-DA or GLSW. Using pyplot inside a running Tk app also creates a hidden extra tk.Tk('matplotlib') root per figure (matplotlib/_backend_tk.py:546), and nothing closes it.
- **Proposal:** Embed a 3-panel diagnostic in Apply & Validate:
(1) clean vs each contaminant group, mean ± IQR, overlaid before and after on shared axes.
(2) |mean_contaminated − mean_clean| per wavelength, before and after, with the excluded or EPO-affected regions shaded.
(3) PCA scores coloured by group, before and after correction, where success means the groups overlap.
Also give a single number: Mahalanobis distance between group centroids before and after. Build the figures with Figure(), not pyplot.
- **Benefit:** The correction's effect on the real question, whether the consolidant signal is gone while the bone signal is kept, becomes visible and defensible. This is the 'do a better job on contaminant removal' request, done through visualization.
- **Evidence:** Code: :60206-60250 (mean-only before/after, plt.subplots, popup). The clean-overview and group mean+IQR plots already exist (tab13 region, roughly :59035-59690) and can be reused. There are 9 plt.subplots/plt.figure calls against 7 plt.close in the GUI. Screenshot: loaded_13_contaminant_analysis_sub3.png. Motivating case: UserGuide.md:8652-8655 (Glyptal).

### One matplotlib house style for all 57 figures, with constrained layout (visual, impact medium, effort M)

- **Problem:** No rcParams, style sheet or style.use call exists anywhere in the GUI or in src/spectral_predict. Styling is set inline on each figure instead: 267 fontsize= arguments, 81 fontweight='bold', raw 'b-'/'r--'/'blue'/'red' lines, and wheat-coloured stats boxes (:22332, :37519, :38875, :39212, :39289). Plot text is DejaVu Sans next to Segoe UI UI text. There are 55 tight_layout calls and no constrained layout, and the Explore PCA 2x2 still collides: 'PC1 (47.8%)' overprints the 'PC1 Loadings' title, 'Principal Component' overprints 'PC2 Loadings', and the wavelength x-labels are clipped. Raw Spectra draws 49 identical translucent blue lines and does not colour by the target, even though the target is loaded.
- **Proposal:** Add spectral_predict/plot_style.py with one rc dict applied at GUI start (and usable by exported code):
- font.family Segoe UI/DejaVu Sans, sizes 9 and 10, titles 10 normal-weight and left-aligned.
- figure.constrained_layout.use True; drop top and right spines; light grid.
- Okabe-Ito colour cycle for categories and viridis for continuous values.
- a neutral grey for reference lines (1:1, thresholds) with the accent colour for the model.
Then delete the per-call fontsize/bold overrides as each plot is touched. Default Explore 'Color by' to the target, with a colorbar, when a target exists.
- **Benefit:** Plots look like one product and read cleanly at any window size. Colouring spectra by %Collagen immediately shows whether the spectra carry the target signal.
- **Evidence:** grep counts: Figure/plt 57, tight_layout 55, constrained 0, fontsize= 267, fontweight='bold' 81. Screenshots: r_explore_pca.png (label collisions), loaded_02_explore.png (monochrome blue spectra), r_dev_results.png. Colour guidance: Crameri, F., Shephard, G.E. & Heron, P.J. (2020) 'The misuse of colour in science communication', Nature Communications 11:5444, doi:10.1038/s41467-020-19160-7; Okabe, M. & Ito, K. (2008) 'Color Universal Design (CUD): how to make figures and presentations that are friendly to colorblind people', jfly.uni-koeln.de/color/. Neither is in Paperpile; I cite them from the web.

### Make the Results leaderboard legible: column-aware number formats, alignment, emphasis (visual, impact medium, effort M)

- **Problem:** Every float goes through f'{val:.6g}' (:32672-32680), so the table shows 8.00268e-05, 0.000799454, 1.56038e-18 and 'nan' next to 1.77363. Numbers are centred, so decimal points do not line up. A constant 'Task' column takes space. Calibration RMSE and R2 come before RMSEcv and R2cv, so the near-perfect calibration numbers draw the eye instead of the ranking metric. Every top row is tinted by a pastel Y-quartile colour (:14888-14891), which looks like random zebra striping and fights the selection highlight. The tree shows about 10 rows while an empty 'Ensemble Model Results' card takes the lower half of the page.
- **Proposal:** Format each column by type: RMSE and MAE to 4 significant figures, R² to 3 decimals, p-values as '<1e-10' or 3 significant figures, and '—' for NaN. Right-align numeric columns. Order the columns Rank, Model, Preprocess, LVs/params, n_vars, then RMSEcv/R2cv (bold header), then calibration metrics. Hide columns that are constant across all rows. Replace the full-row tints with a narrow 'Best-in' column showing a coloured Q1-Q4 chip. Collapse the ensemble card when it is empty so the tree fills the page.
- **Benefit:** Users can compare dozens of candidates at a glance and spot overfit rows (a large gap between calibration and CV) without deciphering scientific notation.
- **Evidence:** Code: :32640-32682 (formatting), :14888-14962 (tags), :14850-14880 (tree construction, no numeric alignment). Screenshot r_results.png: 150 rows from PLS and Ridge on the example data.

### Metric tiles and a proper parity plot in Model Development (reporting, impact medium, effort S)

- **Problem:** Model Development > Results prints its metrics as a monospace text dump ('RMSE: 1.7598 ± 0.1498 ...'). The parity plot uses the generic labels 'Reference Values' / 'Predicted Values' (:37522-37524) rather than the target name and units. Its colour-by defaults to 'Y Value' (:3026-3029), which is redundant on a Y-vs-Y plot and adds a large colorbar. The red dashed 1:1 line and the blue fit line compete, the axes are not square, and there is a wheat stats box. The plot is wider than the viewport, which forces horizontal scrolling.
- **Proposal:** Show a row of 6 KPI tiles (RMSECV ± fold SD, R²cv, RPD, bias, LVs or key hyperparameter, n) in large Segoe UI figures, with the text dump behind a 'Details' expander. Make the parity plot square with equal limits, neutral points coloured by fold or by user group (not by Y), a thin grey 1:1 line, the fit line in the accent colour, and axis labels taken from the target column ('%Collagen, reference' and '%Collagen, predicted (CV)'). Move the stats into the tiles.
- **Benefit:** The headline answer, how good the model is, is readable in one second, and the parity plot can go straight into a paper or slide.
- **Evidence:** Code: :37440-37530 (parity plot), :3026-3029 (defaults 'Y Value'). Screenshots: r_dev_results.png and r_dev_results_scrolled.png.

### 'Export for publication' preset on every plot (reporting, impact medium, effort S)

- **Problem:** The export button on each plot calls figure.savefig(filepath, dpi=300, bbox_inches='tight') on the on-screen figure (:16880-16902). The result keeps GUI sizing (for example 12x4 in), 10-12 pt bold titles and DejaVu fonts, so users must restyle figures by hand for manuscripts. The group's real outputs are journal papers (for example the ATR-FTIR bone collagen screening paper).
- **Proposal:** Add a second menu entry, 'Export for publication…', that re-renders the same figure under an rc context:
- width 85 mm (single column) or 180 mm (double column), height from the aspect ratio.
- 7-8 pt Arial/Helvetica, no titles (the caption carries them), and line widths of 0.75-1.0.
- pdf.fonttype 42 and svg.fonttype 'none', so text stays editable.
Also write a CSV of the plotted data next to the image for reproducibility.
- **Benefit:** Figures go from app to manuscript without a detour through Illustrator or re-plotting in R, and each image comes with its underlying data file.
- **Evidence:** Code: :16860-16905 (_add_plot_export_button). This builds on the house-style idea: the same plot_style module would supply a 'publication' rc variant.

### One heading and stepper system, and fix stale wayfinding text (usability, impact medium, effort S)

- **Problem:** Three heading styles are in use:
(1) a blue left-rule section header plus a blue card title (Import, Configuration, Contaminant).
(2) plain bold '1. Outlier Detection Parameters' with small LabelFrame titles (Quality Check, Prediction, Model Development).
(3) small LabelFrame titles lettered A) B) C) A1) A2) B) C) C1) D) (Calibration Transfer, :54663-55430).
Several on-screen strings still point to tab numbers that no longer exist anywhere in the sidebar UI: 'Main Dataset (Tab 1)' :58303, 'go to Tab 11A' :60385, 'Tab 11C' :61013-61018, and 'Requires interferent library from Tab 11A' on Interference > Method Configuration. The main Configuration page repeats the Run Analysis button at the top of each subtab (:11489, :12234 and others), and it scrolls away.
- **Proposal:** Standardise on one pattern: numbered step headers ('1 Load data', '2 Choose method', '3 Build', '4 Apply'), rendered by _create_section_header, with LabelFrames used only for sub-grouping. Replace the tab-number strings with the sidebar names, for example 'Interference › Library'. Put a sticky footer bar on Analysis Configuration holding Run, Pause and Stop and a one-line summary such as 'PLS, Ridge · 4 preprocessings · 5-fold · ≈150 configs'.
- **Benefit:** Users always know where they are in a workflow and what comes next, and the most important button is always visible.
- **Evidence:** Screenshots: loaded_04_quality_check.png, loaded_11_calibration.png, loaded_12_interference_sub1.png, loaded_13_contaminant_analysis.png, loaded_05_config.png. grep: 4 'Tab N' strings plus the Interference help text.

### Replace the 6-theme header with a dataset context bar; retire lossy themes and emoji icons (visual, impact medium, effort M)

- **Problem:** Six saturated theme buttons (Classic, Sakura, Matcha, Sumi-e, Yuhi, Ocean) take the right half of the header. Theme switching is lossy: _update_widget_colors (:5000-5040) forces every non-card tk.Frame to colors['bg'], which erases the card shadow and accent borders. It also resets every tk.Label to fg=colors['text'], which wipes meaningful colours: the amber BETA badge, success/warning status text and the region-legend swatches. The code still checks card colours for the removed 'Midnight' and 'Obsidian' themes. Accent.TButton hard-codes #0078D4 (:4690), so its buttons stay blue in every theme, and plots do not follow the theme at all. There is no persistent indication of what data is loaded. The about 300 emoji in labels and buttons render as monochrome glyphs of mixed size and baseline under Tk 9.0.4 on Windows. The 'Advanced' sidebar section, which holds Cal Transfer, Interference, Contaminant, Spectral Library and Data Management, is collapsed by default (:5849).
- **Proposal:** Keep one well-tuned light theme, and add a real dark theme later only if wanted. Move any theme choice into a View menu. Use the header space for a context bar: '49 samples × 2151 λ (350-2500 nm) · target %Collagen · regression · 0 excluded · CV 5-fold'. Swap emoji for a single small monochrome icon set rendered at the scaled size, or drop icons from buttons entirely. Expand the Advanced section by default, or rename it 'Instruments & contaminants'.
- **Benefit:** A calmer, more professional header that also answers 'what am I working on?' on every tab, and side modules that are easy to find.
- **Evidence:** Code: :4499-4645 (6 themes), :4690 (hard-coded accent), :5000-5040 (recolour logic, stale Midnight/Obsidian list), :4829-4900 (top bar), :5849 (Advanced collapsed). Screenshot r_theme_sakura_results.png shows card borders gone, the BETA badge grey and the legend swatches blank after switching theme. Emoji census: 300 characters in the U+1F300-1FAFF and U+2600-27BF ranges. The Tk patchlevel, 9.0.4, is measured.

## Lens: architecture-maintainability

**Current state.** Structurally, dasp is expensive to change, and the cost sits in a few places I could measure. Scripts and outputs are in C:\Users\sponheim\AppData\Local\Temp\claude\C--Users-sponheim-git-dasp\4dcdd341-ff73-4633-a31c-e4a9f4e97601\scratchpad\improve\structure\ (methsize.py, coupling2.py, thread_reach.py, dups.py, xgb_time*.py, gui_noncomp.txt).

**The GUI is one class.** SpectralPredictApp runs from spectral_predict_gui_optimized.py:2667 to :61671, about 59k lines. A live app holds 851 methods, 771 Tk variables, 1,501 instance attributes and 3,530 widgets. Construction takes 2.3 s, plus 1.8 s to import the module.
- Four methods are over 1,000 lines: _run_analysis_thread (:27425, 3,763 lines), _run_refined_model_thread (:39685, 2,595), _create_tab4c_model_configuration (:12548, 1,902) and __init__ (:2670, 1,371).
- Change concentrates in the launch/worker path. Since 2026-03-01, 163 of 609 commits touched the GUI. The most-edited methods are __init__ (28 commits, adding Tk vars), oc_progress_wrapper nested in the worker (:29817, 20), _run_analysis_thread (19), _norm_label nested in the worker (:30563, 15), _run_analysis (13) and _check_for_incomplete_run (11).

**Worker threads read live Tk state.** In a static call graph, _run_analysis_thread reaches 45 methods (6,775 lines) that make 518 live Tk-variable reads of 390 distinct variables. The launch snapshot captures only 155 settings (CAPTURABLE_SETTINGS, run_gui_settings.py:86-266). The refine worker reaches another 117 reads. That gap is where most of PR #79's 13 review rounds went, and PROJECT_STATUS §1 says the grid, one-class, NSGA-II and post-search paths still read live Tk state.

**The backend has four search kernels that each re-derive shared logic.** Entry points take run_search 139 parameters, run_one_class_search 41, run_unified_bayesian 31 and run_nsga2_search 19; only 6 parameters are common to all four.
- Each engine has its own CV loop: _run_single_fold/joblib in search.py:4334-4945, cross_val_predict_pooled in unified_bayesian.py:1847-1935, and cross_val_score_with_early_stopping in nsga2_search.py:1497-1523.
- 36 CV splitters are built directly outside cv_utils.build_cv_splitter.
- Helpers are copied between modules: _needs_resampling_pipeline in 3 modules, and _apply_edge_mask_to_data, _get_edge_zone_size and _normalize_preprocess_name in 2 each.
- NSGA-II builds PLS-DA itself (nsga2_search.py:794) instead of calling models.build_model.
- A model name such as CatBoost appears in 14 files; the preprocessing name deriv_snv appears in 13.
- SESSION_LOG_ARCHIVE mentions this "sister-site" bug class about 40 times. One example is the class_weight fix that had to land in 5 places (SESSION_LOG_ARCHIVE.md:3056).

**The pattern that fixes this already works in the repo.** T-51 PR D generated the GUI card, the capture list and the required list from the BUNDLES registry, so "adding a bundle to the registry now needs no GUI work" (PROJECT_STATUS §0). The plan below applies that pattern to settings, CV, models and preprocessing.

**Tests.** On this machine, the non-comprehensive GUI suite (264 tests) took 768.6 s: 263 passed and 1 failed (the known pre-existing failure). The whole suite takes about 38 min, so the 34 "comprehensive" tests account for roughly 25 min. Those tests call run_search directly through harness.run_analysis_direct (tests/gui/harness.py:454), so they do not exercise GUI code at all. The two slowest remaining tests are tiny-data XGBoost and LightGBM refits, at 254 s and 108 s. Timing caveat: the machine was at 100% CPU from other agents while I measured.

**Contaminant Analysis is the cleanest tab to extract, and it is one the user wants improved.** It has 40 methods (2,573 lines), one entry point used from outside and 14 outward calls. Calibration Transfer has 55 methods (4,417 lines) and 10 entry points that Prediction and Multi-Model depend on.

**Proposed order, no rewrite.** Each step is a small PR that pays off by itself:
1. Test hygiene and guard rails (ideas 2, 6, 12).
2. A frozen RunRequest plus one settings registry (ideas 1, 3).
3. A CVPlan object (idea 4), which unblocks grouped CV.
4. Extract the Contaminant Analysis tab, then Interference and Spectral Library, then Calibration Transfer behind a TransferChain service (idea 7).
5. Move row-to-pipeline rebuild into the backend (idea 8).
6. A shared candidate evaluator (idea 5).
7. Model and preprocessing registries (idea 9).
8. A SpectraInput component (idea 10).

### Freeze every engine's inputs at the click into a typed RunRequest; the worker never touches Tk (workflow, impact high, effort L)

- **Problem:** _run_analysis_thread (spectral_predict_gui_optimized.py:27425, 3,763 lines) mixes three jobs: it collects settings from Tk, prepares data, and dispatches to 5 engines. The dispatch points are run_unified_bayesian around offset 2413/3005, run_one_class_search 2528, run_multiclass_simca_search 2749, run_nsga2_search 3282 and run_search 3406. Statically, the worker reaches 518 live `self.<var>.get()` calls on 390 distinct Tk vars. The #79 launch gate freezes only the Bayesian subset (BAYESIAN_REQUIRED_SETTINGS :2640) plus 155 captured names. PROJECT_STATUS §1 notes that grid, one-class grid, NSGA-II and post-search paths 'read live Tk state'. Nearly every Codex block in #79 rounds 9-12 was 'a place where the worker still re-read something the gate had decided'. Testing whether a GUI toggle reaches the backend currently means driving the full click and worker with a monkeypatched backend (tests/gui/test_t51_pr_d_gui.py:112-154).
- **Proposal:** 1. Add src/spectral_predict/run_request.py with frozen dataclasses: RunRequest(task, engine, cv: CVPlan, preprocessing: PreprocessPlan, varsel: VarSelPlan, models: tuple[ModelChoice], grids: dict, validation, imbalance, bayes: BayesOptions | None, nsga: NsgaOptions | None).
2. Add a pure `build_run_request(settings: Mapping[str, Any], data_summary) -> RunRequest` that owns all the parsing now inline in the worker (window lists, '10, 20' grid strings, and so on).
3. The main-thread gate calls capture_gui_settings, then build_run_request. The worker takes only the request plus frozen arrays.
4. Migrate one engine per PR, grid regression first because it has the most reads. Keep the old kwargs call as a one-line adapter: `run_search(X, y, **request.to_run_search_kwargs())`.
5. Move the worker body into `execute_run(request, data, progress_cb)` in the backend, and leave the GUI with result-handling callbacks scheduled through root.after.
- **Benefit:** This removes the whole live-Tk-read bug class for all engines, not only Bayesian. A resumed or queued run then means exactly what was on screen at the click, and batch or queued runs become trivial. Plumbing tests become millisecond dict→request unit tests instead of full GUI launches. The same RunRequest is the natural object for agent scripts (AGENT_COMPOSITION) to build and save next to results, which gives reproducibility.
- **Evidence:** thread_reach.py output: `_run_analysis_thread: reachable methods=45 lines=6775 tkvar.get() sites=518 distinct vars=390`; `_run_refined_model_thread: ... 117 sites`. len(CAPTURABLE_SETTINGS)=155. The run_search call site is spectral_predict_gui_optimized.py around 30831-30990, with ~130 kwargs that mix `self.x.get()` and locals. Churn since 2026-03-01: nested worker helpers oc_progress_wrapper (20 commits), _norm_label (15) and unified_progress_wrapper (9) are among the most-edited GUI code. PROJECT_STATUS.md §1 'Also out of scope and still live'.

### Mechanical guard: fail tests on any Tk access off the main thread, plus a static ratchet in CI (workflow, impact high, effort S)

- **Problem:** The main-thread rule is enforced only by reviewers. SESSION_LOG_ARCHIVE.md:2474-2475 records 'cycle 4' of the same anti-pattern: a fix re-added a worker-thread messagebox.showwarning, which three reviewers had to catch. The worker's call graph has up to 11 messagebox call sites (thread_reach.py; some may sit inside root.after lambdas). The 13 review rounds of #79 were largely a hunt for these reads.
- **Proposal:** (a) In tests/gui/conftest.py, add an autouse fixture that wraps tkinter.Variable.get/set, Misc.after-free widget methods (config, insert, delete) and tkinter.messagebox.* so they record or raise when threading.current_thread() is not threading.main_thread(). Start in 'record' mode, then switch to 'raise' once idea 1 lands.
(b) Add a scripts/check_worker_tk.py AST pass. It is essentially thread_reach.py: find `threading.Thread(target=self.X)` targets, walk self-method calls, and count `self.<tkvar>.get()` and messagebox calls. Run it in CI next to the existing blocking F821 gate, with a checked-in ceiling (518 today) that may only go down.
- **Benefit:** The class of bug behind crash-resume regressions and silent wrong-study runs gets caught at commit time, not after rounds of paid review. The ceiling also gives visible progress on the decomposition.
- **Evidence:** Thread targets at spectral_predict_gui_optimized.py: _run_analysis_thread, _train_ensemble_thread, _run_learning_curve_thread, _run_refined_model_thread (grep 'threading.Thread('). The F821 gate precedent is PR #76 (PROJECT_STATUS §2). The static scan script is scratchpad/improve/structure/thread_reach.py.

### One declarative settings registry that generates Tk vars, capture/required/legacy lists and backend kwargs (workflow, impact high, effort M)

- **Problem:** A new analysis setting has to be added by hand in several places:
- a Tk var in __init__ (:2670, 1,371 lines; the most-edited GUI method, 28 commits since March);
- a widget in a tab builder;
- CAPTURABLE_SETTINGS (run_gui_settings.py:86);
- sometimes BAYESIAN_REQUIRED_SETTINGS or BAYESIAN_CONDITIONAL_SETTINGS (GUI :2640-2654);
- LEGACY_DEFAULTS (run_gui_settings.py:269);
- a `.get()` in the worker;
- a run_search kwarg.
The lists drift. `use_msc` is created at :3339 and whitelisted in CAPTURABLE_SETTINGS, but nothing reads it; grep finds no other use in the GUI. The file's own docstring says 'Adding a new setting requires adding it here'.
- **Proposal:** Create src/spectral_predict/settings_registry.py with `SettingSpec(name, kind: bool|int|float|str|choice, default, legacy_default, section, analysis_defining: bool, required_for: set[engine], only_when: str|None, backend_key: str|None, parse: Callable)`. Generate CAPTURABLE_SETTINGS, BAYESIAN_REQUIRED_SETTINGS, BAYESIAN_CONDITIONAL_SETTINGS and LEGACY_DEFAULTS from it, and have __init__ create vars by looping over it. Migrate one section at a time, preprocessing toggles first. Add a test that fails when a spec has no widget, is never consumed by build_run_request (idea 1), or when a whitelisted var is never read.
- **Benefit:** Adding a setting touches one line in the registry and one widget, instead of five or six lists. Dead toggles like use_msc get caught automatically. Resume restores exactly what matters.
- **Evidence:** The T-51 PR D precedent: 'Adding a bundle to the registry now needs no GUI work. The card, capture and required lists are all generated from BUNDLES' (PROJECT_STATUS §0). grep `use_msc`: only spectral_predict_gui_optimized.py:3339 and run_gui_settings.py. Fixing the #79 round 9 bug required adding `bayes_enable_autoscale` to the whitelist (run_gui_settings.py comment around line 124).

### A CVPlan object passed everywhere, replacing 36 hard-coded splitters (unblocks grouped CV and 'CV Strategy Phase 2') (method, impact high, effort M)

- **Problem:** cv_utils.build_cv_splitter (cv_utils.py:275) handles only kfold, repeated_kfold and loo, and compute_min_train_fold_size raises NotImplementedError for groups (cv_utils.py:218-222). 36 other sites build KFold or StratifiedKFold directly, with fixed folds and seeds:
- nsga2_search.py ×7 (:1493, :2685, :2829, :2970, :3124, :4054);
- tpe_preprocessing_discovery.py ×6;
- ensemble.py ×5 (:351, :607, :854, :1884, :1965);
- variable_selection.py ×3 (:161, :288, :1434);
- ga_pls, ga_preprocessing, ga_lightgbm, wavelength_selection, preprocessing_discovery.py:701, simca.py ×3, diagnostics.py:292;
- two in the GUI (:25480, :25663).
That is why the user's CV choice never reaches variable selection, screening, smart preprocessing or NSGA-II. It is also why grouped CV, called 'the largest single gap', cannot be added without touching around 15 files.
- **Proposal:** 1. Add `@dataclass(frozen=True) class CVPlan(strategy, n_folds, n_repeats, seed, groups: np.ndarray | None)`, with methods `.splitter(task, y)`, `.split(X, y)`, `.inner(train_idx)` (the nested plan for selectors, carrying group labels) and `.min_train_size(n)`.
2. Thread one `cv_plan` argument through the selectors and engines, keeping `cv_folds=`/`random_state=` as deprecated fallbacks that build a kfold plan.
3. Once every site takes a plan, add group_kfold and leave_one_group_out in one place.
4. Add a grep-based test that fails on any new `KFold(` outside cv_utils.
- **Benefit:** Grouped CV for replicates and sites (Border Cave, multi-site FTIR) becomes a small change. The CV the user picks is the CV actually used in every inner loop, which removes an optimistic-bias source in variable selection and preprocessing discovery. Contaminant correction can later be refit inside each fold through the same object.
- **Evidence:** `grep -nE 'KFold\(|LeaveOneOut\(' src/spectral_predict/*.py`: 42 hits, of which 6 are in cv_utils.py. The pain points list 'CV Strategy Phase 2' and T-15 grouped CV. docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md calls it the 'largest single gap'.

### A single evaluate_candidate() kernel shared by grid, Bayesian, NSGA-II and one-class (dedupe the copied helpers first) (method, impact high, effort XL)

- **Problem:** Each engine fits and scores a (preprocessing, subset, model, params) candidate its own way:
- search._run_single_config (search.py:4678, 907 lines) with _run_single_fold (:4334);
- create_unified_objective (unified_bayesian.py:1120, about 1,100 lines), using cross_val_predict_pooled at :1847-1935;
- SpectralOptimizationProblem._evaluate (nsga2_search.py:1071-1650), using cross_val_score_with_early_stopping, with its own model builder _build_model (:794) that hand-builds PLS-DA;
- run_one_class_search (search.py:5779, 1,421 lines) and contamination.run_one_class_cv.
Copied helpers: _needs_resampling_pipeline (search.py:310, unified_bayesian.py:506, nsga2_search.py:126), _apply_edge_mask_to_data (search.py:255, unified_bayesian.py:1091), _get_edge_zone_size (search.py:228, nsga2_search.py:328), _normalize_preprocess_name (unified_bayesian.py:540, nsga2_search.py:3317). There are 39+42+40 task_type branches in search, unified_bayesian and nsga2_search. Every weighting, early-stopping or head-param fix therefore has 3-5 sister sites: the class_weight discriminator was fixed at 5 sites in PR #38 (SESSION_LOG_ARCHIVE.md:3056), and PLS-DA head params had to survive 'rebuild/refit/ensemble/export' in #72.
- **Proposal:** Step 1 (S, no behaviour change): move the four copied helpers into one module (e.g. search_common.py) and import them everywhere.
Step 2: define `CandidateSpec` (preprocess cfg, wavelength indices, model name, params, weighting) and `evaluate_candidate(spec, X, y, cv_plan, *, early_stopping, sample_weight_policy) -> CandidateResult` (pooled OOF predictions, per-fold metrics, fitted-param capture). Grid calls it first; its output must be byte-identical on the existing gold-standard fixtures.
Step 3: the Bayesian objective and NSGA-II _evaluate become proposers that call it. Existing pinned sampler hashes (tests/test_t51_extra_axes_mechanism.py) and the T3 trajectory fixture act as parity guards.
Step 4: fold one-class in with a task strategy (score direction, metrics).
- **Benefit:** A fix lands once and all engines agree. Rankings from grid, Bayesian and NSGA-II become directly comparable. New staples such as PCR and EMSC, and statistical model comparison (block bootstrap CIs, paired permutation tests), can be built on one set of OOF predictions instead of three.
- **Evidence:** dups.py output (scratchpad), grep of cv_utils imports per engine, and the SESSION_LOG_ARCHIVE 'sister site' entries (~40 mentions across the logs, e.g. :2818 NSGA-II `.get('n_components')` on a string, found by Codex after grep missed it).

### Split the GUI test run into fast/default and nightly, and cap booster threads in tests (speed, impact high, effort S)

- **Problem:** The full GUI suite takes about 38 min (SESSION_LOG.md:687). On this machine the 264 non-comprehensive tests took 768.6 s (12:48), so the 34 `comprehensive` tests take roughly 25 min. Those TestAllModelsViaGUI / TestVariableSelectionViaGUI tests (tests/gui/test_comprehensive.py:109-470) call harness.run_analysis_direct, which calls run_search directly (harness.py:454-542, 'bypassing GUI threading'). They are backend benchmarks on the full 2,151-band data, and SESSION_LOG.md:9-19 records test_xgboost_via_gui at about 60 min on Windows CI.
In the fast set, 4 tests take 500 s, 65% of it:
- test_tab7_refit_keeps_bundle_params[XGBoost]: 254 s, on 60×30 synthetic data;
- the same test for [LightGBM]: 108 s;
- test_baseline_regression: 88 s;
- test_quick_plsda_analysis: 50 s.
No `addopts` or marker deselection exists (pyproject.toml:92-96), and pytest-xdist is not installed.
- **Proposal:** 1. In pyproject, set `addopts = "-m 'not comprehensive and not slow'"` and move the comprehensive benchmarks into tests/benchmarks/, run by a nightly or `-m comprehensive` job. They are backend tests and can run without Tk.
2. In tests/conftest.py, set OMP_NUM_THREADS=1 and patch or tag booster n_jobs=1 for unit and parity tests, whose assertions don't depend on threading.
3. Shrink test_workflows' baseline to a 3-model × 2-preprocess fixture on a band-decimated copy of the example data.
4. Add pytest-xdist to the lock for the non-GUI suite: `-n auto --dist loadfile`, since the tests are module-state sensitive (see the reimport_modules fixture from #70).
- **Benefit:** The GUI suite should drop from about 38 min to well under 10 min, by my estimate: 12.8 min minus most of the 500 s spent in the top 4 tests. That makes 'run the GUI suite before merging' routine instead of a background chore, which matters because agent reviewers and the user wait on it every PR.
- **Evidence:** scratchpad/improve/structure/gui_noncomp.txt (`1 failed, 263 passed, 34 deselected ... in 768.60s`, plus the slowest-40 table). `pytest --collect-only -m comprehensive` gives 34/298. The machine was at 100% CPU from concurrent agents during the measurement, so absolute times are inflated; the relative split still holds.

### Extract the Contaminant Analysis tab first, then Interference and Spectral Library, then Calibration Transfer behind a TransferChain service (workflow, impact high, effort M)

- **Problem:** All 15 tabs are methods on one class, and each shares state through the 1,501 attributes. The science tabs the user wants improved are among the easiest to lift out.
- Contaminant: 40 methods / 2,573 lines, reachable from outside only through `_create_tab13_contaminant_analysis`. Its external state is X, colors, notebook, root, data_value_scale, source_data_type and analysis_wl_custom; everything else is its own contam_* state. It makes 14 outward calls: data-type conversion, plot helpers, and _apply_epo_projection / _apply_glsw_weighting / _apply_opls_filter.
- Calibration Transfer: 55 methods / 4,417 lines, but 10 entry points (_apply_transfer_chain, _apply_transfer_with_roi, _load_transfer_models, _move_transfer_up/down, ...) are used by Model Prediction and Multi-Model.
Today a contributor improving contaminant removal has to load a 62k-line file in the editor or agent context, and any edit re-runs the GUI suite's shared-app state.
- **Proposal:** 1. Create src/spectral_predict/gui/ with an `AppContext` protocol: root, colors, notebook, get_working_X(), data-type helpers, add_plot_export_button.
2. Move the contaminant methods verbatim into `ContaminantTab(ctx)` in gui/contaminant_tab.py. SpectralPredictApp keeps a one-line `_create_tab13_contaminant_analysis` that instantiates it, plus thin forwarding attributes for the tests that reach into contam_* names.
3. Next, Interference Removal and Spectral Library as one module; they share interferent_libraries and library_* widgets.
4. For Calibration Transfer, first pull the transfer chain (transfer_models list plus apply/reorder/load) into a non-Tk `TransferChain` class used by the Prediction and Multi-Model tabs, then move the tab.
5. Build tabs lazily on first selection. This also cuts the 2.3 s startup and 3,530 eager widgets.
- **Benefit:** Work on contaminant removal or calibration transfer happens in a 2-4k-line module with a small explicit interface and its own fast headless tests. The backend correction classes (all sklearn TransformerMixin, contaminant_analysis.py:81-2272) can then be offered in-pipeline without touching the analysis tabs.
- **Evidence:** coupling2.py output: `Contaminant: methods=40 lines=2573 shared_state=30 calls_out=14 entry_points_used_outside=1` and `CalTransfer: methods=55 lines=4417 shared_state=124 calls_out=43 entry_points_used_outside=10`. Existing test: tests/gui/test_contaminant_tab.py (531 lines).

### Move 'results row → fitted pipeline' into one backend function and relocate the GUI's estimator wrappers (with a pickle shim) (workflow, impact medium, effort M)

- **Problem:** Model Development refit, ensemble reconstruction and prediction each rebuild models inside the GUI:
- _run_refined_model_thread (:39685, 2,595 lines);
- _reconstruct_models_from_results (:24632, 521 lines);
- _load_model_for_refinement (:36756, 512);
- _collect_refine_hyperparams (:36104, 277).
The GUI module defines six sklearn wrapper classes (WavelengthSubsetWrapper, GAPreprocessWrapper, CombinedPreprocessWrapper and their classifier twins, :2167-2540) and _build_transform_from_config (:1981). It has 57 direct preprocessing-pipeline constructions. search._rebuild_model_from_row (search.py:367) already exists but the GUI doesn't use it everywhere. Parity breaks often enough that there are dedicated suites: test_t20_saved_model_export_parity, test_export_cv_early_stopping_parity, test_multiclass_gui_parity, test_t51_pr_b_tab7_bundle_refit. That last one must set about 20 Tk vars to drive a refit (tests/gui/test_t51_pr_b_tab7_bundle_refit.py:28-55).
- **Proposal:** 1. Create src/spectral_predict/rebuild.py with `rebuild_pipeline(row, *, task, wavelengths, overrides=None) -> Pipeline` and `refit_with_cv(row, X, y, cv_plan, overrides) -> RefitResult`. Put it on the AGENT_COMPOSITION surface.
2. Move the wrapper classes and _build_transform_from_config into src/spectral_predict/wrappers.py, re-exported from the GUI module under the old names.
3. Because models are joblib-pickled (model_io.py:216) and wrappers defined in the GUI script can pickle as `__main__.X` or `spectral_predict_gui_optimized.X`, add a find_class remap in the model_io loader plus a fixture .dasp regression test before moving anything.
4. Refit, ensembles, export and prediction then all call rebuild_pipeline.
- **Benefit:** 'The refit doesn't match the search' stops being a recurring bug class. The Tab 7 refit tests become plain backend tests (the XGBoost one currently costs 254 s through the GUI). Scripted users get the exact pipeline the leaderboard row describes.
- **Evidence:** methsize.py output (method sizes); grep of wrapper classes, found only in the GUI and in ensemble.py docstrings; model_io.py:216 `joblib.dump(model, ...)`; PR #72 'PLS-DA head params survive rebuild/refit/ensemble/export' (PROJECT_STATUS §2).

### ModelSpec and PreprocessSpec registries, so adding a model or a step (PCR, EMSC, in-fold EPO/GLSW) touches one file (method, impact medium, effort L)

- **Problem:** Model knowledge is scattered. The string 'catboost' appears in 14 files: GUI 363 hits, models.py 86, search.py 29, code_generator.py 27, nsga2_search.py 24, cv_utils.py 18, templates/models.py 14, unified_bayesian.py 13, and others. Preprocessing names (e.g. deriv_snv) appear in 13 files. Grids are separate kwargs in run_search (about 90 model-grid params, for example `xgb_min_child_weight_list`). The Optuna space (unified_bayesian.suggest_model_params :781), the NSGA gene decoder (nsga2_search._decode_hyperparameter_genes :662, _build_model :794) and the code generators each re-encode every model. The one attempt to add an in-pipeline correction step is commented out at the run_search call: `# interference_settings=interference_settings,  # DISABLED: Code stashed (broke R² reproducibility)`. PCR and EMSC are still missing.
- **Proposal:** Extend model_registry.py from name lists to `ModelSpec(name, tasks, build(params, task), default_grid(task, n_features), suggest(trial, task, n_features), nsga_genes, codegen_template, supports_early_stopping, weighting_policy)`. Similarly, add `PreprocessSpec(name, params, build_transformer(cfg), grid_expansion, suggest, codegen)` in preprocess.py. Contaminant and interference transformers (EstimatedEPO, ContaminantGLSW, ContaminantOPLSDA, all TransformerMixin) then become ordinary specs fitted inside each CV fold. Migrate one model per PR, starting with a new one (PCR) to prove the path, then PLS.
- **Benefit:** New chemometrics staples ship in hours rather than across 10+ files. Contaminant removal can be scored honestly inside cross-validation instead of by correct-export-reimport. GUI model cards can be generated the way the T-51 bundle card is.
- **Evidence:** Per-file counts from `grep -ic catboost` and `grep -cE 'deriv_snv|snv_deriv'`. Disabled interference kwarg at the run_search call site in _run_analysis_thread (around spectral_predict_gui_optimized.py:30856). BUNDLES registry precedent: search_spaces.py.

### One reusable SpectraInput component to replace seven copies of file-load plus absorbance/reflectance state (usability, impact medium, effort M)

- **Problem:** Each tab re-implements 'pick file(s), read, detect absorbance vs reflectance, convert, preview'. The GUI has 44 `_browse_*`/`_load_*` spectra methods and 83 direct reader calls. The per-tab state repeats as *_original_data_type / *_data_converted / *_type_confidence triples for main, contam_, ct_primary_, ct_satellite_, ct_pred_, ct_export_ and comparison_. There are separate converters for each: _ct_pred_convert_data_type :50820, _comparison_convert_data_type :53444, _contam_convert_data_type :59404, _convert_data_type :21454, and the matching _update_*_data_type_ui methods.
- **Proposal:** Create gui/spectra_input.py: `SpectraInput(parent, ctx, *, allow_reference=False, on_loaded)` wrapping spectral_predict.io.read_spectra. It owns format detection, the abs/refl radio and detect_spectral_data_type, conversion, and preview. It returns a small immutable `LoadedSpectra(X, ids, wavelengths, data_type, converted)`. Replace the contaminant and calibration-transfer loaders first, since they are extracted in idea 7.
- **Benefit:** Every tab gets the same reader coverage and data-type handling. A reader improvement, such as relaxing the 100-wavelength minimum for handheld/MEMS instruments or adding a format, is made once. Users see one consistent import widget instead of seven slightly different ones.
- **Evidence:** `grep -oE 'self\.[a-z_]*(original_data_type|data_converted|type_confidence)\b' | sort -u` gives 18 attributes across 7 prefixes. Also `grep -nE '    def _(browse|load)_...'` → 44 and reader-call grep → 83.

### One thread-budget policy (estimator threads × fold jobs × trials) with a small benchmark (speed, impact medium, effort S)

- **Problem:** get_model defaults every estimator to n_jobs=-1 (models.py:325; hard-coded -1 again at :636-744). Grid CV separately parallelises folds with joblib (search.py:4911-4945, n_jobs_cv), with a partial exception list MODELS_PREFER_SERIAL_CV (:3184), so boosters with n_jobs=-1 can nest inside parallel folds.
On this 24-core box, with 40×2,151 synthetic data and 100 trees:
- XGBoost: 25.3 s per fit at n_jobs=-1 vs 0.76 s at n_jobs=1, and 24.3 s at n_jobs=4;
- LightGBM: 40.4 s vs 0.35 s;
- XGBoost on 60×30 data: 14.9 s vs 0.11 s.
The machine was at 100% CPU from other agents during these runs, which exaggerates OpenMP spin-wait costs; the effect must be re-measured on an idle machine before acting on the ratio. Even so, small-n spectral data gains nothing from intra-model threading, and users running dasp alongside other work will hit exactly this contention.
- **Proposal:** 1. Add `parallel_policy.py`, with one function deciding (estimator_n_jobs, fold_n_jobs, trial_n_jobs) from n_samples, the model and os.cpu_count(). Its default for n < ~500 is 1 thread per estimator and parallelism across folds or configs.
2. Route get_model, build_model, _run_single_config, the Bayesian objective and NSGA-II through it.
3. Add scripts/bench_threads.py (like scratchpad xgb_time2.py) and record results in docs.
4. Expose a single 'CPU cores to use' setting instead of per-path n_jobs literals.
- **Benefit:** Booster grids and Bayesian trials could get substantially faster: pause and stop respond sooner because trials are short, and the GUI stays responsive while a search runs. Tests speed up in the same way; the 254 s XGBoost refit test is almost all thread overhead on tiny data.
- **Evidence:** scratchpad/improve/structure/xgb_time.py and xgb_time2.py outputs (xgboost 3.4.1, cpu_count 24); psutil showed 100.0% CPU during measurement. Known pain point: 'CatBoost trial can run 20+ minutes' (PROJECT_STATUS.md:583).

### Isolate the shared session app: snapshot and restore every Tk var around each GUI test (workflow, impact medium, effort S)

- **Problem:** The GUI suite shares one SpectralPredictApp for speed (tests/gui/conftest.py session_app). Each test resets only X/X_original/y/results_df and the T-51 LEGACY_DEFAULTS vars (conftest gui_app fixture). Leaks are therefore handled one at a time:
- SESSION_LOG.md:689-691: ticking model boxes flips model_tier to 'custom'; setting task_type rewrites imbalance_method; order-sensitive restores are hand-coded in test_t51_pr_d_gui.py::_restore_shared_gui_state.
- PROJECT_STATUS §4 notes that PR D tests leave result-filter values and traces behind.
- One GUI test is already failing on main.
Order-dependent failures cost a full 38-min rerun to diagnose.
- **Proposal:** At session start, record {name: var.get()} for all 771 Tk vars on the app (they are enumerable via vars(app)). In the gui_app fixture teardown, restore them, with task_type first and dependent vars (imbalance_method, model_tier) last, and remove traces added during the test by diffing trace_info(). This costs a few ms per test (771 sets) and replaces the per-file restore helpers. Also, optionally, run the GUI suite in random order weekly to surface remaining leaks.
- **Benefit:** GUI test results stop depending on file order, so a red test means a real regression. New GUI tests need no bespoke cleanup code, which lowers the cost of every GUI PR.
- **Evidence:** time_app.py: `tk vars on app: 771`, app construction 2.31 s (so the shared app is worth keeping; isolation, not re-creation, is the fix). tests/gui/conftest.py lines ~105-125 (LEGACY_DEFAULTS-only reset). SESSION_LOG.md:689-691.
