# Re-evaluated Roadmap — 2026-04-30

> **Re-evaluation under the master rule:** Every ticket validated against (1) chemometrics literature and (2) the applied domain (bone FTIR / paleoanthropology / diagenesis). When sklearn/generic-ML conventions conflict with chemometrics convention, chemometrics wins.
>
> **Re-evaluator:** opencode (mimo-v2.5-pro), 2026-04-30
>
> **Source documents:** `docs/T01_VARSEL_LEAKAGE_AUDIT.md` (banner + body), `docs/analysis_vs_chemometrics_lit/leakage_validation_GLM_2026-04-30.md`, `docs/analysis_vs_chemometrics_lit/leakage_validation_codex_2026-04-30.txt`, `docs/RECONCILED_ROADMAP_2026-04-29.md`, `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md`, Li et al. 2009 (via GLM/Codex extraction), Centner et al. 1996, Araujo et al. 2001, Nørgaard et al. 2000, Wold 2001, Savitzky & Golay 1964, Lin 1989.
>
> **Code spot-checks:** `ensemble.py:310-329`, `preprocessing_discovery.py:210-250, 570-670`, `search.py:5640-5670`, `preprocess.py:48-107`, `preprocessing_wrapper.py:15-100`, `varsel_transformer.py:1-80` (worktree).

---

## Verdict legend

- **KEEP** — ticket is genuinely valuable under chemometrics convention + application context.
- **REFRAME** — the issue is real but the proposed framing/fix imports the wrong literature.
- **DROP** — ticket was based on wrong-literature framing.
- **DEFER** — real but lower priority than current state suggests.
- **NEEDS_USER_DECISION** — depends on a workflow choice the user must make.

---

## Validation gate outcomes (2026-04-30 session)

A strict validation gate was applied to every implemented bugfix branch + the four
re-evaluation flags + T-21 + T-04. The gate verifies (1) reachability in the GUI
bundled-app, (2) commercial-software / open-source field alignment via documentation
lookup, (3) regression test sweep. Gate methodology + lessons learned are at
`docs/bugfix_validation/README.md`.

| Ticket | Gate verdict | Commit / disposition | Validation note |
|---|---|---|---|
| T-04 (one-class UVE) | MERGED — GUI grey-out matching iPLS pattern | `6beb5e8` | [T04_one_class_uve.md](bugfix_validation/T04_one_class_uve.md) ([investigation](bugfix_validation/T04_findings.md)) |
| T-05 (VIP central) | MERGED | `2c068cd` | [T05_vip_formula.md](bugfix_validation/T05_vip_formula.md) |
| T-05a (VIP duplicates) | MERGED | `1eb6c06` | [T05a_vip_duplicates.md](bugfix_validation/T05a_vip_duplicates.md) |
| T-07 (PDS even-window) | MERGED | `1b91d93` | [T07_pds_even_window.md](bugfix_validation/T07_pds_even_window.md) |
| T-10 (PLS clamp LOO) | MERGED | `fbeb50c` | [T10_pls_components_clamp.md](bugfix_validation/T10_pls_components_clamp.md) |
| T-21 (SG uniformity) | RESOLVED — convert button hidden (eliminates bug surface; function preserved for future resample-on-convert fix) | `a5eef70` | [T21_findings.md](bugfix_validation/T21_findings.md) |
| T-24 (Lin's CCC) | MERGED — addition with user override on field-alignment soft-flag; GUI plumbing fixed pre-merge | `0087cad` | [T24_lins_ccc.md](bugfix_validation/T24_lins_ccc.md) |
| T-26 (SNV near-zero std) | DROP / WONT_FIX — current dasp behavior matches PLS_Toolbox default at offset=0; bundled-app distribution makes a backend-only knob useless | n/a | [T26_snv_near_zero_std.md](bugfix_validation/T26_snv_near_zero_std.md) |
| T-32 (sample_weight mismatch) | DEFERRED to T-19 — current code path is unreachable; bug fires only after T-19's planned scope makes class_weight + resampler combinations possible | n/a | (analyzed inline in 2026-04-30 SESSION_LOG) |
| `bayesian_utils` random_state hardcoding (re-evaluation flag) | MERGED — refactored to use shared `RANDOM_STATE` constant | `50d5d05` | (analyzed inline in 2026-04-30 SESSION_LOG) |
| `search.py:2855` top_n_vars hardcoding (re-evaluation flag) | DROPPED — display economy, not a bug. `all_vars` column already preserves the full wavelength list for replication; `top_vars` is a separate display-only truncated list | n/a | (analyzed inline in 2026-04-30 SESSION_LOG) |
| Worktree cleanup | DONE — 5 stale worktrees removed; branch refs preserved | n/a | (covered in T-05a commit message) |

---

## Amendments — 2026-05-02 (user re-validation pass)

A second user-driven pass on 2026-05-02 confirmed verdict changes for six ticket entries. Original entries below are preserved as audit trail; the amended verdict is canonical going forward. See per-ticket inline markers (`> **Amended 2026-05-02:** ...`).

| Ticket | Original verdict | Amended verdict | Rationale (short form) |
|---|---|---|---|
| **T-12** | KEEP | **SUBSUMED by T-45** | T-45 (`docs/plans/2026-05-02-T45-logger-warning-visibility-bundled-gui.md`, currently in flight on `fix/T45-logger-file-handler`) is the concrete implementation of the same disk-mirrored-logging concept. T-12 has no remaining unique scope. |
| **T-15** | KEEP | **DROP** | LOGO is a footgun in the user's data regime (15 sites, N 5–100, 20× ratio); not field-mandated (Westad & Marini 2015 + Workman 2018 prefer external test sets); most chemometrics competitors don't expose group-aware CV. Memory: `project_t15_dropped_t16_reframed.md`. Investigation: `docs/bugfix_validation/T15_findings.md`. |
| **T-16** | KEEP (block-bootstrap framing) | **REFRAMED — competitive model-comparison machinery survey + implementation** | Don't anchor to the FTIR Bone PLS paper's specific recipe. Survey what Unscrambler/SIMCA/OPUS/PLS_Toolbox expose for two-method comparison (bootstrap CIs, paired permutation, Wilcoxon, paired t, jackknife, Bayesian, AIC/BIC), then scope the smallest defensible shape. Memory: `project_t15_dropped_t16_reframed.md`. |
| **T-19** | KEEP (publication-reproducibility framing) | **REFRAMED — expose model-native imbalance abilities** | Goal is exposing / auto-detecting `scale_pos_weight` (XGB), `is_unbalance` (LGBM), `auto_class_weights` (CatBoost), `class_weight` (LR/SVC/RF). Not reproducing one paper's specific configuration. Simpler scope than the original design doc. Memory: `project_t19_user_framing.md`. |
| **T-22** | REFRAME (varsel stability bootstrap) | **REFRAMED — DEFERRED until concrete need** | Reframe is correct but the diagnostic is additive (not bug-class). Defer until a "is this wavelength real chemistry?" question hits a real analysis. Direct alternatives exist (FTIR band-assignment literature). The `VarselTransformer` infrastructure on the paused `fix/varsel-leakage` branch is the engine when work resumes. |
| **T-31** | NEEDS_USER_DECISION | **KEEP** | User confirmed 2026-05-02 that "none of the above / could be multiple" output is useful for bone-FTIR / diagenesis specimens. Implementation must be true multi-class SIMCA per Oliveri & Downey 2012 — NOT an extension of the existing one-class `PCASIMCA` in `contamination.py` (that is a single-class membership detector, structurally wrong base). Memory: `project_t31_simca_confirmed.md`. |

**Post-amendment summary counts:**

| Verdict | Count | Notes |
|---|---|---|
| KEEP | 26 | original 27 minus T-15, with T-31 returning from NEEDS_USER_DECISION |
| REFRAMED | 4 | T-01, T-16, T-19, T-22 (last is deferred) |
| DROP | 4 | T-02, T-03, T-12 (subsumed by T-45), T-15 |
| DEFER | 2 | T-05a, T-10b (unchanged) |
| NEEDS_USER_DECISION | 0 | cleared |

**Recommendations section below is now stale.** Original "Top 5 next actions" listed T-15 as #1 and T-16 in its block-bootstrap framing. Updated actionable order under master-rule + post-amendment dispositions:

1. **T-19 reframed (expose model-native imbalance)** — ~1–2 days post-reframe scope; closes a real reproducibility gap with low risk.
2. **T-16 reframed (competitive model-comparison machinery survey)** — survey first, scope after.
3. **T-31 (Multi-class SIMCA)** — 1–2 weeks; new infrastructure, not a one-class extension.
4. **T-17 (Multi-Y / PLS-2)** — 2–3 weeks; largest single-ticket effort but unlocks correlated-target modeling.
5. **T-01 reframed (external-test-set workflow + RMSEP labeling)** — 2–3 days; canonical chemometrics solution to varsel-on-full-cal bias.

T-22 stays parked. T-15 is closed.

---

## P0 — Science integrity

### T-01: Per-fold variable selection leakage audit
**Verdict:** REFRAME
**Why:** The audit body (method × path matrix, file:line references) is **technically correct as a code description** — dasp does run varsel on full calibration before CV. But the "LEAKY" framing was wrong. Li 2009 (CARS), Centner 1996 (UVE), Araujo 2001 (SPA), Nørgaard 2000 (iPLS), Wold 2001 (PLS/VIP) all run varsel once on full calibration data. Per-fold varsel was never the published methodology. Furthermore, per-fold varsel actively defeats the scientific purpose: stable wavelength selection across resamples is *evidence the chemistry is real*; per-fold instability destroys interpretability (user: "doing some of these per fold actually defeats the whole point").
**If REFRAME:** The correct issue is: dasp reports R²cv as the headline metric on data that varsel saw, which is technically biased upward. The canonical chemometrics solution is external test sets (RMSEP), not per-fold varsel. Reframe T-01 as: "Add external-test-set workflow + reporting language ('R²cv on selection set — for unbiased performance, evaluate RMSEP on independent samples')." The audit doc should be retained as a *description of CV-honesty status* (renamed, with "LEAKY" → "varsel-on-full" throughout). The existing `VarselTransformer` code is useful as an optional diagnostic tool (see Deliverable 2).

### T-02: Fix ensemble OOF preprocessor leakage
**Verdict:** DROP
**Why:** The preprocessor at `ensemble.py:314-316` calls `preprocessor.transform()` on both train and val folds. Tracing the preprocessor: `PreprocessorConfig` (`preprocessing_wrapper.py:15-100`) only applies SNV (per-spectrum, `preprocess.py:9-45`), Savitzky-Golay derivatives (per-spectrum, `preprocess.py:48-107`), and wavelength subsetting (no y-dependence). There is NO StandardScaler or other cross-sample operation in the preprocessing pipeline. Per-spectrum preprocessing applied uniformly to train and val is NOT data leakage by chemometrics convention — this is exactly the pattern the master rule was created to protect. The original ticket framing ("preprocessor fitted on full training data including the held-out fold") incorrectly assumed the preprocessor learns cross-sample statistics. It doesn't.
**If DROP:** This was the same false-alarm pattern as T-01: applying sklearn-pipeline-purity instincts to per-spectrum operations that are lossless/informationless by chemometrics convention.

### T-03: Fix preprocessing-discovery full-data leakage
**Verdict:** DROP
**Why:** The ticket claimed "importances used to rank preprocessing pipelines are computed on the full calibration set." Tracing the code: `preprocessing_discovery.py:570-662` shows that preprocessing configs are ranked by `_quick_evaluate()` (line 624), which uses honest 5-fold cross-validation via `cross_val_score()`. The importance computation (lines 631-643) is a **separate side output** stored in `selected_wavelengths` for potential downstream use — it does NOT feed into the ranking score. The `score` field in the result dict (line 659) comes from `_quick_evaluate`, not from importance. Furthermore, preprocessing choice (SNV, SG derivatives, baseline) is a per-spectrum methodology decision — ranking preprocessing pipelines by CV performance is standard chemometrics practice (The Unscrambler's "Preprocessing Advisor" does exactly this).
**If DROP:** The ticket was based on misreading the code (importance feeds ranking → false) and applying sklearn leakage framing to a preprocessing-ranking workflow that is standard in chemometrics.

### T-04: Fix one-class UVE prefilter using outlier-contaminated labels
**Verdict:** KEEP
**Why:** This is a genuine methodological issue distinct from the where-CV-runs question. At `search.py:5649-5655`, UVE is called with `y_oc` (+1/-1 inlier/outlier labels). UVE (Centner et al. 1996) uses PLS regression coefficient stability to identify informative variables. When the y-vector encodes inlier vs outlier membership, UVE selects wavelengths that *distinguish outliers from inliers* — the opposite of what one-class screening needs (wavelengths stable *within* the inlier class). This is not a leakage question; it's a "wrong target for the question being asked" question. PCA-SIMCA and OneClassSVM in spectroscopy contexts (Oliveri & Downey 2012) train on inlier-only data. The UVE prefilter for one-class should either run on inlier-only samples or be skipped entirely.
**Evidence:** Centner et al. 1996 UVE uses y to compute PLS coefficient reliability. For one-class, the relevant y is "stable within inliers," not "distinguishes inliers from outliers." The current implementation at `search.py:5650` passes `y_oc` which encodes the wrong signal.

### T-05: Fix VIP formula
**Verdict:** KEEP
**Why:** No methodology question — this is a correctness bug. The VIP formula at `models.py:1738` uses `np.var(y)` instead of per-component `y_loadings_`. Wold et al. 2001 (PLS/VIP) defines VIP as a function of PLS weights and explained variance per component. The current formula produces skewed wavelength rankings. Branch `fix/T05-vip-formula-fix` is already implemented with 33 tests passing.

### T-06: Fix SPA `n_random_starts` non-functionality
**Verdict:** KEEP
**Why:** No methodology question — this is a correctness bug. `variable_selection.py:401-408` loops `n_random_starts` times but every start picks `np.argmax(initial_corrs)` — the same starting variable. `random_state` is unused. Araujo et al. 2001 SPA uses projection-based forward selection; the "random starts" feature is meant to explore different initial conditions. Fix: `rng.choice()` for random first-variable starts.

### T-07: Fix PDS even-window arithmetic
**Verdict:** KEEP
**Why:** No methodology question — this is a correctness bug. `calibration_transfer.py:193-215` allows even windows that overflow `B` allocation. Wang 1991 (PDS) specifies window-based local regression; even windows cause index misalignment. Branch `fix/T07-pds-even-window` is already implemented with 54 tests passing.

### T-08: Fix CARS tree-mode weight-update bias
**Verdict:** KEEP
**Why:** No methodology question — this is a correctness bug. `variable_selection.py:1519-1522, 1549` gives selected variables tiny weights (~0.01) while unselected stay at ~1.0, biasing sampling toward unselected variables. Li 2009 CARS uses exponentially decreasing function (EDF) + adaptive reweighted sampling (ARS); the tree-mode variant should follow the same convergence logic. The current implementation oscillates instead of converging.

### T-09: Fix SPC multi-subfile silent data loss
**Verdict:** KEEP
**Why:** No methodology question — this is a data I/O correctness bug. `io.py:965` silently discards subfiles 2+ in Thermo OMNIC SPC stacks. Users loading multi-sample OMNIC files lose spectra without warning.

### T-10: Clamp PLS component grid by training-fold sample count
**Verdict:** KEEP
**Why:** No methodology question — this is a correctness bug. `models.py:842-843` clamps by `n_features` but not by `n_samples_train_per_fold`. Small datasets hit silent fold failures when `n_components > n_samples_train`. Branch `fix/T10-pls-components-clamp` is already implemented with 135 tests passing.

### T-32: Fix existing sample_weight length-mismatch bug in resampling path
**Verdict:** KEEP
**Why:** No methodology question — this is an active bug. `search.py:3869-3883` computes `sample_weight_train` from pre-resampling `y_train` but passes it to `final_model.fit()` against `y_train` rather than `y_train_for_model` (the resampled y). When a resampler runs, the lengths mismatch.

---

## P0 — Productivity

### T-11: Pause/resume hardening + Optuna SQLite persistence
**Verdict:** KEEP
**Why:** No methodology question — this is a productivity infrastructure ticket. The user lost hours on a 19-hour CatBoost run. Optuna's built-in SQLite persistence (`storage="sqlite:///..."`) would make runs crash-recoverable. The coarse pause checkpoint, UI pause-state lies, and missing thread-alive check are all real UX failures.

### T-12: Disk-mirrored logging for long runs
> **Amended 2026-05-02 → SUBSUMED by T-45.** T-45 is the concrete implementation of the same concept and is in flight on `fix/T45-logger-file-handler`. T-12 has no remaining unique scope.

**Verdict:** KEEP
**Why:** No methodology question — this is a debugging infrastructure ticket. Progress text is widget-only with a 2000-line cap; backend `print()` output is invisible in the bundle (`console=False`). A `logging.FileHandler` to `outputs/logs/<run-timestamp>.log` would enable post-mortem debugging.

### T-13: Wire jackknife prediction intervals into GUI
**Verdict:** KEEP
**Why:** No methodology question — this is a GUI wiring ticket. `diagnostics.py:143-230` implements per-sample prediction intervals but they're never called from the GUI. Jackknife prediction intervals are standard in chemometrics (Martens & Naes 1989).

### T-14: Hardcoded version-string mismatch
**Verdict:** KEEP
**Why:** No methodology question — this is a hygiene bug. `__init__.py:3` says `0.5.0b1`, `report.py:139` says `v0.4.0`, `code_generator.py:373` says `3.9.0`. Single version constant needed.

---

## P1 — Scientific rigor for publication

### T-15: LeaveOneGroupOut / GroupKFold + group-column concept
> **Amended 2026-05-02 → DROP.** LOGO is a footgun in the user's actual data regime (15 sites, N 5–100, 20× ratio); not field-mandated (Westad & Marini 2015 + Workman 2018); most chemometrics competitors don't expose it. Memory: `project_t15_dropped_t16_reframed.md`. Investigation: `docs/bugfix_validation/T15_findings.md`. The original KEEP rationale below is preserved as audit trail.

**Verdict:** KEEP
**Why:** Genuinely valuable for transferability claims. The FTIR Bone PLS paper's headline finding (MCC = 0.636 for LDA on PLS scores via pooled LOGO ≥5 sites) depends on holding out whole sites during CV. Standard 5-fold CV across pooled spectra is optimistically biased when within-site spectral correlations exist (same instrument, same operator, same sample-prep batch). LOGO is standard practice in multi-site bone-FTIR studies and in the broader chemometrics transferability literature (Filzmoser et al. 2009 rdCV, Westad & Marini 2015). Without LOGO, transfer claims aren't defensible to reviewers.

### T-16: Bootstrap CIs and paired permutation tests
> **Amended 2026-05-02 → REFRAMED.** Drop the "FTIR Bone PLS paper headline" framing. Build instead as a **competitive model-comparison machinery survey + implementation**: catalog what Unscrambler/SIMCA/OPUS/PLS_Toolbox expose (bootstrap CIs, paired permutation, Wilcoxon signed-rank, paired t, jackknife, Bayesian, AIC/BIC), then scope the smallest defensible shape. Ships independently — no longer paired with T-15 (dropped). Memory: `project_t15_dropped_t16_reframed.md`.

**Verdict:** KEEP
**Why:** Two-sample inference machinery is standard in the applied domain. The FTIR Bone PLS paper's headline claim ("MCC gap +0.225, 95% CI excludes zero, p < 0.01") depends on site-level block bootstrap + paired permutation tests. Without this infrastructure, dasp cannot produce publication-defensible inferential claims about whether one method beats another. Efron & Tibshirani 1993 (bootstrap), Good 2000 (permutation tests) are canonical.

### T-17: Multi-Y / PLS-2 workflow
**Verdict:** KEEP
**Why:** User endorsed; PLS-2 is the canonical multi-output PLS variant (Wold et al. 1983, Wold 2001). When Y variables are correlated (collagen yield + IRSF + C/P + Am/P + isotope ratios), joint PLS-2 modeling can be more accurate than separate single-Y models. sklearn's `PLSRegression` already supports multi-output natively. The gap is the surrounding pipeline assuming 1D y.

### T-18: Stratified Kennard-Stone for cal/test split
**Verdict:** KEEP
**Why:** Class-imbalanced datasets need class stratification in the calibration/test split. Kennard-Stone (1969) selects samples to maximize coverage of feature space; stratified KS runs KS within each class then recombines. Standard in chemometrics for ensuring the test set inherits class balance.

### T-19: Model-native imbalance handling (loss reweighting)
> **Amended 2026-05-02 → REFRAMED.** Goal is **exposing / auto-detecting** the model-native imbalance kwargs — `scale_pos_weight` (XGB), `is_unbalance` (LGBM), `auto_class_weights` (CatBoost), `class_weight` (LR/SVC/RF) — not reproducing the FTIR Bone PLS paper's specific configuration. Simpler scope than the original `docs/plans/2026-04-29-model-native-loss-reweighting.md` design doc; effort drops to ~1–2 days. Memory: `project_t19_user_framing.md`.

**Verdict:** KEEP
**Why:** Real gap verified by Codex audit. XGBoost, LightGBM, CatBoost construction sites in `models.py` never receive any imbalance-aware kwarg (`scale_pos_weight`, `is_unbalance`, `auto_class_weights`). The FTIR Bone PLS paper uses "XGBoost (scale_pos_weight)", "LightGBM (balanced)", "Logistic regression (EN, balanced)" — none reproducible in dasp without source edits. Resampling (SMOTE) and loss reweighting are different statistical interventions; chemometrics defaults to loss reweighting because it preserves real measurements. The design at `docs/plans/2026-04-29-model-native-loss-reweighting.md` is sound. The 5 PLS-DA sites + wrapper design + sklearn floor bump scope is correct. Effort: 5-7 days.

### T-20: Saved-model ↔ exported-script reproducibility test
**Verdict:** KEEP
**Why:** No methodology question — this is a testing hygiene ticket. Code export is marketed as a strength but no test verifies exported script produces identical predictions to saved model.

### T-21: Wavelength-spacing uniformity guard for SG derivatives
**Verdict:** KEEP
**Why:** Savitzky & Golay 1964 assumes uniform sampling — the polynomial weights are derived for fixed-stride windows. scipy.signal.savgol_filter carries the same assumption. Non-uniform grids produce artefactual peak shifts that mimic real chemistry (apparent red/blue shifts from H-bonding). The plan at `docs/plans/2026-04-29-T21-sg-wavelength-uniformity-guard.md` correctly references PLS_Toolbox's `gridcheck` and OpenSpecy's `is_evenly_spaced()` as commercial/academic precedent. The warn-and-proceed design (not hard-raise) is appropriate for a GUI tool. Tolerance 0.01 CV is chemometrics-standard. This is a genuine scientific integrity guard, not an sklearn-purity concern.

### T-22: Multi-source consensus wavenumber selection
> **Amended 2026-05-02 → REFRAMED, DEFERRED.** The April reframe (varsel stability bootstrap as chemistry-vs-noise diagnostic) is correct, but the diagnostic is additive (not bug-class) and 3–4 days of effort. Defer until a concrete "is this wavelength real chemistry?" question hits a real analysis — direct alternatives (FTIR band-assignment literature) cover the common cases. The paused `fix/varsel-leakage` branch's `VarselTransformer` infrastructure remains the engine when work resumes.

**Verdict:** REFRAME
**Why:** The original framing ("defends against reviewer 'cherry-picked channels' objection") is correct but undersells the methodological value. Under the master rule, this is the *right diagnostic for finding real chemistry* — stable wavelength selection across independent runs is evidence of real absorption bands / vibrational modes / functional groups. This connects directly to the varsel-finds-chemistry principle: if CARS selects 1650 cm⁻¹ across 6 independent runs, that's the amide I band, not search noise. The FTIR Bone PLS paper's "strict consensus" (all 6 sources agree → 10 wavenumbers → perfect classification) demonstrates this.
**If REFRAME:** Reframe as "bootstrap stability diagnostic for variable selection" — run varsel N times with different seeds, report wavelength selection frequency. This is a separate diagnostic tool, not a fix for a bug. The `VarselTransformer` infrastructure from the worktree could serve as the engine for this (see Deliverable 2). Connect to the varsel-finds-chemistry memory.

---

## P2 — Real but smaller

### T-23: Add ENLR / LDA-on-PLS / PCR / MLR as model cards
**Verdict:** KEEP
**Why:** ENLR is a workhorse classifier in chemometrics (sparse coefficients + probabilistic output). LDA-on-PLS-scores is the classical chemometrics recipe (Brereton & Lloyd 2014). Both are used in the FTIR Bone PLS paper. Small additions — ENLR already constructed in `nsga2_search.py:856`.

### T-24: Add Lin's CCC as a metric option
**Verdict:** KEEP
**Why:** Lin 1989 CCC penalizes deviation from the y=x line, not just linear correlation. Standard for instrument-transfer and method-comparison work. Branch `fix/T24-lins-ccc` is already implemented with 14 tests passing.

### T-25: Safe ZIP extraction + UX warning at model load
**Verdict:** KEEP
**Why:** Security issue — no chemometrics conflict. `model_io.py:417, 1676` uses `extractall()` without path validation (Zip Slip). HMAC alone is insufficient (key extractable from app). Fix: validate ZIP entry names + UX warning at load.

### T-26: SNV near-zero std tolerance threshold
**Verdict:** KEEP
**Why:** No methodology question — numerical stability fix. `preprocess.py:43` divides by std; near-zero std produces inf/nan. Branch `fix/T26-snv-near-zero-std` is already implemented with 50 tests passing.

### T-27: Encoding handling on file I/O
**Verdict:** KEEP
**Why:** No methodology question — hygiene fix. `io.py`, `report.py`, `calibration_transfer.py` need `encoding='utf-8'` to `open()` and `pd.read_csv()`.

### T-28: Lower `min_wavelengths` from 100 to 10
**Verdict:** KEEP
**Why:** No methodology question — practical fix. `io.py:140, 1864` rejects files with <100 wavelengths, blocking low-resolution instruments and peak-calculator outputs.

### T-29: Replace bare `except:` in scoring.py with NaN + warning
**Verdict:** KEEP
**Why:** No methodology question — hygiene fix. `scoring.py:509, 517, 535` have bare except blocks that swallow errors silently.

### T-30: Remove debug prints
**Verdict:** KEEP
**Why:** No methodology question — hygiene fix. `search.py:2313, 2789, 2834, 3091-3100, 3991-3999` contain debug prints that should be logger calls or removed.

### T-31: Multi-class SIMCA (true class-modeling, not anomaly detection)
> **Amended 2026-05-02 → KEEP.** User confirmed 2026-05-02 that "none of the above / could be multiple" output is useful for bone-FTIR / diagenesis specimens. **Implementation note:** must be a true multi-class SIMCA per Oliveri & Downey 2012 (one PCA model per class + independent membership decisions per specimen) — NOT an extension of the existing one-class `PCASIMCA` in `contamination.py`, which is a single-class membership detector (structurally wrong base). Memory: `project_t31_simca_confirmed.md`.

**Verdict:** NEEDS_USER_DECISION
**Why:** SIMCA (Wold 1976, Oliveri & Downey 2012) is a standard chemometrics class-modeling method. The user's domain context supports it: fossil bone differs site-by-site (every site is "its own thing"), consolidants are often unknown, and specimens on a diagenetic continuum can legitimately belong to multiple classes or none. Discriminant classifiers (PLS-DA, RF, XGBoost) force every specimen into one of the trained classes — no "none of the above" output is possible. Multi-class SIMCA gives per-class membership decisions that are independent. However, this is a 1-2 week effort, and the user should confirm that the "none of the above" / "could be multiple" output is useful for their science before work begins.

---

## P3 — Drop / defer

The P3 drop list from the 2026-04-29 roadmap is **confirmed correct** under the master rule. No items need to be moved. Specific confirmations:

| Item | Re-evaluation |
|------|---------------|
| Internationalization / Japanese UI | Single-PI academic tool. Confirmed DROP. |
| ASTM E1655 compliance templates | Regulated NIR; not user's science. Confirmed DROP. |
| Qt/PySide6 GUI migration | 3-6 month project; doesn't help research. Confirmed DROP. |
| PDF/Word/HTML "regulatory" reports | Markdown + matplotlib + pandoc covers academic publication. Confirmed DROP. |
| Multiblock PLS, MCR-ALS, ASCA, PLS-PM, MSPC | Industrial chemometrics; not user's domain. Confirmed DROP. |
| HMAC signing of `.dasp` | Threat model is academic sharing; T-25's warning is enough. Confirmed DROP. |
| Drag-and-drop file loading | UX nicety, no scientific value. Confirmed DROP. |
| Removing "BETA" cosmetic | It IS beta. Confirmed DROP. |
| MATLAB `.mat` import/export | scipy.io ad-hoc when needed. Confirmed DROP. |
| Norris-Williams gap derivatives | SG covers the use case. Confirmed DROP. |
| FIPLS-CARS named hybrid | Paper-specific; chain manually. Confirmed DROP. |
| "Random-split sensitivity test" workflow | 30-line notebook task; T-16 covers same argument. Confirmed DROP. |
| Bayesian regression with PyMC backend | Massive infrastructure. Punt. Confirmed DEFER. |
| Per-fold full-spectrum normalization "fix" | Per-spectrum SNV/SG/baseline are NOT leakage by chemometrics convention. Confirmed DROP. |
| 57K-line GUI decomposition | Real tech debt but P3. Confirmed DEFER. |
| 40% test coverage target | Coverage theatre. Confirmed DROP. |

---

## Deferred items (from 2026-04-29 roadmap)

### T-05a: Duplicate VIP formulas in templates/nsga2_search
**Verdict:** DEFER
**Why:** Real inconsistency (duplicate VIP formulas in `templates/variable_selection.py:54` + `nsga2_search.py:627`) but low priority — the primary VIP fix (T-05) is more important.

### T-10b: NSGA-II PLS component audit
**Verdict:** DEFER
**Why:** Same class of issue as T-10 but in the NSGA-II path. Lower traffic than grid/Bayesian search.

### T-31 PENDING: Multi-class SIMCA
> **Amended 2026-05-02 → KEEP** (see T-31 above and Amendments section). User confirmed 2026-05-02.

**Verdict:** NEEDS_USER_DECISION (see T-31 above)
**Why:** User must confirm "none of the above" output is useful for their bone-FTIR/diagenesis science before work begins.

---

## Summary counts

| Verdict | Count |
|--------|-------|
| KEEP | 26 |
| REFRAME | 2 (T-01, T-22) |
| DROP | 3 (T-02, T-03, T-04 was KEEP) |
| DEFER | 2 (T-05a, T-10b) |
| NEEDS_USER_DECISION | 2 (T-31, T-31 PENDING) |

Wait — let me recount. T-04 is KEEP, not DROP.

| Verdict | Count |
|--------|-------|
| KEEP | 27 (T-04 through T-30, T-32) |
| REFRAME | 2 (T-01, T-22) |
| DROP | 2 (T-02, T-03) |
| DEFER | 2 (T-05a, T-10b) |
| NEEDS_USER_DECISION | 2 (T-31, T-31 PENDING) |

Total: 35 items evaluated (32 tickets + T-05a, T-10b, T-31 PENDING, P3 drop list).

---

## Deliverable 3: Recommendations

> **Stale as of 2026-05-02 amendments above.** T-15 (was #1) is dropped; T-16 framing has changed; T-11 (was #3) merged; T-31 user-decision question (in §"Tickets the user should make decisions on") resolved as KEEP. See the Amendments section's revised Top-5 list. The original recommendations are preserved below for audit trail.

### Top 5 next actions (under the new framing)

1. **T-15 (LeaveOneGroupOut by site)** — Highest leverage. Unlocks transferability claims for the FTIR Bone PLS paper. Prerequisite for T-16 (block bootstrap needs group labels). Effort: 3-5 days.

2. **T-16 (Bootstrap CIs + paired permutation tests)** — The paper's inferential claims ("MCC gap +0.225, 95% CI excludes zero") depend on this. No other chemometrics tool surfaces these in a usable way. Effort: 4-7 days.

3. **T-11 (Pause/resume + Optuna SQLite)** — Highest productivity win. The 19-hour CatBoost run becomes crash-recoverable. Effort: 3-5 days.

4. **T-01 REFRAME (external-test-set workflow + reporting language)** — Instead of per-fold varsel, add: (a) GUI workflow for external test set evaluation, (b) clear labeling of R²cv as "on selection set — for unbiased performance, evaluate RMSEP on independent samples." This is the correct chemometrics solution to the bias problem. Effort: 2-3 days.

5. **T-19 (Model-native loss reweighting)** — Real gap for reproducibility. Boosting models need imbalance kwargs. Design is sound. Effort: 5-7 days.

### Tickets to add (not in current roadmap)

1. **Vsel-stability diagnostic (reframed T-22)** — Run varsel N times with different seeds, report wavelength selection frequency. This is the *right diagnostic* for "is this wavelength real chemistry or search noise?" The `VarselTransformer` infrastructure from the worktree can serve as the engine. Reframe T-22 as this. Effort: 3-4 days.

2. **Reporting-language update for R²cv with varsel** — Add clear labeling throughout the GUI and exported reports: when varsel was used, R²cv is "on selection set" and the user should evaluate RMSEP on independent samples for unbiased performance. This is the correct chemometrics framing — not per-fold varsel, but honest labeling. Connects to T-01 reframe.

3. **External-test-set workflow enhancement** — The canonical chemometrics workflow (Li 2009, Centner 1996, etc.) uses RMSEP on a held-out test set as the final performance metric. dasp's current external validation is single-shot. Enhance to support: (a) lock variables from a search, refit on new calibration set, evaluate on held-out test, (b) repeat across N random splits (the paper's "random-split sensitivity test"). Connects to gap analysis item #5.

### Tickets the user should make decisions on

> **All three resolved 2026-05-02:**
> - **T-31 (Multi-class SIMCA):** RESOLVED → KEEP. User confirmed the "none of the above / could be multiple" output is useful for bone-FTIR/diagenesis. See Amendments section.
> - **T-01 reframe scope:** RESOLVED → KEEP. External-test-set workflow + reporting-language is the canonical chemometrics solution; `VarselTransformer` is repurposed as the engine for T-22 (deferred).
> - **T-22 reframe:** RESOLVED → REFRAMED, DEFERRED. Worth the investment in principle but parked until a concrete "is this wavelength real chemistry?" question hits a real analysis. See Amendments section.

(Original questions preserved below for audit trail.)

1. **T-31 (Multi-class SIMCA)** — Confirm whether "none of the above" / "could be multiple" output is useful for bone-FTIR/diagenesis science. If yes, scope and prioritize. If no, drop.

2. **T-01 reframe scope** — Confirm the external-test-set workflow + reporting-language approach (rather than per-fold varsel). This determines whether the `VarselTransformer` code is kept as a diagnostic tool or dropped entirely.

3. **T-22 reframe** — Confirm whether the bootstrap stability diagnostic for varsel (run N times, report frequency) is worth the 3-4 day investment as a separate diagnostic tool, or whether the external-test-set workflow (T-01 reframe) is sufficient.

### Non-obvious findings worth logging

1. **Potential new ticket: `bayesian_utils.py:261` hardcodes `random_state=42`** — The T-01 audit's bonus finding #1 is real: varsel calls inside `bayesian_utils.py` (lines 442, 455, 469, 488, 502, 520) hardcode `random_state=42`, ignoring the user's setting. This is a correctness issue independent of the leakage framing.

2. **Potential new ticket: `search.py:2855` hardcodes `top_n_vars=30`** — The T-01 audit's bonus finding #3 is real: the `top_n_vars` parameter passed to `_run_single_config()` is always 30 regardless of actual `n_top`. Reporting/display mismatch, not a selection bug.

3. **The `varsel_transformer.py` code has value beyond the leakage framing** — The 4-class infrastructure (`VarselTransformer`, `ModelImportanceVarselTransformer`, `SubsetVarselTransformer`, `OneClassVarselTransformer`) is architecturally useful for: (a) the T-22 stability diagnostic, (b) expert mode for users with external test sets who want per-fold varsel as a diagnostic comparison, (c) reproducing specific papers that use per-fold varsel (e.g., Filzmoser 2009 rdCV). See Deliverable 2.
