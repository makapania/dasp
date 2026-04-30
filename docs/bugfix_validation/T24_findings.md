# T-24 findings: Lin's Concordance Correlation Coefficient as a regression metric

**Branch:** `fix/T24-lins-ccc` (HEAD `ccc34c0`)
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30
**Worktree:** `.worktrees/T24-lins-ccc/`

This is the investigation phase. No verdict, no merge.

T-24 is structurally different from T-05/T-07/T-10/T-26 — it is an **addition**, not a bugfix.
dasp main does not compute CCC anywhere; the branch adds it as an always-computed regression
metric. So the validation question is not "is the fix correct?" but rather:

1. Is the formula bit-correct vs Lin (1989)?
2. Is CCC a recognized metric in chemometrics / NIR / FTIR / the user's applied domain (bone
   FTIR, paleoanthropology, isotopes)?
3. Do leading commercial chemometrics tools (PLS_Toolbox, SIMCA, Unscrambler) report CCC
   alongside R², RMSE, RPD?
4. Will the GUI user actually see CCC in the bundled-app workflow, or is it a hidden export-
   only column?

---

## 1. What the branch actually changes

Source / test diff vs main, summary:

| File | Change |
|------|--------|
| `src/spectral_predict/scoring.py` | Adds `lins_ccc(y_true, y_pred) -> float` helper (~70 LOC including docstring); adds `"CCC", "CCCcv"` to regression `metric_cols` in `create_results_dataframe` |
| `src/spectral_predict/search.py` | Imports `lins_ccc`; computes `cal_ccc` and `ccc_cv`; emits `result["CCC"]` and `result["CCCcv"]` in regression result dict |
| `src/spectral_predict/unified_bayesian.py` | Imports `lins_ccc`; computes `cal_ccc` / `ccc_cv` per Optuna trial; stores via `trial.set_user_attr('CCC', ...)` / `'CCCcv'`; emits in `convert_study_to_dataframe` |
| `src/spectral_predict/nsga2_search.py` | Imports `lins_ccc`; computes `CCC` (calibration) in `_compute_calibration_metrics`; computes `CCCcv` (pooled CV) in `_compute_nir_metrics`; emits in `convert_nsga2_to_v1_format` |
| `src/spectral_predict/templates/validation.py` | Adds `_lins_ccc()` helper to exported standalone validation scripts; computes/prints `ccc` from pooled CV predictions; computes/prints `cal_ccc` from final-model calibration |
| `tests/test_scoring.py` | New `TestLinsCCC` class with 14 tests (was 13 in plan; one extra finite-sample ddof-disambiguation test was added per Codex feedback) |
| `docs/PROJECT_STATUS.md`, `docs/SESSION_LOG.md` | Standard session-protocol updates |

Total: 8 files, +276/-8 lines. Source change is concentrated in 4 result-construction sites
(search.py / unified_bayesian.py / nsga2_search.py / templates/validation.py) plus the helper
in scoring.py — exactly the 4 sites the plan called for.

### 1a. The helper itself (`scoring.py:369-439`)

```python
def lins_ccc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return float("nan")

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()                   # ddof=0
    var_pred = y_pred.var()                   # ddof=0
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))   # ddof=0

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator == 0.0:
        return 1.0
    if var_true == 0.0 or var_pred == 0.0:
        return 0.0

    return float(2.0 * cov / denominator)
```

### 1b. Wiring sites (verified line-by-line)

| File | Site | Cal CCC / CV CCC |
|------|------|-------------------|
| `search.py:4362` | `_run_single_config` regression CV block, after `bias_cv` | `ccc_cv = lins_ccc(all_y_test, all_y_pred)` (pooled) |
| `search.py:4484, :4522, :4744, :4748` | Calibration block + result dict | `cal_ccc = lins_ccc(y, y_pred_cal)`; `result["CCC"]`, `result["CCCcv"]` |
| `unified_bayesian.py:1277, :1437, :1442` | Optuna objective + trial.user_attrs | Same pattern; `cal_ccc`, `ccc_cv`, stored as user attrs |
| `unified_bayesian.py:2056, :2059` | `convert_study_to_dataframe` | `row["CCC"]`, `row["CCCcv"]` from trial user attrs |
| `nsga2_search.py:2908, :3481` | `_compute_nir_metrics`, `_compute_calibration_metrics` | `'CCCcv': float(lins_ccc(y, y_pred_cv.ravel()))`, `metrics['CCC'] = lins_ccc(y, y_pred.ravel())` |
| `nsga2_search.py:3655, :3691` | `convert_nsga2_to_v1_format` | `row['CCC']`, `row['CCCcv']` |
| `templates/validation.py:60-77` | Adds `_lins_ccc` helper to exported template | matches scoring.py implementation |
| `templates/validation.py:125, :195, :237` | Both metric computation block and final-model template print CCC | exported scripts now print `ccc` and `cal_ccc` |

Coverage of all 4 result-construction paths is complete. Pooled CCCcv (not averaged) is used
throughout, matching how RMSEcv / R²cv are computed in this codebase.

### 1c. What is NOT done — GUI tooltip + sort direction

The plan (Task 7) called for two GUI changes:

1. Add `'CCC'` and `'CCCcv'` tooltip entries in `spectral_predict_gui_optimized.py:1437-1448`
   (the `'metrics': { ... }` dict).
2. Add `'CCC'` and `'CCCcv'` to `higher_is_better_cols` set in `_sort_results_by_column`
   around line 27964.

Neither change is in the branch. `Grep CCC` in the GUI file returns only CSS hex color matches
(`#CCCCCC`) — no functional CCC references. This is a real planned-but-not-executed gap and
matters for distribution-model verdict (Section 6).

### 1d. Tests added

14 tests in `tests/test_scoring.py::TestLinsCCC` covering:
- Perfect prediction → 1.0 (`test_perfect_prediction_returns_one`)
- Perfect anti-correlation → -1.0 (`test_perfect_anticorrelation_returns_minus_one`)
- Bias-only `ŷ = y + 5`: Pearson r = 1, CCC < 1 (`test_bias_only_below_one_even_when_pearson_is_one`)
- Scale-only `ŷ = 2y` (zero-mean y): closed-form CCC = 0.8 (`test_scale_only_below_one_even_when_pearson_is_one`)
- Stochastic scale-only at large n: CCC ≈ 0.8 (`test_known_closed_form_scale_only`)
- **Codex-recommended** finite-sample ddof check: `y=[0,1,2], pred=[1,2,3]` → CCC = 4/7 (`test_ccc_finite_sample_ddof_zero_exact`) — disambiguates ddof=0 from ddof=1
- Range bounds [-1, 1] on random pairs (`test_range_within_bounds_random_inputs`)
- Symmetry in arguments (`test_symmetry_in_arguments`)
- NaN propagation (`test_nan_in_inputs_returns_nan`)
- Length-mismatch raises (`test_length_mismatch_raises`)
- Constant predictions / truth → 0.0 (`test_constant_predictions_returns_zero`,
  `test_constant_truth_returns_zero`)
- Both equal constants → 1.0 (`test_both_constant_and_equal_returns_one`)
- Accepts list / pandas Series inputs (`test_accepts_lists_and_pandas_series`)

All 14 pass on the branch (verified — Section 8).

---

## 2. Canonical Lin (1989) formula

**Reference:** Lin, L. I.-K. (1989). "A concordance correlation coefficient to evaluate
reproducibility." *Biometrics* 45(1), 255–268.

The canonical definition:

```
CCC = (2 σ_xy) / (σ_x² + σ_y² + (μ_x − μ_y)²)
```

Equivalently, using `σ_xy = ρ σ_x σ_y`:

```
CCC = (2 ρ σ_x σ_y) / (σ_x² + σ_y² + (μ_x − μ_y)²)
```

Range: `[-1, 1]`. CCC = 1 iff predictions fall exactly on the 1:1 line. CCC = -1 iff perfectly
anti-concordant (mirror image across origin with equal variances and zero bias).

**Lin's decomposition** (Section 3 of Lin 1989, restated more accessibly in the
NCSS / yardstick / DescTools docs):

```
CCC = ρ · C_b
```

where `ρ` is the Pearson correlation (precision component) and `C_b ∈ [0, 1]` is the bias
correction factor (accuracy component, i.e. how close the predictions are to the 1:1 line):

```
C_b = 2 / (v + 1/v + u²)
v = σ_x / σ_y                     # scale shift
u = (μ_x − μ_y) / sqrt(σ_x σ_y)   # location shift
```

This decomposition is the standard pedagogical framing — CCC = (precision) × (accuracy).

**Variances are population (`ddof=0`).** Lin's original derivation in §2 of the 1989 paper
uses MLE-style population variances; the `ddof=0` choice in dasp matches Lin (1989), the
yardstick R package, the metrica R package, the NIRPY tutorial, and the Wikipedia
formulation. Using `ddof=1` is also seen in some implementations (a finite-sample correction)
but is not Lin's original; switching to `ddof=1` would change the closed-form
`y_pred = 2y → CCC = 0.8` test, so the choice is non-trivially encoded into the test contract.

---

## 3. What the branch implements vs canonical

Side-by-side mapping:

| Lin (1989) symbol | Branch code | Match? |
|-------------------|-------------|--------|
| `μ_x` | `mean_true = y_true.mean()` | ✓ |
| `μ_y` | `mean_pred = y_pred.mean()` | ✓ |
| `σ_x²` | `var_true = y_true.var()` (ddof=0) | ✓ |
| `σ_y²` | `var_pred = y_pred.var()` (ddof=0) | ✓ |
| `σ_xy` | `cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))` (ddof=0) | ✓ |
| `(μ_x − μ_y)²` | `(mean_true - mean_pred) ** 2` | ✓ |
| `2 σ_xy / (σ_x² + σ_y² + (μ_x − μ_y)²)` | `2.0 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2)` | ✓ |

**Bit-correct.** No ambiguity, no rearrangement that loses precision, no factor mixed up. The
implementation is the formula in Section 2 of Lin (1989), exactly.

The two degenerate-input branches (returning 0.0 or 1.0) are an explicit display convention
on top of Lin's formula. Lin (1989) defines CCC as undefined when variance is zero. The
branch's choice is documented in the docstring as a "DISPLAY CONVENTION for ranking"
(scoring.py:383-386). Codex flagged this is acceptable but should be labeled as a convention,
not Lin's definition — the docstring does this.

---

## 4. Empirical demonstration of correctness

I ran the branch's helper against five test cases. All match the analytic expectation:

```
1. CCC(y, y)         = 1.0                 # perfect agreement
2. CCC(y, -y)        = -1.0                # perfect anti-concordance (zero-mean inputs)
3. CCC(uncorrelated) = -0.0079             # near zero for random independent inputs (n=1000)
4. y_pred = y + 5
   Pearson r = 1.0
   CCC       = 0.4096                      # < 1 because of bias offset (R² would also be < 1
                                            # but Pearson r is fooled — exactly the case CCC
                                            # was designed to catch)
5. y_pred = 2y, y zero-mean
   Pearson r = 1.0
   CCC       = 0.8                         # closed-form: 4σ²/5σ² = 0.8 (matches Lin §2 exactly)
6. y=[0,1,2], pred=[1,2,3]
   CCC       = 0.5714286... = 4/7          # finite-sample exact, distinguishes ddof=0 vs ddof=1
                                            # (ddof=1 would give a different value)
```

Cases 1–3 confirm the bounds are tight and zero-handling is sane. Case 4 is the canonical
"why CCC and not Pearson r" demonstration — Pearson r is bias-blind; CCC catches the bias.
Case 5 is the closed-form scale-only check (Lin §2, Eq. 2). Case 6 is the Codex-recommended
ddof-disambiguation case.

The implementation is bit-correct on all canonical test inputs.

---

## 5. Field-alignment check (the critical T-24 question)

This is where T-24 differs from T-05/T-10/T-26. The fix is bit-correct — the question is
whether dasp should compute this metric at all for its user base.

### 5a. Is CCC a recognized metric in regression / method comparison?

**Yes, unambiguously.** CCC is the standard "agreement between two continuous variables"
metric since Lin's 1989 *Biometrics* paper. It is canonical in:

- Clinical chemistry / biochemistry / lab medicine (method comparison: gold standard vs new
  technique)
- Bioassay validation
- Imaging reproducibility studies
- Any "method A vs method B" agreement assessment

Lin (1989) has 5,000+ citations. Bland & Altman (1986) is the alternative for the same
problem class (limits-of-agreement plot rather than a single coefficient).

### 5b. Is CCC a standard chemometrics / NIR / spectroscopy metric?

**Mixed evidence — emerging but not yet universal.**

- **NIRPY Research** (a community NIR/chemometrics tutorial site at nirpyresearch.com) has a
  dedicated tutorial titled "The Concordance Correlation Coefficient" that explicitly
  recommends CCC for NIR regression: "*The CCC can be informative for the quantification of
  regression results in spectroscopy. ... particularly when the prediction data shows a
  correlation that doesn't sit exactly on the 1:1 line.*" This is direct evidence that CCC is
  in the active NIR-chemometrics conversation.

- **R `metrica` package** lists CCC explicitly (#10 of 48 regression metrics) as an
  "agreement" metric in its `available_metrics_regression` vignette. Not chemometrics-specific
  but available in R's general regression evaluation toolkit.

- **R `yardstick`** (tidymodels) provides `ccc()` as a first-class regression metric. Formula
  matches Lin (1989) exactly. Not chemometrics-specific but heavily used in applied stats.

- **R `DescTools::CCC()`** and **R `epiR::epi.ccc()`** provide CCC for general method comparison.

- **R `lcc` package** is dedicated to longitudinal CCC estimation.

- **Wikipedia / Real Statistics / NCSS / MedCalc** all describe CCC in standard agreement-metric
  terms.

- **scikit-learn**: no native CCC. `keras3` has `metric_concordance_correlation()`. The
  metric is recognized in modern ML evaluation suites.

### 5c. Do leading commercial chemometrics tools report CCC?

**Probably no — but I cannot fully verify by remote documentation lookup.**

I searched for explicit mentions of CCC in the documentation of PLS_Toolbox (Eigenvector
Research), SIMCA (Sartorius/Umetrics), and Unscrambler (CAMO/Aspen). None of those vendor
documentation searches surfaced CCC as a standard reported metric. Their default report
metrics are RMSEC / RMSECV / RMSEP / R² / bias / slope / SEP / SEC.

Caveats: vendor documentation is only partially indexed by web search; PLS_Toolbox user-guide
material is behind license walls; SIMCA's user-guide PDF was searched but didn't surface CCC
in the indexed snippets. So this is a "weak no" — *I could not find evidence that any of the
three flagship commercial chemometrics programs reports CCC by default.*

This contrasts with R² / RMSE / Bias / RPD / RER, which appear in **all** of those programs'
default reports.

### 5d. Open-source chemometrics packages (the more rigorous evidence)

I searched the documentation pages for the chemometrics-focused R/Python packages (the same
list used in the T-05 validation):

| Package | Domain | CCC included? |
|---------|--------|---------------|
| R `mdatools` (Kucheryavskiy) | Chemometrics | **No** explicit CCC in default regression summary (verified via mda.tools/docs/regression-coefficients.html — shows coefficients/SE/t/p, not agreement metrics; `summary()` for regression models reports R²/RMSE/bias/SEP, no CCC) |
| R `rchemo` | Chemometrics, PLS/locally-weighted | No explicit CCC in regression metrics docs |
| R `prospectr` | NIR preprocessing | No regression metrics — preprocessing-focused |
| R `pls` (mevik) | PLS regression | No CCC in `summary.mvr` output |
| R `chemometrics` (Filzmoser) | Chemometrics tutorial | No CCC by default |
| R `simplerspec` (Baumann, NIR) | NIR | Did not surface CCC in default metrics |
| Python `scikit-learn` | Generic ML | No native CCC (would need user implementation) |
| R `yardstick` (tidymodels) | Generic ML | **Yes** — first-class `ccc()` |
| R `metrica` | Crop/agricultural prediction | **Yes** — listed as one of 48 metrics |

**The chemometrics-specific packages (`mdatools`, `rchemo`, `pls`, `chemometrics`) do not
include CCC by default in their regression report tables.** The general ML evaluation
packages (`yardstick`, `metrica`, `keras3`) do.

This pattern matches the user's master rule's caveat 4: leading chemometrics-specific tools
default to RMSE / R² / RPD / Bias / SEP, not CCC. But CCC is increasingly visible in the NIR
tutorial space (NIRPY) and is canonical in adjacent fields (clinical chemistry, ML).

### 5e. Bone FTIR / paleoanthropology / isotope domain check

I could not find a paper combining bone FTIR + PLS regression + CCC reporting in the user's
specific domain. Searches for combinations of "concordance correlation coefficient", "bone
collagen", "FTIR", "isotope", "paleoanthropology" (2023–2025) returned zero direct hits.

The user's domain papers I am aware of (and that the search results corroborate) — e.g. France
et al. *Radiocarbon* 2014 "Rapid quantification of bone collagen content by ATR-FTIR
spectroscopy", Lebon et al. *Scientific Reports* 2018 (universal apatite crystallinity
curve), Dal Sasso et al. *Scientific Reports* 2020 (FTIR-ATR archaeological collagen) —
report R² / RMSE / SEC / SECV / bias, not CCC. The *Hungarian cemeteries* diagenesis paper
(Springer 2022) and the bioapatite stable-isotope methodology papers (ScienceDirect 2019,
2020) similarly report R² / RMSE / regression slopes.

**So in the user's specific domain, CCC is not currently a reported metric.** It is used
adjacent to the domain (bone density studies, biomedical method-comparison work that bone
researchers might cite from clinical chemistry contexts), but the bone FTIR / collagen
calibration literature has not adopted it as standard.

### 5f. Net field-alignment assessment

CCC is:

- **Universally accepted** as the standard agreement-coefficient since Lin (1989). Formula is
  not contested. dasp's implementation is bit-correct vs the original paper.
- **Standard** in clinical chemistry / lab medicine method comparison.
- **Emerging but not yet universal** in chemometrics — the NIR tutorial space (NIRPY)
  recommends it, the general ML eval packages have first-class CCC, but the chemometrics-
  specific R packages (`mdatools`, `rchemo`, `pls`) and the flagship commercial programs
  (PLS_Toolbox, SIMCA, Unscrambler) do not appear to report it by default.
- **Not yet adopted** in the user's specific applied domain (bone FTIR / paleoanthropology /
  isotopes). Domain papers report R² / RMSE / Bias.

This is a different shape than T-05 (where every chemometrics tool agrees on the canonical
VIP formula and dasp main was simply wrong) or T-26 (where dasp main matched PLS_Toolbox's
default behavior). T-24's evidence is "CCC is broadly correct and increasingly visible in
NIR tutorials, but the user's specific domain doesn't currently use it, and the leading
commercial programs don't report it by default."

The roadmap's stated rationale ("the FTIR Bone PLS paper that anchors the roadmap reports
CCC alongside RMSE/R² as the standard concordance metric in chemometrics") would be the
strongest counter-evidence — but I do not have access to verify that specific paper from
this validation. The main agent should confirm that paper actually reports CCC if the
verdict relies on it.

---

## 6. GUI / distribution-model check (the T-26 lesson)

dasp ships as a bundled Inno Setup desktop app to non-technical users. There is no Python
REPL, so a metric only useful to power users running scripts is dead code for the actual user
base.

### 6a. Where CCC will surface to the user

Path 1 — **results-table column**: `_populate_results_table_inner` at GUI line 28219 sets
`columns = [c for c in results_df.columns if c not in INTERNAL_COLUMNS]`. CCC and CCCcv
are not in `INTERNAL_COLUMNS`, so they will automatically appear as columns when the user
runs a regression search. **The branch DOES surface CCC to the GUI user automatically — no
explicit GUI plumbing required for the column to appear.**

Path 2 — **CSV export**: results dataframe is saved with all columns; CCC and CCCcv will be
in the CSV.

Path 3 — **exported standalone validation scripts**: the branch updated
`templates/validation.py` so generated scripts compute and print CCC alongside RMSE/R²/MAE.
Users who export a script for reproducibility get CCC printed in the script's output.

### 6b. Where CCC is missing GUI affordances

Two pieces of GUI plumbing were planned but not implemented in the branch:

1. **Tooltip dictionary** (`spectral_predict_gui_optimized.py:1437-1448`). Hovering over the
   `CCC` or `CCCcv` column header in the results table will show no tooltip (the dict has no
   entry for these columns). For comparison, hovering over `RMSE` shows "RMSE (Root Mean
   Square Error) — Average prediction error in original units. Lower is better." A user
   seeing an unfamiliar `CCC` column with no tooltip is reasonably going to be confused.

2. **`higher_is_better_cols` set** (line 27964). This set determines the default sort
   direction when the user clicks a column header. `RMSE` is not in the set → ascending sort
   (lowest first, which is best). `R2` and `RPD` are in the set → descending sort (highest
   first, which is best). `CCC` and `CCCcv` are not in the set → they will sort ascending by
   default, putting the *worst* models first when the user clicks the column header. This is
   counter-intuitive and inconsistent with how R²/RPD/RER sort.

Both of these are simple 2-line additions (one entry per metric in each location). The plan
explicitly listed them as Task 7. They were skipped or omitted between plan and merge.

### 6c. Distribution-model verdict shape

The CCC column itself **will** appear automatically in the GUI results table — that's the
first-order distribution-model criterion. The branch satisfies the bundled-app requirement at
the "is this metric reachable to the GUI user?" level.

But the user experience is degraded relative to existing metrics: no tooltip explaining what
CCC means, sort direction backwards. A user encountering CCC for the first time will:

1. See a column called `CCC` with values they don't immediately recognize (range -1 to 1,
   not 0 to 1 like R²).
2. Hover for tooltip — get nothing.
3. Click to sort — sort the wrong way (lowest first).

This is finishable polish — 4 lines of code, 5 minutes of work — but it is a real gap
between what the plan specified and what the branch delivered.

---

## 7. Test results

### 7a. Branch (`pytest tests/test_scoring.py -v -k ccc`)

```
14 passed, 16 deselected in 1.02s
```

All 14 CCC tests pass on the branch. Full `test_scoring.py` run: 30 passed.

### 7b. Main (same test selector)

```
0 selected (16 deselected) in 0.61s
```

Main has zero CCC tests because main has no CCC implementation. There is nothing to fail —
the test class doesn't exist on main. (The investigation main doesn't yet have the file
contents; this is the correct state for an addition-not-a-fix branch.)

### 7c. Discriminating power

For an addition like T-24, the right question is "does the test suite verify the formula is
canonical, not just self-consistent?" The 14 tests cover:

- **Three closed-form numerical checks** that distinguish the canonical formula from common
  alternatives:
  - Identity → 1.0
  - Anti-correlation → -1.0
  - Scale-only `ŷ = 2y`, zero-mean → 0.8 (Lin §2 closed form)
  - Codex-added: `y=[0,1,2], pred=[1,2,3]` → 4/7 (distinguishes ddof=0 vs ddof=1)
- **Distinguishing tests**:
  - `test_bias_only_below_one_even_when_pearson_is_one` — verifies the metric actually
    catches what Pearson r misses
- **Robustness tests**: NaN propagation, length mismatch, range bounds, symmetry
- **Degenerate-case contracts**: constant predictions, constant truth, both equal constants

The Codex-added ddof-disambiguation test is critical — without it, `ddof=0` and `ddof=1`
would be impossible to distinguish from the test suite alone (the 0.8 closed-form for
`ŷ = 2y` is a ratio that cancels the variance scale factor). With it, the suite encodes Lin's
ddof=0 choice as a contract.

The test suite is well-discriminating for an *addition*. It cannot however verify the metric
is "the right metric for chemometrics" — only that it is implemented as Lin (1989) defines
it. That higher-level question is the field-alignment question in Section 5.

---

## 8. Codex review reference (pre-master-rule)

`docs/bugfix_validation/codex_reviews/T24_lins_ccc.txt` (lines 9332–9412). Verdict was
**APPROVE_WITH_CHANGES**. No blocking issues. Five strong suggestions:

1. **Auxiliary result paths mandatory, not conditional.** Codex pointed out that the plan
   left `unified_bayesian.py` and `nsga2_search.py` as "open questions" but those clearly
   construct user-visible regression result rows.
   **Resolution: the branch addressed this.** All four result-construction sites
   (search.py, unified_bayesian.py, nsga2_search.py, templates/validation.py) emit CCC/CCCcv.
   Verified line-by-line in Section 1.

2. **Add ddof-disambiguation test.** The closed-form `ŷ = 2y → CCC = 0.8` cancels the
   variance-scale factor, so it doesn't distinguish ddof=0 from ddof=1.
   **Resolution: the branch addressed this.** `test_ccc_finite_sample_ddof_zero_exact`
   (`y=[0,1,2], pred=[1,2,3] → 4/7`) is in the test suite.

3. **Update exported validation template** if "matches Model Development exactly" is a
   contract.
   **Resolution: the branch addressed this.** `templates/validation.py` includes a copy
   of `_lins_ccc()` and prints `ccc` (CV) and `cal_ccc` (calibration) in the generated
   scripts.

4. **Keep pooled CCCcv, not averaged.** Per-fold concordance averaging is mathematically
   wrong because fold means/variances/biases differ.
   **Resolution: the plan and branch both pool.** `ccc_cv = lins_ccc(all_y_test, all_y_pred)`
   uses the pooled CV vectors at search.py:4362, matching how RMSEcv and R²cv are computed.

5. **Degenerate convention 0.0 vs NaN.** Codex preferred 0.0 for GUI sortability. Recommended
   the docstring label this a display convention, not Lin's definition.
   **Resolution: the branch addressed this.** The scoring.py docstring at line 383–386
   explicitly says: "*This is a DISPLAY CONVENTION for ranking; Lin (1989) defines CCC as
   undefined when variance is zero.*"

Codex nits:

- search.py already imports from scoring, just add `lins_ccc` to the existing line.
  **Resolution: the branch addressed this.** Verified at search.py:70:
  `from .scoring import create_results_dataframe, add_result, compute_specificity, lins_ccc`.

- The `# Cross-validation metrics (test fold averages)` comment at search.py:4742 is wrong
  because the code computes pooled metrics; fix while adding CCC.
  **Resolution: the branch addressed this.** Comment now reads `# Cross-validation metrics
  (pooled across folds)` (verified at search.py:4745).

- GUI sort/tooltip plumbing (lines 1437 + 27964).
  **NOT RESOLVED.** This is the only Codex nit not actioned in the branch (Section 6b).

So 4 of 5 strong suggestions and 2 of 3 nits were actioned. The unresolved item is the GUI
tooltip + sort wiring, which is small but real for the bundled-app distribution model.

---

## 9. Notes / uncertainties for the main agent

Items the verdict-writer should weigh:

1. **The formula is bit-correct and the test discriminates ddof.** This part is clean — Lin
   (1989) §2 is unambiguous, the implementation matches, the test catches the canonical
   counter-examples. Section 4's empirical run-through confirms.

2. **The 4-site coverage is complete.** The branch wires CCC into all four regression result-
   construction paths the plan identified (search.py, unified_bayesian.py, nsga2_search.py,
   templates/validation.py). Codex's main concern about auxiliary sites was addressed.

3. **The GUI plumbing gap is real but small.** Two short additions (tooltip dict + sort
   set). The CCC column will appear automatically in the GUI table because columns are
   discovered from the dataframe schema, but tooltip and sort-direction will be wrong.
   Verdict could be APPROVE_WITH_GUI_FOLLOWUP if the user wants merge now and a 2-line
   follow-up; or APPROVE_AFTER_GUI_FIX if the user wants merge to be polished first.

4. **The field-alignment evidence for the user's specific domain is weak.** I could not find
   bone FTIR / paleoanthropology / isotope papers that report CCC. Domain papers consistently
   report R² / RMSE / Bias. So adding CCC to dasp's regression metrics introduces a column
   that is *correct chemometrics but not currently used in the user's published-paper context*.
   The roadmap claims the anchor FTIR-Bone-PLS paper reports CCC; I cannot verify that paper
   from this investigation. **The main agent should confirm whether the user's actual workflow
   / target journal venues use CCC in reported tables before treating the column as
   user-visible value rather than user-visible noise.**

5. **The leading commercial chemometrics programs likely don't report CCC.** Vendor docs for
   PLS_Toolbox / SIMCA / Unscrambler did not surface CCC in my searches. (Caveat: those docs
   are partially behind license walls, so absence is "weak no.") The user's master rule
   caveat 4 says "If we are doing things that software costing tens of thousands of dollars
   does not do, it should be a red flag." T-24 is in that zone — though "adding an
   informational metric column" is much lower stakes than "inventing a numerical-protection
   pattern" (T-26's red flag) or "diverging from the canonical formula" (T-05's bug).

6. **Adjacent-field adoption is real.** R `yardstick` (tidymodels), R `metrica`, NIRPY
   Research's NIR tutorial all explicitly include CCC. So while chemometrics-specific
   programs lag, the broader applied-stats / agricultural / NIR-tutorial space has adopted
   CCC. Whether dasp positions itself with the conservative chemometrics tradition or with
   the more modern agricultural / general-ML evaluation toolkit is partly a positioning
   decision.

7. **The `ddof=0` choice is locked in by the test.** If anyone later wants to switch to
   `ddof=1` (the Bessel-corrected sample-variance convention used by some implementations),
   the `test_ccc_finite_sample_ddof_zero_exact` test will fail. That's the right behavior
   for a contract, but worth flagging in case someone audits later — `ddof=0` is the
   canonical Lin choice and matches yardstick/metrica/Wikipedia.

8. **CCC convention for degenerate inputs (0.0 / 1.0) is documented as a display convention,
   not Lin's definition.** Lin (1989) leaves CCC undefined for zero-variance inputs. The
   branch's choice to return 0.0 (one-sided degeneracy) and 1.0 (both-equal-constants) is
   pragmatic for GUI sorting. The docstring labels it correctly. No issue.

9. **Schema impact is benign.** The two new columns appear at the end of the calibration and
   CV metric blocks, after RPD/Bias/RER. Old saved CSVs without CCC columns will load fine
   in pandas (missing columns become NaN if the GUI tries to access them). No back-compat
   risk identified.

10. **No regression risk on existing metrics.** None of the existing metric computations
    were modified; CCC is purely additive. The full `tests/test_scoring.py` run shows 30
    passed, including the existing 16 non-CCC tests.

---

## Sources

- Lin, L. I.-K. (1989). *A concordance correlation coefficient to evaluate reproducibility.*
  Biometrics 45(1), 255–268. https://www.jstor.org/stable/2532051
- Lin, L. I. (2000). *A note on the concordance correlation coefficient.* Biometrics 56,
  324–325.
- [NIRPY Research — The Concordance Correlation Coefficient](https://nirpyresearch.com/concordance-correlation-coefficient/)
  (NIR-tutorial advocacy for CCC)
- [yardstick (tidymodels) — `ccc()` reference](https://yardstick.tidymodels.org/reference/ccc.html)
- [yardstick num-ccc.R source](https://github.com/tidymodels/yardstick/blob/main/R/num-ccc.R)
- [metrica R package — available regression metrics](https://cran.r-project.org/web/packages/metrica/vignettes/available_metrics_regression.html)
  (lists CCC as #10 of 48)
- [DescTools::CCC reference](https://search.r-project.org/CRAN/refmans/DescTools/html/CCC.html)
- [epiR::epi.ccc reference](https://rdrr.io/cran/epiR/man/epi.ccc.html)
- [Wikipedia — Concordance correlation coefficient](https://en.wikipedia.org/wiki/Concordance_correlation_coefficient)
- [Real Statistics — Lin's CCC](https://real-statistics.com/reliability/interrater-reliability/lins-concordance-correlation-coefficient/)
- [NCSS / PASS chapter on Lin's CCC](https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Lins_Concordance_Correlation_Coefficient.pdf)
- [keras3 metric_concordance_correlation](https://keras3.posit.co/reference/metric_concordance_correlation.html)
- [mdatools mda.tools docs](https://mda.tools/docs/regression-coefficients.html)
- [Eigenvector PLS_Toolbox docs](https://www.eigenvectordocs.com/index.php?title=PLS_Toolbox_Topics)
- [Sartorius SIMCA 15 user guide PDF](https://www.sartorius.com/download/544940/simca-15-user-guide-en-b-00076-sartorius-data.pdf)
- France, C. A. M. et al. (2014). *Rapid Quantification of Bone Collagen Content by ATR-FTIR
  Spectroscopy.* Radiocarbon 56(2): 513–520.
- Lebon, M. et al. (2018). *A universal curve of apatite crystallinity for the assessment of
  bone integrity and preservation.* Scientific Reports 8: 13367.
- Dal Sasso, G. et al. (2020). *Linking structural and compositional changes in
  archaeological human bone collagen: an FTIR-ATR approach.* Scientific Reports 10: 17188.
