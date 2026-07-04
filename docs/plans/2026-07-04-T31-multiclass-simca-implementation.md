# T-31 — Multi-Class SIMCA / Class-Modeling: TDD Implementation Plan

**Branch:** `feat/T31-multiclass-simca` (off `origin/main`)
**Spec:** `docs/superpowers/specs/2026-07-04-T31-multiclass-class-modeling.md` (review-hardened; Codex 5.5 + DeepSeek V4 Pro on the plan, Codex 5.5 + MiniMax M3 on the spec)
**Status:** not started
**Effort:** ~1–2 weeks (new infrastructure + GUI + export + tests)
**Execution model:** Opus orchestrates; delegated coding subtasks go to GLM-5.2 write-mode workers (`opencode-call`, HALT-OR-BLOCK). Opus drives TDD, reviews every task diff, and commits per task. **Multi-family review at every major checkpoint (mandatory — user directive):** each phase boundary gets a panel of **Codex + ≥2 orthogonal families** (from GLM 5.2 / Kimi / DeepSeek V4 Pro / MiniMax M3 — rotate so a family that wrote code doesn't review its own), plus the full **pr-review-toolkit** at merge-readiness. Single-reviewer gates are not sufficient. Do NOT auto-merge — user greenlight only.

---

## 1. Goal

Add a new **`multiclass_simca`** task ("Multi-Class Class Modeling"): fit one per-class membership model, emit a per-sample × per-class **decision matrix** → "belongs to A / B / both / novel (none-of-the-above)". Flagship engine DD-SIMCA; pluggable per-class engines (OCSVM/IF/LOF/EE) each calibrated to a level-α p-value. Wold + supervised variable selection, preprocessing grid, leaderboard, GUI, code export.

**Out of scope (v1.1):** preprocessing-discovery TPE/exhaustive proxies; parametric tail-fit calibration smoothing; per-target-optimal search.

**Non-negotiable invariants:** never edit `search.py`'s single-Y path; existing regression/classification/one-class paths stay byte-identical; **chemometric-standards-not-ML leakage rule** (spec §5.4 / decision #9); no new deps without `pyproject.toml`; `.venv312` only.

Every task is TDD: write the failing test first, confirm it FAILS on unfixed code (reject tautological tests via revert-and-confirm-FAIL), then implement. Spec section refs and §9 test-matrix items are cited per task.

---

## 2. Key formulas (from spec §5.1, verified against `PCASIMCA`)

```
Per class k, PCA(n_components=nk) on class-k samples (post scaling mode §5.5):
  T2_i   = sum_a t_{i,a}^2 / max(lambda_a, 1e-10)          # diagonal Hotelling T2
  Q_i    = sum_j (x_ij - xhat_ij)^2, xhat = scores@components_ + mean_   # SPE
  chi2 fit (MoM): scipy chi2.fit(v, floc=0, method='mm'); fallback df=2*mean^2/var, scale=var/(2*mean)
  p_T2   = clip(1 - chi2.cdf(T2; t2_df, scale=t2_scale), 1e-300, 1)
  p_Q    = clip(1 - chi2.cdf(Q;  q_df,  scale=q_scale ), 1e-300, 1)
  fisher = -2*(ln p_T2 + ln p_Q)
  accept iff fisher <= chi2(4).ppf(1-alpha)      # == margin (thr - fisher) >= 0
  p_joint = chi2(4).sf(fisher)                    # the value used in the P matrix + ranking

Non-SIMCA engine (OCSVM/IF/LOF/EE), per class:
  score s normalized "higher = more normal" (pin per engine, §5.3; IsolationForest score_samples AS-IS — do NOT negate)
  null = cross-fit engine scores on train-only inliers (m >= 10)
  p    = (1 + #{null <= s}) / (m + 1);  accept iff p >= alpha
```

alpha is GLOBAL. Only n_components is tuned per class (filter the existing PCA-SIMCA grid to n_components; inject orchestrator alpha).

---

## Phase A — engine + CV/novelty harness + persistence (`src/spectral_predict/simca.py`, `model_io.py`)

Two co-equal load-bearing risks: the CV/novelty harness AND persistence.

### A1 — `PCASIMCA.p_joint` accessor + numeric fixtures + docstring fix
- **Test:** (a) `p_joint = chi2(4).sf(fisher_stat)` in [0,1], accept iff `p_joint >= alpha` matches existing `predict`; (b) **numeric reference fixtures** pinning T²/Q/`p_joint`/decision on a fixed small matrix (compute once, freeze); (c) **Fisher-vs-Bonferroni calibration test** (§9.6) — on synthetic data the Fisher-combined accept rate stays within a stated tolerance of the per-axis-Bonferroni (α/2 each) accept rate. (§9.13, §9.6)
- **Impl:** add `PCASIMCA.p_joint(X)` returning `stats.chi2.sf(fisher_stat, 4)` (reuse `decision_function` internals). Fix the `_fit_chi2` docstring ("MLE" → "method-of-moments `method='mm'`", `contamination.py:184-188`).
- **Done:** accessor + numeric fixtures + Fisher/Bonferroni test green; docstring accurate. No behavior change to `decision_function`.

### A2 — `MultiClassClassModel` core (SIMCA engine only)
- **Test:** fit on synthetic K=3 with a **fixed** `n_components` (int or per-class dict — `"per_class_cv"` is not wired until A5); `decision_matrix(X)` returns `(P (n,K) in [0,1], A (n,K) bool)`; `predict` yields `single/multiple/novel` with the "≥2 of K" rule; novel row → all-reject. **`n_k < 10` → class marked unmodelable, its column preserved (not dropped) and flagged** (§8). **False-rejection reliability (§9.5):** K=3 sizes {5, 30, 100} → empirical false-rejection at α=0.05 ∈ [0.02, 0.10] per modeled class; the size-5 class is flagged unmodelable. (§9.2, §9.5)
- **Impl:** `MultiClassClassModel(engine="pca-simca", alpha=0.05, n_components=<int|dict>, scaling="per_class", min_class_samples=10, engine_params=None)`. `fit(X,y)` stores `classes_` (ordered), enforces `min_class_samples` (mark unmodelable, keep column), fits one engine per modeled class on that class's samples, builds `P` from `p_joint`; `A = P >= alpha`. `predict` → argmax `p_joint` over accepted. (`n_components="per_class_cv"` becomes the public default in A5, not here.)
- **Done:** core API + unmodelable handling + false-rejection test green; decision rule per spec §5.2.

### A3 — scaling modes (`per_class` / `global` / `none`)
- **Test:** `scaling="none"` single-class model == bare `PCASIMCA` accept/reject bit-for-bit (functional-equivalence anchor); `per_class` fits a scaler per class; `global` one scaler; all fit train-only. (§9.3)
- **Impl:** `scaling` param; StandardScaler fit per §5.5; store fitted scaler(s) on the orchestrator.
- **Done:** three modes green; equivalence test passes only with `scaling="none"`.

### A4 — non-SIMCA per-class p-value calibration
- **Test:** for each of OCSVM/IF/LOF/EE, held-out-inlier accept rate ∈ [1−α−0.10, 1−α+0.10] at n ∈ {10,30,100}; assert IsolationForest direction not inverted. (§9.7)
- **Impl:** calibration object per §5.3 — per-engine normalized score ("higher=more normal", pin the exact call; **IF `score_samples` AS-IS, NOT negated** — GPT-5.5 caught the inversion bug), cross-fit null on train-only inliers, `p=(1+#{null≤s})/(m+1)`, min m=10 else class unmodelable.
- **Done:** all four engines produce valid level-α p-values feeding the same `P` matrix.

### A5 — two-stage nested leakage-safe CV
- **Test:** A.1 tuning samples are disjoint from the A.2 evaluation samples they feed (assert via instrumentation); per-class n_components selected under global α (α never varies); reuses `run_one_class_cv` one-vs-rest. (§9.4)
- **Impl:** for each A.2 outer-train split, run A.1 (`run_one_class_cv`, `compute_calibration=False`, class k=+1 rest=−1, select balanced_accuracy) to pick each class's n_components; then fit K models + scaling + calibration + varsel on that split only; score held-out pool from all classes vs all K; pool. **Filter the existing PCA-SIMCA grid to n_components only and inject the orchestrator α** (global α; α never tuned). **Wire `n_components="per_class_cv"` as the public constructor default here** (A2 used fixed values).
- **Done:** nested CV green; leakage instrumentation test passes; `"per_class_cv"` default resolves through A.1.

### A6 — LOCO / external novelty evaluation mode
- **Test:** LOCO refits all K−1 models with the held-out class excluded; no-class rate = fraction of held-out-class samples accepted by 0 of the K−1 remaining. Synthetic held-out 4th class → ≥80% novelty (non-overlap). **Baseline contract (GPT-5.5):** the PLS-DA baseline forces the 4th-class samples into a **specific trained class** — assert the forced class identity AND its reported probability/confidence, not just "forces a class." (§9.1, §9.8)
- **Impl:** `evaluate_novelty(X, y, mode="loco"|"external", external=None)` per spec §5.4.
- **Done:** novelty mode green; acceptance-target test green.

### A7 — dedicated multilabel metrics
- **Test:** each metric matches a hand-computed value on a fixed decision matrix (explicit denominators per spec §7): per-class acceptance-sensitivity, rejection-specificity, no-class rate, ambiguity rate, exact-set rate, efficiency, pairwise confusion; Wilson CI bounds; α-sweep tradeoff-curve shape. Empty/small-class denominators defined. (§7)
- **Impl:** new `multiclass_simca_metrics(...)` in `simca.py` (NOT `one_class_metrics` — inverted; NOT `compute_imbalance_metrics` — single-label). α-sweep AUC of `(novelty_rate, 1−false_rejection_rate)` for ranking.
- **Done:** metrics green with pinned reference values.

### A8 — persistence (`.dasp` multi-class format + predict schema + compat gate)
- **Test:** fit → `save_model` → `load_model` → `predict` reproduces `p_values`/`decision_matrix`/`summary_label`/`accepted_classes` exactly. **Compat gate (GPT-5.5 reword — can't make an *old* build raise from new code):** the **current** build raises `NotImplementedError` when `load_model`/`predict_with_model` encounters an **unsupported/unknown `task_type`** (guards forward-compat, and stops the current silent regression-path fall-through). (§9.11)
- **Impl:** pickle the whole `MultiClassClassModel` as `model.pkl` (owns engines/calibrators/scaler(s)/varsel-mask/α/registry/column-order); mirror scaler to `autoscaler.pkl`; `metadata.json` += `task_type`, `class_names`, `engine_family`, `alpha`, `varsel_path`, `scaling`. Extend `predict_with_model`/`predict_with_uncertainty` to return the §6 schema shapes. Add the `task_type` validator gate.
- **Done:** round-trip + compat-gate tests green.

**Phase A gate (MAJOR checkpoint — multi-family):** Codex + ≥2 orthogonal families (GLM/Kimi/DeepSeek/MiniMax) on `simca.py` + `model_io.py` — the load-bearing math/CV/calibration/persistence. Fold findings with discriminating tests (revert-and-confirm-FAIL). GLM 5.2 (if it wrote the code) does not review its own diff.

---

## Phase B — variable selection (`simca.py` / `variable_selection.py`)

### B1 — Wold modeling + discriminating power
- **Test:** `MPOW_j = 1 − s_resid/s_total` matches a hand value; DPOW macro-averaged one-vs-rest is stable across resamples (Spearman ρ ≥ 0.8); "balanced" (max `MPOW·DPOW`) retains ≥2/3 known-relevant variables. (§9.10)
- **Impl:** per spec §5.6 — modeling power per class from PCA residuals; discriminating power via pairwise cross-fit residual ratios, K>2 = macro-avg OVR; selection modes modeling/discriminating/balanced.
- **Done:** Wold varsel green; classical-Wold-1976 provenance documented.

### B2 — Wold diagnostic plot data
- **Test:** plot-data function returns per-variable MPOW/DPOW arrays of correct shape/order for K classes.
- **Impl:** data provider (GUI renders in D2).
- **Done:** plot-data green.

### B3 — supervised prefilter wiring (tagged + novelty-gated)
- **Test:** `varsel_path=supervised` runs `importance/spa/cars/cars-tree/ga/vcpa-iriv` on the multi-class label; **external-novel-class guard** shows no degradation of novelty rate vs full-spectra baseline. (§9.9)
- **Impl:** reuse the `implemented_oc_varsel` methods as a global prefilter; tag `varsel_path` on results.
- **Done:** both varsel paths green; novelty guard passes.

**Phase B gate (multi-family):** Codex + ≥2 orthogonal families on the varsel diff (Wold math + supervised-novelty guard).

---

## Phase C — search + task-type wiring (`search.py`, `scoring.py`, `unified_bayesian.py`, registries)

### C1 — task-type plumbing + fall-through audit
- **Test:** `multiclass_simca` never falls through to the classification/regression path at ANY branch site; a parametric audit test enumerates the branch sites. (§9.12)
- **Impl:** thread `multiclass_simca` through `scoring.create_results_dataframe` (:549), `compute_composite_score` (:14), `unified_bayesian.convert_study_to_dataframe` (~:3019), and the hardcoded branches in `model_registry.py:74-103`, `model_config.py:158-181`, `cv_utils.py:275-340`, `code_generator.py`, `templates/validation.py`.
- **Done:** plumbing + audit green.

### C2 — `run_multiclass_simca_search`
- **Test:** preprocessing × engine grid runs; per-class n_components auto-tuned inside each row (no G^K blowup); rows keyed `(preprocessing, engine, varsel_path)`; ranked by α-sweep AUC, NaN-safe; unmodelable class → row flagged, never silently dropped. (§8 edge cases)
- **Impl:** new function patterned on `run_one_class_search` (:5594); grid-engine-only guard; reuses Phase A CV.
- **Done:** search green on synthetic multi-class.

### C3 — result columns + composite score
- **Test:** result dataframe has the multiclass column set **including an explicit `engine_family` column per row** (spec decision #3) and `varsel_path`; `compute_composite_score` ranks by the §7 metric.
- **Impl:** `create_results_dataframe('multiclass_simca')` column set (with `engine_family`, `varsel_path`); scoring branch. C2 populates `engine_family` on every emitted row.
- **Done:** green; `engine_family` present and correct on all rows.

**Phase C gate (MAJOR checkpoint — multi-family):** Codex + ≥2 orthogonal families on the search + task-type-plumbing diff (dispatcher/fall-through risk); **end-to-end smoke** — real multi-class spectral set through `run_multiclass_simca_search` + a genuinely held-out novel class; save→load→predict round-trip reproduces the decision matrix at max|diff|=0.

---

## Phase D — GUI + code export (`spectral_predict_gui_optimized.py`, `code_generator.py`)

### D1 — 5th task radio + controls
- **Test (GUI):** selecting "Multi-Class Class Modeling" shows the engine picker + α + n_components + varsel-path controls; hides one-class/classification-only widgets; disambiguated from the existing one-class PCA-SIMCA entry point.
- **Impl:** add radio value `multiclass_simca` at the task-radio block (~:6462-6465 on this branch); extend `_on_task_type_changed` (~:16773 on this branch — NOT ~18398, which is data-loading); new control panel. *(GUI line numbers differ on this main-based branch from the earlier T-17-branch scout — re-grep each symbol at edit time.)*
- **Done:** GUI toggle green.

### D2 — results view (decision matrix + Wold plots)
- **Test (GUI):** results render the per-sample decision matrix + single/multiple/novel labels + Wold MPOW/DPOW diagnostic plots; CSV export includes the decision matrix.
- **Impl:** dispatch patterned on the one-class detection branch (~:29535 on this branch; :29497 is the empty-results guard — re-grep at edit time) on a worker thread (queue + `root.after` poller); results Treeview + plots.
- **Done:** results view green.

### D3 — code export
- **Test:** exported script + notebook reproduce the in-app decision matrix bit-identically.
- **Impl:** `code_generator` multiclass templates gated on `task_type=="multiclass_simca"`.
- **Done:** export round-trip green.

**Phase D gate:** GUI smoke (`import spectral_predict_gui_optimized`, tab renders, run completes on synthetic data).

---

## 3. Merge-readiness

- **Full multi-family whole-diff pass** — Codex + ≥2 orthogonal families (GLM/Kimi/DeepSeek/MiniMax) on the cumulative diff, THEN the full pr-review-toolkit (code-reviewer, silent-failure-hunter, pr-test-analyzer, type-design-analyzer). The two layers are orthogonal — the toolkit has caught HIGHs the cross-family passes missed and vice versa.
- **Merge gate:** local diff-failure-set vs `origin/main` (main red on cloud CI since 2025-10-27) — PR adds **zero new failures**. Run the full suite ex-GUI on HEAD and on `origin/main`, diff the failure sets.
- Commit per task with explicit `git add` paths (never `git add -A` — many untracked scratch files); CRLF `git diff --stat` check after GUI edits. Commit before any control-yielding op. Do NOT auto-merge — await user greenlight.

## 4. Open confirmations (carry into build)
- Confirm the DD-SIMCA (Pomerantsev & Rodionova) / class-modeling (Oliveri & Downey 2012) detail against the primary sources (§5.1).
- The ≥80% / ≥60% acceptance targets (§1) are placeholders until calibrated against a real multi-class dataset.
- Global-vs-per-class autoscaling default is `per_class` (§5.5); revisit if real-data behavior argues otherwise.
