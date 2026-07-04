# Design Spec — T-31 Multi-Class SIMCA / Class-Modeling

**Status:** REVIEW-HARDENED draft — findings from two cross-family rounds folded in; ready for a TDD implementation plan (`docs/plans/`).
**Ticket:** T-31 (confirmed KEEP by user 2026-05-02; memory `project_t31_simca_confirmed.md`).
**Repo:** `C:\Users\mspon\git\dasp`, Python 3.12 (`.venv312`).
**Planning artifact:** `~/.claude/plans/i-belive-one-of-curried-ripple.md` (approved; all design decisions below are locked with the user unless marked OPEN).
**Review trail:** plan reviewed by Fable (low), Codex 5.5 (high), DeepSeek V4 Pro (max); THIS spec reviewed by Codex 5.5 (high) + MiniMax M3 (max) — all anchors independently verified correct across reviewers; formula §5.1 verified against `PCASIMCA` line-by-line.

---

## 1. Problem / motivation

dasp's classifiers (PLS-DA, RF, XGBoost, …) are **discriminant** — every specimen is forced into one of the trained classes via argmax. For the user's science this silently misleads:

- **Fossil-bone sites don't generalize** — a sample from an unexcavated site D gets forced into A/B/C.
- **Unknown consolidants** (preservation chemicals) produce spectra no trained class has seen, yet a discriminant model still reports a confident class.

Both are textbook **class-modeling** cases: a specimen can legitimately belong to one class, several, or **none of the above**. That capability does not exist in dasp today. The one-class path (`PCASIMCA` etc.) models a *single* class (inlier-vs-everything); the classification path forces exactly one label. T-31 fills the missing middle: **K independent per-class membership models + a decision rule producing a per-sample × per-class matrix** → "A", "B", "both", or "novel".

**Acceptance target** (`docs/RECONCILED_ROADMAP_2026-04-29.md:325-349`; concrete numbers set here per MiniMax): train on 3 known classes, evaluate on a held-out 4th class. On a **non-overlapping** 4th-class distribution the SIMCA path flags **≥80%** of its samples as "no class"; on a **partially overlapping** 4th class, **≥60%**. In the same test a PLS-DA baseline forces essentially all 4th-class samples into one of the 3 trained classes (assert the specific forced class + its reported confidence).

---

## 2. Approved design (locked decisions)

1. **Spec-first process:** this spec → cross-family review → TDD plan (`docs/plans/`) → phased build with per-phase Codex + cross-family review, merge-gate vs `origin/main`. No auto-merge.
2. **New task type** `multiclass_simca` ("Multi-Class Class Modeling") — a 5th radio alongside Regression / Classification / One-Class. It gets its own metrics, results columns, and decision-matrix display, because classification metrics/UI assume exactly one label per sample and cannot express none/multiple.
3. **General class-modeling, not SIMCA-only.** Flagship engine = DD-SIMCA (`PCASIMCA`); pluggable per-class engines reuse the existing one-class roster — **EllipticEnvelope ≈ UNEQ**, **OneClassSVM ≈ SVDD**, IsolationForest, LOF. **All engines ship in v1, each with per-class p-value calibration** (§5.3) so accept/reject is a real level-α test for every engine, not a raw nu/contamination heuristic. Tag `engine_family` per result row.
4. **Search parity on the meaningful axes:** preprocessing grid (SNV / SG-derivatives / baseline / autoscale) + per-class n_components tuning + full leaderboard / CSV / code export. Preprocessing-discovery TPE/exhaustive proxies deferred to v1.1.
5. **Variable selection — BOTH paths in v1**, tagged `varsel_path`:
   - **Supervised prefilter** (`importance`/`spa`/`cars`/`cars-tree`/`ga`) on the genuine multi-class label — parity with the classification path.
   - **Wold SIMCA-native** modeling power + discriminating power (+ diagnostic plots).
   - Supervised selects for *discrimination*; Wold for *within-class model quality*. They can disagree; supervised must pass an external-novel-class regression test proving it does not degrade novelty detection (§9).
6. **α is GLOBAL**; only **n_components** is tuned per class. Per-class α would give classes different χ²(4) quantiles, making "≥2 accept = ambiguous" incoherent (§5.1). **Implementation note (Codex):** the existing `get_one_class_model_grids()` PCA-SIMCA grid varies α ∈ {0.05, 0.01} — A.1 must **filter that grid to n_components only** and inject the orchestrator-level α, or global-α is silently violated.
7. **Chemometric standards, not ML standards, on leakage** (§5.4) — per-spectrum SNV/SG/baseline are *not* leakage; column-autoscale, per-class calibration, and varsel fitting are train-only inside folds; HP tuning is separated from evaluation folds.

**OPEN (this spec must resolve — see §5.5):** global vs per-class autoscaling. DeepSeek leans per-class (textbook SIMCA); Codex leans global (one interpretable mask per leaderboard row). Recommendation in §5.5.

---

## 3. Existing code to reuse (all anchors verified)

- **Per-class engine primitive:** `PCASIMCA` (`contamination.py:62-301`) — DD-SIMCA with diagonal T², Q/SPE, method-of-moments χ² fits, Fisher-combined acceptance. `decision_function` (238-274) returns `joint_threshold_ − fisher_stat` (≥0 = in-class). Factories `build_one_class_model` (361-390), `get_one_class_model_grids` (393-446).
- **Per-class tuning sub-step:** `run_one_class_cv` (`contamination.py:532-880`) IS reusable one-vs-rest (class k = +1, rest = −1, `compute_calibration=False`, select on balanced_accuracy) — it is NOT reusable as the joint evaluation loop (it collapses all non-target classes into one outlier blob).
- **Supervised varsel:** the methods in `implemented_oc_varsel` (`search.py:6054`).
- **Task-type plumbing (broader than first mapped — Codex):** `scoring.py` (`create_results_dataframe` :549, `compute_composite_score` :14), `unified_bayesian.py` (`convert_study_to_dataframe` ~:3019), `search.py` (`run_one_class_search` :5594 as template), **plus** `model_registry.py:74-103`, `model_config.py:158-181`, `cv_utils.py:275-340`, `code_generator.py`, `templates/validation.py`.
- **Persistence:** `.dasp` = zipped `metadata.json` + `model.pkl` (+ `preprocessor.pkl`, `label_encoder.pkl`, one-class `scaler.pkl`/`pca_reducer.pkl`) — `model_io.py:200-260`, predict paths ~512-573 / 797-816, `predict_with_uncertainty` 851-959.
- **Multi-class metrics reference** (>2 classes, single-label only): `scoring.compute_imbalance_metrics` :635-765.

---

## 4. Module API — `src/spectral_predict/simca.py`

```
class MultiClassClassModel(BaseEstimator, ClassifierMixin):
    def __init__(self, engine="pca-simca", alpha=0.05,
                 n_components="per_class_cv", scaling="OPEN §5.5",
                 engine_params=None): ...

    def fit(self, X, y):                 # y = class labels (K classes)
        # per class k: select train-only inliers (y == k),
        #   fit engine on that class, calibrate per-class p-value (§5.3)
        # stores: self.classes_ (ordered), self.models_[k], self.calibrators_[k],
        #   self.scaling_, self.varsel_mask_, self.alpha_
    def decision_matrix(self, X) -> (P, A):
        # P: (n, K) per-class p-values; A: (n, K) bool accept (p >= alpha)
    def predict(self, X) -> labels + summary:
        # per row: n_accept = A[i].sum(); 0→"novel", 1→that class, >=2→"multiple"
        # ranked single-best (optional hard label) = argmax_k P[i,k] over accepted
```

The orchestrator **owns all per-class artifacts** (engines, calibrators, scaling, varsel mask, α, class registry + column order) so it pickles as a single `model.pkl` (§6).

---

## 5. Method detail

### 5.1 DD-SIMCA per-class engine (exact, from `PCASIMCA`)

Per class *k*, fit `PCA(n_components=nk)` on that class's samples. For sample *i*:

- **Hotelling T²** (diagonal, exact for orthogonal PCA scores): `T²_i = Σ_a t_{i,a}² / λ_a`, with `λ_a = explained_variance_` floored at 1e-10.
- **Q / SPE:** `Q_i = Σ_j (x_ij − x̂_ij)²`, `x̂ = scores @ components_ + mean_`.
- **Data-driven χ² fits** to training T² and Q: **scipy `stats.chi2.fit(values, floc=0, method='mm')` (method-of-moments) first**, then a **manual MoM fallback** `df = 2·mean²/var`, `scale = var/(2·mean)` if scipy raises (degenerate-Q `zero_guard` → df=2, scale=1e-10). *(The current `PCASIMCA._fit_chi2` docstring wrongly says "MLE first" — fix that stale docstring as a small cleanup in the plan; the code is MoM.)*
- **p-values:** `p_T2 = 1 − F_χ²(T²; t2_df, scale=t2_scale)`, `p_Q` likewise, clipped `[1e-300, 1]`.
- **Fisher combination:** `fisher_stat = −2(ln p_T2 + ln p_Q)`. Accept iff `fisher_stat ≤ χ²(4).ppf(1−α)` — equivalently the code's decision margin `joint_threshold_ − fisher_stat ≥ 0`.
- **Joint p-value for the decision matrix (Codex):** `p_joint = χ²(4).sf(fisher_stat)`. The `P` matrix (§4) is built from `p_joint` (SIMCA) — a real p-value — not the raw margin.

**Citations (MiniMax correction):** the DD-SIMCA T²/Q + acceptance math is **Pomerantsev & Rodionova**; **Oliveri & Downey 2012** is the *multi-class class-modeling decision framing* (belongs-to-none/one/several). Cite both correctly; confirm against the primary sources before finalizing.

**Independence approximation — quantify, don't hand-wave (both reviewers).** Fisher assumes T²⊥Q; they are correlated (underfit PCA inflates Q, depresses T²), so the joint tail is fatter than χ²(4). Pomerantsev's own formulation uses a T²∩Q acceptance region with per-axis Bonferroni α/2, not the Fisher shortcut. **Phase-A calibration test:** on synthetic data compare Fisher-combined accept rate to per-axis-Bonferroni accept rate and pin the gap below a stated tolerance. Plus numeric reference tests pinning T²/Q/p_joint/decision on a fixed small matrix.

**Minimum per-class n for a reliable MoM χ² fit (MiniMax — HIGH).** The `var/(2·mean)` MoM is brittle at small n (a class of 5 fits far worse than a class of 200), so the *same* nominal α gives *different* empirical false-rejection rates per class — which silently undermines cross-class comparability. Require **n_k ≥ 10** for a class to be modeled by the MoM path (the `n≥3` PCA floor is too low here); classes below threshold are marked unmodelable (§8). Phase-A test: K=3 of sizes {5, 30, 100} → empirical false-rejection at α=0.05 must lie in [0.02, 0.10] per modeled class; the size-5 class is flagged unmodelable.

**Global α** (decision #6): every per-class SIMCA shares `χ²(4).ppf(1−α)`, giving a **common decision level** across classes. Cross-class *ranking* uses the calibrated `p_joint`, not raw margins (raw margins reflect class-specific fitted nulls and the independence approximation, so they are not literally comparable).

### 5.2 Decision rule (Oliveri & Downey 2012)

Independent per-class membership at global level α. Per sample: accepted-by-0 → **novel**; by-1 → assigned to that class; by-**≥2 of K** → **multiple/ambiguous** (state this "≥2 of K" definition in the API docstring — for K=2 the only multiple case is "both"). When a single hard label is needed, **argmax `p_joint`/calibrated p-value over accepted classes** (valid because §5.1's `p_joint` and §5.3's calibrated p-values are on a common [0,1] scale).

### 5.3 Per-class p-value calibration for non-SIMCA engines

OCSVM/IF/LOF/EE `decision_function`/`score_samples` are uncalibrated (nu/contamination boundaries). Define a **calibration object** per class:

- **Score convention — normalize to "higher = more normal" per engine (must pin or a sign inverts):** `OneClassSVM.decision_function` (≥0 inlier, use as-is), `EllipticEnvelope.decision_function` / `LocalOutlierFactor(novelty=True).decision_function` (<0 outlier, use as-is), **`IsolationForest`: use `score_samples` AS-IS — do NOT negate.** GPT-5.5-verified via smoke test: sklearn `IsolationForest.score_samples` already gives higher=more normal (normal ≈ −0.44 > outlier ≈ −0.71), and `decision_function` (positive=inlier) is equivalent as-is. Negating would invert the anomalous tail. Pin the exact call per engine; a phase-A test asserts a far outlier gets the *lower* normalized score and *lower* p-value.
- **Null distribution:** collect the engine's normalized scores on **train-only inliers**, cross-fit (out-of-fold, after HP selection) so a sample never scores against a model trained on it.
- **p-value:** for test score *s*, `p = (1 + #{null ≤ s}) / (m + 1)` (add-one smoothing; *s* in "higher = more normal" convention so the lower tail is anomalous). **Accept iff p ≥ α.**
- **Minimum calibration size** `m` (tie to the §5.1 n_k ≥ 10 rule); below it, mark the class engine unmodelable rather than emit a coarse p-value (K-fold on n=12 gives resolution 0.083 — too coarse near α=0.05). Note a parametric tail fit (half-Gaussian/GEV/Weibull) as a v1.1 smoothing hook.

**Phase-A test:** held-out-inlier accept rate ∈ [1−α−0.10, 1−α+0.10] for each engine at n ∈ {10, 30, 100}. SIMCA yields a calibrated `p_joint` natively (no extra calibration).

### 5.4 CV design — two stages, leakage-safe (decisions #6/#7)

- **A.1 per-class HP tuning — nested inside A.2 (Codex — else it leaks):** for each A.2 outer-train split, tune each class's `n_components` using **only that split** (reuse `run_one_class_cv` one-vs-rest, `compute_calibration=False`, select on balanced_accuracy). A single global A.1 before the outer loop would leak held-out samples into model selection. Global α is fixed, never tuned. **Caveat (MiniMax):** one-vs-rest tunes n_components against the *pooled* non-class distribution, not the specific competing classes — fine for novelty, mildly suboptimal for within-K confusion; the K-wise trade-off is only *evaluated* in A.2, not optimized. State this explicitly.
- **A.2 outer evaluation:** stratified K-fold. Per fold: fit K per-class models with the (fold-tuned) params; fit scaling + per-class calibration + varsel **on fold-train only**; score the held-out pool from **all** classes against **all K** models; pool decisions.
- **Novelty evaluation (the acceptance target) — LOCO refits ALL K (MiniMax):** hold an entire class out; **refit all K per-class models with that class excluded**; the no-class rate is the fraction of held-out samples accepted by **0 of the remaining K−1** models. (Omitting only the held-out class's own model but still scoring against the other K−1 would let one of them "accept" and contaminate the rate.)
- **Leakage rule (decision #7 / user):** per-spectrum SNV/SG/baseline are applied outside folds per chemometrics convention (not leakage); **column-autoscale, calibration, and varsel are fit train-only inside folds**; A.1 tuning uses only the outer-train split of the estimate it feeds.

### 5.5 RESOLVED — three explicit scaling modes (was OPEN; both reviewers concur)

Textbook SIMCA column-autoscales **within each class**; current `PCASIMCA.fit` applies **no** column scaling (StandardScaler is wired only for OCSVM/EE/LOF, `contamination.py:663-665`). Both reviewers endorse per-class default; Codex flagged a §5.4↔§11 contradiction that this resolves. **A `scaling` parameter with three modes, each fit train-only inside folds:**

- **`per_class`** (SIMCA default) — a StandardScaler fit on **each class's inliers only**; textbook-correct, makes modeling power interpretable.
- **`global`** — one StandardScaler fit on **fold-train across all classes**; use for cross-engine leaderboard comparability.
- **`none`** — no column scaling; reproduces bare `PCASIMCA` exactly (the functional-equivalence anchor, §8).

The fitted scaler(s) are persisted (`autoscaler.pkl`, §6) so predict-time transforms match. **No contradiction with §11:** the "full-X autoscale before CV is a chemometrics convention" clause is a *legacy discriminant-path* convention; for class-modeling, scaling is a defined, fold-internal, mode-controlled step (§11 updated accordingly).

### 5.6 Wold variable selection (decision #5)

Per class, from the PCA residuals:
- **Modeling power** `MPOW_j = 1 − s_j,resid / s_j,total` (classical **Wold 1976** — state explicitly this is distinct from the DD-SIMCA per-variable T²/Q framework).
- **Discriminating power** — pairwise cross-fit residual ratios (residual of class-i samples on class-j's model vs their own). **K>2 aggregation (pinned now, per both reviewers): macro-average the one-vs-rest per-variable discriminating power** (each variable's DPOW = mean over the K−1 pairwise ratios for its class), with documented handling of asymmetric pairwise ratios. **"Balanced" mode** selects variables maximizing `MPOW · DPOW` (or a stated weighted combination).
- Selection modes: modeling / discriminating / balanced, + diagnostic plots. **Phase-B test:** DPOW ranking is stable across resamples (Spearman ρ ≥ 0.8 over 20 resamples on K=3 synthetic data) and "balanced" retains ≥2/3 of the known-relevant variables.
- The reused **supervised prefilter** path is tagged `varsel_path=supervised` and gated by §9's novelty test.

---

## 6. Persistence format (co-equal Phase-A risk)

No `.dasp` precedent exists for a dict/list of models. **Format:** the `MultiClassClassModel` orchestrator is pickled whole as `model.pkl` (it owns per-class engines, calibrators, the fitted scaler(s) — also mirrored to `autoscaler.pkl` for clarity, varsel mask, α, class registry + column order) — avoids JSON-serializing sklearn objects. `metadata.json` gains `task_type="multiclass_simca"`, `class_names`, ordered `engine_family`, `alpha`, `varsel_path`, `scaling`.

**Exact prediction schema (Codex):** extend `predict_with_model` / `predict_with_uncertainty` for `multiclass_simca` to return, with shapes: `p_values (n,K)`, `decision_matrix (n,K) bool`, `summary_label (n,)` ∈ {single-class-name, "multiple", "novel"}, `accepted_classes (n,)` list-of-lists, and optional `best_class (n,)`. Round-trip test asserts **all** of these reproduce the in-app values.

**Backward-compat gate (MiniMax):** every current downstream site branches `if task_type == "one_class"/"classification"` and otherwise falls through to the **regression** path — so an *older* build loading a `multiclass_simca` `.dasp` would emit regression nonsense, not an error. Add an explicit `task_type` validator in `load_model` / `predict_with_model` / `predict_with_uncertainty` that raises `NotImplementedError("this build does not support multiclass_simca; upgrade to ≥X.Y.Z")`. Pin a test.

**Round-trip test** (fit → save → load → predict reproduces the in-app decision matrix) is a Phase-A deliverable, not downstream.

---

## 7. Search + GUI + export

- **Search:** new `run_multiclass_simca_search` (patterned on `run_one_class_search`) — preprocessing grid × engine, per-class n_components auto-tuned inside each row (never a G^K product), NaN-safe ranking, leaderboard rows keyed `(preprocessing, engine, varsel_path)`. **Ranking metric (MiniMax — pin it):** α-sweep **AUC of `(novelty_rate, 1 − false_rejection_rate)`** (honest across α, small-class-robust), tie-break by min per-class n. Not aggregate novelty at a fixed α (cheats on small classes).
- **Task-type plumbing:** thread `multiclass_simca` through every branch site in §3; **add a task-type audit checklist + tests that it never falls through to the classification path.**
- **Metrics — dedicated, NOT reused** (`one_class_metrics` inverts sensitivity/specificity; `compute_imbalance_metrics` is single-label). **Define each with explicit denominators (both reviewers) before the TDD plan:** per-class acceptance-sensitivity = accepted / true-class members; per-class rejection-specificity = rejected / non-class samples; no-class (novelty) rate = novel / held-out-novel; multi-class (ambiguity) rate = (≥2 accepts) / N; exact-set rate = correct-accept-set / N; efficiency = geomean(sens·spec); pairwise-class confusion (behavior under multi-accept samples stated). Report the **novelty-vs-in-class-sensitivity tradeoff curve across α** (a single α is gameable; sample-level false-rejection is indistinguishable from novelty). **Empty/small-class denominators:** define the value when there is no external novel class and when a class has too few samples; report **Wilson-interval CIs** and enforce the §5.1 min per-class n.
- **GUI:** 5th task radio; engine picker + α/n_components; results view = decision matrix + single/multiple/novel labels + Wold diagnostic plots. Disambiguate from the existing one-class PCA-SIMCA entry point (labels/help).
- **Code export:** script + notebook templates reproducing the in-app decision matrix.

---

## 8. Edge cases

- **Unmodelable / too-small class:** numerical threshold is **n_k ≥ 10** for the MoM χ² fit / calibration (§5.1, §5.3), not just the `PCASIMCA` n≥3 PCA floor. Below it, mark the class unmodelable / lower its fold count / mark the row invalid with a stated reason — **never silently drop a decision-matrix column** (both reviewers). Define the degenerate-Q `zero_guard` meaning in the multi-class context.
- **Functional-equivalence guard (Codex/MiniMax — tighten the clause):** a single-class `MultiClassClassModel(engine="pca-simca", scaling="none", same α + n_components)` produces **identical accept/reject scores** to bare `PCASIMCA` (not bit-identity; `scaling="none"` is required because bare `PCASIMCA` does no column scaling). Add separate tests for `scaling="per_class"`.

---

## 9. Test matrix

1. Synthetic 3-class + **held-out 4th class**: SIMCA path (LOCO/external mode) labels ≥80% of a non-overlapping 4th class (≥60% partial-overlap) "no class"; PLS-DA baseline forces a trained class.
2. Decision-rule edge cases (0 / 1 / ≥2 acceptances); global-α coherence.
3. Per-class-engine **functional-equivalence** (`scaling="none"`, matching α/n_components) to bare `PCASIMCA`; plus a separate `scaling="per_class"` test.
4. **Leakage:** autoscale, calibration, varsel fit train-only inside folds; A.1 tuning nested per outer-train split, disjoint from A.2 evaluation samples.
5. **Per-class MoM reliability:** K=3 sizes {5, 30, 100} → empirical false-rejection at α=0.05 ∈ [0.02, 0.10] per modeled class; size-5 flagged unmodelable (n<10).
6. **Fisher approximation:** Fisher-combined accept rate vs per-axis-Bonferroni accept rate within a stated tolerance on synthetic data.
7. Non-SIMCA calibration: held-out-inlier accept rate ∈ [1−α−0.10, 1−α+0.10] per engine at n ∈ {10, 30, 100}; IsolationForest direction not inverted.
8. **LOCO refits all K−1**: no-class rate = fraction of held-out-class samples accepted by 0 of the K−1 remaining models.
9. **Supervised-varsel novelty guard:** external-novel-class test shows `varsel_path=supervised` does not degrade novelty rate below the full-spectra baseline.
10. **Wold DPOW** ranking stable across resamples (Spearman ρ ≥ 0.8); "balanced" retains ≥2/3 known-relevant variables.
11. Persistence round-trip reproduces `p_values`/`decision_matrix`/`summary_label`/`accepted_classes`; old-build load raises `NotImplementedError`.
12. Plumbing: `multiclass_simca` never falls through to classification at any task-type branch site.
13. Numeric reference tests pinning T²/Q/`p_joint`/decision on a fixed matrix.

---

## 10. Phases (for the TDD plan to detail)

- **Phase A — engine + CV/novelty harness + persistence** (`simca.py` + `model_io.py`): `MultiClassClassModel`, decision rule, per-class calibration, two-stage leakage-safe CV + LOCO/external novelty mode, dedicated metrics, `.dasp` multi-class format. Two co-equal load-bearing risks: the CV/novelty harness AND persistence.
- **Phase B — Wold varsel + diagnostics** + reused supervised prefilter (tagged, novelty-gated).
- **Phase C — search + task-type wiring** (`run_multiclass_simca_search`, all branch sites + audit).
- **Phase D — GUI + code export.**

---

## 11. Guardrails

- Never edit `search.py`'s single-Y path; existing regression/classification/one-class paths stay byte-identical (diff/gold guard). No new deps without `pyproject.toml`. `.venv312` only. Never edit `.env`.
- **Chemometrics conventions are not bugs** — **per-spectrum** ops (SNV, SG derivatives, baseline) are applied outside folds and are not leakage. **Column-autoscale is different here:** for the class-modeling path it is a defined, mode-controlled (`per_class`/`global`/`none`), **fold-internal** step (§5.5) — this does *not* contradict the legacy discriminant path's "full-X autoscale before CV" convention, which stays as-is for regression/classification. Both discriminant and class-modeling are legitimate paradigms; do not "correct" one toward the other.
- Merge gate = local diff-failure-set vs `origin/main` (main red on cloud CI since 2025-10-27); PR adds zero new failures. Commit per phase before yielding control; explicit `git add` paths (never `git add -A` — many untracked scratch files); CRLF `git diff --stat` check after GUI edits. Do NOT auto-merge — await user greenlight.

---

## 12. Next step

Spec is review-hardened (two cross-family rounds folded in). Next: write the **TDD implementation plan** in `docs/plans/` (Phase A→D task breakdown with the tests in §9), on its own branch off `main`. Two open confirmations to carry into the plan: (a) the Oliveri & Downey 2012 / Pomerantsev & Rodionova primary-source detail (§5.1); (b) the concrete acceptance-rate targets in §1 are placeholders pending a real multi-class dataset to calibrate against.
