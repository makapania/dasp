# Codex review of Kimi's mixed-type target plan

**Review date:** 2026-04-21
**Reviewing:** `docs/plans/2026-04-21-mixed-type-target-kimi-plan.md`
**Reviewer:** GPT-5.4 via Codex CLI
**Repo tip:** `9ea4267` (main)

---

## Overall Verdict: Revise

Do not ship the plan as written. The diagnosis is mostly right, but the proposed "Centralized" implementation has a blocking NaN regression, and the plan under-specifies several label-encoding consistency points. I'd ship a narrower, explicit fix first.

## Recommended Option

Choose **Validation-Wide**, not Narrow or Centralized.

It fixes all three reported bugs with limited blast radius: coerce mixed-type classification labels before validation splitting, before validation metrics, and in refine validation. Narrow leaves bugs #2 and #3 unfixed. Centralized is directionally good, but migrating scattered `self.y` consumers in a 45K-line GUI is too risky as the first fix, especially because some consumers still need raw missing-value semantics.

## Blocking Issues

1. The Centralized plan's target-column assignment hook would break missing-target detection if applied before the existing `nan_count = self.y.isna().sum()` at `spectral_predict_gui_optimized.py:16487`. `astype(str)` turns missing values into `"nan"`, so the GUI would stop warning and downstream filters would stop dropping those rows.

2. Do not mutate `self.validation_y = self.validation_y.astype(str)` globally unless missing values were already removed. Manual validation creation at `19871` does not drop missing targets, unlike automatic creation at `19648`. Use local coerced arrays/Series preserving nulls, or coerce only after `pd.isna` filtering.

3. Backend direct callers remain exposed. `search.py` uses `LabelEncoder.fit_transform(y_np)` at `975` and `3232`. GUI coercion protects GUI calls, but library calls with mixed object labels can still crash.

## NaN → `"nan"` Risk

The plan flags the risk, but its proposed handling is not sufficient. There are real downstream consumers relying on post-coercion NaN detection: `search.py:419`, `search.py:3070`, `search.py:3701`, GUI Bayesian/NSGA validation filters at `26836` and `27060`, and refine validation at `37101`.

Fix with a helper that preserves nulls, e.g. coerce only non-null values, or always filter nulls before coercion in local validation arrays.

## Line-Number Accuracy

Mostly accurate.

- Training coercion exists at `25986-26001`.
- Bayesian crash sites are exactly `26848` and `26849`; `26849` is the likely validation-side crash.
- Stale heuristic exists at `19609`.
- Validation storage is `19719`.
- Refine validation reads raw validation labels at `37098`.
- Refine training coercion is already present at `35224-35238`.

## Test Plan

The proposed Bayesian test is weak if it only calls `run_unified_bayesian`; the reported crash is in the GUI post-processing block after Bayesian returns. It may pass without exercising `26849`.

Better tests:

- Directly test `_validation_stratified` with `pd.Series([1, "1", 2, "2", ...], dtype=object)`.
- Directly test `_validation_spxy` with mixed labels.
- Test `compute_validation_metrics_for_top_models` with a minimal fitted classifier and mixed `y_val`.
- Add a null-preservation test proving `np.nan` is still dropped after coercion.

## Task-Type Switch Reset

I would not auto-clear the validation set solely on regression/classification switches. The holdout indices are still a valid holdout; only representativeness/stratification assumptions may be stale. Warn or mark the validation set as "created under previous task type" and recommend recreation. Auto-reset is surprising UX and can discard manual selections.

## Additional Missed Sites

Prediction/validation display paths also use raw validation labels with `np.unique`, notably `40267`, `40453`, and `40644`. They may fail later when using a validation set for prediction plots/statistics.
