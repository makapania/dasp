# LOO CV popup analysis — when is the warning actually warranted?

**Date:** 2026-04-21
**Reviewer:** GPT-5.4 via Codex CLI
**Trigger:** User complaint that LOO compute warnings fire spuriously — popup says "very expensive" on runs that are actually cheap.
**Scope:** Read-only diagnostic. No code changes.

---

## Recommendation

**Popup #1 — "Very High Compute Cost" (`total_fits > 50_000`)** — **rewrite threshold and estimate.** Valuable in principle, but `total_fits` is inflated for unified Bayesian because preprocessing is sampled inside each trial, not looped outside. A 50-sample LOO run with 4 models and 300 Bayesian trials is closer to `50 × 300 × 4 = 60,000` CV fits, not `600,000`.

**Popup #2 — "High Compute Cost" (`total_fits > 10_000`)** — **adjust threshold / downgrade.** This is the noisy one. Fit count alone is a poor proxy: 10k PLS fits can be seconds; 10k CatBoost fits can be hours. Use estimated wall-clock and only warn when expected time crosses a user-meaningful threshold.

**Popup #3 — LOO + stochastic (RF/LightGBM/XGBoost/CatBoost/MLP)** — **rewrite, not remove.** This warning is about **statistical reliability**, not compute. LOO folds train on nearly identical data so stochastic seed/model variance can dominate fold differences. Firing as a modal popup for a cheap 50-sample run is disruptive. Make it a non-blocking progress-log warning unless the same run also exceeds time thresholds.

## Preprocessing count heuristic

`n_preprocessing = 10` is only accidentally close.

Default grid-search GUI config (`raw=False, snv=True, sg1=True, sg2=True, sg3=False, sg4=False, deriv_snv=True, window=[17]`) yields:
- `snv` = 1
- `sg1`: deriv + snv_deriv + deriv_snv = 3
- `sg2`: deriv + snv_deriv + deriv_snv = 3
- **Total = 7**, not 10.

If all 5 window boxes are selected, default derivatives become `2 × 3 × 5 + 1 = 31`.

Unified Bayesian has `PREPROCESSING_OPTIONS = 14`, but each trial chooses **one** preprocessing option. Do **not** multiply Bayesian trials by `n_preprocessing`.

## Bayesian work estimate

Unified Bayesian does **not** appear to prune trials. `run_unified_bayesian()` creates an Optuna study without a pruner and calls `study.optimize(..., n_trials=n_trials)`. Boosting early stopping exists in `cv_utils.py`, but it is disabled under `LeaveOneOut` because one-sample test folds cannot serve as eval sets. So Bayesian cost is not "mostly pruned"; the inflation comes mainly from multiplying by preprocessing.

## Replacement heuristic — fit count AND estimated seconds

```python
PER_FIT_SECONDS = {
    "PLS": 0.01,
    "PLS-DA": 0.02,
    "Ridge": 0.01,
    "Lasso": 0.03,
    "ElasticNet": 0.03,
    "RandomForest": 0.15,
    "SVR": 0.20,
    "SVC": 0.20,
    "LightGBM": 0.35,
    "XGBoost": 0.50,
    "CatBoost": 0.75,
    "MLP": 0.50,
}
```

Scale by data size:
```python
size_factor = max(0.5, min(20.0, (n_samples * n_features) / (100 * 2000)))
```

Suggested modal thresholds:
- `>= 60 min`: "Very High Compute Cost" modal, ask yes/no.
- `>= 15 min`: "High Compute Cost" modal, ask yes/no.
- `>= 5 min`: log/status warning only, no modal.
- LOO stochastic: modal only when `n_samples >= 100` or estimated time `>= 15 min`; otherwise non-blocking progress-log warning.

## Code change

In `spectral_predict_gui_optimized.py:22852-22873`, replace:

```python
# Rough preprocessing count: grid search tests ~10-20 configs; use 10 as default
n_preprocessing = 10
n_trials = 1
if self.optimization_method.get() == 'unified':
    try:
        n_trials = self.n_unified_trials.get()
    except Exception:
        n_trials = 300
```

with:

```python
window_count = max(1, sum([
    self.window_7.get(), self.window_11.get(), self.window_17.get(),
    self.window_23.get(), self.window_31.get()
]) + len([w for w in self.window_custom.get().split(',') if w.strip()]))

if self.optimization_method.get() == 'unified':
    n_trials = self.n_unified_trials.get()
    n_preprocessing = 1  # preprocessing is sampled inside each Bayesian trial
else:
    n_derivs = sum([self.use_sg1.get(), self.use_sg2.get(),
                    self.use_sg3.get(), self.use_sg4.get()])
    per_deriv = 1 + int(self.use_snv.get()) + int(self.use_deriv_snv.get())
    n_preprocessing = int(self.use_raw.get()) + int(self.use_snv.get()) + n_derivs * per_deriv * window_count
    n_trials = 1
```

Then replace the `total_fits > 50_000 / 10_000` gates at `spectral_predict_gui_optimized.py:22888-22914` with estimated-minute gates.

`src/spectral_predict/cv_utils.py:146` can keep `estimate_total_cv_fits()` as a raw fit counter, but add a separate helper such as `estimate_cv_wall_time_seconds(...)`; do not overload fit count with runtime semantics.

## Edge cases not covered by the new heuristic

- GPU acceleration (reduces wall-clock drastically for GBM/MLP)
- Unusually large wavelength counts after preprocessing (size_factor is rough)
- Custom hyperparameter grids (user expands search space via config tab)
- Variable subset expansion (UVE/SPA/CARS inner CV adds fits not captured here)
- GA / smart preprocessing (exponential search space)
- User hardware differences (per-fit-seconds map is calibrated to a median machine)

Still much less noisy than fixed 10k/50k fit-count thresholds.
