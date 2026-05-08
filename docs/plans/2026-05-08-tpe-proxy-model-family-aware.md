# Handoff: Model-family-aware TPE proxy

**Status:** ready for planning. Next agent should produce a detailed implementation plan; user will then have Codex and Kimi K2.6 evaluate that plan before any code is written.

**Working directory:** `C:/Users/mspon/git/dasp` on branch `main` (currently 6-7 commits ahead of `origin/main`, not pushed).

**One-line goal:** make `_quick_evaluate` (the TPE preprocessing-discovery proxy) pick its model family based on what the user has enabled downstream — tree-family models get a tree-family proxy, linear/PLS-family models get a PLS proxy — so the proxy's preprocessing preferences align with the actual downstream model and TPE stops fighting itself.

---

## Why this matters (empirical motivation)

User runs spectral_predict on n~50 NIR chemometrics datasets with PLS as the primary downstream model. We discovered today that the current TPE proxy (hardcoded LightGBM) misbehaves in two distinct ways:

### Symptom 1 — proxy collapses to mean-prediction at n<50

Default LightGBM `min_child_samples=20` requires both children of any split to hold ≥20 samples. With 5-fold CV on n=49 (n_train_per_fold=39 for most folds, 32 after a 9-sample SPXY holdout), no split is legal. The tree degenerates to a single leaf predicting the training mean. **Every preprocessing config returns the same RMSE** (= y std-derived mean-prediction RMSE).

User's GUI showed `RMSE=6.890843325188513` for all 10 entries in "TPE Top Preprocessing Configurations" regardless of which preprocessing was applied. Confirmed via per-trial DIAG instrumentation: 30/30 first completed trials had identical `t.value=-6.890843325188513` even though X fingerprints differed wildly per trial.

### Symptom 2 — when proxy IS informative, it ranks WRONG preprocessings for the downstream model

Commit `9b9d244` (today, now reverted at `b879b52`) "fixed" the collapse by scaling `min_child_samples = max(2, n_train_per_fold // 5)`. The proxy started returning distinct scores. But end-to-end A/B on user's actual SPXY 20% workflow showed **the fix made downstream model quality WORSE** — proxy preferences misaligned with PLS preferences.

**A/B numbers (3 splits, gap ≤ 0.02 chemometrics-strict filter, BoneCollagen example folder):**

| Split | n_pre passing | n_post passing | PRE best R²pred | POST best R²pred | delta |
|---|---|---|---|---|---|
| **SPXY 20%** (user's actual setup) | 73 | 30 | **0.9722** | 0.9405 | **−0.0317** PRE wins large |
| Stratified 20% | 149 | 144 | 0.9520 | 0.9492 | −0.0028 tied |
| Random 20% | 121 | 107 | 0.9526 | 0.9576 | +0.0050 tied |

Root cause:
- LightGBM at small n votes for `snv_deriv3_w13+autoscale` (PRE-FIX top 5 ALL `snv_deriv3+autoscale`)
- PLS (the actual downstream model) prefers `snv_deriv2_w15+autoscale` (PRE-FIX top 5 dominated by this)
- Diversity-blind selection (pre-fix mean-prediction collapse) accidentally fed the canonical chemometrics winner to the main grid because `select_diverse_configs` falls back to "1-best-per-preprocessing-type" when scores tie
- Signal-driven selection (post-fix) filtered out PLS's preferred family in favor of LightGBM's preference

**Conclusion:** the fix doesn't work without aligning the proxy to the downstream model family. That's what this handoff is for.

A/B harness for reproducing: `tools/_repro_tpe_fix_downstream_ab.py`. Saved val_df CSVs from the 3 splits live at `tools/_tpe_fix_ab_arm_PRE_*.csv` / `_POST_*.csv` (user's machine).

---

## Architectural context the next agent must internalize

### Recent autoscale fix is load-bearing (commits `3a4e502` + `ca987b4` + `2791d7a`, today)

The user (Makapania) fixed a separate bug in **exhaustive** preprocessing where `chromosome_to_transform`'s closure called `StandardScaler().fit_transform(X)` at the end when the autoscale gene was set. `compute_validation_metrics_for_top_models` calls the closure twice (once on X_train, once on X_val), so each call refit a fresh scaler on its own input → val features centered to val's stats rather than train's → R²pred collapsed.

**The fix establishes a pattern this handoff must respect:** closures handle ONLY per-spectrum operations (SNV, SG derivatives, baseline, smoothing); cross-sample operations (StandardScaler, fitted PCA, MSC reference) are applied OUTSIDE the closure with proper fit-on-train, transform-on-val state.

For the TPE proxy work specifically, this matters because:
- PLS is **stateful and cross-sample** (it fits LV decomposition on training data)
- LightGBM is **stateful and cross-sample** (it learns histograms / split thresholds)
- The proxy is wrapped in `cross_val_score`, which sklearn handles correctly via clone + fit-per-fold, so this isn't a concern at the proxy level
- BUT: any preprocessing applied INSIDE the proxy (right now there's none — `_quick_evaluate` just CV-scores the model on already-preprocessed X) must respect the per-spectrum-vs-cross-sample distinction. New agent should keep this in mind if extending the proxy to do its own preprocessing

### TPE call chain

```
GUI: tpe_preprocess=True, tpe_preprocess_n_trials=75, ...
  → spectral_predict_gui_optimized.py:28627-28634 (and similar nearby) calls run_search(...)
    → src/spectral_predict/search.py:1845 (the TPE block) calls run_tpe_preprocessing_discovery(...)
      → src/spectral_predict/tpe_preprocessing_discovery.py:run_tpe_preprocessing_discovery
        → _objective(trial) per Optuna trial
          → _apply_full_preprocessing(X, ...)  [per-spectrum, no train/val concern]
          → _quick_evaluate(X_eval, y, task_type, cv_folds)
            → CURRENT: hardcoded LightGBM (with PLS / LogReg / IsolationForest fallback if LGBM not importable)
            → PROPOSED: branch on proxy_family → tree (LGBM with adaptive mcs) | linear (PLS / LogReg)
```

There's also a multistart sibling `evaluate_config_with_seed(X, y, task_type, cv_folds, random_state)` at the same file (~line 252) used by `run_tpe_multistart_preprocessing_discovery`. **Same change must apply uniformly to both** — the previous fix attempt did this and the next plan should too.

### What `_quick_evaluate` does today (post-revert)

```python
# tpe_preprocessing_discovery.py:133-260 (approx)
def _quick_evaluate(X, y, task_type, cv_folds) -> float:
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
        # ... LightGBM with hardcoded n_estimators=50, max_depth=4 (no min_child_samples
        # override post-revert)
        # uses cross_val_score with cv=int (non-shuffled KFold)
    except Exception:
        # Fallback path: PLS for regression, LogReg+StandardScaler for classification,
        # -inf for one_class
```

Multistart sibling `evaluate_config_with_seed` is similar but uses shuffled KFold/StratifiedKFold with the supplied `random_state`.

### Display lie banner just landed (`258fc00`, today)

When `np.std([t.value for t in completed_trials]) < 1e-9` (proxy collapsed to a constant), the GUI now shows an honest "configs selected by random+diverse sampling, not by RMSE ranking" banner instead of misleading per-config RMSE numbers. **This banner becomes dead code on the regression path under model-family-aware routing** because PLS doesn't have the collapse failure mode. Leave it in place as defense in depth — don't remove during this work.

---

## The proposed design (rough — next agent should refine)

### Core change: `_quick_evaluate` branches on `proxy_family`

```python
def _quick_evaluate(X, y, task_type, cv_folds, *, proxy_family='linear') -> float:
    # New keyword-only parameter; backward-compatible default = 'linear' (PLS)
    if proxy_family == 'tree':
        # LightGBM with adaptive min_child_samples = max(2, n_train_per_fold // 5)
        # (the formerly-reverted 9b9d244 fix, now correctly aligned because the
        # downstream model is in the tree family)
    elif proxy_family == 'linear':
        # PLS for regression: PLSRegression(n_components=clamped, scale=False)
        # LogReg+StandardScaler for classification (chemometrics-canonical for binary)
        # IsolationForest for one_class (regardless of family — no PLS one-class variant)
    else:
        raise ValueError(f"unknown proxy_family: {proxy_family}")
```

### Plumbing: `run_tpe_preprocessing_discovery` accepts `proxy_family`

```python
def run_tpe_preprocessing_discovery(..., proxy_family: str = 'linear'):
    ...
    score = _quick_evaluate(X_eval, y, task_type, cv_folds, proxy_family=proxy_family)
```

Same change to `run_tpe_multistart_preprocessing_discovery` and `evaluate_config_with_seed`.

### Family resolver: `search.py` picks family from `models_to_test`

```python
TREE_FAMILY = {'LightGBM', 'XGBoost', 'CatBoost', 'RandomForest'}
LINEAR_FAMILY = {'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet'}
# SVM (kernel) and MLP (neural) intentionally not categorized — they're slow proxies
# and SVM is poor as a coarse search guide; default them to 'linear' (PLS) which is
# chemometrics-canonical and fast.

def _resolve_tpe_proxy_family(models_to_test) -> str:
    if not models_to_test:
        return 'linear'
    has_tree = any(m in TREE_FAMILY for m in models_to_test)
    has_linear = any(m in LINEAR_FAMILY for m in models_to_test)
    if has_tree and not has_linear:
        return 'tree'
    if has_linear and not has_tree:
        return 'linear'
    # Mixed → default to linear (chemometrics-canonical PLS)
    return 'linear'
```

Then in `run_search` at `search.py:1845-onwards`, pass `proxy_family=_resolve_tpe_proxy_family(models_to_test)` into both TPE call sites.

### Verification target

After implementation, on user's BoneCollagen + PLS workflow at SPXY 20%:
- proxy_family resolves to `'linear'`
- TPE proxy = PLS (no collapse, distinct scores, no autoscale-bias)
- Top-N includes `snv_deriv2_w15+autoscale` (the canonical PLS winner that the LightGBM-with-adaptive-mcs fix was filtering out)
- Best passing R²pred at gap ≤ 0.02 ≥ 0.97 (matching or beating the pre-fix 0.9722)

For a hypothetical user running LightGBM-only:
- proxy_family resolves to `'tree'`
- TPE proxy = LightGBM with adaptive `min_child_samples`
- Proxy preferences align with LightGBM downstream (no mismatch)
- Distinct scores, audit RMSE column meaningful

---

## Open design questions for the next agent to decide

These are deliberate forks; the user expects each to be answered explicitly in the plan:

1. **Mixed-family default.** User has both PLS and LightGBM enabled. Plan above defaults to `'linear'` because (a) PLS is chemometrics-canonical and (b) PLS proxy is faster and (c) the diversity selector ensures all preprocessing types still reach the main grid where LightGBM evaluates them properly. Alternative: pick whichever family has more models enabled, or run two TPE searches (one per family) and union the results. **Decide:** which?

2. **GUI exposure.** Plan above is silent auto-routing — no new GUI control. Alternative: add a "Proxy model family" dropdown to the TPE Preprocessing card with options Auto / Linear / Tree, defaulting to Auto. Pro: user can override. Con: another configuration burden.

3. **One-class proxy choice.** Plan keeps IsolationForest (the existing fallback) regardless of `proxy_family` because there's no PLS one-class variant and tree-family one-class is well-served by IF. Alternative: branch one_class too — IF for tree family, OneClassSVM for kernel users (no kernel family in the resolver currently). **Decide:** keep IF unconditionally, or extend?

4. **Exact-model-match vs family-level routing.** Plan groups models into families. Alternative: use the user's primary model directly with their hyperparameter settings (e.g., user chose PLS with n_components=8 → proxy uses PLSRegression(n_components=8)). Pro: tightest possible alignment. Con: requires plumbing user's hyperparameter choices through the TPE layer; complicates the proxy contract; for tree models means a slow proxy that defeats TPE's speed advantage. **Decide:** family-level (simpler) or exact-match (tighter)?

5. **Classification proxy details.** Plan uses `LogisticRegression(max_iter=1000)` on `StandardScaler(X)` for the linear-family classifier proxy. Alternative: PLS-DA implemented as `PLSRegression` followed by threshold-tuning. PLS-DA is more chemometrics-canonical but harder to score (continuous output → threshold). LogReg avoids the threshold question and is generally equivalent in ranking power. **Decide:** LogReg or PLS-DA?

6. **What about Bayesian preprocessing search?** TPE preprocessing-discovery is the focus here. There's also a Bayesian path (`unified_bayesian.py`). Does it have the same proxy problem? Quick read of `unified_bayesian.py` would tell us. **Verify:** does Bayesian use a proxy or fit the actual user-chosen model? If proxy, same fix needed; if actual model, no change needed.

7. **Tests for the family resolver.** Edge cases to pin: empty `models_to_test`, single-model lists for each family, mixed lists, unknown model name (e.g., a typo or new model added later). **Decide:** what does the resolver do for unknown model names? Fallback to `'linear'` is the proposed default; raise ValueError is more strict.

---

## Codex's prior review of the related "exhaustive baseline+smoothing checkboxes" plan

Earlier today the user asked Codex to review a different plan (adding baseline + smoothing checkboxes to **exhaustive** preprocessing). Codex returned `NEEDS_CHANGES` with five items, of which **three are directly relevant to this TPE proxy plan** because they touch the same data flow:

1. **Pass-through wiring incomplete.** New parameters added to a leaf function need to flow through the GUI → `run_search` → `run_tpe_preprocessing_discovery` chain. Codex specifically flagged this as a common miss. The proxy_family parameter must be threaded through three layers, not just one.

2. **Result-CSV column naming.** Don't add new uppercase fields when lowercase fields already exist (`baseline_method`, `smoothing`, etc.). For the proxy work specifically: if we want to record which proxy was used per row, populate an existing field if one fits or pick a clearly-namespaced new one (`tpe_proxy_family`).

3. **GUI reload restoration is partial.** GUI reload (`gui:33910-34024`) detects from display name only, doesn't restore params from result columns. If we add a proxy_family column for audit purposes, make sure reload either restores it or enforces a default.

The other two Codex items were specific to baseline/smoothing dimensions and don't apply here. But the plumbing-discipline lesson does — next agent should explicitly verify all three layers of the TPE call chain are updated.

Codex's review prompt + verdict were captured in conversation; reproduce by re-running the codex-reviewer agent on the new plan when it's drafted.

---

## Implementation file paths (for the next agent)

Files that definitely need editing:
- `src/spectral_predict/tpe_preprocessing_discovery.py` — `_quick_evaluate` (line ~133), `evaluate_config_with_seed` (line ~252), `run_tpe_preprocessing_discovery` signature (line ~410), `run_tpe_multistart_preprocessing_discovery` signature (line ~750)
- `src/spectral_predict/search.py` — TPE call sites at line ~1845 and ~5798 (the second is the one_class TPE path); add `_resolve_tpe_proxy_family` helper near top of file
- `tests/test_tpe_preprocessing_discovery.py` — extend `TestQuickEvaluateDirect` class with family-routing tests; add a new `TestProxyFamilyResolver` class for the family resolver

Files to read for context but probably not edit:
- `src/spectral_predict/preprocessing_discovery.py` — `select_diverse_configs` at line 769 (the diversity selector that masks the collapse symptom downstream)
- `src/spectral_predict/preprocess.py` — `build_preprocessing_pipeline` at line ~479 (canonical pipeline rebuilder; reference for ordering conventions)
- `tools/_repro_tpe_fix_downstream_ab.py` — A/B harness; reuse for verification
- `tools/_repro_tpe_top10_rmse.py` — direct backend repro on BoneCollagen
- `tools/_phase2_isolated_arm.py` — process-isolated single-arm runner

Recent commit timeline (most recent first):
- `258fc00` — TPE display banner when proxy collapses (today)
- `b879b52` — Reverted `9b9d244` after empirical regression on SPXY (today)
- `2791d7a` / `ca987b4` / `3a4e502` — Exhaustive autoscale train/val asymmetry fix (today, by user)
- `9b9d244` — TPE proxy fix attempt (today; reverted; relevant because the tree-family branch in this plan IS this fix, restored conditionally)
- `f2b0561` — GUI runtime label cleanup + 2025 Model Samples A/B tool (today)
- `50db937` — Doc-only PR #58/#59 record

---

## Verification plan (what the new agent's plan must include)

1. **Unit:** family resolver returns expected family for each model name and combination
2. **Integration:** `run_tpe_preprocessing_discovery` with `proxy_family='tree'` on n=40 produces distinct, monotonically-improving RMSEs (no collapse banner fires)
3. **Integration:** `run_tpe_preprocessing_discovery` with `proxy_family='linear'` on n=40 also produces distinct scores (PLS doesn't have the collapse failure mode at all)
4. **End-to-end via A/B harness:** rerun `tools/_repro_tpe_fix_downstream_ab.py` (modify the patched `_quick_evaluate` to use the new proxy_family routing) on BoneCollagen + PLS at SPXY 20%; assert best passing R²pred at gap≤0.02 is ≥0.97 and that `snv_deriv2_w15+autoscale` is in the top-5
5. **Cross-family verification:** same A/B harness but with hypothetical LightGBM-only enabled list; assert proxy_family routes to 'tree' and scores are non-degenerate
6. **No-regression:** existing 33 tests in `test_tpe_preprocessing_discovery.py` continue to pass

---

## Risks the new agent must call out explicitly

1. **Wall-time impact.** PLS proxy is faster than LightGBM for n<100 (no joblib overhead, no histogram building). Tree proxy under model-family routing only fires when user explicitly enables tree models, so the typical chemometrics workflow gets a wall-time *improvement*. But the planning agent should benchmark to confirm.
2. **PLS edge cases.** PLS with `n_components > min(n_samples, n_features)` raises. The existing fallback path clamps `n_components = max(2, min(10, X.shape[1]//10, X.shape[0]//2))` — reuse this. Watch for X with zero variance or near-singular matrices (the existing `np.any(np.std(X_eval, axis=0) < 1e-10)` early exit already handles this case).
3. **Reproducibility.** PLS is deterministic; LightGBM with `random_state=RANDOM_STATE` is also deterministic. Family-routed proxy should remain reproducible — no new RNG sources. Verify via the existing `test_deterministic_output_with_seed` test in `TestReproducibility`.
4. **Stochastic proxy comparison across families.** For users who switch between linear and tree proxies on the SAME data, scores will differ (different models). This is expected, not a regression. Tests should not assert cross-family score agreement.

---

## What the new agent should produce

A detailed implementation plan including:
1. Exact diff sketch for `_quick_evaluate`, `evaluate_config_with_seed`, both `run_tpe_*` signatures, both TPE call sites in `search.py`, and the new resolver
2. Explicit answers to all 7 design questions above
3. Test plan with expected test counts per class
4. Verification battery using the A/B harness with concrete pass/fail thresholds
5. Backward compatibility statement (user said BC for old result CSVs is NOT a concern; document this)
6. Estimated effort and commit-by-commit breakdown so the work can be split and reviewed atomically

After the plan is drafted, user will pass it to Codex CLI and Kimi K2.6 (via `peer-review` skill or `opencode-call`) for adversarial review before any code is written.

---

## End of handoff. Good luck.
