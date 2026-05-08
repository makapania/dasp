# Implementation Plan — Model-family-aware TPE proxy

**Status:** REVISED 2026-05-08 after Codex + Kimi K2.6 adversarial review. NEEDS_CHANGES verdicts folded in. User scope-locked to TPE only — Basic path explicitly out of scope.

**Working dir:** `C:/Users/mspon/git/dasp` on `main`.

**Authoritative handoff:** `docs/plans/2026-05-08-tpe-proxy-model-family-aware.md`. This file refines that handoff into an implementable, reviewable plan.

**One-line goal:** make TPE preprocessing-discovery's proxy model match the family of the user's downstream models — tree-family proxy (LightGBM) when only tree models are enabled, linear-family proxy (PLS / LogReg) otherwise — so the proxy stops voting for preprocessings that the actual downstream model doesn't want.

---

## 1. Decisions on the seven open design questions

The handoff explicitly demands answers to these. Each answer is accompanied by *why*; reviewers should challenge any of these on its merits, not just the implementation.

### Q1 — Mixed-family default (PLS + LightGBM both enabled)

**Decision: default to `'linear'` (PLS proxy).**

Rationale:
1. **Chemometrics-canonical.** PLS is the field-standard regression model for NIR/spectral data (`CLAUDE.md` puts PLS first in the model list; `feedback_chemometrics_conventions.md` reinforces). The proxy should bias toward the canon, not the ML default.
2. **Empirical record.** Today's A/B in `tools/_tpe_fix_ab_arm_*_spxy.csv` shows the LightGBM proxy actively worse than the pre-collapse-fix random+diverse fallback for PLS downstream (R²pred 0.9405 vs 0.9722). The "diversity selector + uninformative scores" mode that helped the user's actual workflow is what the linear-family proxy will reproduce, but with informative scoring on top of it.
3. **Wall-time.** PLS at n<100 is faster than LightGBM by a measurable margin (no joblib JIT first-touch, no histogram building). Default linear means the typical chemometrics workflow gets a wall-time *improvement* relative to today.
4. **Diversity selector saves the LightGBM users.** Because `select_diverse_configs` returns one best-per-preprocessing-type, the main grid will still evaluate every preprocessing type with the user's actual LightGBM. The proxy's role is to find good *windows* within each type and rank across types — and PLS does that competently for the tree-family configs too because what matters at the proxy layer is whether the preprocessing exposes signal, not whether tree splits or linear projections happen to consume it best.

Alternatives explicitly rejected:
- **Pick the family with the most enabled models.** Sounds democratic but creates a non-monotonic UI: user enables one extra LightGBM and the proxy flips, changing top-N silently. Not acceptable.
- **Run two TPE searches and union the results.** 2× wall-time. The diversity selector + n_top=10 already exposes enough preprocessing variety that running TPE twice doesn't reliably add value.

### Q2 — GUI exposure

**Decision: silent auto-routing for v1. No new GUI control.**

Rationale:
1. **The user's framing is "expose, not reproduce"** (`feedback_t19_user_framing.md`). The user wants the system to do the right thing automatically; the proxy choice is plumbing, not a methodology knob.
2. **One more checkbox is one more way to misconfigure.** A wrong manual pick would be worse than a sensible auto-routing.
3. **Audit-trail via `tpe_proxy_family` column** (see §3 plumbing) lets the user see what got picked without surfacing it as a control.

Future-proofing: if reviewer evidence demands a control later, adding a dropdown is a one-commit change. Don't ship it preemptively.

### Q3 — One-class proxy choice

**Decision: keep IsolationForest unconditionally for `task_type='one_class'`, regardless of `proxy_family`.**

Rationale:
1. **No PLS one-class variant.** PLS is supervised; PCA-SIMCA exists but is not a fast surrogate.
2. **OneClassSVM is slow and kernel-sensitive** — bad proxy properties.
3. **IF is what the existing fallback already uses** (`evaluate_config_with_seed` line 364–402); that path is well-tested and matches the H1 fix from the post-Phase-4 review.
4. **Family-routing only meaningfully changes the regression / classification path.** One-class doesn't have the proxy/downstream-mismatch problem because one-class downstream is itself tree-family-leaning (IF, LOF) or PCA-based — IF as proxy is family-aligned by accident.

### Q4 — Exact-model-match vs family-level routing

**Decision: family-level routing.**

Rationale:
1. **Proxy contract is "fast and approximately right."** Exact-match would mean the LightGBM downstream user gets a LightGBM proxy with the user's hyperparameter choices — which is no longer a proxy, it's the actual model run inside the TPE loop. That defeats the speed advantage of TPE.
2. **Hyperparameter plumbing complexity.** The user's `lightgbm_n_estimators_list`, etc., are *grids*, not single values. "Use the user's hyperparameter" requires picking which one — median? first? most permissive? — and any choice introduces a new config surface.
3. **Family alignment is sufficient for the empirical observation.** The user's pain isn't "the LightGBM proxy uses different hyperparameters than my downstream LightGBM"; it's "the LightGBM proxy votes against PLS." Family alignment fixes that.

### Q5 — Classification proxy details

**Decision: `LogisticRegression(max_iter=1000, n_jobs=1)` on `StandardScaler(X)` for the linear-family classifier proxy. Match the existing fallback path (line 209–215) exactly.**

Rationale:
1. **Already proven path.** The existing T-37 sklearn fallback uses precisely this combination because PLS-DA-as-PLSRegression-with-threshold introduces a threshold-tuning question that has no clean answer at proxy speed.
2. **Ranking equivalence.** LogReg and PLS-DA rank preprocessings near-identically on chemometrics datasets in practice; the proxy doesn't need PLS-DA's calibration properties.
3. **No new dependencies.** sklearn already imports.

For the multistart sibling (`evaluate_config_with_seed`), pass `random_state=random_state` to `LogisticRegression` (already done at line 359) — keep that.

### Q6 — Bayesian preprocessing search proxy parity

**Decision: NO change to `unified_bayesian.py`. Bayesian is not affected.**

Verification (read of `src/spectral_predict/unified_bayesian.py:1145–1529`):
- The Bayesian objective at `unified_bayesian.py:1145` calls `apply_preprocessing(...)` then **builds the user's actual model via `build_model(model_name, params, task_type=task_type)` at line 1529** and CV-scores that model directly.
- There is no proxy. Bayesian is fitting the actual user-chosen model in the loop, with hyperparameter search built in. That's the design.
- No proxy/downstream mismatch can exist.

This is good news: the proxy fix scopes cleanly to one module.

### Q7 — Resolver behavior on unknown model names

**Decision: unknown model names are silently treated as `'linear'`-family contributors. No exception, no warning.**

Rationale:
1. **Forward compatibility.** When a future ticket adds a new model, the resolver should not crash existing TPE runs.
2. **Safe-direction default.** If we're going to misclassify, misclassifying as linear is the safer error: PLS proxy is faster and matches the chemometrics canon.
3. **Loud failure on bad names is a footgun in production.** A typo in a saved-CSV reload (the GUI's reload path) shouldn't silently break TPE and instead route to a sensible default.

Edge cases pinned by tests:
- `models_to_test=None` → `'linear'` (no info → safe default).
- `models_to_test=[]` → `'linear'` (same).
- `models_to_test=['UnknownModel']` → `'linear'` (forward compat).
- `models_to_test=['UnknownModel', 'LightGBM']` → `'tree'` (one tree-family member is enough).
- `models_to_test=['UnknownModel', 'PLS']` → `'linear'` (one linear-family member is enough).
- `models_to_test=['LightGBM', 'PLS']` → `'linear'` (mixed → linear default per Q1).
- `models_to_test=['SVR']` / `['MLP']` / `['SVM']` / `['NeuralBoosted']` → `'linear'` (uncategorized → linear default; rationale: SVR/SVM are kernel-method, MLP is neural — both are slow proxies and not in tree family; routing to PLS is the existing T-37 chemometrics-aligned default).

---

`★ Insight ─────────────────────────────────────`
The handoff's seven-question split is deliberate: each question represents a different axis of risk (silent UX, perf, scope creep, future-proofing). Answering them in writing **before** touching code lets reviewers attack design choices that are hard to see in a diff.
The Q4 decision (family-level not exact-match) is the most consequential — it's why the proxy stays a proxy and not a slow stand-in for the user's actual hyperparameter grid. If a reviewer pushes back, this is the question that decides whether the whole approach scales.
`─────────────────────────────────────────────────`

---

## 2. Diff sketches (line numbers reference current `main` at `0b7f0c8`)

### 2.1 `src/spectral_predict/tpe_preprocessing_discovery.py`

#### 2.1.1 New constants and resolver near top of file (after line ~59)

```python
# Model-family-aware proxy routing (2026-05-08).
# See docs/plans/2026-05-08-tpe-proxy-model-family-aware-IMPLEMENTATION.md.
TREE_FAMILY_MODELS = frozenset({
    'LightGBM', 'XGBoost', 'CatBoost', 'RandomForest',
})
LINEAR_FAMILY_MODELS = frozenset({
    'PLS', 'PLS-DA', 'Ridge', 'Lasso', 'ElasticNet',
})
# SVM/SVR/MLP/NeuralBoosted are intentionally uncategorized — they default to
# 'linear' via the resolver's mixed/unknown rule. See Q7 of the implementation plan.

VALID_PROXY_FAMILIES = frozenset({'tree', 'linear'})


def resolve_tpe_proxy_family(models_to_test) -> str:
    """Pick the TPE proxy family from the user's enabled-models list.

    Returns 'tree' iff the only enabled models are in TREE_FAMILY_MODELS;
    otherwise returns 'linear' (the chemometrics-canonical PLS / LogReg path).

    Unknown model names are silently routed to 'linear' for forward compat.
    Empty/None inputs route to 'linear' (no information → safe default).
    """
    if not models_to_test:
        return 'linear'
    has_tree = any(m in TREE_FAMILY_MODELS for m in models_to_test)
    has_linear = any(m in LINEAR_FAMILY_MODELS for m in models_to_test)
    if has_tree and not has_linear:
        return 'tree'
    return 'linear'
```

Public API (no leading underscore) so `search.py` can import it without the
private-symbol smell.

#### 2.1.2 `_quick_evaluate` — branch on `proxy_family` (current line 133–218)

The current function has two paths: try-LightGBM and except-fallback (PLS / LogReg / `-inf` for one_class). Restructure to:

1. Resolve `proxy_family` into a model+CV factory.
2. Single try/except where the except branch only covers runtime errors (CV crash, OOM), not proxy-family selection.

Proposed signature change:

```python
def _quick_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int,
    *,
    proxy_family: str = 'linear',
) -> float:
    """Cross-validated evaluation using a family-aware proxy model.

    Parameters
    ----------
    proxy_family : {'tree', 'linear'}, default 'linear'
        Model family for the TPE proxy.
        - 'tree': LightGBM with adaptive min_child_samples (the previously-
          reverted 9b9d244 fix, now correctly aligned because the downstream
          model is tree-family).
        - 'linear': PLS for regression, LogReg+StandardScaler for classification.
        Both branches use IsolationForest for `task_type='one_class'` (no PLS
        one-class variant; see Q3 of the implementation plan).

    Notes
    -----
    Default is 'linear' (PLS / LogReg) — the chemometrics-canonical proxy.
    Backward-compatible with all existing callers that don't pass the parameter.
    """
    if proxy_family not in VALID_PROXY_FAMILIES:
        raise ValueError(
            f"unknown proxy_family={proxy_family!r}; "
            f"expected one of {sorted(VALID_PROXY_FAMILIES)}"
        )
    import warnings

    n_samples = X.shape[0]
    cv_folds = min(cv_folds, n_samples // 2)
    cv_folds = max(2, cv_folds)

    # One-class: family-independent, IsolationForest only.
    if task_type == 'one_class':
        return _quick_evaluate_oneclass(X, y, cv_folds)

    if proxy_family == 'tree':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError:
            # Tree proxy requested but LGBM not installed — fall back to the
            # linear path rather than failing. Log once. Same rationale as
            # the existing fallback path (T-37).
            return _quick_evaluate_linear(X, y, task_type, cv_folds)
        return _quick_evaluate_tree(X, y, task_type, cv_folds, LGBMRegressor, LGBMClassifier)

    return _quick_evaluate_linear(X, y, task_type, cv_folds)
```

Three helpers (private, _underscore-prefixed):

```python
def _quick_evaluate_tree(X, y, task_type, cv_folds, LGBMRegressor, LGBMClassifier) -> float:
    """Tree-family LightGBM proxy with adaptive min_child_samples.

    Restores the formerly-reverted 9b9d244 fix: scale min_child_samples to
    fold size so trees can actually grow on small chemometrics datasets.
    Aligned with downstream tree-family models, so the proxy's preference
    matches what those models will pick.
    """
    import warnings

    n_samples = X.shape[0]
    n_train_per_fold = n_samples - (n_samples // cv_folds)
    # Scale min_child_samples so both children of any split fit. Floor at 2.
    adaptive_mcs = max(2, n_train_per_fold // 5)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        if task_type == 'classification':
            model = LGBMClassifier(
                n_estimators=50, max_depth=4,
                min_child_samples=adaptive_mcs,
                random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
            )
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        else:
            model = LGBMRegressor(
                n_estimators=50, max_depth=4,
                min_child_samples=adaptive_mcs,
                random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
            )
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
        return scores.mean()


def _quick_evaluate_linear(X, y, task_type, cv_folds) -> float:
    """Linear-family proxy: PLS regression / LogReg+StandardScaler classification."""
    if task_type == 'regression':
        n_components = min(10, X.shape[1] // 10, X.shape[0] // 2)
        n_components = max(2, n_components)
        pls = PLSRegression(n_components=n_components, scale=False)
        scores = cross_val_score(pls, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
        return scores.mean()
    elif task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=1),
        )
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy')
        return scores.mean()
    else:  # one_class — should not reach here; guarded above
        return -np.inf


def _quick_evaluate_oneclass(X, y, cv_folds) -> float:
    """One-class proxy: LightGBM-supervised-on-y_oc when LGBM available; else IF.

    Mirrors the existing one_class branches in _quick_evaluate (LGBM
    StratifiedKFold path) and evaluate_config_with_seed (IF fallback). Family-
    routing does not change one-class behavior.
    """
    import warnings
    n_outliers = int(np.sum(y == -1))
    if n_outliers < 2:
        return 0.0
    n_splits = min(cv_folds, n_outliers)

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return _quick_evaluate_oneclass_iforest(X, y)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            model = LGBMClassifier(
                class_weight='balanced',
                n_estimators=50, max_depth=3,
                random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
            )
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')
            return scores.mean()
    except Exception:
        return _quick_evaluate_oneclass_iforest(X, y)


def _quick_evaluate_oneclass_iforest(X, y) -> float:
    """IsolationForest fallback for one-class proxy. Mirrors evaluate_config_with_seed:364-402."""
    try:
        from sklearn.ensemble import IsolationForest
        X_inlier = X[y != -1]
        if len(X_inlier) < 5:
            return 0.0
        clf = IsolationForest(
            contamination='auto', random_state=RANDOM_STATE,
            n_estimators=50, n_jobs=1,
        )
        clf.fit(X_inlier)
        preds = clf.predict(X)
        inlier_mask = y != -1
        outlier_mask = y == -1
        if inlier_mask.sum() == 0 or outlier_mask.sum() == 0:
            return 0.0
        inlier_recall = (preds[inlier_mask] == 1).mean()
        outlier_recall = (preds[outlier_mask] == -1).mean()
        return float((inlier_recall + outlier_recall) / 2)
    except Exception:
        return -np.inf
```

This decomposition deliberately:
- Keeps `_quick_evaluate`'s existing positional signature stable (just adds keyword-only `proxy_family` with a backward-compatible default).
- Pulls the LGBM-vs-fallback try/except apart so a runtime CV failure with the tree proxy doesn't silently fall through to PLS (mixing objectives, the same H1-equivalent footgun the multistart fix already closed for `evaluate_config_with_seed`).
- Makes the one-class path family-independent (Q3).

#### 2.1.3 `evaluate_config_with_seed` — same family branch (current line 221–402)

Identical structural change. Proposed signature:

```python
def evaluate_config_with_seed(
    X, y, task_type, cv_folds, random_state,
    *, proxy_family: str = 'linear',
) -> float:
```

The current implementation has the same try-LGBM-then-fallback split. Restructure to mirror `_quick_evaluate`:

```python
def evaluate_config_with_seed(X, y, task_type, cv_folds, random_state, *, proxy_family='linear'):
    if proxy_family not in VALID_PROXY_FAMILIES:
        raise ValueError(...)

    n_samples = X.shape[0]
    cv_folds = min(cv_folds, n_samples // 2)
    cv_folds = max(2, cv_folds)

    if task_type == 'one_class':
        return _evaluate_with_seed_oneclass(X, y, cv_folds, random_state)

    if proxy_family == 'tree':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError:
            return _evaluate_with_seed_linear(X, y, task_type, cv_folds, random_state)
        try:
            return _evaluate_with_seed_tree(X, y, task_type, cv_folds, random_state,
                                            LGBMRegressor, LGBMClassifier)
        except Exception:
            # LGBM installed but failed for THIS seed (OOM / numerical) —
            # return -inf rather than silently falling through to a different
            # family. Same rationale as the existing post-Phase-4 fix.
            return float('-inf')

    return _evaluate_with_seed_linear(X, y, task_type, cv_folds, random_state)
```

Three siblings (`_evaluate_with_seed_tree`, `_evaluate_with_seed_linear`, `_evaluate_with_seed_oneclass`) factor identically to the helpers above but use `random_state=random_state` and `shuffle=True` in the CV splitters (matching the current behavior at line 290–322).

The tree branch gets the same `min_child_samples = max(2, n_train_per_fold // 5)` adaptation.

The one-class path **stays identical to the current behavior** (LGBM-with-seeded-RS or IF fallback) — `proxy_family` is a no-op for one_class. This preserves the H1 fix and the CV-failure-fall-through semantics that DeepSeek's review already validated.

#### 2.1.4 `run_tpe_preprocessing_discovery` — accept and thread `proxy_family` (current line 405–757)

Signature change (additive, non-breaking):

```python
def run_tpe_preprocessing_discovery(
    X, y,
    task_type='regression',
    n_trials=75,
    n_startup_trials=20,
    n_top=10,
    cv_folds=5,
    enable_autoscale=True,
    enable_baseline=True,
    enable_smoothing=True,
    smoothing_window=17,
    smoothing_polyorder=2,
    progress_callback=None,
    random_state=RANDOM_STATE,
    skip_diversity=False,
    *,
    proxy_family: str = 'linear',
) -> List[Dict[str, Any]]:
```

`*` separator keeps `proxy_family` keyword-only — no positional confusion with `skip_diversity` for callers that pass that.

In the body:
1. The `_objective(trial)` closure (line 488) closes over `proxy_family` and passes it to `_quick_evaluate`:
   ```python
   score = _quick_evaluate(X_eval, y, task_type, cv_folds, proxy_family=proxy_family)
   ```
2. The header-print (line 467–475) gains a line: `print(f"  Proxy family: {proxy_family}")`.
3. Each cfg dict in `all_configs` (line 620–634) records the family it was scored with:
   ```python
   '_tpe_proxy_family': proxy_family,
   '_tpe_proxy_model_name': 'lightgbm' if proxy_family == 'tree' else 'pls' if task_type == 'regression' else 'logreg',
   ```
   The `'importance_method': 'lightgbm'` line at 627 should change to `'importance_method': '<proxy_model_name>'` to keep the existing field accurate. **Audit:** verify no consumer relies on the literal `'lightgbm'` string here.
4. Replace the proxy-collapse banner (line 670–696) so it no longer hardcodes the LightGBM diagnosis. The new wording:
   - Tree family: keep the existing diagnosis ("min_child_samples=...") but use the adaptive value: "LightGBM tree proxy returned identical scores — n_train_per_fold too small even with adaptive min_child_samples".
   - Linear family: PLS doesn't have this collapse mode. If `np.std(values) < 1e-9` for the linear path, that's *suspicious* (constant X across configs?) and the banner should say "Proxy returned identical scores — possible constant-X or numerical issue; configs selected by random+diverse sampling."

   Implementation: branch the banner text on `proxy_family`. Keep the banner-firing predicate (`np.std(values) < 1e-9`) identical; only the diagnostic text differs.

#### 2.1.5 `run_tpe_multistart_preprocessing_discovery` — accept and thread `proxy_family` (current line 788–1028)

Same additive signature change:
```python
def run_tpe_multistart_preprocessing_discovery(
    X, y,
    task_type='regression',
    n_trials=75,
    n_top=10,
    cv_folds=5,
    enable_autoscale=True,
    enable_baseline=True,
    enable_smoothing=True,
    smoothing_window=17,
    smoothing_polyorder=2,
    n_starts=5,
    per_start_pool=7,
    n_seeds=5,
    progress_callback=None,
    controller=None,
    *,
    proxy_family: str = 'linear',
) -> List[Dict[str, Any]]:
```

In the body:
1. Pass `proxy_family=proxy_family` into the per-start `run_tpe_preprocessing_discovery` call (line 875).
2. Pass `proxy_family=proxy_family` into `evaluate_config_with_seed` inside `_eval_fn` (line 934).
3. Header print gains the family line.
4. Each result_config gets the same `_tpe_proxy_family` annotation post-rescore (parallel to the multistart_halt_reason annotation at line 970).

### 2.2 `src/spectral_predict/search.py`

#### 2.2.1 Import the resolver at the top of the file

```python
from .tpe_preprocessing_discovery import (
    run_tpe_preprocessing_discovery,
    run_tpe_multistart_preprocessing_discovery,
    resolve_tpe_proxy_family,
)
```

The current code does these imports lazily inside the TPE blocks at line 1898 and 5837. **Keep the lazy imports**, just add `resolve_tpe_proxy_family` to those import statements. (Top-level import of the module triggers optuna import which the test suite already accommodates via skipif; lazy is the established pattern here, do not change it.)

#### 2.2.2 Regression/classification TPE block (current line 1872–1950)

After the header prints and before the wrapper dispatch:
```python
proxy_family = resolve_tpe_proxy_family(models_to_test)
print(f"  Proxy family: {proxy_family} (resolved from models_to_test={models_to_test})")
```

Then thread into both call sites (line 1920 and 1937):
```python
discovered_configs = run_tpe_multistart_preprocessing_discovery(
    ...
    proxy_family=proxy_family,
)
```

```python
discovered_configs = run_tpe_preprocessing_discovery(
    ...
    proxy_family=proxy_family,
)
```

#### 2.2.3 One-class TPE block (current line 5836–5887)

`run_one_class_search` doesn't have a `models_to_test` parameter — the equivalent is `enabled_models`. Per Q3, the one-class proxy is family-independent (always IsolationForest), so the `proxy_family` value passed into the one-class call site is **functionally a no-op** but should still be wired for plumbing consistency:

```python
proxy_family = resolve_tpe_proxy_family(enabled_models)  # no-op for OC, but plumbed for audit
```

This way, the audit trail (`_tpe_proxy_family` column) reflects what *would* have been picked if the family branch mattered. If a future change makes one-class family-aware, the plumbing is already in place.

### 2.3 Audit trail — `tpe_proxy_family` survives into the result CSV

**Codex MEDIUM + Kimi MEDIUM convergent finding:** `_tpe_*` fields do NOT auto-flow through to the result CSV. The existing pattern is *manual per-key extraction*, not pattern-based. `scoring.py:445-500` (`create_results_dataframe`) uses a fixed column schema; `search.py:5130/6205/6769` builds `result` dicts by listing only specific `preprocess_cfg` keys (e.g. `tpe_score`, `tpe_multistart_halt_reason`).

The earlier draft was wrong on this. Corrected plumbing path — every step is explicit:

1. **`tpe_preprocessing_discovery.py` writes** `cfg['_tpe_proxy_family']` and `cfg['_tpe_proxy_model_name']` on each returned config (§2.1.4 unchanged).

2. **`search.py` regression/classification `preprocess_configs` build (lines 1984, 1998):** add explicit copy from `cfg` into the `preprocess_configs` dict. New unmasked column name `tpe_proxy_family` (drops the leading underscore — matches the existing `_tpe_baseline_method → baseline_method` unmask convention):
   ```python
   "tpe_proxy_family": cfg.get("_tpe_proxy_family"),
   "tpe_proxy_model_name": cfg.get("_tpe_proxy_model_name"),
   ```

3. **`search.py` regression/classification result-row dicts (lines 5130, 5164):** add explicit forward from `preprocess_cfg` into the `result` dict:
   ```python
   "tpe_proxy_family": preprocess_cfg.get("tpe_proxy_family"),
   "tpe_proxy_model_name": preprocess_cfg.get("tpe_proxy_model_name"),
   ```

4. **`search.py` one-class `preprocess_configs` build (lines 5904, 5917):** mirror step 2.

5. **`search.py` one-class result-row dicts (lines 6205, 6262, 6769, 6848):** mirror step 3 at all four sites — full-spectrum AND variable-subset rows.

6. **GUI reload at `gui:33910/33940/34011/34021/34032`:** verified by Codex — won't choke on missing `tpe_proxy_family` column for old CSVs because reload parses display affixes and known columns like `Autoscale`, not arbitrary `_tpe_*` fields. **No reload-path changes needed.**

7. **`scoring.py:445-500` (`create_results_dataframe`):** Codex/Kimi did not flag this as needing schema extension because the result dicts feed in directly with their full key set. Verify by re-running existing TPE integration tests after the §2.3 changes — if a column is silently dropped, that test will surface it.

**Pre-implementation grep (required):**
```
grep -n "_tpe_baseline_method\|tpe_score\|tpe_multistart_halt_reason" src/spectral_predict/search.py
```
Use the output to double-check that the line numbers above are still correct against current `main` before editing — they may shift with intervening commits.

---

## 3. Test plan

Total new tests: **~14 across 2 new classes + 2 existing classes**. Existing 33 tests must continue to pass.

### 3.1 New class: `TestProxyFamilyResolver` in `tests/test_tpe_preprocessing_discovery.py`

7 tests covering the Q7 edge cases:

```python
class TestProxyFamilyResolver:
    def test_none_returns_linear(self):
        assert resolve_tpe_proxy_family(None) == 'linear'

    def test_empty_list_returns_linear(self):
        assert resolve_tpe_proxy_family([]) == 'linear'

    def test_pure_tree_returns_tree(self):
        assert resolve_tpe_proxy_family(['LightGBM']) == 'tree'
        assert resolve_tpe_proxy_family(['XGBoost', 'CatBoost', 'RandomForest']) == 'tree'

    def test_pure_linear_returns_linear(self):
        assert resolve_tpe_proxy_family(['PLS']) == 'linear'
        assert resolve_tpe_proxy_family(['Ridge', 'Lasso', 'ElasticNet', 'PLS-DA']) == 'linear'

    def test_mixed_returns_linear(self):
        assert resolve_tpe_proxy_family(['LightGBM', 'PLS']) == 'linear'

    def test_unknown_returns_linear(self):
        assert resolve_tpe_proxy_family(['SomeFutureModel']) == 'linear'

    def test_unknown_plus_tree_returns_tree(self):
        assert resolve_tpe_proxy_family(['SomeFutureModel', 'LightGBM']) == 'tree'

    def test_uncategorized_models_route_to_linear(self):
        # SVR/SVM/MLP/NeuralBoosted are intentionally uncategorized
        for name in ['SVR', 'SVM', 'MLP', 'NeuralBoosted']:
            assert resolve_tpe_proxy_family([name]) == 'linear'
```

### 3.2 Extend `TestQuickEvaluateDirect` (line 455 of test file)

Add 4 family-routing tests:

```python
def test_quick_evaluate_linear_family_regression(self):
    np.random.seed(42)
    X = np.random.randn(40, 60).astype(np.float64)
    y = 3.0 * X[:, 5] - 2.0 * X[:, 30] + np.random.randn(40) * 0.3
    score = _quick_evaluate(X, y, 'regression', 3, proxy_family='linear')
    assert np.isfinite(score)
    # Linear path is PLS — should NOT collapse
    # (sanity: re-run on different X yields different score)

def test_quick_evaluate_tree_family_regression(self):
    np.random.seed(42)
    X = np.random.randn(40, 60).astype(np.float64)
    y = 3.0 * X[:, 5] - 2.0 * X[:, 30] + np.random.randn(40) * 0.3
    score = _quick_evaluate(X, y, 'regression', 3, proxy_family='tree')
    assert np.isfinite(score)

def test_quick_evaluate_invalid_family_raises(self):
    np.random.seed(42)
    X = np.random.randn(20, 30).astype(np.float64)
    y = np.random.randn(20)
    with pytest.raises(ValueError, match='proxy_family'):
        _quick_evaluate(X, y, 'regression', 3, proxy_family='quantum')

def test_tree_family_adaptive_mcs_yields_distinct_scores_at_small_n(self):
    """The motivating bug: at n<50, tree proxy with default mcs collapses.
    With adaptive mcs, scores must differ across X variants."""
    np.random.seed(42)
    n = 49  # below the user's reported collapse threshold
    X1 = np.random.randn(n, 60).astype(np.float64)
    X2 = X1 + np.random.randn(n, 60) * 0.5  # different signal
    y = 2.0 * X1[:, 10] + np.random.randn(n) * 0.2
    s1 = _quick_evaluate(X1, y, 'regression', 5, proxy_family='tree')
    s2 = _quick_evaluate(X2, y, 'regression', 5, proxy_family='tree')
    assert s1 != s2  # at n=49, default mcs would tie at -RMSE_mean; adaptive must differ
```

### 3.3 New class: `TestEvaluateConfigWithSeedFamilyRouting`

Mirror tests for the multistart sibling — 3 tests:

```python
class TestEvaluateConfigWithSeedFamilyRouting:
    def test_linear_family_regression_seeded(self):
        ...

    def test_tree_family_regression_seeded(self):
        ...

    def test_invalid_family_raises(self):
        ...
```

### 3.4 Extend `TestSearchIntegration` (line 295)

2 tests for end-to-end resolver wiring:

```python
def test_search_with_pls_routes_to_linear_proxy(self, monkeypatch):
    """run_search with models_to_test=['PLS'] passes proxy_family='linear' into TPE."""
    captured = {}
    def spy(*args, **kwargs):
        captured['proxy_family'] = kwargs.get('proxy_family', '<unset>')
        return []
    import spectral_predict.tpe_preprocessing_discovery as tpe_mod
    monkeypatch.setattr(tpe_mod, 'run_tpe_preprocessing_discovery', spy)
    # ... call run_search with tpe_preprocess=True, models_to_test=['PLS']
    assert captured['proxy_family'] == 'linear'

def test_search_with_lightgbm_only_routes_to_tree_proxy(self, monkeypatch):
    ...
```

### 3.5 Existing tests — semantics shift, NOT no-regression

**Codex MEDIUM finding:** the earlier draft claimed "33 existing tests continue to pass" — that's mechanically true but semantically misleading. The default behavior changes (LightGBM-primary → PLS-primary), and several existing tests have *names* that assert "default" or "unchanged" without specifying which proxy family they expect.

Tests requiring explicit semantic update:
- `tests/test_tpe_multistart.py:184` — `test_quick_evaluate_unchanged_at_default`. **Rename and repin** to `test_quick_evaluate_default_is_linear_family`. Add a sibling `test_quick_evaluate_tree_family_explicit` that passes `proxy_family='tree'` and pins LightGBM behavior.
- `tests/test_tpe_preprocessing_discovery.py:462,469,476` — `TestQuickEvaluateDirect.test_regression_returns_finite` / `test_classification_returns_finite` / `test_one_class_with_outliers_returns_finite`. These currently call `_quick_evaluate(...)` without the new arg. Keep them — they now pin the new linear default (and the family-independent one_class path). Add explicit `proxy_family='tree'` companion tests as part of the §3.2 additions.
- `tests/test_t44_autoscale_wiring.py` — autoscale wiring tests. PLS is stable on autoscale, so behavior is preserved on the linear default. Re-run to confirm; no expected churn.
- `tests/test_tpe_preprocessing_discovery.py:385` `test_deterministic_output_with_seed` (`TestReproducibility`). PLS is deterministic, so reproducibility is preserved on the linear default. Re-run to confirm.

**No silent semantic shifts allowed.** Every test that's affected by the default flip gets either renamed or has its expectation explicitly updated, with a comment line citing the 2026-05-08 default-flip commit SHA.

### 3.6 Test count summary

- New tests: **~14** (7 resolver + 4 quick_evaluate + 3 evaluate_config_with_seed + 2 search integration; integration tests may need a couple of extra cases for one-class)
- Existing tests preserved: 33 in `test_tpe_preprocessing_discovery.py` + 12 in `test_tpe_multistart.py` + 17 in `test_t44_autoscale_wiring.py` = 62 baseline must remain green.

`★ Insight ─────────────────────────────────────`
The `test_tree_family_adaptive_mcs_yields_distinct_scores_at_small_n` test is the load-bearing one — it directly pins the symptom from `PROJECT_STATUS.md` (n=49 collapse). If this test passes today with `proxy_family='tree'`, we have empirical proof the tree branch fixes Symptom 1. If it fails, the adaptive-mcs formula needs revisiting *before* shipping.
`─────────────────────────────────────────────────`

---

## 4. Verification battery (beyond unit/integration tests)

The handoff demands verification *with the A/B harness* against concrete pass/fail thresholds. Implementation plan:

### 4.1 Linear-family verification: BoneCollagen + PLS at SPXY 20%

**Tool:** `tools/_repro_tpe_fix_downstream_ab.py` — modify the patched `_quick_evaluate` to use `proxy_family='linear'` instead of the formerly-tested adaptive-mcs LightGBM.

**Pass/fail thresholds** (pinned from the handoff §"Verification target"):
- TPE proxy returns distinct scores (no collapse banner fires).
- Top-N includes `snv_deriv2_w15+autoscale` (the canonical PLS winner).
- **Best passing R²pred at gap ≤ 0.02 ≥ 0.97** (matches or beats pre-fix 0.9722).
- Wall-clock not significantly worse than the pre-fix (< 1.5× pre-fix wall-time).

### 4.2 Tree-family verification: BoneCollagen + LightGBM-only at SPXY 20%

**Tool:** same harness with `models_to_test=['LightGBM']`.

**Pass/fail thresholds:**
- `proxy_family` resolves to `'tree'`.
- Distinct scores (no collapse).
- Top-N includes preprocessings that the LightGBM downstream actually likes (proxy/downstream alignment); we don't have a pre-pinned list because today's data doesn't have a clean LightGBM downstream A/B, but the key check is **scores are non-degenerate** (proxy didn't collapse, audit RMSE column meaningful).

### 4.3 Cross-split sanity: stratified 20% and random 20%

Linear-family on both partitions. Pass: best passing R²pred remains ≥ pre-fix figures (0.9520 stratified, 0.9526 random — within ±0.005 noise band).

### 4.4 Reproducibility cross-check

Run the linear-family path twice with the same `random_state`; assert byte-identical top-N. PLS is deterministic and `select_diverse_configs` is deterministic, so this should hold trivially. Pin it as a CI test (already covered by `test_deterministic_output_with_seed`, but worth re-running explicitly).

### 4.5 Wall-time benchmark

On the user's actual workflow (BoneCollagen, PLS, n=49, n_trials=75):
- Pre-fix (today's main): wall-time T0
- Post-fix (linear path): wall-time T1, expect T1 ≤ T0 (PLS faster than LightGBM)
- Post-fix (tree path): wall-time T2, expect T2 ≈ T0 (LightGBM with adaptive mcs ≈ default LightGBM perf)

Don't ship if T1 > T0 by more than 20% — that would contradict Q1's wall-time argument.

---

## 5. Backward compatibility

Per the handoff, **BC for old result CSVs is NOT a concern** — explicitly waived by the user. Document this in the implementation commit message.

What this means concretely:
- Old CSVs without the `_tpe_proxy_family` column reload fine (the GUI reload at `gui:33910-34024` doesn't choke on missing columns; if it does, that's a separate bug).
- Old CSVs are not retroactively annotated.
- Saved studies / SQLite resume continues to work because `proxy_family` is search-time-only metadata, not stored in trial.params.

What this does NOT mean:
- We do not break the public function signatures. `proxy_family` is keyword-only with a default. Existing scripts in `tools/` that call `run_tpe_preprocessing_discovery(...)` without the new arg get the linear-family proxy automatically — which is also the chemometrics-canonical path, so this is a *no-regression-for-tools* default.

`★ Insight ─────────────────────────────────────`
Backward compat for the public API is non-negotiable even when CSV BC is waived. The user has tools in `tools/_repro_tpe_*.py` that call these functions directly; breaking them would break the very harness we use to verify the fix.
`─────────────────────────────────────────────────`

---

## 6. Estimated effort and commit-by-commit breakdown

The work splits cleanly into atomic commits, each individually reviewable:

### Commit 1 — pure refactor: factor `_quick_evaluate` into helpers (no behavior change)

- Extract `_quick_evaluate_oneclass`, `_quick_evaluate_oneclass_iforest` from current `_quick_evaluate` body.
- No new parameters yet. No behavior change. Tests must remain bit-identical pass.
- Same refactor for `evaluate_config_with_seed`.
- **Size:** ~200 LOC moved, 0 LOC behavior change.
- **Verification:** all 62 existing tests pass; manual diff inspection shows extracted helpers contain identical logic.

### Commit 1.5 — A/B harness pre-fix (BLOCKER from Codex/Kimi)

**Both reviewers flagged independently:** `tools/_repro_tpe_fix_downstream_ab.py:59` defines `_quick_evaluate_prefix_emulation(X, y, task_type, cv_folds)` (4 positional args). Once commit 2 adds keyword-only `proxy_family` and `_objective` calls `_quick_evaluate(..., proxy_family=...)`, the patched function raises `TypeError: unexpected keyword argument 'proxy_family'`. **This breaks the A/B harness the §4 verification battery depends on.**

- Update `tools/_repro_tpe_fix_downstream_ab.py:59,192,207`: change the patched function signature to accept `**kwargs` (and ignore `proxy_family` if present, or branch on it for explicit emulation arms):
  ```python
  def _quick_evaluate_prefix_emulation(X, y, task_type, cv_folds, **kwargs):
      # Pre-fix emulation ignores proxy_family — it always reproduced the
      # hardcoded-LightGBM behavior that was main as of today.
      ...
  ```
- **Size:** ~10 LOC, single file.
- **Verification:** harness runs cleanly on today's main with the kwarg-tolerant signature (no semantic change yet — the kwarg isn't passed until commit 2).
- **Order:** ship this BEFORE commit 2 so the harness keeps working continuously across the default flip.

### Commit 2 — add resolver and `proxy_family` parameter (linear default = old behavior)

- Add `TREE_FAMILY_MODELS`, `LINEAR_FAMILY_MODELS`, `VALID_PROXY_FAMILIES`, `resolve_tpe_proxy_family`.
- Add `proxy_family: str = 'linear'` keyword-only to `_quick_evaluate`, `evaluate_config_with_seed`, `run_tpe_preprocessing_discovery`, `run_tpe_multistart_preprocessing_discovery`.
- Body changes: `_quick_evaluate` and `evaluate_config_with_seed` branch on `proxy_family`. Default `'linear'` reproduces today's PLS-fallback behavior **for the linear path** but **does NOT change today's primary tree path** because today the primary path is LightGBM-fallback-PLS.
- **Critical caveat:** with `proxy_family='linear'` as default, the *default* behavior changes — today's non-fallback path is LightGBM, but the new default is PLS. This is the intended fix, BUT it's why we want Codex+Kimi to review *before* shipping. **The default change IS the user-visible behavior change** that fixes the Symptom 2 regression.
- **Size:** ~150 LOC additive + ~50 LOC restructured.
- **Verification:** `TestProxyFamilyResolver` (7 tests) + extended `TestQuickEvaluateDirect` (4 tests) + extended `TestEvaluateConfigWithSeedFamilyRouting` (3 tests) all pass. Existing 62 tests pass (PLS-fallback path is well-covered).

### Commit 3 — wire resolver into `search.py` (regression/classification + one-class)

- Import `resolve_tpe_proxy_family` (lazy, in both TPE blocks at lines ~1898 and ~5837).
- Resolve `proxy_family = resolve_tpe_proxy_family(models_to_test)` (or `enabled_models` for OC) once per TPE block.
- Pass into both single-start and multistart wrappers.
- Carry `_tpe_proxy_family` / `_tpe_proxy_model_name` through `preprocess_configs` build.
- **Size:** ~30 LOC.
- **Verification:** `TestSearchIntegration` 2 new tests pass.

### Commit 4 — verification battery (no source changes)

- Run `tools/_repro_tpe_fix_downstream_ab.py` modified for linear / tree / cross-split.
- Save outputs to `tools/_tpe_proxy_family_ab_*.csv` (or similar).
- Document in `docs/SESSION_LOG.md` per CLAUDE.md session protocol.
- Update `docs/PROJECT_STATUS.md` reflecting the new state.
- **Size:** doc-only.

### Commit 5 — peer review feedback fold-in (if any)

Reserved slot for whatever Codex / Kimi flag in the review of this plan and/or the resulting code.

### Total: 5-6 atomic commits, ~410 LOC source change + ~260 LOC test additions

Updated post-review (2026-05-08): added commit 1.5 (A/B harness pre-fix, BLOCKER from Codex+Kimi). Source LOC bumped slightly for the explicit `tpe_proxy_family` audit-column copies at 6 result-row sites.

Estimated review burden: 2 cross-family review cycles (per project pattern from PR #54, #55, #57, #58, #59), 1-2 days end-to-end including verification battery.

---

## 7. Risks the new agent must call out (carrying forward from the handoff)

These are documented in the handoff §"Risks the new agent must call out explicitly" and the implementation accepts them as flagged, with the following clarifications:

### 7.1 Wall-time impact

Linear default makes typical chemometrics workflows *faster* (PLS < LightGBM). Tree path only fires when user enables tree models exclusively, where today's behavior is already LightGBM. Net wall-time: improvement-or-tie everywhere. **Action:** include wall-time numbers in commit 4's verification log.

### 7.2 PLS edge cases

`n_components > min(n_samples, n_features)` raises. Existing clamp `n_components = max(2, min(10, X.shape[1]//10, X.shape[0]//2))` reused verbatim from current fallback (line 203–205). Zero-variance early exit at line 542 already covers near-singular X. **No new edge-case code needed.**

### 7.3 Reproducibility

PLS is deterministic; LightGBM with `random_state=RANDOM_STATE` (or seeded `random_state` in the multistart path) is deterministic. Family-routing introduces no new RNG sources. `test_deterministic_output_with_seed` continues to pass on the linear default; should also be re-run with `proxy_family='tree'` explicitly (added test).

### 7.4 Stochastic proxy comparison across families

If the same dataset is run twice — once with PLS-only enabled, once with LightGBM-only enabled — the proxy scores differ because the proxy differs. **This is by design**, not a regression. Tests must NOT cross-compare scores across families.

### 7.5 NEW risk: behavior change on default

The current default behavior (when LightGBM is installed) is LightGBM proxy. The new default is PLS proxy — a **silent behavior change** for any caller that doesn't pass `proxy_family`. This is the entire point of the fix, but it is a behavior change. Mitigation:
- The change is documented in the function docstring with a "default changed in 2026-05-08" note.
- Tools and tests that need the old behavior pass `proxy_family='tree'` explicitly.
- The chain of evidence (Symptom 2 A/B, today's session) justifies the default flip.

**Per-tool impact (Kimi LOW finding):**
- `tools/_repro_tpe_top10_rmse.py` — was named/designed to reproduce the LightGBM mean-prediction collapse. After the default flip, this tool will produce distinct PLS scores and **no longer reproduces the bug it was named for**. Add a banner-print at the top of that tool: *"Note: as of 2026-05-08, default proxy is PLS; pass `proxy_family='tree'` to reproduce historic LightGBM collapse behavior."* Optionally rename later.
- `tools/_repro_tpe_fix_downstream_ab.py` — patched harness covered by commit 1.5; behavior preserved.
- `tools/dump_tpe_top10_configs.py` — dump tool; semantically unchanged because dumps reflect whatever the proxy picks. Any saved-CSV consumers should be checked.
- `tools/repro_tpe_multistart_one_class.py` — one-class path; family-routing is a no-op for one_class (Q3), so behavior is preserved.

### 7.6 NEW risk: tree path's adaptive-mcs is the formerly-reverted `9b9d244`

The tree branch carries the formerly-reverted `min_child_samples = max(2, n_train_per_fold // 5)` formula. The revert at `b879b52` happened because the formula made things worse *at the search level* — but the diagnosis was "wrong proxy family", not "wrong formula". Restoring the formula *only* on the tree path (where it's family-aligned) is correct, but the reviewer should re-confirm by re-running the SPXY 20% A/B with `models_to_test=['LightGBM']` (Verification §4.2) and assert the formula doesn't have a *separate* regression on tree-family downstreams.

`★ Insight ─────────────────────────────────────`
Risk 7.6 is the most subtle. The 9b9d244 fix wasn't wrong on its own; it was misaligned with the user's downstream. Restoring it under the tree branch is the minimal correct change, but we lack pre-pinned LightGBM downstream A/B numbers to compare against. Verification 4.2 must produce those numbers as part of shipping.
`─────────────────────────────────────────────────`

---

## 8. Pre-implementation checklist (to be completed by the implementing agent)

- [ ] This plan reviewed by Codex CLI and Kimi K2.6 via `peer-review` skill or `opencode-call`.
- [ ] All design decisions in §1 confirmed or amended.
- [ ] Sister-site grep for `_tpe_baseline_method` confirms CSV column ingestion path.
- [ ] No new dependencies needed (sklearn LogisticRegression, sklearn IsolationForest, lightgbm — all already imported elsewhere in the module).
- [ ] `tools/_repro_tpe_fix_downstream_ab.py` runs cleanly on today's main as a baseline before any changes.
- [ ] `tests/test_tpe_preprocessing_discovery.py` runs cleanly on today's main (62 tests baseline).
- [ ] `docs/PROJECT_STATUS.md` updated with the planning-complete status before commit 1.

---

## 9. What this plan does NOT do (explicit non-goals)

- **Does NOT touch `preprocessing_discovery.py:669` (Basic Preprocessing Discovery / `smart_preprocess=True`).** Kimi's MEDIUM finding correctly identified that the Basic path has the same proxy/downstream-mismatch bug as TPE — same hardcoded LightGBM `_quick_evaluate` at `preprocessing_discovery.py:669`, same n<50 collapse mode. Per user direction (2026-05-08), Basic is **explicitly out of scope** for this PR. Rationale: the user wants TPE made multi-model; Basic is a separate concern that can be addressed in a follow-up if its bug becomes user-visible. The existing `Exhaustive Preprocessing` path (`ga_preprocessing.py:289` `evaluate_fitness`) already exposes a `fitness_model` parameter and `_evaluate_with_actual_model` path, so users with the Basic-path bug can switch to Exhaustive if needed. **File a follow-up ticket** if the Basic-path bug ever surfaces in a user-facing R²pred regression.
- Does NOT change `unified_bayesian.py` (Q6 confirmed: not affected — `unified_bayesian.py:1529` builds the user's actual model via `build_model`, no proxy exists).
- Does NOT add a GUI dropdown for proxy family (Q2: silent auto-routing for v1).
- Does NOT remove the proxy-collapse banner shipped at `258fc00` (handoff §"Display lie banner": leave in place as defense in depth).
- Does NOT change the multistart per-study `skip_diversity=True` default (Phase 4 H2 fix; orthogonal to this work).
- Does NOT touch the chromosome / closure scaffolding from the autoscale fix (3a4e502/ca987b4) — that's per-spectrum-only and doesn't intersect with the proxy.
- Does NOT change `n_components` clamp logic or `n_estimators=50, max_depth=4` LightGBM hyperparameters (no scope creep).
- Does NOT rename `proxy_family` to align with Exhaustive's `fitness_model` parameter (naming-consistency follow-up; out of scope here).

---

## 10. Reviewer-question answers (post-review, 2026-05-08)

The original draft asked seven questions. Both reviewers landed; here's what each settled:

1. **Linear-family default (Q1):** ACCEPTED. Neither reviewer pushed back. Codex and Kimi both implicitly endorsed via their LOW-severity confirmations of the call-site enumeration.
2. **Unknown → linear (Q7):** ACCEPTED. No reviewer flagged this as needing strict-error semantics.
3. **`_quick_evaluate` decomposition (Q3):** ACCEPTED. Codex re-read the helper extraction; no new error paths introduced. Kimi confirmed the autoscale closure pattern is preserved.
4. **`tpe_proxy_family` audit field reload (Q4):** ACCEPTED — metadata-only sufficient. Codex verified `gui:33910/33940/34011/34021/34032` won't choke on missing column.
5. **Atomic commit breakdown (Q5):** **AMENDED.** A new commit 1.5 (A/B harness pre-fix) inserted per Codex/Kimi BLOCKER finding. Five-or-six commits now, depending on how reviewer-feedback fold-in is counted.
6. **Cross-file dispatcher angle (Codex):** **CONFIRMED COMPLETE.** Two `src/` TPE call sites (`search.py:1920/1937` and `5857/5874`); four `tools/` direct callers; many tests. No production sites missed. Codex flagged `preprocessing_discovery.py:669` as a separate non-TPE symbol with the same name (different module) — out of scope per user direction.
7. **Sister-site sweep angle (Kimi):** **ONE SITE FLAGGED, OUT-OF-SCOPED.** `preprocessing_discovery.py:669` (`smart_preprocess=True` / Basic Preprocessing Discovery in the GUI) has the same proxy/downstream bug. Per user direction 2026-05-08, scope locked to TPE only — Basic is filed as a follow-up consideration in §9.

---

## 11. Pre-review state and what changed

This document was reviewed by:
- **Codex CLI** (cross-file dispatcher angle) — verdict: NEEDS_CHANGES, 4 actionable findings (1 HIGH, 2 MEDIUM, 1 MEDIUM defaults)
- **Kimi K2.6 via opencode-call** (sister-site sweep angle) — verdict: NEEDS_CHANGES, 1 BLOCKER (convergent with Codex HIGH), 2 MEDIUM (1 unique sister-site, 1 convergent CSV plumbing)

Convergent findings between the two reviewers:
1. **HIGH/BLOCKER** — A/B harness monkeypatch will TypeError on the new `proxy_family` kwarg. Folded into commit 1.5.
2. **MEDIUM** — `_tpe_proxy_family` does not auto-flow to result CSV; needs manual copies at 6 explicit sites. Folded into §2.3.

Unique findings:
- **Kimi MEDIUM:** `preprocessing_discovery.py:669` sister-site has the same bug. Out-of-scoped to §9 per user direction.
- **Codex MEDIUM:** test name semantics shift on default flip. Folded into §3.5.

Plan is now ready for implementation. Any further reviewer cycle is at the user's discretion — the high-confidence convergent fixes are folded; the remaining decisions (Basic out-of-scope) are user-locked.

---

## End of plan.
