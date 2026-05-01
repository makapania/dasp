# T-37: TPE Quick Preprocessing Discovery (Sketch)

> **Status:** ROUGH PLAN — depends on T-36 (autoscale toggle) shipping first so it has the dimension to search over.

**Goal:** Replace the half-built basic-preprocessing-discovery + GA + exhaustive paths with a single TPE-based quick search that covers a richer space (preproc × window × autoscale × baseline × smoothing) in ~75-100 trials. Suitable as the new default discovery path before main grid/Bayesian/NSGA search.

**Source:** Conversation 2026-04-30. The user observed that `preprocessing_discovery.py`, `ga_preprocessing.py`, and the deprecated paths all do roughly the same thing on slightly different search spaces (84 vs 238 combos), with no module covering the autoscale/baseline/smoothing dimensions jointly. DeepSeek V4 Pro audit confirmed: GA evaluates ~6× more configs than the entire 238-point space contains, providing zero benefit over exhaustive on a discrete categorical space with no neighborhood structure.

---

## Background

### Why TPE, not GA, on this space

- Search space is small (≤ 238 points) and categorical — no smooth neighbor structure between e.g. `snv_deriv2` and `deriv1_snv`. GA crossover/mutation produce random jumps, not informed steps.
- TPE handles mixed categorical/integer natively, builds independent KDEs per categorical level, and learns "given preproc=X, which window/autoscale combo does well?"
- Optuna is already a project dependency (used for Bayesian model search). Reusing TPESampler costs ~150 LOC of wrapper code.

### Why this beats current options

| | Basic discovery | GA preprocessing | **Proposed TPE quick-discovery** |
|---|---|---|---|
| Search space | 14 × 6 = 84 | 14 × 17 = 238 | 14 × ~10 × 2 × 4 × 2 ≈ 2,240 conceptual cells |
| Strategy | Exhaustive sweep | GA / exhaustive / smart-2-stage | TPE with 75-100 trials |
| Includes autoscale | No | No | **Yes** (post T-36) |
| Includes baseline toggle | No | No | **Yes** |
| Includes smoothing toggle | No | No | **Yes** |
| Diversity selection | Yes | No | Yes (port from basic) |
| Derivative-aware windows | No | Yes (smart mode only) | Yes (port from GA) |
| Multi-seed robustness | No | Yes (smart mode only) | Yes (port from GA) |
| One-class support | Yes | No | **Yes** |
| Runtime | 1-3 min | 2-5 min | 1-2 min (target) |

### What gets retired

- `ga_preprocessing.py` — superseded entirely by T-37; useful insights ported (DERIVATIVE_WINDOW_RANGES, multi-seed robustness logic).
- `preprocessing_discovery.py` — either extended in-place to become the TPE path, or refactored into `preprocessing_quick_search.py` with the basic exhaustive path retained as an opt-in "thorough mode" for users who want certainty over speed.

---

## Architecture sketch

### Public API

```python
def run_tpe_preprocessing_discovery(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,                    # 'regression' | 'classification' | 'one_class'
    n_trials: int = 75,                # GUI dropdown: 50 / 75 / 100 / 150
    n_startup_trials: int = 20,        # Random startup before TPE kicks in
    n_top: int = 10,                   # Diverse top-N returned to caller
    cv_folds: int = 5,
    enable_autoscale: bool = True,
    enable_baseline: bool = True,
    enable_smoothing: bool = True,
    progress_callback: Callable | None = None,
    random_state: int = 42,
) -> List[PreprocessingConfig]:
    """Quick TPE-based preprocessing discovery — returns top-N diverse configs."""
```

### Search dimensions

- `preprocess_type` — categorical (14 options matching `PREPROCESSING_CANDIDATES`)
- `window` — categorical, drawn from a derivative-aware list (port `DERIVATIVE_WINDOW_RANGES` from `ga_preprocessing.py:75-80`)
- `autoscale` — boolean (depends on T-36)
- `baseline_method` — categorical: `None`, `'als'`, `'polynomial'`, `'rubber_band'`, `'airpls'`
- `smoothing` — boolean (window/polyorder use defaults; full Bayesian search of those is in main `run_unified_bayesian`)

### Trial fitness

LightGBM proxy with PLS fallback (mirror `_quick_evaluate` at `preprocessing_discovery.py:669`). 5-fold CV. Returns RMSE for regression, balanced-accuracy for classification, balanced-accuracy on inlier/outlier for one-class.

### Diverse top-N selection

After all trials complete, port `select_diverse_configs` from `preprocessing_discovery.py:769`. Goal: top N must include configs from different preprocessing families, not all variants of the single best.

### GUI exposure

- Replace existing "Smart Preprocessing" + "GA / Exhaustive" checkboxes with a single "Quick Preprocessing Discovery" toggle + n_trials dropdown.
- Optional advanced subpanel: enable/disable autoscale, baseline, smoothing dimensions individually.
- Tooltip: explains TPE-based search, expected runtime, top-N output.

---

## Tasks (high-level — refine before execution)

1. Build `tpe_preprocessing_discovery.py` with the public API above.
2. Port `_quick_evaluate` and `select_diverse_configs` from `preprocessing_discovery.py`.
3. Port `DERIVATIVE_WINDOW_RANGES` and multi-seed robustness from `ga_preprocessing.py`.
4. Wire into `search.py` as a pre-step before main search (replaces `discover_preprocessing` call site).
5. GUI consolidation: single toggle + n_trials dropdown.
6. Tests: TPE convergence, diversity guarantee, one-class path, runtime regression.
7. Deprecation notes on the retired modules.
8. Validation note documenting field-aligned defaults.

---

## Open questions

1. Should baseline params (`lam`, `p` for ALS) be Optuna-tuned per trial or fixed at sensible defaults? Default to fixed for speed; revisit if results look noisy.
2. For one-class, should we use the `y_oc` ±1 encoding directly, or score on per-fold held-out outliers? Mirror what `preprocessing_discovery.py` does.
3. Should `n_trials` scale with dataset size? Quick rule: `n_trials = max(50, min(150, 5 × n_search_dimensions × n_categorical_levels))`. Probably overengineering.

---

## Sequencing

- **Blocked on:** T-36 (autoscale toggle). The autoscale dimension needs to exist as a real pipeline option before TPE can search over it.
- **Blocks:** T-38 (dead code cleanup). T-37 absorbs the useful pieces of `ga_preprocessing.py` so it can be retired.
