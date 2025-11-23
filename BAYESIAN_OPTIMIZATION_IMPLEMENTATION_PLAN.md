# Bayesian Optimization Implementation Plan for DASP

**Created:** 2025-01-23
**Objective:** Add Bayesian hyperparameter optimization as ADDITIVE feature (zero risk to existing grid search)
**Expected Benefit:** 100x speedup with same or better performance
**Timeline:** 7 days (3 phases)

---

## Executive Summary

### Current State
- Grid search tests 5,000+ hyperparameter combinations
- Takes 11+ hours for comprehensive searches
- Hyperparameters defined in `model_config.py`

### Target State
- **ADDITIVE ONLY** - Zero modifications to existing grid search
- New `run_bayesian_search()` function using Optuna
- 30-50 trials achieves same results (100x speedup)
- GUI radio button to select Grid vs Bayesian mode

---

## 1. ARCHITECT PERSPECTIVE

### 1.1 Core Design Principle: SEPARATION

```
Grid Search (UNCHANGED)
├── run_search()              # DO NOT TOUCH
├── get_model_grids()         # DO NOT TOUCH
└── _run_single_config()      # DO NOT TOUCH

Bayesian Search (NEW)
├── run_bayesian_search()     # New parallel function
├── bayesian_config.py        # Optuna search spaces
└── bayesian_utils.py         # Helper functions
```

**Key:** Bayesian REUSES existing infrastructure:
- Same CV framework (`_run_single_config`)
- Same preprocessing pipeline
- Same result format
- Same scoring system

### 1.2 New Files

```
src/spectral_predict/
├── bayesian_config.py       # NEW (~200 lines)
└── bayesian_utils.py        # NEW (~150 lines)

search.py                    # MODIFIED - add run_bayesian_search() (~400 lines)
spectral_predict_gui_optimized.py  # MODIFIED - add GUI (~100 lines)

tests/
├── test_bayesian_search.py  # NEW (~300 lines)
├── test_grid_regression.py  # NEW (~200 lines)
└── test_bayesian_validation.py  # NEW (~250 lines)
```

**Total:** ~1,900 lines added, **0 lines deleted**

### 1.3 API Design

```python
def run_bayesian_search(
    X, y, task_type,
    n_trials=30,              # NEW: Optuna trials budget
    timeout=None,             # NEW: Optional time limit
    random_state=42,          # NEW: Reproducibility
    # All existing parameters from run_search()
    folds=5,
    models_to_test=None,
    tier='standard',
    # ... (60+ existing parameters)
):
    """
    Bayesian hyperparameter optimization using Optuna.

    Returns same DataFrame format as run_search() - fully compatible.
    """
```

### 1.4 GUI Integration Points

**Point 1: Add Optimization Mode Selection**
```python
# Line ~870: Variable declarations
self.optimization_mode = tk.StringVar(value="grid")
self.bayesian_n_trials = tk.IntVar(value=30)

# Line ~3818: UI section (BEFORE Model Tier)
ttk.Radiobutton(text="🔲 Grid Search", value="grid")
ttk.Radiobutton(text="🚀 Bayesian (Fast)", value="bayesian")
ttk.Spinbox(textvariable=self.bayesian_n_trials, from_=10, to=200)
```

**Point 2: Dispatch Logic**
```python
# Line ~11184: Where run_search() is called
if self.optimization_mode.get() == "bayesian":
    results_df = run_bayesian_search(X, y, n_trials=self.bayesian_n_trials.get(), ...)
else:
    results_df = run_search(X, y, ...)  # Existing - unchanged
```

### 1.5 Search Space Design

**Advantage:** Continuous space vs discrete grid

| Parameter | Grid | Bayesian | Benefit |
|-----------|------|----------|---------|
| learning_rate | [0.05, 0.1, 0.2] | log-uniform(0.01, 0.3) | Finds 0.127 |
| max_depth | [3, 6, 9] | int-uniform(3, 9) | Finds 5 or 7 |
| subsample | [0.7, 0.85, 1.0] | uniform(0.7, 1.0) | Finds 0.83 |

**Result:** Better parameters with fewer trials

---

## 2. DEBUGGER PERSPECTIVE

### 2.1 Critical Risks & Mitigations

**RISK 1: Grid Search Regression** ⚠️ HIGHEST PRIORITY
- **What:** Accidental changes break existing grid search
- **Impact:** CATASTROPHIC - production workflows fail
- **Mitigation:**
  - ✅ Zero modifications to `run_search()` function
  - ✅ Comprehensive regression test suite
  - ✅ Side-by-side validation before release

**RISK 2: Result Format Incompatibility**
- **What:** Bayesian results don't match grid search format
- **Impact:** GUI crashes, reports fail
- **Mitigation:**
  - ✅ Reuse `create_results_dataframe()`
  - ✅ Schema validation: `assert df_bayesian.columns == df_grid.columns`

**RISK 3: Non-Reproducible Results**
- **What:** Same data produces different results each run
- **Impact:** Debugging impossible, user trust lost
- **Mitigation:**
  ```python
  sampler = optuna.samplers.TPESampler(seed=random_state)
  study = optuna.create_study(sampler=sampler)
  ```

**RISK 4: Small Dataset Edge Cases**
- **What:** n_samples=20, PLS suggests n_components=50
- **Impact:** Crash with "components > samples"
- **Mitigation:**
  ```python
  # Dynamic constraints in search space
  max_components = min(max_n_components, n_samples - 1, n_features)
  trial.suggest_int('n_components', 2, max_components)
  ```

**RISK 5: Single Hyperparameter Models**
- **What:** Ridge has only 'alpha' - Bayesian overhead unnecessary
- **Impact:** Slower than grid for simple models
- **Mitigation:**
  ```python
  # Fallback to grid for simple models
  SIMPLE_MODELS = ['Ridge', 'Lasso']
  if model_name in SIMPLE_MODELS and n_trials > grid_size:
      return run_grid_for_simple_model(...)
  ```

**RISK 6: Optuna Database Locking**
- **What:** Parallel runs conflict on SQLite
- **Impact:** "Database is locked" errors
- **Mitigation:**
  ```python
  # Use in-memory storage
  study = optuna.create_study(storage=None)
  ```

### 2.2 Regression Testing Strategy

**Test Suite 1: Grid Search Unchanged**
```python
def test_grid_search_exact_match():
    """CRITICAL: Verify grid search identical before/after."""
    df1 = run_search(X, y, random_state=42)
    df2 = run_search(X, y, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)
```

**Test Suite 2: Bayesian Reproducibility**
```python
def test_bayesian_reproducibility():
    """Same seed gives same results."""
    df1 = run_bayesian_search(X, y, n_trials=20, random_state=42)
    df2 = run_bayesian_search(X, y, n_trials=20, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)
```

**Test Suite 3: Performance Comparison**
```python
def test_bayesian_competitive():
    """Bayesian achieves ≥95% of grid performance."""
    grid_df = run_search(X, y, tier='standard')  # 200 trials
    bayes_df = run_bayesian_search(X, y, n_trials=30)  # 30 trials

    grid_best = grid_df['RMSE'].min()
    bayes_best = bayes_df['RMSE'].min()

    assert bayes_best <= grid_best * 1.05  # Within 5%
```

### 2.3 Edge Cases

1. **Zero features after wavelength restriction**
   - Validation: Skip invalid trials with `raise optuna.TrialPruned()`

2. **Binary vs multiclass classification**
   - Task-aware constraints in search space

3. **Derivatives reduce feature count**
   - Preprocess-aware feature counting

4. **Very small datasets (n < 20)**
   - Conservative parameter ranges

---

## 3. IMPLEMENTATION PLAN

### PHASE 1: Core Backend (Days 1-3)

**Goal:** Working `run_bayesian_search()` function

#### Task 1.1: Create `bayesian_config.py` (4 hours)
```python
def get_bayesian_search_space(model_name, trial, tier='standard'):
    """Define Optuna search space for each model."""

    if model_name == 'XGBoost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }
    # ... 10 more models
```

**Deliverable:** Search spaces for all 11 model types

#### Task 1.2: Create `bayesian_utils.py` (4 hours)
```python
def create_optuna_study(direction='minimize', random_state=42):
    """Create reproducible Optuna study."""
    sampler = optuna.samplers.TPESampler(seed=random_state)
    return optuna.create_study(direction=direction, sampler=sampler)

def convert_optuna_params(optuna_params, model_name):
    """Convert Optuna params to sklearn-compatible dict."""
    # Handle type conversions, rounding, etc.
```

**Deliverable:** Helper functions

#### Task 1.3: Implement `run_bayesian_search()` (8 hours)

**Structure:**
```python
def run_bayesian_search(...):
    # 1. Setup (same as run_search)
    X_np = X.values
    y_np = y.values
    wavelengths = X.columns.values

    # 2. For each preprocessing config:
    for preprocess_cfg in preprocess_configs:
        X_preprocessed = apply_preprocessing(X_np, preprocess_cfg)

        # 3. For each model type:
        for model_name in models_to_test:
            # Create Optuna study
            study = create_optuna_study(random_state=random_state)

            # Define objective function
            def objective(trial):
                params = get_bayesian_search_space(model_name, trial)
                model = build_model(model_name, params)

                # REUSE existing CV framework
                result = _run_single_config(
                    X_preprocessed, y_np, wavelengths,
                    model, model_name, params,
                    preprocess_cfg, cv_splitter, task_type, ...
                )

                return result['RMSE']  # Minimize RMSE

            # Optimize
            study.optimize(objective, n_trials=n_trials)

            # Extract best result
            best_params = study.best_params
            best_result = _run_single_config(..., best_params, ...)

            # Add to results
            df_results = add_result(df_results, best_result)

    # 4. Return same format as run_search()
    return df_results, label_encoder
```

**Key:** Reuses `_run_single_config()` - no code duplication

**Deliverable:** Working Bayesian search backend

#### Task 1.4: Unit Tests (4 hours)
```python
tests/test_bayesian_search.py:
- test_search_space_validity()
- test_single_model_optimization()
- test_multimodel_optimization()
- test_preprocessing_integration()
- test_edge_cases()
```

**Deliverable:** 20+ passing tests

#### Task 1.5: CLI Testing (2 hours)
```python
# Manual validation
from spectral_predict.search import run_bayesian_search

results = run_bayesian_search(X, y, models_to_test=['XGBoost'], n_trials=20)
print(results.head())
```

**Deliverable:** Verified working backend

---

### PHASE 2: GUI Integration (Days 4-5)

**Goal:** Add Bayesian option to GUI

#### Task 2.1: Add GUI Controls (3 hours)

**File:** `spectral_predict_gui_optimized.py`

**Step 1: Variables (Line ~870)**
```python
self.optimization_mode = tk.StringVar(value="grid")
self.bayesian_n_trials = tk.IntVar(value=30)
```

**Step 2: UI Section (Line ~3818, BEFORE Model Tier)**
```python
# Optimization Mode Card
opt_card = self._create_card(content_frame, title="Optimization Method 🆕")

# Radio buttons
ttk.Radiobutton(text="🔲 Grid Search (Exhaustive)", value="grid")
ttk.Radiobutton(text="🚀 Bayesian (100x Faster)", value="bayesian")

# Bayesian settings
ttk.Spinbox(textvariable=self.bayesian_n_trials, from_=10, to=200)

# Info text
info = "💡 Bayesian: 30 trials for quick, 100 for thorough, 200 for publication"
ttk.Label(text=info)
```

**Deliverable:** GUI controls added

#### Task 2.2: Mode Toggle Handler (1 hour)
```python
def _on_optimization_mode_changed(self):
    """Enable/disable Bayesian settings."""
    if self.optimization_mode.get() == 'grid':
        # Disable Bayesian controls
        for widget in self.bayesian_settings_frame.winfo_children():
            widget.config(state='disabled')
    else:
        # Enable Bayesian controls
        for widget in self.bayesian_settings_frame.winfo_children():
            widget.config(state='normal')
```

**Deliverable:** Interactive mode switching

#### Task 2.3: Dispatch Logic (2 hours)

**File:** `spectral_predict_gui_optimized.py` (Line ~11184)

```python
# Conditional dispatch
if self.optimization_mode.get() == 'bayesian':
    from spectral_predict.search import run_bayesian_search

    self._log_progress("🚀 BAYESIAN OPTIMIZATION MODE")
    self._log_progress(f"Trials: {self.bayesian_n_trials.get()}")

    results_df, label_encoder = run_bayesian_search(
        X_filtered, y_filtered,
        task_type=task_type,
        n_trials=self.bayesian_n_trials.get(),
        # ... all existing parameters ...
    )
else:
    # Existing grid search (UNCHANGED)
    from spectral_predict.search import run_search
    results_df, label_encoder = run_search(...)
```

**Deliverable:** Working GUI dispatch

#### Task 2.4: Progress Callback (2 hours)
```python
# In run_bayesian_search():
def callback(study, trial):
    if progress_callback:
        progress_callback({
            'stage': 'bayesian_optimization',
            'message': f'Trial {trial.number}/{n_trials} - Score: {trial.value:.4f}',
            'current': trial.number,
            'total': n_trials
        })

study.optimize(objective, n_trials=n_trials, callbacks=[callback])
```

**Deliverable:** Real-time progress updates

#### Task 2.5: GUI Testing (2 hours)
- Switch between Grid/Bayesian modes
- Verify progress updates
- Check result display
- Confirm grid search unaffected

**Deliverable:** Working end-to-end GUI

---

### PHASE 3: Testing & Validation (Days 6-7)

**Goal:** Production-ready validation

#### Task 3.1: Grid Regression Tests (4 hours)

**File:** `tests/test_grid_regression.py`

```python
def test_grid_unchanged():
    """Grid search EXACTLY identical before/after."""
    df1 = run_search(X, y, tier='quick', random_state=42)
    df2 = run_search(X, y, tier='quick', random_state=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_all_grid_parameters():
    """Every grid parameter still works."""
    # Test 50+ parameter combinations
```

**Deliverable:** Grid search 100% validated

#### Task 3.2: Bayesian Validation (4 hours)

**File:** `tests/test_bayesian_validation.py`

```python
def test_reproducibility():
    """Same seed = same results."""
    df1 = run_bayesian_search(X, y, n_trials=20, random_state=42)
    df2 = run_bayesian_search(X, y, n_trials=20, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_performance():
    """Bayesian competitive with grid."""
    grid_df = run_search(X, y)  # 200 trials
    bayes_df = run_bayesian_search(X, y, n_trials=30)  # 30 trials

    assert bayes_df['RMSE'].min() <= grid_df['RMSE'].min() * 1.05

def test_all_models():
    """All 11 models work."""
    for model in ALL_MODELS:
        df = run_bayesian_search(X, y, models_to_test=[model], n_trials=10)
        assert len(df) > 0
```

**Deliverable:** Bayesian fully validated

#### Task 3.3: Integration Testing (4 hours)

**Scenarios:**
1. Full workflow: Load → Bayesian → Results → Model Dev
2. Ensemble training with Bayesian models
3. Report generation
4. Save/load compatibility

**Deliverable:** End-to-end validation

#### Task 3.4: Performance Benchmark (3 hours)

**File:** `benchmarks/bayesian_vs_grid_benchmark.py`

```python
datasets = [('small', 50, 100), ('medium', 200, 500), ('large', 500, 2000)]
models = ['PLS', 'XGBoost', 'LightGBM']

for dataset, model in product(datasets, models):
    # Grid search
    t0 = time.time()
    df_grid = run_search(...)
    grid_time = time.time() - t0

    # Bayesian
    t0 = time.time()
    df_bayes = run_bayesian_search(..., n_trials=30)
    bayes_time = time.time() - t0

    print(f"Speedup: {grid_time / bayes_time:.1f}x")
```

**Expected:**
- XGBoost: 100-150x speedup
- LightGBM: 100-150x speedup
- PLS: 10-20x speedup

**Deliverable:** Performance report

#### Task 3.5: Documentation (3 hours)

**Create:** `docs/BAYESIAN_OPTIMIZATION_GUIDE.md`
- When to use Grid vs Bayesian
- How Bayesian optimization works
- Configuration guide
- Troubleshooting

**Update:** `USER_MANUAL.md`
- Add optimization mode section
- Add performance comparison
- Add best practices

**Deliverable:** Complete documentation

---

## Timeline Summary

| Phase | Duration | Tasks | Lines Added |
|-------|----------|-------|-------------|
| Phase 1: Backend | 3 days | 5 tasks | ~800 lines |
| Phase 2: GUI | 1.5 days | 5 tasks | ~200 lines |
| Phase 3: Testing | 2.5 days | 5 tasks | ~900 lines |
| **TOTAL** | **7 days** | **15 tasks** | **~1,900 lines** |

---

## Dependencies

**Add to requirements.txt:**
```
optuna>=3.5.0
plotly>=5.18.0  # Optional: for visualizations
```

**Installation:**
```bash
pip install optuna plotly
```

---

## Success Criteria

**Phase 1 Complete:**
- [ ] `run_bayesian_search()` works standalone
- [ ] All 11 models have search spaces
- [ ] Returns same DataFrame format
- [ ] Reproducible (random_state works)
- [ ] 20+ unit tests pass

**Phase 2 Complete:**
- [ ] GUI has Grid/Bayesian radio buttons
- [ ] Trials spinbox controls n_trials
- [ ] Progress updates work
- [ ] Results display correctly
- [ ] Grid search regression tests pass

**Phase 3 Complete:**
- [ ] All regression tests pass (grid unchanged)
- [ ] Bayesian reproducibility confirmed
- [ ] Achieves 95%+ grid performance
- [ ] 100x speedup confirmed
- [ ] Documentation complete

**Production Ready:**
- [ ] All 50+ tests passing
- [ ] Benchmarks validated
- [ ] User manual updated
- [ ] Zero grid search modifications

---

## Rollback Plan

**If implementation fails:**

1. Delete `bayesian_config.py`
2. Delete `bayesian_utils.py`
3. Remove `run_bayesian_search()` from `search.py`
4. Remove GUI controls from `spectral_predict_gui_optimized.py`

**Risk:** ZERO - Grid search untouched

---

## Key Design Decisions

**Decision 1: Separate Function vs Mode Parameter**
- ✅ **Chosen:** Separate `run_bayesian_search()` function
- ❌ **Rejected:** Add `optimization_mode` parameter to `run_search()`
- **Reason:** Zero risk to existing code, easier to test independently

**Decision 2: Continuous vs Discrete Search Space**
- ✅ **Chosen:** Continuous distributions (log-uniform, uniform)
- ❌ **Rejected:** Discrete grid (same as current)
- **Reason:** Bayesian advantage is exploring between grid points

**Decision 3: In-Memory vs Persistent Storage**
- ✅ **Chosen:** In-memory (storage=None)
- ❌ **Rejected:** SQLite database
- **Reason:** No persistence needed, avoids locking issues

**Decision 4: Reuse vs Reimplement CV**
- ✅ **Chosen:** Reuse `_run_single_config()`
- ❌ **Rejected:** Reimplement CV in Bayesian code
- **Reason:** No duplication, guaranteed identical results

---

## Next Steps

**To Begin Implementation:**

1. **Approve this plan** - Review and confirm approach
2. **Install Optuna** - `pip install optuna`
3. **Start Phase 1, Task 1.1** - Create `bayesian_config.py`

**Questions to resolve:**
- Which models are highest priority? (XGBoost, LightGBM, CatBoost?)
- Preferred default n_trials? (30, 50, 100?)
- Should visualization dashboard be included? (Phase 4)

---

**Ready to begin when approved!** 🚀
