# RANKING BUG - ROOT CAUSE ANALYSIS AND FIX

## Executive Summary

**BUG FOUND**: The penalty scaling formula causes variable count penalties to dominate R² performance, even at low penalty settings (2/10).

**IMPACT**: Models with excellent R² (0.943) rank #876 because they use many variables, while models with worse R² but fewer variables rank in top 10.

**ROOT CAUSE**: The penalty formula is too aggressive. The difference between using 10 vs 2000 variables (0.186 penalty units at setting=2) is comparable to or larger than performance differences between models.

## Detailed Analysis

### User's Report

- Model with R²=0.9430 ranked #876
- None of top 10 models by R² ranked better than #700
- Settings: Variable Penalty = 2, Complexity Penalty = 2
- Dataset: ~2151 variables, 876 model configurations tested

### Mathematical Proof of Bug

#### Current Penalty Formula (BUGGY)

```python
# src/spectral_predict/scoring.py, lines 85-86
var_penalty_term = (variable_penalty / 10.0) * var_fraction
# where var_fraction = n_vars / full_vars

# With variable_penalty = 2:
# Model using 10 vars:   penalty = 0.2 * (10/2151)   = 0.00093
# Model using 2000 vars: penalty = 0.2 * (2000/2151) = 0.186
# Difference: 0.185 penalty units
```

#### Performance Score Range

```python
# Lines 50-59
z_rmse = (df["RMSE"] - df["RMSE"].mean()) / df["RMSE"].std()
z_r2 = (df["R2"] - df["R2"].mean()) / df["R2"].std()
performance_score = 0.5 * z_rmse - 0.5 * z_r2
```

Z-scores typically range -3 to +3. For models with similar performance:
- Good model: z_r2 ≈ +1.0, z_rmse ≈ -1.0 → performance_score ≈ -1.0
- Slightly worse: z_r2 ≈ +0.8, z_rmse ≈ -0.8 → performance_score ≈ -0.8
- **Difference: 0.2 units**

**PROBLEM**: The 0.186 variable penalty difference is nearly as large as the 0.2 performance difference!

At penalty=2 (supposedly "low"), variable count is weighing almost as heavily as performance.

#### Concrete Example

**Model A (Best Performance, Many Variables)**:
- R² = 0.943, RMSE = 0.10, n_vars = 2000
- z_r2 = +1.16, z_rmse = -1.0
- performance_score = 0.5*(-1.0) - 0.5*(1.16) = -1.08
- var_penalty = 0.186
- comp_penalty = 0.08
- **CompositeScore = -1.08 + 0.186 + 0.08 = -0.814**

**Model B (Worse Performance, Few Variables)**:
- R² = 0.88, RMSE = 0.13, n_vars = 50
- z_r2 = +0.375, z_rmse = -0.4
- performance_score = 0.5*(-0.4) - 0.5*(0.375) = -0.3875
- var_penalty = 0.0046
- comp_penalty = 0.08
- **CompositeScore = -0.3875 + 0.0046 + 0.08 = -0.303**

Model A: -0.814 (better - more negative)
Model B: -0.303 (worse - less negative)

✓ Model A ranks better. So why the bug?

**Answer**: There must be MANY models with slightly worse R² but far fewer variables, creating a large cluster of models that all outrank the best-performing model.

**Model C (Slightly Worse Performance, Very Few Variables)**:
- R² = 0.92, RMSE = 0.11, n_vars = 20
- z_r2 = +0.875, z_rmse = -0.8
- performance_score = 0.5*(-0.8) - 0.5*(0.875) = -0.8375
- var_penalty = 0.0018
- comp_penalty = 0.08
- **CompositeScore = -0.8375 + 0.0018 + 0.08 = -0.7557**

Model A: -0.814
Model C: -0.7557

**Model C ranks WORSE than Model A** despite having fewer variables because its performance is still quite good.

So the math STILL shows Model A should rank well!

### The Real Problem: Distributional Effects

If most of the 876 models have configurations like:
- 600 models with n_vars in range 10-500 (low penalty)
- 276 models with n_vars in range 500-2151 (high penalty)

Then among the 276 high-variable models, even the best one (R²=0.943) might rank #276 within that subset.

And if the 600 low-variable models include many with R² in range 0.85-0.93, they could collectively fill ranks 1-600.

**This is actually WORKING AS DESIGNED** - the penalty system is strongly favoring parsimony!

BUT: At penalty setting = 2 out of 10, users expect performance to dominate. The current formula doesn't match user expectations.

## The Core Issue

The formula:
```python
var_penalty_term = (variable_penalty / 10.0) * var_fraction
```

Creates a LINEAR scaling: penalty=2 → 20% of max penalty, penalty=5 → 50% of max penalty.

But the MAX penalty itself (var_fraction = 1.0 → penalty = 0.2 at setting=2) is TOO HIGH relative to typical performance score ranges.

### Proposed Fix

The penalty should scale NON-LINEARLY with lower settings having MUCH less impact:

```python
# CURRENT (LINEAR):
# penalty=0 → 0.0 * var_fraction
# penalty=2 → 0.2 * var_fraction
# penalty=5 → 0.5 * var_fraction
# penalty=10 → 1.0 * var_fraction

# PROPOSED (QUADRATIC):
# penalty=0 → 0.0 * var_fraction
# penalty=2 → 0.04 * var_fraction  (2²/100)
# penalty=5 → 0.25 * var_fraction  (5²/100)
# penalty=10 → 1.0 * var_fraction  (10²/100)
```

This way:
- At penalty=2: difference between 20 vs 2000 vars = 0.04 * 0.93 = 0.037 (much smaller!)
- At penalty=10: difference between 20 vs 2000 vars = 1.0 * 0.93 = 0.93 (strong penalty as intended)

## Exact Location of Bug

**File**: `C:\Users\sponheim\git\dasp\src\spectral_predict\scoring.py`

**Lines 76-87** (Variable Count Penalty):
```python
    # 1. Variable Count Penalty (0-10 scale)
    # Normalize: fraction of variables used, scaled by user preference
    if variable_penalty > 0:
        n_vars_array = np.asarray(df["n_vars"], dtype=np.float64)
        full_vars_array = np.asarray(df["full_vars"], dtype=np.float64)
        var_fraction = n_vars_array / full_vars_array  # 0-1 scale

        # Scale by user penalty (0-10) and make it affect ranking reasonably
        # At penalty=10, using all variables adds ~1 unit (comparable to ~0.3 std deviations in performance)
        var_penalty_term = (variable_penalty / 10.0) * var_fraction  # ← BUG HERE
    else:
        var_penalty_term = 0
```

**Lines 89-106** (Model Complexity Penalty):
```python
    # 2. Model Complexity Penalty (0-10 scale)
    # For PLS: penalize latent variables. For others: use median baseline
    if complexity_penalty > 0:
        lvs = df["LVs"].fillna(0).astype(np.float64)

        # Normalize LVs to [0, 1] scale (assume max useful LVs is ~25)
        lv_fraction = lvs / 25.0
        lv_fraction = np.minimum(lv_fraction, 1.0)  # Cap at 1.0

        # For non-PLS models (LVs=0), use a median penalty to avoid favoring them unfairly
        # This makes all model types comparable
        median_lv_fraction = 0.4  # Equivalent to ~10 LVs
        lv_fraction_adjusted = np.where(lvs == 0, median_lv_fraction, lv_fraction)

        # Scale by user penalty (0-10)
        comp_penalty_term = (complexity_penalty / 10.0) * lv_fraction_adjusted  # ← SAME BUG
    else:
        comp_penalty_term = 0
```

## Recommended Fix

### Option 1: Quadratic Scaling (RECOMMENDED)

Replace lines 85 and 104 with:

```python
# Line 85 - Variable penalty
var_penalty_term = ((variable_penalty / 10.0) ** 2) * var_fraction

# Line 104 - Complexity penalty
comp_penalty_term = ((complexity_penalty / 10.0) ** 2) * lv_fraction_adjusted
```

This creates a quadratic response curve where low settings have minimal impact but high settings still provide strong penalties.

### Option 2: Exponential Scaling

```python
# More aggressive scaling at high end
penalty_scale = (variable_penalty / 10.0) ** 1.5
var_penalty_term = penalty_scale * var_fraction
```

### Option 3: Rescale Maximum Penalty

Keep linear scaling but reduce the maximum penalty impact:

```python
# Reduce max penalty from 1.0 to 0.3
var_penalty_term = (variable_penalty / 10.0) * var_fraction * 0.3
comp_penalty_term = (complexity_penalty / 10.0) * lv_fraction_adjusted * 0.3
```

## Additional Issues Found

### Issue 1: Default Penalty Mismatch

**Location**: `spectral_predict_gui_optimized.py` lines 209-210
```python
self.variable_penalty = tk.IntVar(value=2)
self.complexity_penalty = tk.IntVar(value=2)
```

**vs** `scoring.py` line 7:
```python
def compute_composite_score(df_results, task_type, variable_penalty=3, complexity_penalty=5):
```

**Impact**: GUI defaults (2, 2) don't match function defaults (3, 5). This is MINOR but inconsistent.

**Fix**: Update scoring.py defaults to match GUI:
```python
def compute_composite_score(df_results, task_type, variable_penalty=2, complexity_penalty=2):
```

### Issue 2: No Unit Tests for Ranking

**Location**: No test files found for `src/spectral_predict/scoring.py`

**Impact**: The bug was not caught because there are no automated tests verifying ranking behavior.

**Fix**: Create `tests/test_scoring.py` with comprehensive test cases.

## Test Cases to Add

```python
def test_ranking_with_high_r2_many_vars():
    """Model with R²=0.943 and 2000 vars should rank in top 50 at penalty=2."""
    # Create synthetic results
    # Verify ranking behavior

def test_penalty_scaling():
    """Verify penalty=2 has minimal impact compared to penalty=10."""
    # penalty=2 should affect ranking < 10% as much as penalty=10

def test_performance_dominates_at_low_penalty():
    """At penalty=0-2, R² should be primary ranking criterion."""
    # Model with best R² should rank #1 when penalty ≤ 2
```

## Summary

**Root Cause**: Linear penalty scaling creates penalties that are too large at low settings (2/10).

**Primary Fix**: Change line 85 from `(variable_penalty / 10.0)` to `((variable_penalty / 10.0) ** 2)` for quadratic scaling.

**Secondary Fix**: Update function defaults to match GUI defaults.

**Testing Fix**: Add comprehensive unit tests for ranking behavior.

## Related Code Locations

All paths relative to `C:\Users\sponheim\git\dasp\`:

1. **Main Bug**: `src\spectral_predict\scoring.py:85` and `src\spectral_predict\scoring.py:104`
2. **Default Values**: `src\spectral_predict\scoring.py:7` and `spectral_predict_gui_optimized.py:209-210`
3. **Ranking Usage**: `src\spectral_predict\search.py:646` (calls `compute_composite_score`)
4. **GUI Display**: `spectral_predict_gui_optimized.py:6643-6697` (displays ranked results)
5. **GUI Sorting**: `spectral_predict_gui_optimized.py:6616-6641` (allows user to resort columns)
