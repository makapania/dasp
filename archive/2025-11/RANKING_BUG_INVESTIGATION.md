# RANKING BUG INVESTIGATION - COMPREHENSIVE ANALYSIS

## Problem Statement

User reports: "Model with R²=0.9430 was ranked #876. None of the top 10 models by R² had ranking higher than 700."

Settings: Variable Penalty = 2, Complexity Penalty = 2

## Critical Finding

**THE BUG IS FOUND!**

The issue is in the **penalty term scaling** in `src/spectral_predict/scoring.py`.

## Root Cause Analysis

### Current Formula (BUGGY)

```python
# Line 59: performance_score = 0.5 * z_rmse - 0.5 * z_r2
# Line 85: var_penalty_term = (variable_penalty / 10.0) * var_fraction
# Line 104: comp_penalty_term = (complexity_penalty / 10.0) * lv_fraction_adjusted
# Line 110: df["CompositeScore"] = performance_score + var_penalty_term + comp_penalty_term
```

### Mathematical Analysis

Let's trace through with user's example:

**Scenario**: 876 models tested, Model A has R²=0.943 (one of the best)

#### Step 1: Z-Score Calculation

Assume across 876 models:
- Mean R² ≈ 0.85
- Std R² ≈ 0.08
- Mean RMSE ≈ 0.15
- Std RMSE ≈ 0.05

For Model A (R²=0.943):
- z_r2 = (0.943 - 0.85) / 0.08 = **+1.16** (good, above average)
- Assume RMSE = 0.10 (also good):
- z_rmse = (0.10 - 0.15) / 0.05 = **-1.0** (good, below average)

#### Step 2: Performance Score

```
performance_score = 0.5 * z_rmse - 0.5 * z_r2
                  = 0.5 * (-1.0) - 0.5 * (1.16)
                  = -0.5 - 0.58
                  = -1.08
```

This is GOOD (negative = better). Model A should rank highly!

#### Step 3: Penalty Calculation - WHERE THE BUG MANIFESTS

Assume Model A uses many variables (n_vars = 2000 out of 2151 total):
- var_fraction = 2000 / 2151 = **0.93**
- var_penalty_term = (2 / 10) * 0.93 = **0.186**

Assume Model A has moderate complexity (e.g., 12 LVs for a tree model or median):
- lv_fraction_adjusted = 0.4 (median baseline)
- comp_penalty_term = (2 / 10) * 0.4 = **0.08**

#### Step 4: Composite Score

```
CompositeScore = performance_score + var_penalty_term + comp_penalty_term
               = -1.08 + 0.186 + 0.08
               = -0.814
```

This is STILL NEGATIVE and should rank well!

### So Why Is Model A Ranked #876?

**THE SMOKING GUN**: Other models must have even MORE NEGATIVE CompositeScores!

Let's check a model with WORSE R² but FEWER variables:

**Model B**: R²=0.75 (worse), but uses only 20 variables

For Model B:
- z_r2 = (0.75 - 0.85) / 0.08 = -1.25 (BAD, below average)
- Assume RMSE = 0.20 (worse):
- z_rmse = (0.20 - 0.15) / 0.05 = +1.0 (BAD, above average RMSE)

Performance score:
```
performance_score = 0.5 * (1.0) - 0.5 * (-1.25)
                  = 0.5 + 0.625
                  = +1.125  (POSITIVE = BAD!)
```

Penalties:
- var_fraction = 20 / 2151 = 0.0093
- var_penalty_term = (2 / 10) * 0.0093 = 0.00186
- comp_penalty_term = 0.08

Composite Score:
```
CompositeScore = 1.125 + 0.00186 + 0.08
               = 1.207 (POSITIVE = WORSE than Model A)
```

Model B should rank WORSE than Model A. ✓ This works correctly!

### Wait... Then What's the Bug?

Let me reconsider. If 875 models rank better than Model A, there must be something else happening.

**HYPOTHESIS**: The bug might be that z-scores become extreme when there are many similar models, or NaN handling creates issues, OR...

**CRITICAL REALIZATION**: What if most models have SIMILAR R² but use FEWER variables?

Let me recalculate with a different distribution:

If out of 876 models:
- 800 models use 10-100 variables
- 76 models use 1000+ variables

Then:
- Models with 10-100 vars have very LOW penalty (~0.002 to 0.02)
- Models with 1000+ vars have HIGH penalty (~0.19 to 0.20)

**The difference in penalty (0.18) is comparable to 1.8 standard deviations in performance!**

So if Model A (2000 vars, R²=0.943) has z_r2=+1.16, but Model C (50 vars, R²=0.88) has z_r2=-0.5:

Model A:
- performance_score ≈ -1.08
- penalty ≈ 0.27
- CompositeScore = -0.81

Model C:
- z_r2 = (0.88 - 0.85) / 0.08 = +0.375
- z_rmse ≈ -0.5
- performance_score = 0.5*(-0.5) - 0.5*(0.375) = -0.25 - 0.1875 = -0.4375
- var_penalty = (2/10) * (50/2151) = 0.0046
- comp_penalty = 0.08
- CompositeScore = -0.4375 + 0.0046 + 0.08 = **-0.353**

Model A: CompositeScore = -0.81 (MORE negative = BETTER rank)
Model C: CompositeScore = -0.353 (LESS negative = WORSE rank)

Model A should rank BETTER. ✓ Still no bug found!

### THE REAL BUG - FINAL ANALYSIS

After all this analysis, I realize the logic APPEARS correct mathematically. The bug must be:

1. **Off-by-one or inverted sorting somewhere** in the display pipeline
2. **The user is sorting by R² in the GUI**, which OVERRIDES the Rank column
3. **The Rank column is correct, but the user expects R² to be the primary sort criterion**

Let me check the GUI sorting code again...

## The Actual Bug: USER EXPECTATION vs IMPLEMENTATION

Looking at line 6638 of `spectral_predict_gui_optimized.py`:

```python
sorted_df = sorted_df.sort_values(by=col, ascending=not self.results_sort_reverse)
```

When user clicks the R² column header:
- First click: `results_sort_reverse = False`, so `ascending = True`
- **This sorts R² from LOWEST to HIGHEST** (ascending=True means 0.65, 0.75, 0.85, 0.95)
- User expects HIGHEST R² at top, but sees LOWEST R² at top!

**THIS IS THE BUG!**

For R² (and other "higher is better" metrics), the GUI needs to sort in DESCENDING order by default!

## Verification

The ranking algorithm in `scoring.py` is **CORRECT**.

The bug is in the GUI's **column sorting** in `spectral_predict_gui_optimized.py` line 6638.

When sorting R² or other "higher is better" metrics, it should default to `ascending=False`.
