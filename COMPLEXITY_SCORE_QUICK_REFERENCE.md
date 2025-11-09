# Unified Complexity Score - Quick Reference

## TL;DR

**What**: Single 0-100 score combining model type, variable count, LVs, and preprocessing  
**Why**: Current scoring only considers LVs + variables, ignores model/preprocessing complexity  
**How**: Weighted average: 25% model + 30% variables + 25% LVs + 20% preprocessing  
**Implementation**: TRIVIAL (~1-2 hours, ~100 lines of code)  
**Recommendation**: YES - Add as new column, don't replace existing CompositeScore  

---

## Formula (One-Liner)

```python
Complexity = 0.25×Model + 0.30×Variables + 0.25×LVs + 0.20×Preprocessing
```

---

## Component Scores (0-100)

### 1. Model Type (25% weight)

| Model | Score | Why |
|-------|-------|-----|
| PLS / PLS-DA | 20 | Linear, interpretable, chemometrics standard |
| Ridge | 25 | Linear with regularization |
| Lasso | 30 | Sparse linear (automatic feature selection) |
| RandomForest | 60 | Ensemble, harder to interpret |
| MLP | 80 | Neural network, black box |
| NeuralBoosted | 85 | Boosted neural nets, most complex |

### 2. Variable Complexity (30% weight)

```python
var_fraction = n_vars / total_vars
score = min(100, var_fraction * 100 * (1 + 0.5 * var_fraction))
```

| Variables | Fraction | Score | Interpretation |
|-----------|----------|-------|----------------|
| 10 / 500 | 2% | 2.0 | Very sparse, highly interpretable |
| 50 / 500 | 10% | 10.5 | Sparse, good for production |
| 100 / 500 | 20% | 22.0 | Moderate sparsity |
| 500 / 500 | 100% | 100 | Full spectrum, complex |

### 3. Latent Variables (25% weight)

```python
if LVs exists:
    score = min(100, (LVs - 2) / 48 * 100)
else:
    score = 50  # Median for non-PLS
```

| LVs | Score | Typical Use |
|-----|-------|-------------|
| 2 | 0 | Minimal dimensionality reduction |
| 6 | 8.3 | Standard PLS (sweet spot) |
| 20 | 37.5 | Higher dimensionality |
| 50+ | 100 | Excessive (usually overfitting) |

### 4. Preprocessing (20% weight)

| Method | Score | Why |
|--------|-------|-----|
| raw | 0 | Direct spectral data, easiest to interpret |
| snv | 20 | Simple normalization |
| deriv (1st) | 50 | Removes baseline, loses physical meaning |
| snv → deriv (1st) | 60 | Combined normalization + derivative |
| deriv (2nd) | 70 | Higher derivative, harder to interpret |
| deriv → snv (2nd) | 85 | Most complex preprocessing |

---

## Example Calculations

### Example 1: Simple PLS Model
```
Model: PLS, LVs: 6, n_vars: 50/500, Preprocess: raw
→ Model: 20
→ Variables: (50/500) × 100 × 1.05 = 10.5
→ LVs: (6-2)/48 × 100 = 8.3
→ Preprocessing: 0

Complexity = 0.25×20 + 0.30×10.5 + 0.25×8.3 + 0.20×0
           = 5.0 + 3.15 + 2.08 + 0.0
           = 10.2 ✓ (Very Simple)
```

### Example 2: Moderate PLS with Derivatives
```
Model: PLS, LVs: 6, n_vars: 100/500, Preprocess: deriv(1st)
→ Model: 20
→ Variables: (100/500) × 100 × 1.10 = 22.0
→ LVs: 8.3
→ Preprocessing: 50

Complexity = 0.25×20 + 0.30×22 + 0.25×8.3 + 0.20×50
           = 5.0 + 6.6 + 2.08 + 10.0
           = 23.7 ✓ (Simple-Moderate)
```

### Example 3: Complex Neural Network
```
Model: NeuralBoosted, n_vars: 500/500, Preprocess: deriv_snv(2nd)
→ Model: 85
→ Variables: 100 × 1.50 = 100 (capped)
→ LVs: 50 (default for non-PLS)
→ Preprocessing: 85

Complexity = 0.25×85 + 0.30×100 + 0.25×50 + 0.20×85
           = 21.25 + 30 + 12.5 + 17
           = 80.8 ✓ (Very Complex)
```

---

## Interpretation Guide

| Score | Level | Description | Use When |
|-------|-------|-------------|----------|
| 0-20 | **Very Simple** | PLS with few LVs and variables, raw data | Production deployment, regulatory requirements |
| 20-40 | **Simple** | PLS with moderate parameters, maybe SNV | Good balance, easy to explain to stakeholders |
| 40-60 | **Moderate** | PLS with derivatives or Ridge/Lasso | Domain expertise available, R&D setting |
| 60-80 | **Complex** | RandomForest or MLP with subset | Performance critical, interpretability less important |
| 80-100 | **Very Complex** | Neural networks with full spectrum | Black box acceptable, only need predictions |

---

## Integration

### Code Changes (in scoring.py)

```python
# Add to compute_composite_score() after line 96:
df["ComplexityScore"] = compute_unified_complexity_score(df)
```

### Results Table Column Order

**Before**:
```
Rank | RMSE | R2 | Model | LVs | n_vars | Preprocess | ...
```

**After**:
```
Rank | ComplexityScore | RMSE | R2 | Model | LVs | n_vars | Preprocess | ...
```

### No Breaking Changes

- CompositeScore (performance + complexity penalty): **unchanged**
- Rank: **unchanged** (still based on CompositeScore)
- ComplexityScore: **new column** for user reference

---

## Use Cases

### 1. Filter for Deployable Models
```python
# Show only simple models suitable for production
df[df['ComplexityScore'] < 40]
```

### 2. Find Best Simple Model
```python
# Sort by complexity, find top performer in simple range
df[df['ComplexityScore'] < 30].sort_values('Rank').head(1)
```

### 3. Performance vs Complexity Trade-off
```python
# Plot Pareto frontier
plt.scatter(df['ComplexityScore'], df['RMSE'])
plt.xlabel('Complexity (higher = more complex)')
plt.ylabel('RMSE (lower = better)')
```

### 4. Compare Model Families
```python
# Compare PLS complexity across preprocessing methods
df[df['Model'] == 'PLS'].groupby('Preprocess')['ComplexityScore'].mean()
```

---

## Pros & Cons

### ✅ Advantages
- Comprehensive (4 dimensions vs current 2)
- Intuitive (0-100 scale, higher = more complex)
- Trivial to implement (~100 lines)
- Backward compatible (new column)
- Enables filtering and trade-off analysis

### ⚠️ Limitations
- Hardcoded weights (25/30/25/20)
- Doesn't count hyperparameters (n_estimators, hidden layers)
- Model complexity scores somewhat subjective
- Fixed 2-50 LV normalization range

### 🔧 Future Enhancements
- Make weights configurable in GUI
- Add hyperparameter counting
- Adaptive LV normalization
- Percentile rankings ("simpler than 75% of models")

---

## Decision Matrix

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Implementation Effort** | ⭐⭐⭐⭐⭐ | Trivial, 1-2 hours |
| **User Value** | ⭐⭐⭐⭐ | High, enables filtering and trade-offs |
| **Risk** | ⭐⭐⭐⭐⭐ | Very low, no breaking changes |
| **Interpretability** | ⭐⭐⭐⭐ | 0-100 scale is intuitive |
| **Comprehensiveness** | ⭐⭐⭐⭐ | Covers 4 major dimensions |

**Overall**: ⭐⭐⭐⭐ (Highly Recommended)

---

## Final Recommendation

**✅ IMPLEMENT NOW**

Add `ComplexityScore` column alongside `CompositeScore` using proposed formula.

**Phase 1 (Now)**: Core implementation + unit tests (~1-2 hours)  
**Phase 2 (Later)**: GUI integration (display, color coding, filters)  
**Phase 3 (Future)**: Configurable weights, hyperparameter counting  
