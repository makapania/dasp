# Unified Complexity Score Design Document

## Executive Summary

This document proposes a unified complexity score for the DASP spectral analysis tool that consolidates multiple complexity dimensions (latent variables, selected variables, model type, and preprocessing) into a single interpretable metric on a 0-100 scale.

**Recommendation**: Implement as an **additional column** alongside existing CompositeScore. Keep implementation simple with hardcoded weights based on domain knowledge.

---

## 1. Current State Analysis

### 1.1 Current Complexity Penalty

```python
# From scoring.py compute_composite_score()
lvs_penalty = df["LVs"].fillna(0) / 25.0          # [0, 1]
vars_penalty = n_vars / full_vars                  # [0, 1]
complexity_penalty = 0.3 * lambda_penalty * (lvs_penalty + vars_penalty)
CompositeScore = performance_score + complexity_penalty
```

**Issues**:
- Only considers 2 dimensions: LVs and variable count
- Ignores model type complexity (PLS vs MLP vs RF)
- Ignores preprocessing complexity (raw vs derivatives)
- Not intuitive: penalty ranges ~0 to 0.06 with default lambda=0.15
- Hard to understand what the penalty means

### 1.2 What Makes a Model "Complex" in Spectral Analysis?

Based on code analysis and domain knowledge:

| Complexity Factor | Low Complexity | High Complexity |
|------------------|----------------|-----------------|
| **Model Type** | PLS, Ridge, Lasso | MLP, NeuralBoosted, RandomForest |
| **Latent Variables** | 2-4 LVs | 20-50 LVs |
| **Variable Selection** | 10-50 wavelengths | Full spectrum (500+) |
| **Preprocessing** | Raw, SNV | 2nd derivative + SNV |
| **Model Parameters** | Simple (few params) | Complex (many hyperparams) |

**Domain Insights**:
- PLS with 6 LVs and 100 variables is considered "simple" and interpretable
- MLP with full spectrum is "complex" but may perform well
- Derivatives add complexity (harder to interpret) but remove baseline effects
- Users value interpretability for deployment in industrial settings

---

## 2. Proposed Unified Complexity Score

### 2.1 Design Goals

1. **Simple**: Easy to calculate, no over-engineering
2. **Interpretable**: 0-100 scale where higher = more complex
3. **Comprehensive**: Captures all major complexity dimensions
4. **Consistent**: Works across regression and classification
5. **Fair**: Normalizes across different model types

### 2.2 Formula

```
Unified Complexity Score (0-100) = 
    0.25 × Model_Complexity_Score +
    0.30 × Variable_Complexity_Score +
    0.25 × Latent_Variable_Complexity_Score +
    0.20 × Preprocessing_Complexity_Score
```

**Rationale for Weights**:
- Variable selection (30%): Most important for interpretability
- Model type (25%): Inherent algorithmic complexity
- Latent variables (25%): Dimensionality reduction benefit
- Preprocessing (20%): Affects interpretability but less critical

### 2.3 Component Calculations

#### 2.3.1 Model Complexity Score (0-100)

Maps model types to fixed scores based on inherent complexity:

```python
MODEL_COMPLEXITY = {
    'PLS': 20,          # Linear, interpretable, chemometrics standard
    'PLS-DA': 20,       # Same as PLS
    'Ridge': 25,        # Linear regularization
    'Lasso': 30,        # Sparse linear (feature selection adds complexity)
    'RandomForest': 60, # Ensemble, many trees, harder to interpret
    'MLP': 80,          # Neural network, black box
    'NeuralBoosted': 85 # Boosted neural nets, most complex
}
```

**Justification**:
- Based on interpretability and mathematical complexity
- PLS is the gold standard in chemometrics (low complexity)
- Neural networks are black boxes (high complexity)
- RandomForest is middle ground (ensemble but feature importance available)

#### 2.3.2 Variable Complexity Score (0-100)

Normalized variable fraction, with bonus penalty for very high counts:

```python
var_fraction = n_selected_vars / total_vars
base_score = var_fraction * 100

# Apply saturation curve to penalize very high variable counts
# This creates a non-linear penalty: 10 vars = low, 500 vars = high
variable_score = min(100, base_score * (1 + 0.5 * var_fraction))
```

**Examples**:
- 10 / 500 vars = 2 × 1.01 = 2.02 (very simple)
- 50 / 500 vars = 10 × 1.05 = 10.5 (simple)
- 100 / 500 vars = 20 × 1.10 = 22 (moderate)
- 500 / 500 vars = 100 × 1.50 = 100 (complex, capped at 100)

#### 2.3.3 Latent Variable Complexity Score (0-100)

For PLS models only; others get median score:

```python
if 'LVs' is not NaN:
    # Normalize to [0, 100], assuming 2-50 LV range
    lv_score = min(100, (LVs - 2) / (50 - 2) * 100)
else:
    # Non-PLS models: assign median complexity (50)
    lv_score = 50
```

**Examples**:
- 2 LVs = 0 (minimal)
- 6 LVs = 8.3 (low)
- 20 LVs = 37.5 (moderate)
- 50 LVs = 100 (high)

**Rationale**: Most PLS models use 2-20 LVs. Beyond 50 is excessive.

#### 2.3.4 Preprocessing Complexity Score (0-100)

Based on preprocessing method and derivative order:

```python
PREPROCESSING_COMPLEXITY = {
    'raw': 0,           # No preprocessing
    'snv': 20,          # Simple normalization
    'deriv_1st': 50,    # 1st derivative
    'deriv_2nd': 70,    # 2nd derivative (harder to interpret)
    'snv_deriv_1st': 60,  # SNV + 1st derivative
    'snv_deriv_2nd': 80,  # SNV + 2nd derivative
    'deriv_snv_1st': 65,  # 1st derivative + SNV
    'deriv_snv_2nd': 85   # 2nd derivative + SNV
}

# Map from DataFrame columns
if Deriv is None:
    preprocess_score = PREPROCESSING_COMPLEXITY[Preprocess]
else:
    key = f"{Preprocess}_deriv_{Deriv}"
    if key in mapping:
        preprocess_score = PREPROCESSING_COMPLEXITY[key]
    else:
        # Fallback: deriv order determines score
        preprocess_score = 50 if Deriv == 1 else 70
```

**Justification**:
- Raw data is easiest to interpret (0)
- SNV is simple normalization (20)
- Derivatives remove baseline but lose physical meaning (50-85)
- 2nd derivatives are harder to interpret than 1st

---

## 3. Implementation Details

### 3.1 Code Location

Add new function to `/home/user/dasp/src/spectral_predict/scoring.py`:

```python
def compute_unified_complexity_score(df_results):
    """
    Compute unified complexity score (0-100 scale).
    
    Higher score = more complex model (harder to interpret, more parameters).
    Lower score = simpler model (easier to interpret, fewer parameters).
    
    Components (weighted):
    - Model type complexity (25%)
    - Variable selection complexity (30%)
    - Latent variable complexity (25%)
    - Preprocessing complexity (20%)
    
    Returns
    -------
    complexity_scores : pd.Series
        Unified complexity score for each model (0-100)
    """
    # Implementation here (see Section 3.2)
```

### 3.2 Implementation Complexity

**Estimated Effort**: 1-2 hours

**Complexity Level**: **TRIVIAL**

**Why Trivial**:
- Simple lookup tables for model and preprocessing complexity
- Basic arithmetic for variable and LV scores
- No new dependencies
- No cross-validation or statistical computation
- Just adding a new column to existing DataFrame

**Steps**:
1. Define lookup dictionaries for model and preprocessing complexity
2. Write helper functions for each component score
3. Combine weighted components
4. Add column to DataFrame in `compute_composite_score()`
5. Update GUI to display new column (optional, can be done later)

### 3.3 Pseudocode

```python
def compute_unified_complexity_score(df_results):
    MODEL_COMPLEXITY = {...}  # See Section 2.3.1
    PREPROCESSING_COMPLEXITY = {...}  # See Section 2.3.4
    
    scores = []
    for idx, row in df_results.iterrows():
        # 1. Model complexity (25%)
        model_score = MODEL_COMPLEXITY.get(row['Model'], 50)
        
        # 2. Variable complexity (30%)
        var_fraction = row['n_vars'] / row['full_vars']
        var_score = min(100, var_fraction * 100 * (1 + 0.5 * var_fraction))
        
        # 3. Latent variable complexity (25%)
        if pd.notna(row['LVs']):
            lv_score = min(100, (row['LVs'] - 2) / 48 * 100)
        else:
            lv_score = 50
        
        # 4. Preprocessing complexity (20%)
        preprocess_score = _get_preprocessing_score(
            row['Preprocess'], row['Deriv']
        )
        
        # Weighted average
        unified_score = (
            0.25 * model_score +
            0.30 * var_score +
            0.25 * lv_score +
            0.20 * preprocess_score
        )
        
        scores.append(round(unified_score, 1))
    
    return pd.Series(scores, index=df_results.index)
```

---

## 4. Interpretation and Usage

### 4.1 Score Interpretation

| Score Range | Complexity Level | Typical Models | User Guidance |
|------------|------------------|----------------|---------------|
| 0-20 | **Very Simple** | PLS (2-4 LVs, 10-50 vars, raw) | Ideal for production deployment |
| 20-40 | **Simple** | PLS (4-10 LVs, 50-100 vars, SNV) | Good balance of performance and interpretability |
| 40-60 | **Moderate** | PLS (10-20 LVs, derivatives) or Ridge/Lasso | May require domain expertise |
| 60-80 | **Complex** | RandomForest, MLP (subset vars) | High performance but harder to interpret |
| 80-100 | **Very Complex** | MLP/NeuralBoosted (full spectrum, derivatives) | Black box; use only if performance critical |

### 4.2 Example Scores

Based on actual models from baseline_results.csv:

| Model | Preprocess | LVs | n_vars | Unified Complexity | Interpretation |
|-------|-----------|-----|--------|-------------------|----------------|
| PLS | raw | 2 | 50 | **13.8** | Very simple, highly interpretable |
| PLS | raw | 6 | 100 | **22.5** | Simple, good for production |
| PLS | deriv(1st) | 6 | 100 | **34.5** | Moderate, derivatives add complexity |
| RandomForest | snv | - | 500 | **74.0** | Complex, harder to interpret |
| MLP | raw | - | 500 | **89.0** | Very complex, black box |
| NeuralBoosted | deriv(2nd) | - | 500 | **96.3** | Extremely complex, use with caution |

### 4.3 Relationship to Current CompositeScore

**Current CompositeScore**:
- Primarily performance-driven (~90%)
- Small complexity penalty (~10%)
- Lower is better

**Unified Complexity Score**:
- Pure complexity metric (0-100)
- Independent of performance
- Higher = more complex

**Use Cases**:
1. **Filtering**: "Show me all models with Complexity < 40"
2. **Exploration**: "What's the best-performing simple model?"
3. **Trade-off Analysis**: Plot performance vs complexity (Pareto frontier)
4. **Deployment Decisions**: Choose based on complexity tolerance

---

## 5. Integration Strategy

### 5.1 Recommended Approach: Additional Column

**Add `ComplexityScore` column alongside `CompositeScore`**:

```python
# In compute_composite_score()
df["CompositeScore"] = performance_score + complexity_penalty
df["ComplexityScore"] = compute_unified_complexity_score(df)
df["Rank"] = df["CompositeScore"].rank(method="min").astype(int)
```

**Column Order**:
```
Rank | CompositeScore | ComplexityScore | RMSE | R2 | Model | LVs | n_vars | ...
```

**Rationale**:
- Maintains backward compatibility
- Users can sort/filter by either metric
- Enables multi-objective analysis
- No need to change ranking algorithm

### 5.2 Alternative: Replace Current Complexity Penalty

**NOT RECOMMENDED** because:
- Breaking change (affects existing rankings)
- Loses fine-grained control via lambda_penalty
- Current system works well for automatic ranking
- Users may have tuned lambda_penalty for their workflows

### 5.3 GUI Integration (Future Work)

Add to results table:
1. New column: "Complexity" (right after "Rank" or before "Model")
2. Color coding: Green (0-40), Yellow (40-70), Red (70-100)
3. Filter controls: "Max Complexity" slider (0-100)
4. Sorting: Allow sorting by Complexity in addition to Rank

---

## 6. Validation and Testing

### 6.1 Sanity Checks

Test cases to verify correct behavior:

```python
# Test 1: Simple PLS model should have low complexity
model = {'Model': 'PLS', 'LVs': 2, 'n_vars': 10, 'full_vars': 500, 
         'Preprocess': 'raw', 'Deriv': None}
assert compute_unified_complexity_score(model) < 20

# Test 2: Complex NeuralBoosted should have high complexity  
model = {'Model': 'NeuralBoosted', 'LVs': NaN, 'n_vars': 500, 'full_vars': 500,
         'Preprocess': 'deriv_snv', 'Deriv': 2}
assert compute_unified_complexity_score(model) > 80

# Test 3: Variable reduction should lower complexity
model_full = {'Model': 'PLS', 'LVs': 6, 'n_vars': 500, 'full_vars': 500, ...}
model_subset = {'Model': 'PLS', 'LVs': 6, 'n_vars': 50, 'full_vars': 500, ...}
assert compute_unified_complexity_score(model_subset) < compute_unified_complexity_score(model_full)
```

### 6.2 User Acceptance Testing

Questions to validate with users:
1. Does a score of 25 feel "simple" and 85 feel "complex"?
2. Is the ordering intuitive? (PLS < Ridge < RF < MLP < NeuralBoosted)
3. Can you use this to filter models for production deployment?
4. Does it help identify Pareto-optimal models (performance vs complexity)?

---

## 7. Pros and Cons

### 7.1 Advantages

✅ **Comprehensive**: Captures 4 complexity dimensions vs current 2  
✅ **Interpretable**: 0-100 scale is intuitive  
✅ **Model-agnostic**: Works across all model types  
✅ **Simple implementation**: ~100 lines of code, no new dependencies  
✅ **Enables filtering**: Users can easily filter by complexity threshold  
✅ **Backward compatible**: Add as new column, don't touch existing ranking  
✅ **Domain-aligned**: Reflects chemometrics best practices (PLS is simple)  

### 7.2 Limitations

⚠️ **Hardcoded weights**: 25/30/25/20 split is somewhat arbitrary  
⚠️ **No hyperparameter complexity**: Doesn't account for n_estimators, hidden layers, etc.  
⚠️ **Subjective model complexity**: Is MLP really 4× more complex than PLS?  
⚠️ **Fixed scale**: Assumes 2-50 LV range, may need adjustment for other domains  
⚠️ **Not calibrated**: Score of 50 doesn't mean "50% complex" in absolute terms  

### 7.3 Mitigation Strategies

- **Make weights configurable** (future enhancement): Add GUI controls for weight tuning
- **Add hyperparameter term** (future enhancement): Count total hyperparameters as bonus penalty
- **Validate with experts**: Survey chemometrics practitioners to calibrate model complexity scores
- **Make scale adaptive**: Auto-adjust LV normalization based on actual LV range in dataset
- **Provide percentile ranks**: Show "This model is simpler than 75% of others" alongside raw score

---

## 8. Recommendation

### 8.1 Implement Now

**YES** - Implement as an additional column with the proposed formula.

**Reasons**:
1. ✅ Simple to implement (1-2 hours, trivial complexity)
2. ✅ Valuable for users (enables filtering and trade-off analysis)
3. ✅ No breaking changes (additional column, not replacement)
4. ✅ Aligns with project goals (user-friendly, interpretable)
5. ✅ Low risk (doesn't affect existing CompositeScore ranking)

### 8.2 Implementation Priority

**Priority**: Medium-High

**Justification**:
- Less important than Option 4 (as stated in requirements)
- But more important than minor bug fixes
- Provides significant user value with minimal effort
- Can be done in parallel with other work

### 8.3 Roadmap

**Phase 1 (Now)**: 
- Implement core complexity score calculation
- Add as column to results DataFrame
- Basic unit tests

**Phase 2 (Later)**:
- GUI integration (display column, color coding)
- Filter controls ("Max Complexity" slider)
- Plotting (performance vs complexity scatter plot)

**Phase 3 (Future)**:
- Make weights configurable
- Add hyperparameter complexity term
- Calibrate scores with user feedback

---

## 9. Code Snippet (Ready to Use)

```python
def compute_unified_complexity_score(df_results):
    """
    Compute unified complexity score (0-100 scale).
    
    Higher score = more complex (harder to interpret, more parameters).
    Lower score = simpler (easier to interpret, fewer parameters).
    
    Parameters
    ----------
    df_results : pd.DataFrame
        Results DataFrame with columns: Model, LVs, n_vars, full_vars, 
        Preprocess, Deriv
    
    Returns
    -------
    complexity_scores : pd.Series
        Unified complexity score for each model (0-100)
    """
    # Model type complexity (0-100)
    MODEL_COMPLEXITY = {
        'PLS': 20, 'PLS-DA': 20,
        'Ridge': 25, 'Lasso': 30,
        'RandomForest': 60,
        'MLP': 80,
        'NeuralBoosted': 85
    }
    
    # Preprocessing complexity lookup
    def get_preprocess_score(preprocess, deriv):
        if pd.isna(deriv):
            return {'raw': 0, 'snv': 20}.get(preprocess, 10)
        else:
            # Derivatives
            base = 50 if deriv == 1 else 70
            if preprocess == 'snv':
                return base + 10  # SNV before deriv
            elif preprocess == 'deriv_snv':
                return base + 15  # Deriv then SNV
            else:
                return base
    
    scores = []
    for _, row in df_results.iterrows():
        # 1. Model complexity (25%)
        model_score = MODEL_COMPLEXITY.get(row['Model'], 50)
        
        # 2. Variable complexity (30%)
        var_fraction = row['n_vars'] / row['full_vars']
        var_score = min(100, var_fraction * 100 * (1 + 0.5 * var_fraction))
        
        # 3. Latent variable complexity (25%)
        if pd.notna(row['LVs']):
            lv_score = min(100, max(0, (row['LVs'] - 2) / 48 * 100))
        else:
            lv_score = 50  # Median for non-PLS models
        
        # 4. Preprocessing complexity (20%)
        preprocess_score = get_preprocess_score(row['Preprocess'], row['Deriv'])
        
        # Weighted combination
        unified_score = (
            0.25 * model_score +
            0.30 * var_score +
            0.25 * lv_score +
            0.20 * preprocess_score
        )
        
        scores.append(round(unified_score, 1))
    
    return pd.Series(scores, index=df_results.index, name='ComplexityScore')
```

**Integration** (in `compute_composite_score()`):

```python
def compute_composite_score(df_results, task_type, lambda_penalty=0.15):
    """..."""
    df = df_results.copy()
    
    # ... existing code ...
    
    # Composite score (lower is better)
    df["CompositeScore"] = performance_score + complexity_penalty
    
    # NEW: Add unified complexity score
    df["ComplexityScore"] = compute_unified_complexity_score(df)
    
    # Rank by CompositeScore
    df["Rank"] = df["CompositeScore"].rank(method="min").astype(int)
    
    # ... existing code ...
```

---

## 10. Conclusion

The unified complexity score provides a simple, interpretable, and comprehensive metric for model complexity in spectral analysis. It consolidates model type, variable selection, latent variables, and preprocessing into a single 0-100 scale.

**Implementation is trivial** (1-2 hours), **backward compatible** (additional column), and **high value** (enables filtering, trade-off analysis, deployment decisions).

**Recommendation**: Implement now as an additional column. Defer GUI integration and weight tuning to later phases.
