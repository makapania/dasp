# Machine Learning Models in Spectral Predict

**Comprehensive Guide to ML Algorithms, Configuration, and Ensemble Methods**

---

## Table of Contents

1. [Overview](#overview)
2. [Tiered Configuration System](#tiered-configuration-system)
3. [Individual Model Documentation](#individual-model-documentation)
4. [Intelligent Ensemble Methods](#intelligent-ensemble-methods)
5. [Preprocessing Recommendations](#preprocessing-recommendations)
6. [Usage Examples](#usage-examples)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Comparison with Commercial Software](#comparison-with-commercial-software)

---

## Overview

Spectral Predict now includes **11 state-of-the-art machine learning algorithms** for spectral analysis, from traditional linear methods to modern gradient boosting. This provides capabilities that match or exceed commercial spectroscopy software packages.

### Available Algorithms

| Model | Type | Complexity | Best For |
|-------|------|------------|----------|
| **PLS** | Linear | Low | Linear relationships, interpretability |
| **Ridge** | Linear | Low | Multicollinearity, stable predictions |
| **Lasso** | Linear | Low | Sparse features, variable selection |
| **ElasticNet** | Linear | Low | Balance between Ridge and Lasso |
| **RandomForest** | Tree ensemble | Medium | Robust non-linear relationships |
| **MLP** | Neural network | High | Deep non-linearity, large datasets |
| **NeuralBoosted** | Hybrid boosting | Medium | Interpretable non-linearity |
| **SVR** | Kernel methods | Medium | Small-medium datasets, non-linear |
| **XGBoost** | Gradient boosting | Medium | High performance, feature importance |
| **LightGBM** | Gradient boosting | Medium | Fast training, large datasets |
| **CatBoost** | Gradient boosting | Medium | Robust to overfitting |

### Key Advantages

**Compared to commercial software:**

1. **More Algorithm Diversity**: 11 models vs. typical 3-5 in commercial packages
2. **Modern Gradient Boosting**: XGBoost, LightGBM, CatBoost (not in Unscrambler/JMP)
3. **Neural Boosted**: JMP-inspired interpretable boosting
4. **Intelligent Ensembles**: Region-aware weighted ensembles and mixture of experts
5. **Automated Hyperparameter Tuning**: 100+ configurations tested automatically
6. **Open Source**: Full transparency and customizability
7. **Python Ecosystem**: Easy integration with scikit-learn, pandas, numpy

---

## Tiered Configuration System

The tiered system balances **performance** with **computational efficiency**. Each tier provides optimized hyperparameter grids based on benchmarking across spectral datasets.

### Tier Overview

| Tier | Duration | Models | Configs | Use Case |
|------|----------|--------|---------|----------|
| **Quick** | 3-5 min | 3 | 6 | Rapid testing, preliminary analysis |
| **Standard** | 10-15 min | 4 | 29 | Daily analysis, most users |
| **Comprehensive** | 20-30 min | 7 | 81 | Thorough analysis, publications |
| **Experimental** | 45-90 min | 11 | 100-200+ | Method comparison, research |

### Quick Tier

**Duration:** 3-5 minutes
**Models:** PLS, Ridge, XGBoost
**Total Configurations:** 6

**When to use:**
- Quick data quality checks
- Preliminary analysis
- Testing preprocessing pipelines
- Time-constrained analysis

**Example:**
```python
from spectral_predict.models import get_model_grids

grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='quick'
)
# Returns grids for PLS (3 configs), Ridge (2), XGBoost (1)
```

### Standard Tier (DEFAULT)

**Duration:** 10-15 minutes
**Models:** PLS, Ridge, ElasticNet, XGBoost
**Total Configurations:** 29

**When to use:**
- Daily routine analysis
- Production workflows
- When good-enough results are sufficient
- Most spectroscopy applications

**Configuration breakdown:**
- PLS: 8 component values (2, 4, 6, 8, 10, 12, 16, 20)
- Ridge: 4 alpha values (0.01, 0.1, 1.0, 10.0)
- ElasticNet: 9 configs (3 alpha × 3 l1_ratio)
- XGBoost: 8 configs (2 n_estimators × 2 learning_rate × 2 max_depth)

**Example:**
```python
grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='standard'  # Default
)
```

### Comprehensive Tier

**Duration:** 20-30 minutes
**Models:** PLS, Ridge, ElasticNet, XGBoost, LightGBM, SVR, NeuralBoosted
**Total Configurations:** 81

**When to use:**
- Research publications
- Method development
- When optimal performance is critical
- Final model selection for deployment

**Configuration breakdown:**
- PLS: 12 configs (extended component range)
- Ridge: 5 configs (includes 0.001 alpha)
- ElasticNet: 20 configs (4 alpha × 5 l1_ratio)
- XGBoost: 27 configs (full 3×3×3 grid)
- LightGBM: 4 configs (optimized)
- SVR: 5 configs (RBF and linear kernels)
- NeuralBoosted: 8 configs (learning rate and activation variants)

**Example:**
```python
grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='comprehensive'
)
```

### Experimental Tier

**Duration:** 45-90 minutes
**Models:** All 11 models
**Total Configurations:** 100-200+

**When to use:**
- Exhaustive method comparison
- Exploring new datasets
- Research and benchmarking
- When computation time is not a constraint

**Includes all models:**
- All models from Comprehensive tier
- Plus: Lasso, RandomForest, MLP, CatBoost

**Example:**
```python
grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='experimental'
)
```

### Customizing Tiers

You can **customize any tier** by:

1. **Enabling/disabling specific models:**

```python
grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='standard',
    enabled_models=['PLS', 'XGBoost', 'NeuralBoosted']
)
```

2. **Overriding hyperparameters:**

```python
grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='standard',
    # Override XGBoost defaults
    n_estimators_list=[50, 100, 200, 500],
    learning_rates=[0.01, 0.05, 0.1, 0.2],
    # Override Ridge defaults
    ridge_alphas_list=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
)
```

3. **Creating custom tier combinations:**

```python
# Fast gradient boosting tier
grids = get_model_grids(
    task_type='regression',
    n_features=2000,
    tier='quick',  # Use quick defaults
    enabled_models=['XGBoost', 'LightGBM', 'CatBoost']  # But only boosting models
)
```

---

## Individual Model Documentation

### 1. Partial Least Squares (PLS)

**What it is:**
PLS regression finds latent components that maximize covariance between predictors (spectra) and response (target). It's the gold standard in spectroscopy.

**When to use:**
- Linear relationships between spectra and target
- High multicollinearity (typical in spectral data)
- When interpretability is important (VIP scores)
- As a baseline for comparison

**Strengths:**
- Handles more predictors than samples
- Robust to multicollinearity
- Excellent interpretability via VIP scores
- Fast training and prediction
- Well-established in spectroscopy

**Weaknesses:**
- Assumes linear relationships
- May underperform with complex non-linear patterns
- Performance limited by component count

**Hyperparameters tuned:**
- `n_components`: Number of latent components
  - Quick: [5, 10, 15]
  - Standard: [2, 4, 6, 8, 10, 12, 16, 20]
  - Comprehensive: [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50]

**Feature importance method:**
- **VIP (Variable Importance in Projection) scores**
- Measures each wavelength's contribution across all components
- VIP > 1.0 indicates above-average importance
- Directly interpretable for spectral bands

**Example configuration:**
```python
from spectral_predict.models import get_model

model = get_model('PLS', task_type='regression', n_components=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Get VIP scores
from spectral_predict.models import get_feature_importances
vip_scores = get_feature_importances(model, 'PLS', X_train, y_train)
```

---

### 2. Ridge Regression

**What it is:**
Linear regression with L2 regularization. Shrinks coefficients toward zero to reduce overfitting and handle multicollinearity.

**When to use:**
- Linear relationships with multicollinearity
- When all features may be relevant (no sparsity needed)
- Fast baseline model
- Stable predictions needed

**Strengths:**
- Fast training and prediction
- Handles multicollinearity well
- More stable than ordinary least squares
- Closed-form solution (no iterative optimization)

**Weaknesses:**
- Assumes linearity
- Doesn't perform feature selection (all features retained)
- May underperform PLS for spectral data

**Hyperparameters tuned:**
- `alpha`: Regularization strength
  - Quick: [0.1, 1.0]
  - Standard: [0.01, 0.1, 1.0, 10.0]
  - Comprehensive: [0.001, 0.01, 0.1, 1.0, 10.0]
  - Higher alpha = more regularization

**Feature importance method:**
- **Absolute coefficient values**
- Larger |coefficient| = more important wavelength
- Can be sensitive to feature scaling

---

### 3. Lasso Regression

**What it is:**
Linear regression with L1 regularization. Performs feature selection by driving some coefficients exactly to zero.

**When to use:**
- Sparse relationships (few wavelengths matter)
- Automatic feature selection needed
- When interpretability requires minimal features

**Strengths:**
- Automatic feature selection
- Interpretable (only relevant features have non-zero coefficients)
- Handles multicollinearity by selecting one from correlated group

**Weaknesses:**
- May arbitrarily select one from correlated features
- Can be unstable with highly correlated features
- ElasticNet often preferred for spectral data

**Hyperparameters tuned:**
- `alpha`: Regularization strength
  - Quick: [0.1]
  - Standard: [0.01, 0.1, 1.0]
  - Comprehensive: [0.001, 0.01, 0.1, 1.0]

**Feature importance method:**
- **Absolute coefficient values**
- Zero coefficients = feature not selected

**Note:** ElasticNet is typically preferred over pure Lasso for spectral data.

---

### 4. ElasticNet Regression

**What it is:**
Combines L1 (Lasso) and L2 (Ridge) regularization. Balances feature selection with stability.

**When to use:**
- Correlated features with potential sparsity
- When both regularization and feature selection are desired
- More stable alternative to Lasso
- Default choice over pure Lasso or Ridge

**Strengths:**
- More stable than Lasso with correlated features
- Performs feature selection like Lasso
- Handles multicollinearity like Ridge
- Best of both worlds for spectral data

**Weaknesses:**
- Two hyperparameters to tune (more complex than Ridge/Lasso)
- Still assumes linearity

**Hyperparameters tuned:**
- `alpha`: Overall regularization strength
  - Quick: [0.1, 1.0]
  - Standard: [0.01, 0.1, 1.0]
  - Comprehensive: [0.001, 0.01, 0.1, 1.0]
- `l1_ratio`: Balance between L1 and L2
  - Quick: [0.5] (balanced)
  - Standard: [0.3, 0.5, 0.7]
  - Comprehensive: [0.1, 0.3, 0.5, 0.7, 0.9]
  - 0 = pure Ridge, 1 = pure Lasso, 0.5 = balanced

**Feature importance method:**
- **Absolute coefficient values**
- More stable than Lasso for correlated features

---

### 5. Random Forest

**What it is:**
Ensemble of decision trees trained on bootstrap samples with random feature subsets. Averages predictions across trees.

**When to use:**
- Non-linear relationships
- Robust predictions needed (resistant to outliers)
- Large datasets (>100 samples)
- When training time is not critical

**Strengths:**
- Handles non-linearity naturally
- Built-in feature importance
- Robust to outliers and noise
- No feature scaling needed
- Minimal hyperparameter tuning

**Weaknesses:**
- Can be slow with high-dimensional spectral data
- May overfit small datasets
- Less interpretable than linear models
- Large memory footprint

**Hyperparameters tuned:**
- `n_estimators`: Number of trees
  - Quick: [200]
  - Standard: [200, 500]
  - Comprehensive: [100, 200, 500]
- `max_depth`: Maximum tree depth
  - Quick: [None] (unlimited)
  - Standard: [None, 30]
  - Comprehensive: [None, 15, 30]

**Feature importance method:**
- **Gini importance (built-in)**
- Based on total reduction in node impurity
- Naturally handles feature interactions
- Well-established for tree-based models

---

### 6. Multi-Layer Perceptron (MLP)

**What it is:**
Feed-forward neural network with hidden layers. Learns complex non-linear mappings through backpropagation.

**When to use:**
- Highly non-linear relationships
- Large datasets (>200 samples)
- When maximum predictive power is needed
- Sufficient computation resources available

**Strengths:**
- Universal function approximator
- Can learn very complex patterns
- State-of-the-art for some spectral tasks

**Weaknesses:**
- Requires more data (prone to overfitting on small datasets)
- Slow training (iterative optimization)
- Many hyperparameters to tune
- Less interpretable
- Sensitive to initialization and scaling

**Hyperparameters tuned:**
- `hidden_layer_sizes`: Network architecture
  - Quick: [(64,)]
  - Standard: [(64,), (128, 64)]
  - Comprehensive: [(64,), (128, 64)]
- `alpha`: L2 regularization
  - Quick: [1e-3]
  - Standard: [1e-3]
  - Comprehensive: [1e-4, 1e-3]
- `learning_rate_init`: Initial learning rate
  - Quick: [1e-3]
  - Standard: [1e-3]
  - Comprehensive: [1e-3, 1e-2]

**Feature importance method:**
- **Average absolute weight of first layer**
- Heuristic approximation
- Less reliable than other methods

**Note:** Requires careful tuning and sufficient data. Often NeuralBoosted is preferred for spectral data.

---

### 7. Neural Boosted Regression (NeuralBoosted)

**What it is:**
Gradient boosting with small neural networks as weak learners. Inspired by JMP's Neural Boosted methodology. Combines boosting's power with neural networks' flexibility.

**When to use:**
- Non-linear relationships with interpretability needed
- Medium-sized datasets (50-500 samples)
- When PLS is too simple but MLP is too complex
- Robust performance without extensive tuning

**Strengths:**
- Better non-linearity than PLS
- More interpretable than MLP
- Robust to overfitting (small weak learners)
- Feature importance via aggregated weights
- Early stopping prevents overtraining
- Good out-of-the-box performance

**Weaknesses:**
- Slower than PLS (iterative boosting)
- More complex than traditional methods
- Relatively new (less established in spectroscopy)

**Hyperparameters tuned:**
- `n_estimators`: Number of boosting rounds
  - Quick: [100]
  - Standard: [100]
  - Comprehensive: [100, 150]
- `learning_rate`: Shrinkage parameter
  - Quick: [0.3] (empirically optimal)
  - Standard: [0.1, 0.2]
  - Comprehensive: [0.1, 0.2, 0.3]
- `hidden_layer_size`: Nodes in weak learner
  - Quick: [3]
  - Standard: [3, 5]
  - Comprehensive: [3, 5]
  - Small values (1-5) maintain weak learner property
- `activation`: Activation function
  - Quick: ['tanh']
  - Standard: ['tanh', 'identity']
  - Comprehensive: ['tanh', 'identity']

**Feature importance method:**
- **Aggregated absolute weights across all weak learners**
- Similar to PLS VIP concept but for non-linear model
- More reliable than single MLP importance

**Best practices:**
- Keep `hidden_layer_size` small (3-5)
- Use `learning_rate` 0.1-0.3
- Enable `early_stopping=True` (default)
- Works well with standard preprocessing

---

### 8. Support Vector Regression (SVR)

**What it is:**
Kernel-based regression that maps data to high-dimensional space. Finds optimal hyperplane with margin.

**When to use:**
- Small to medium datasets (<500 samples)
- Non-linear relationships
- When robustness to outliers is needed

**Strengths:**
- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile kernel functions (RBF, linear, polynomial)
- Robust to outliers (epsilon-insensitive loss)

**Weaknesses:**
- Slow training on large datasets (>1000 samples)
- Sensitive to hyperparameter choices
- Harder to interpret than linear models
- Requires feature scaling

**Hyperparameters tuned:**
- `kernel`: Kernel function
  - Quick: ['rbf']
  - Standard: ['rbf', 'linear']
  - Comprehensive: ['rbf', 'linear']
- `C`: Regularization parameter
  - Quick: [1.0]
  - Standard: [1.0, 10.0]
  - Comprehensive: [0.1, 1.0, 10.0]
  - Higher C = less regularization
- `gamma`: RBF kernel parameter
  - Quick: ['scale']
  - Standard: ['scale']
  - Comprehensive: ['scale', 'auto']

**Feature importance method:**
- **Linear kernel:** Absolute coefficient values
- **RBF kernel:** Weighted sum of support vectors (approximation)
- Less direct than other methods

---

### 9. XGBoost

**What it is:**
Gradient boosting with decision trees. Highly optimized implementation with regularization and advanced features.

**When to use:**
- High predictive performance needed
- Feature importance analysis
- Structured/tabular data (excellent for spectral data)
- When you want state-of-the-art results

**Strengths:**
- Often best out-of-the-box performance
- Fast training with GPU support
- Built-in feature importance
- Handles missing values
- Built-in regularization prevents overfitting
- Widely used in competitions

**Weaknesses:**
- More hyperparameters to tune
- Can overfit small datasets
- Less interpretable than linear models
- Requires installation of xgboost package

**Hyperparameters tuned:**
- `n_estimators`: Number of boosting rounds
  - Quick: [100]
  - Standard: [100, 200]
  - Comprehensive: [50, 100, 200]
- `learning_rate`: Shrinkage rate
  - Quick: [0.1]
  - Standard: [0.05, 0.1]
  - Comprehensive: [0.05, 0.1, 0.2]
- `max_depth`: Maximum tree depth
  - Quick: [6]
  - Standard: [3, 6]
  - Comprehensive: [3, 6, 9]

**Feature importance method:**
- **Gain-based importance (built-in)**
- Total gain across all splits using each feature
- Accounts for feature interactions
- Reliable and widely used

---

### 10. LightGBM

**What it is:**
Gradient boosting framework by Microsoft. Uses histogram-based tree learning for speed and efficiency.

**When to use:**
- Large datasets (fast training)
- High-dimensional data
- When training speed is critical
- Similar use cases as XGBoost

**Strengths:**
- Faster training than XGBoost
- Lower memory usage
- Excellent for large datasets
- Built-in categorical feature support
- Good default hyperparameters

**Weaknesses:**
- Can overfit small datasets (<100 samples)
- Less robust to noise than XGBoost on very small data
- Requires lightgbm package

**Hyperparameters tuned:**
- `n_estimators`: Number of boosting rounds
  - Quick: [100]
  - Standard: [100, 200]
  - Comprehensive: [50, 100, 200]
- `learning_rate`: Shrinkage rate
  - Quick: [0.1]
  - Standard: [0.1]
  - Comprehensive: [0.05, 0.1, 0.2]
- `num_leaves`: Maximum leaves per tree
  - Quick: [31] (default)
  - Standard: [31, 50]
  - Comprehensive: [31, 50, 70]

**Feature importance method:**
- **Split-based importance (built-in)**
- Number of times feature is used in splits
- Similar concept to XGBoost

---

### 11. CatBoost

**What it is:**
Gradient boosting by Yandex. Handles categorical features natively and uses ordered boosting to reduce overfitting.

**When to use:**
- When overfitting is a concern
- Mixed numerical/categorical features
- Good general-purpose choice
- Alternative to XGBoost/LightGBM

**Strengths:**
- Robust to overfitting (ordered boosting)
- Good default hyperparameters
- Handles categorical features natively
- Often works well without tuning

**Weaknesses:**
- Slower training than LightGBM
- Fewer community resources than XGBoost
- Requires catboost package

**Hyperparameters tuned:**
- `iterations`: Number of boosting rounds
  - Quick: [100]
  - Standard: [100, 200]
  - Comprehensive: [50, 100, 200]
- `learning_rate`: Shrinkage rate
  - Quick: [0.1]
  - Standard: [0.1]
  - Comprehensive: [0.05, 0.1, 0.2]
- `depth`: Tree depth
  - Quick: [6]
  - Standard: [4, 6]
  - Comprehensive: [4, 6, 8]

**Feature importance method:**
- **PredictionValuesChange (built-in)**
- Change in prediction value when feature is removed
- Robust and reliable

---

## Intelligent Ensemble Methods

Beyond training individual models, Spectral Predict implements **advanced ensemble strategies** that intelligently combine predictions.

### Why Ensembles?

Single models have limitations:
- Some models excel at predicting low values, others high values
- Different models capture different patterns
- Simple averaging treats all models equally (suboptimal)

**Solution:** Intelligent ensembles that adapt to prediction context.

### Region-Based Analysis Concept

**Key insight:** Models often have **regional specialization** - they perform differently across the target range.

**Example with soil nitrogen prediction:**
- **PLS** excels at mid-range values (1-3% N)
- **Random Forest** better at low values (<1% N)
- **XGBoost** handles high values (>3% N) well

**Region-based analysis:**
1. Divide target space into regions (e.g., 5 quantiles)
2. Measure each model's performance in each region
3. Assign higher weights where models perform best
4. Adapt predictions based on predicted value's region

---

### 1. RegionAwareWeightedEnsemble

**What it does:**
Assigns dynamic weights to models based on regional performance. Each model gets a weight function that varies across the target space.

**How it works:**
1. Cross-validation determines each model's regional errors
2. Convert errors to weights (lower error = higher weight)
3. Normalize weights per region (sum to 1)
4. During prediction, apply region-specific weights

**When to use:**
- Multiple complementary models available
- Models show regional specialization
- Want to leverage each model's strengths
- Default ensemble choice

**Example:**
```python
from spectral_predict.ensemble import RegionAwareWeightedEnsemble

# Assume you have fitted models
models = [pls_model, rf_model, xgb_model]
model_names = ['PLS', 'RandomForest', 'XGBoost']

# Create and fit ensemble
ensemble = RegionAwareWeightedEnsemble(
    models=models,
    model_names=model_names,
    n_regions=5,  # Divide target space into 5 regions
    cv=5  # 5-fold CV for computing weights
)
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)

# Analyze model profiles
profiles = ensemble.get_model_profiles()
for model_name, profile in profiles.items():
    print(f"{model_name}:")
    print(f"  Type: {profile['specialization']}")
    print(f"  Best regions: {profile['best_regions']}")
    print(f"  Weights: {profile['weights']}")
```

**Output interpretation:**
- **Specialist models:** High variance in regional weights (excel in specific ranges)
- **Generalist models:** Low variance in weights (consistent across ranges)

---

### 2. MixtureOfExpertsEnsemble

**What it does:**
Assigns each region to its best-performing model ("expert"). Uses hard or soft gating.

**How it works:**
1. Identify best model for each region
2. Optionally use soft gating (weighted combination in each region)
3. During prediction, route to appropriate expert based on region

**When to use:**
- Clear expert models for different regions
- Want interpretable "specialist" assignments
- Models have distinct strengths
- Alternative to weighted ensemble

**Soft vs Hard Gating:**
- **Hard:** Use only the best model per region (sharp transitions)
- **Soft:** Weighted combination biased toward best model (smooth transitions)

**Example:**
```python
from spectral_predict.ensemble import MixtureOfExpertsEnsemble

ensemble = MixtureOfExpertsEnsemble(
    models=models,
    model_names=model_names,
    n_regions=5,
    soft_gating=True  # Use weighted combination (smoother)
)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

# See expert assignments
assignments = ensemble.get_expert_assignments()
for region, info in assignments.items():
    print(f"{region}:")
    print(f"  Primary expert: {info['primary_expert']}")
    print(f"  Weights: {info['weights']}")
```

---

### 3. StackingEnsemble

**What it does:**
Trains a meta-model on base model predictions. Traditional stacking with optional region-aware features.

**How it works:**
1. Generate cross-validated predictions from base models
2. Optionally add region features (one-hot encoded + predicted value)
3. Train meta-model (default: Ridge) on these features
4. Meta-model learns optimal combination

**When to use:**
- Want to learn optimal model combination
- Have sufficient data (>100 samples)
- Traditional ensemble approach preferred
- Comparison with region-aware methods

**Standard vs Region-Aware:**
- **Standard:** Meta-features = base model predictions only
- **Region-aware:** Adds region information (which region prediction falls in)

**Example:**
```python
from spectral_predict.ensemble import StackingEnsemble
from sklearn.linear_model import Ridge

# Standard stacking
ensemble = StackingEnsemble(
    models=models,
    model_names=model_names,
    meta_model=Ridge(alpha=1.0),
    region_aware=False,
    cv=5
)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

# Region-aware stacking
ensemble_region = StackingEnsemble(
    models=models,
    model_names=model_names,
    meta_model=Ridge(alpha=1.0),
    region_aware=True,
    n_regions=5,
    cv=5
)
ensemble_region.fit(X_train, y_train)
predictions_region = ensemble_region.predict(X_test)
```

---

### Ensemble Factory Function

**Convenient creation:**
```python
from spectral_predict.ensemble import create_ensemble

# Create any ensemble type
ensemble = create_ensemble(
    models=models,
    model_names=model_names,
    X=X_train,
    y=y_train,
    ensemble_type='region_weighted',  # or 'mixture_experts', 'stacking', etc.
    n_regions=5,
    cv=5
)

predictions = ensemble.predict(X_test)
```

**Available ensemble types:**
- `'simple_average'`: Baseline (equal weights)
- `'region_weighted'`: RegionAwareWeightedEnsemble
- `'mixture_experts'`: MixtureOfExpertsEnsemble
- `'stacking'`: Standard stacking
- `'region_stacking'`: Region-aware stacking

---

### Ensemble Visualization

**Analyze ensemble behavior:**
```python
from spectral_predict.ensemble_viz import (
    plot_regional_performance,
    plot_ensemble_weights,
    plot_model_specialization_profile,
    plot_prediction_comparison
)

# 1. Regional performance heatmap
predictions_dict = {
    'PLS': pls_predictions,
    'RandomForest': rf_predictions,
    'XGBoost': xgb_predictions
}

fig, axes = plot_regional_performance(
    analyzer=ensemble.analyzer_,
    y_true=y_test,
    predictions_dict=predictions_dict,
    metric='rmse',
    save_path='outputs/regional_performance.png'
)

# 2. Ensemble weights visualization
fig, axes = plot_ensemble_weights(
    ensemble=ensemble,
    save_path='outputs/ensemble_weights.png'
)

# 3. Model specialization profiles
fig, axes = plot_model_specialization_profile(
    ensemble=ensemble,
    save_path='outputs/specialization.png'
)

# 4. Prediction comparison
fig, axes = plot_prediction_comparison(
    y_true=y_test,
    predictions_dict=predictions_dict,
    ensemble_pred=ensemble.predict(X_test),
    save_path='outputs/predictions.png'
)
```

---

## Preprocessing Recommendations

### Savitzky-Golay Window Size Guidelines

Window size critically affects feature preservation vs. noise reduction.

**General guidelines:**

| Window Size | Effect | Best For |
|-------------|--------|----------|
| 5-11 | Minimal smoothing, preserves sharp features | Raman, sharp peaks |
| 15-25 | Balanced smoothing | NIR, most applications |
| 31+ | Heavy smoothing, may lose features | Very noisy data |

**By spectral type:**

**VIS-NIR (350-2500 nm):**
- **Recommended:** Window 7-19 (default in Standard tier)
- **Quick tier:** Window 11 (balanced)
- **Comprehensive tier:** Windows 7, 15, 25

```python
from spectral_predict.model_config import PREPROCESSING_DEFAULTS

visnir_windows = PREPROCESSING_DEFAULTS['savitzky_golay']['standard']['window_lengths']
# [7, 19]
```

**NIR only (1000-2500 nm):**
- **Recommended:** Window 11-25
- Broader peaks than VIS-NIR, can tolerate more smoothing

**Raman spectroscopy:**
- **Recommended:** Window 5-15
- Sharp peaks require minimal smoothing
- Window >15 may blur important peaks

**Polynomial order:**
- **Default:** 2nd order (quadratic)
- Rarely needs to be changed
- Higher orders (3-4) can introduce artifacts

### Preprocessing Methods by Tier

**Standard tier:**
- Methods: `['raw', 'snv', 'deriv1', 'deriv2']`
- SG windows: `[7, 19]`
- Derivative orders: `[1, 2]`

**Comprehensive tier:**
- Methods: `['raw', 'snv', 'deriv1', 'deriv2', 'snv_deriv1', 'snv_deriv2']`
- SG windows: `[7, 15, 25]`
- Derivative orders: `[1, 2]`

**Quick tier:**
- Methods: `['raw', 'snv']`
- SG windows: `[11]`
- Derivative orders: `[1]`

### Feature Scaling

**Models requiring scaling:**
- Ridge, Lasso, ElasticNet
- SVR
- MLP, NeuralBoosted

**Models NOT requiring scaling:**
- PLS (has built-in scaling)
- Tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)

**Recommendation:** Use StandardScaler for consistency:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Usage Examples

### Example 1: Basic Usage with Tier Selection

```python
from spectral_predict.models import get_model_grids
from sklearn.model_selection import cross_val_score

# Load your spectral data
X, y = load_spectral_data()  # Shape: (n_samples, n_wavelengths)

# Get model grids for standard tier
grids = get_model_grids(
    task_type='regression',
    n_features=X.shape[1],
    tier='standard'  # 10-15 minute analysis
)

# Evaluate each model configuration
results = []
for model_name, configs in grids.items():
    for model, params in configs:
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring='neg_root_mean_squared_error'
        )
        results.append({
            'model': model_name,
            'params': params,
            'rmse': -scores.mean(),
            'rmse_std': scores.std()
        })

# Find best model
best = min(results, key=lambda x: x['rmse'])
print(f"Best model: {best['model']}")
print(f"Best params: {best['params']}")
print(f"RMSE: {best['rmse']:.4f} ± {best['rmse_std']:.4f}")
```

### Example 2: Advanced Usage with Custom Model List

```python
from spectral_predict.models import get_model_grids

# Custom configuration: Only gradient boosting models
grids = get_model_grids(
    task_type='regression',
    n_features=X.shape[1],
    tier='comprehensive',  # Use comprehensive hyperparameters
    enabled_models=['XGBoost', 'LightGBM', 'CatBoost']  # But only these models
)

# Or: Custom hyperparameters for specific models
grids = get_model_grids(
    task_type='regression',
    n_features=X.shape[1],
    tier='standard',
    # Override XGBoost defaults with more extensive search
    n_estimators_list=[50, 100, 200, 500],
    learning_rates=[0.01, 0.05, 0.1, 0.2, 0.3]
)
```

### Example 3: Ensemble Usage

```python
from spectral_predict.models import get_model_grids, get_model
from spectral_predict.ensemble import create_ensemble
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train multiple models
pls = get_model('PLS', n_components=10)
pls.fit(X_train, y_train)

xgb = get_model('XGBoost')
xgb.fit(X_train, y_train)

nb = get_model('NeuralBoosted', learning_rate=0.2)
nb.fit(X_train, y_train)

models = [pls, xgb, nb]
model_names = ['PLS', 'XGBoost', 'NeuralBoosted']

# Create region-aware ensemble
ensemble = create_ensemble(
    models=models,
    model_names=model_names,
    X=X_train,
    y=y_train,
    ensemble_type='region_weighted',
    n_regions=5,
    cv=5
)

# Predict
ensemble_pred = ensemble.predict(X_test)

# Compare with individual models
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("Individual model performance:")
for model, name in zip(models, model_names):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"  {name}: RMSE={rmse:.4f}, R²={r2:.4f}")

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_r2 = r2_score(y_test, ensemble_pred)
print(f"\nEnsemble: RMSE={ensemble_rmse:.4f}, R²={ensemble_r2:.4f}")
```

### Example 4: CLI Usage

**Basic analysis with standard tier:**
```bash
spectral-predict \
  --spectra data/spectra.csv \
  --reference data/reference.csv \
  --id-column sample_id \
  --target nitrogen \
  --tier standard
```

**Quick analysis (testing):**
```bash
spectral-predict \
  --spectra data/spectra.csv \
  --reference data/reference.csv \
  --id-column sample_id \
  --target nitrogen \
  --tier quick
```

**Comprehensive analysis for publication:**
```bash
spectral-predict \
  --spectra data/spectra.csv \
  --reference data/reference.csv \
  --id-column sample_id \
  --target nitrogen \
  --tier comprehensive \
  --folds 10 \
  --outdir outputs/comprehensive_analysis
```

**Custom model selection:**
```bash
spectral-predict \
  --spectra data/spectra.csv \
  --reference data/reference.csv \
  --id-column sample_id \
  --target nitrogen \
  --tier standard \
  --models PLS XGBoost NeuralBoosted
```

---

## Performance Benchmarks

### Configuration Counts by Tier

| Tier | Models | Total Configs | Est. Time |
|------|--------|---------------|-----------|
| Quick | 3 | 6 | 3-5 min |
| Standard | 4 | 29 | 10-15 min |
| Comprehensive | 7 | 81 | 20-30 min |
| Experimental | 11 | 100-200+ | 45-90 min |

### Standard Tier Breakdown

| Model | Configs | Description |
|-------|---------|-------------|
| PLS | 8 | Components: 2, 4, 6, 8, 10, 12, 16, 20 |
| Ridge | 4 | Alpha: 0.01, 0.1, 1.0, 10.0 |
| ElasticNet | 9 | 3 alpha × 3 l1_ratio |
| XGBoost | 8 | 2 n_estimators × 2 learning_rate × 2 max_depth |
| **Total** | **29** | |

### Comprehensive Tier Breakdown

| Model | Configs | Description |
|-------|---------|-------------|
| PLS | 12 | Extended component range |
| Ridge | 5 | Extended alpha range |
| ElasticNet | 20 | 4 alpha × 5 l1_ratio |
| XGBoost | 27 | Full 3×3×3 grid |
| LightGBM | 4 | Optimized grid |
| SVR | 5 | RBF + linear kernels |
| NeuralBoosted | 8 | Learning rates + activations |
| **Total** | **81** | |

### Timing Estimates

**Based on typical spectral dataset:**
- 100 samples
- 2000 wavelengths
- 5-fold cross-validation
- Single CPU core

**Actual times vary with:**
- Dataset size (more samples = longer)
- Number of features
- Hardware (CPU/GPU)
- Cross-validation folds

**Tips for faster analysis:**
1. Use Quick tier for exploration
2. Reduce number of CV folds (3 instead of 5)
3. Use GPU for XGBoost/LightGBM
4. Enable parallel processing (`n_jobs=-1`)

---

## Comparison with Commercial Software

### Feature Matrix

| Feature | Spectral Predict | Unscrambler | JMP Pro | SIMCA | TQ Analyst |
|---------|------------------|-------------|---------|-------|------------|
| **Linear Models** |
| PLS | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ridge | ✅ | ❌ | ✅ | ❌ | ❌ |
| Lasso | ✅ | ❌ | ✅ | ❌ | ❌ |
| ElasticNet | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Non-Linear Models** |
| Random Forest | ✅ | ❌ | ✅ | ❌ | ❌ |
| Neural Networks | ✅ | ✅ (limited) | ✅ | ❌ | ❌ |
| Neural Boosted | ✅ | ❌ | ✅ | ❌ | ❌ |
| SVR | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Gradient Boosting** |
| XGBoost | ✅ | ❌ | ❌ | ❌ | ❌ |
| LightGBM | ✅ | ❌ | ❌ | ❌ | ❌ |
| CatBoost | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Ensemble Methods** |
| Region-aware | ✅ | ❌ | ❌ | ❌ | ❌ |
| Mixture of Experts | ✅ | ❌ | ❌ | ❌ | ❌ |
| Stacking | ✅ | ❌ | Limited | ❌ | ❌ |
| **Other Features** |
| Open Source | ✅ | ❌ | ❌ | ❌ | ❌ |
| Python Integration | ✅ | Limited | Limited | ❌ | ❌ |
| Tiered Configs | ✅ | ❌ | ❌ | ❌ | ❌ |
| Auto Hyperparameter Tuning | ✅ | Limited | Limited | ❌ | Limited |

### Unique Features in Spectral Predict

1. **Modern Gradient Boosting**: XGBoost, LightGBM, CatBoost not available in traditional spectroscopy software

2. **Intelligent Ensembles**: Region-aware methods that adapt to prediction context

3. **Tiered System**: Optimized configurations for different time budgets

4. **Full Transparency**: Open source, see exactly what each model does

5. **Python Ecosystem**: Easy integration with pandas, numpy, matplotlib, scikit-learn

6. **No License Costs**: Free to use, modify, and distribute

7. **Reproducible**: All code available, exact algorithm specifications

8. **Extensible**: Easy to add custom models or preprocessing methods

### When to Use Spectral Predict vs Commercial Software

**Use Spectral Predict when:**
- Want state-of-the-art gradient boosting models
- Need intelligent ensemble methods
- Prefer open-source transparency
- Want Python integration
- Budget is limited
- Need to customize algorithms
- Want to automate analysis pipelines

**Use Commercial Software when:**
- Need validated regulatory compliance (pharma, FDA)
- Require vendor support contracts
- Team already trained on commercial tools
- Need specialized features (e.g., SIMCA's MVDA)
- Prefer GUI over code
- Need integration with specific instruments

**Best of Both Worlds:**
- Use Spectral Predict for model development and research
- Export final models for production use
- Use commercial software for regulatory submissions
- Validate Spectral Predict models against commercial tools

---

## Summary

Spectral Predict provides **11 machine learning algorithms** organized into a **tiered configuration system** that balances performance with computational efficiency. From traditional PLS to modern gradient boosting, researchers now have access to state-of-the-art methods for spectral analysis.

**Key takeaways:**

1. **Start with Standard tier** (10-15 min) for most applications
2. **PLS remains excellent** for linear relationships and interpretability
3. **NeuralBoosted** offers interpretable non-linearity
4. **XGBoost/LightGBM** provide top performance for complex patterns
5. **Region-aware ensembles** can improve predictions by 5-15%
6. **Customize tiers** by enabling/disabling models or overriding hyperparameters
7. **Window size matters** - use 7-19 for VIS-NIR, 5-15 for Raman

**Recommended workflow:**
1. Quick tier for initial exploration (5 min)
2. Standard tier for production analysis (15 min)
3. Comprehensive tier for publications (30 min)
4. Ensemble top models for final predictions

**Questions or issues?**
- Documentation: `/docs/` directory
- Issues: GitHub repository
- CLI help: `spectral-predict --help`

---

**Last Updated:** November 10, 2025
**Version:** 2.0
**License:** MIT
