# Part II: Running Analyses

This section provides comprehensive documentation for configuring and executing spectral analyses in Spectral Predict V1. You will learn how to set up analysis parameters, monitor progress, and interpret results.

---

## Chapter 4: Analysis Configuration Tab

The Analysis Configuration tab (Tab 4) is the central control panel for all analysis settings. It is organized into five sub-tabs to help you navigate the many configuration options efficiently.

```
+------------------------------------------------------------------+
|  Analysis Configuration                                           |
|  +--------------------------------------------------------------+|
|  | Basic Settings | Variable Selection | Model Config |          ||
|  |                | Ensemble Methods   | Validation   |          ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  [Run Analysis Button - Available on all sub-tabs]               |
|                                                                   |
+------------------------------------------------------------------+
```

### 4.1 Basic Settings Sub-Tab

The Basic Settings sub-tab contains fundamental configuration options that affect all analyses.

#### 4.1.1 Target Variable Selection

The target variable dropdown allows you to select which column from your reference data should be used as the prediction target.

| Setting | Description |
|---------|-------------|
| Target Variable | The column containing values to predict (e.g., protein content, moisture, species class) |
| Auto-Detection | If only one numeric column exists, it is automatically selected |

**Important**: For classification tasks, ensure your target variable contains categorical values (text labels or discrete integers). For regression tasks, use continuous numeric values.

#### 4.1.2 Cross-Validation Configuration

Cross-validation (CV) is the primary method for evaluating model performance. The CV folds setting determines how many subsets your data is split into for training and validation.

| CV Folds | Recommended For | Description |
|----------|-----------------|-------------|
| 3 | Small datasets (<30 samples) | Minimal but fast |
| 5 | Medium datasets (30-100 samples) | Standard default |
| 10 | Large datasets (>100 samples) | More robust estimates |
| Leave-One-Out | Very small datasets (<20 samples) | Each sample used as test once |

**Guidelines for CV Fold Selection**:
- 50+ samples: Use 5-fold cross-validation
- 100+ samples: Use 10-fold cross-validation
- <50 samples: Consider Leave-One-Out (LOO)

#### 4.1.3 Penalty Settings

Spectral Predict uses a composite scoring system that can incorporate penalties for model complexity and variable count. Both penalties use a 0-10 scale.

**Variable Count Penalty (0-10)**

| Value | Effect |
|-------|--------|
| 0 | Only performance (R^2/Accuracy) matters - no penalty for using many wavelengths |
| 1-3 | Light penalty - exploration-friendly, slightly favors parsimonious models |
| 4-6 | Moderate penalty - balances performance with simplicity |
| 7-10 | Strong penalty - strongly prefers models using fewer wavelengths |

**Model Complexity Penalty (0-10)**

| Value | Effect |
|-------|--------|
| 0 | Only performance matters - no penalty for complex models |
| 1-3 | Light penalty - allows complex models with slight preference for simpler ones |
| 4-6 | Moderate penalty - balances performance with interpretability |
| 7-10 | Strong penalty - strongly prefers simpler models (fewer latent variables, shallower trees) |

**Scoring Formula**:
```
CompositeScore = PerformanceScore + VariablePenaltyTerm + ComplexityPenaltyTerm

Where:
- PerformanceScore (penalties=0): -R2cv (lower is better)
- PerformanceScore (penalties>0): 0.5*z(RMSEcv) - 0.5*z(R2cv)
- VariablePenaltyTerm = (penalty/10)^2 * (n_vars/full_vars)
- ComplexityPenaltyTerm = (penalty/10)^2 * complexity_fraction
```

#### 4.1.4 Wavelength Restriction

You can restrict the wavelength range used for analysis without affecting your imported data.

| Option | Description |
|--------|-------------|
| Enable Restriction | Checkbox to activate wavelength filtering |
| Analysis Range | Min and max wavelength values (e.g., 800-2500 nm) |
| Custom Regions | Specify multiple regions (e.g., "2000-2100, 2200-2300") |

**Preset Buttons**:
- UV (10-400 nm)
- VIS (400-800 nm)
- NIR (800-2500 nm)
- MIR (2500-25000 nm)

#### 4.1.5 Preprocessing Methods

Select which preprocessing transformations to test during analysis.

| Method | Description | Best For |
|--------|-------------|----------|
| Raw | No preprocessing | Baseline comparison |
| SNV | Standard Normal Variate | Scatter correction, solid samples |
| SG1 (1st derivative) | First derivative with Savitzky-Golay smoothing | Baseline offset removal |
| SG2 (2nd derivative) | Second derivative with Savitzky-Golay smoothing | Peak enhancement, overlapping peaks |
| SG3 (3rd derivative) | Third derivative | Complex spectral features |
| SG4 (4th derivative) | Fourth derivative | Rarely needed |
| deriv_snv | Derivative then SNV | Combined baseline issues |

**Derivative Window Sizes**:

| Window Size | Characteristics |
|-------------|-----------------|
| 7 | Preserves sharp features, more noise |
| 11 | Balance of smoothing and feature preservation |
| 17 | Standard default - good for most spectral data |
| 23 | More smoothing |
| 31 | Heavy smoothing |

**Tip**: Select Window=17 as the default starting point. Enable multiple window sizes only for comprehensive analyses.

#### 4.1.6 Pre-Processing Steps (Optional)

These optional steps are applied before the main transformations:

**Baseline Correction**:
- Polynomial: Fits and removes polynomial baseline (degree 1-5)
- AsLS: Asymmetric Least Squares baseline correction

**Smoothing**:
- Savitzky-Golay smoothing applied before derivative/SNV transforms
- Configure window size (7-25) and polynomial order (2-4)

#### 4.1.7 Smart Preprocessing Discovery

Enable automated preprocessing optimization using NSGA-II-style intelligence.

| Setting | Description | Default |
|---------|-------------|---------|
| Enable Discovery | Activates exhaustive preprocessing testing | Off |
| Importance Method | cars_tree, model_specific, lightgbm, or vip | model_specific |
| Top Configs | Number of preprocessing configurations to pass to grid search | 7 |

**What It Tests**: 14 preprocessing types x 6 window sizes = 84 combinations
**Runtime**: Approximately 3-4 minutes

---

### 4.2 Variable Selection Sub-Tab

The Variable Selection sub-tab configures which wavelengths to include in your models.

#### 4.2.1 Subset Analysis

**Top-N Variable Analysis**:
Test models using only the N most important wavelengths.

| N Value | Description |
|---------|-------------|
| N=10 | Minimal feature set |
| N=20 | Small subset |
| N=50 | Moderate subset |
| N=100 | Standard subset |
| N=250 | Large subset |
| N=500 | Very large subset |
| N=1000 | Nearly full spectrum |

**Spectral Region Analysis**:
Test models using auto-detected spectral regions.

| Depth | Regions Tested |
|-------|----------------|
| Shallow | 5 regions |
| Medium | 10 regions |
| Deep | 15 regions |
| Thorough | 20 regions |

**Advanced Region Options**:
- Test all regions individually (default tests approximately half)
- Test pairwise combinations (e.g., 45 pairs for top 10 regions)

#### 4.2.2 Variable Selection Methods

Select one or more sophisticated wavelength selection techniques:

| Method | Speed | Best For |
|--------|-------|----------|
| Feature Importance | Fast | Quick ranking, interpretable results |
| SPA (Successive Projections) | Fast | Minimizing collinearity |
| UVE (Uninformative Variable Elimination) | Fast | Filtering noisy variables |
| UVE-SPA Hybrid | Fast | Combined noise + collinearity reduction |
| iPLS (Interval PLS) | Medium | Region-based interpretation |
| Forward iPLS | Medium | Interval combination |
| Backward iPLS | Medium | Interval elimination |
| CARS (PLS-based) | Slow | Best for linear models |
| CARS-Tree | Slow | Best for tree models (RF, LightGBM, XGBoost) |
| VCPA-IRIV | Slowest | Best compact subset |
| GA (Genetic Algorithm) | Slowest | Comprehensive evolutionary search |

**Method Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| UVE Cutoff | 1.0 | Threshold multiplier (0.7-1.5) |
| UVE Components | Auto | Number of PLS components |
| iPLS Intervals | 20 | Number of spectral intervals |
| Fwd iPLS Max Combine | 5 | Maximum intervals to combine |
| GA Population | 64 | Genetic algorithm population size |
| GA Generations | 100 | Number of evolution generations |
| GA Runs | 5 | Independent runs for stability |

---

### 4.3 Model Configuration Sub-Tab

The Model Configuration sub-tab controls which machine learning models to test and their hyperparameters.

#### 4.3.1 Model Tiers

Select a preset tier to automatically configure model selections:

| Tier | Models Included | Recommended For |
|------|-----------------|-----------------|
| Quick | PLS, RandomForest | Rapid testing, daily QC, preliminary analysis |
| Standard | PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM | Most users, routine work, daily analysis |
| Comprehensive | Standard + XGBoost, CatBoost, NeuralBoosted | Thorough analysis, research, publications |
| Experimental | All models including SVR, MLP | Exploration, method comparison, no time constraints |
| Custom | User-selected | When you need specific model combinations |

**Classification Tiers**:

| Tier | Classification Models |
|------|----------------------|
| Quick | PLS-DA, RandomForest |
| Standard | PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost |
| Comprehensive | Standard + SVM, MLP, NeuralBoosted |
| Experimental | All classification models |

#### 4.3.2 Optimization Method

Choose how hyperparameters are searched:

| Method | Description | When to Use |
|--------|-------------|-------------|
| Grid Search | Tests all parameter combinations exhaustively | When thoroughness matters more than speed |
| Bayesian Optimization | Intelligent sampling of parameter space | Recommended for most analyses |
| NSGA-II Multi-Objective | Pareto optimization for error, parsimony, and complexity | When you need trade-off exploration |

**Bayesian Optimization Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Trials | 500 | Number of optimization iterations |

**NSGA-II Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population | 60 | Size of each generation |
| Generations | 120 | Number of evolutionary cycles |
| Selection | Min Error | How to select final solution (Min Error, Balanced, Knee Point) |
| Mode | NSGA-II w/Guidance | Standard NSGA-II or guided variant |

#### 4.3.3 Available Models

**Core Models (Linear)**:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| PLS | Partial Least Squares | n_components (latent variables), max_iter, tol |
| PLS-DA | PLS Discriminant Analysis (classification) | n_components, C (LogisticRegression) |
| Ridge | L2 regularized linear regression | alpha (regularization strength), solver, tol |
| Lasso | L1 regularized linear regression (sparse) | alpha, selection, tol |
| ElasticNet | Combined L1+L2 regularization | alpha, l1_ratio, selection, tol |

**Tree-Based Models**:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| RandomForest | Ensemble of decision trees | n_estimators, max_depth, min_samples_split |
| XGBoost | Extreme Gradient Boosting | n_estimators, learning_rate, max_depth, subsample |
| LightGBM | Light Gradient Boosting Machine | n_estimators, learning_rate, num_leaves, max_depth |
| CatBoost | Categorical Boosting | iterations, learning_rate, depth, l2_leaf_reg |

**Other Models**:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| MLP | Multi-Layer Perceptron (neural network) | hidden_layer_sizes, alpha, learning_rate_init |
| SVR/SVM | Support Vector Regression/Machine | kernel, C, gamma, epsilon |
| NeuralBoosted | Gradient boosting with neural networks | n_estimators, learning_rate, hidden_layer_size |

#### 4.3.4 Advanced Model Options

Each model type has collapsible sections for detailed hyperparameter configuration:

**Example: PLS Hyperparameters**
- Max Latent Variables: 2-100 (default: 8)
- Max Iterations: 500, 1000, or custom
- Tolerance: 1e-7, 1e-6 (default), 1e-5

**Example: Random Forest Hyperparameters**
- Number of Trees: 100, 200, 500 (default), custom
- Maximum Depth: None (unlimited), 30, custom
- Min Samples Split: 2, 5, 10, 20
- Max Features: sqrt (default), log2, None (all)
- Bootstrap: True (default), False

---

### 4.4 Ensemble Methods Sub-Tab

Ensemble methods combine predictions from multiple models to improve overall performance. This feature is available for regression tasks only.

#### 4.4.1 Enabling Ensembles

Check "Enable Ensemble Methods" to activate ensemble training after the main analysis completes.

#### 4.4.2 Available Ensemble Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Simple Average | Equal weight to all models | Baseline ensemble, stable predictions |
| Region-Aware Weighted | Dynamic weights based on prediction region | When models excel in different Y-value ranges |
| Mixture of Experts | Selects best model per region | Highly specialized model combination |
| Stacking Ensemble | Meta-learner combines predictions | Complex prediction scenarios |
| Stacking + Region Features | Stacking with region-aware meta-features | Advanced region-dependent optimization |

#### 4.4.3 Ensemble Configuration

| Setting | Options | Description |
|---------|---------|-------------|
| Number of Regions | 3, 5 (default), 7, 10 | How many Y-value ranges to divide data into |
| Top N Models | 3-20 (default: 10) | Number of models to include from ranked list |
| Models per Quartile | 1-10 (default: 5) | For quartile-based specialist selection |

**Workflow**:
```
1. Run main analysis
2. Top N models automatically selected
3. Each ensemble method trained
4. Results compared in Ensemble Results section
5. Best ensemble identified and available for saving
```

---

### 4.5 Validation Sub-Tab

The Validation sub-tab configures holdout validation for independent model testing.

#### 4.5.1 Enabling Validation

Check "Enable Validation Set" to hold out samples from training for independent evaluation.

#### 4.5.2 Validation Set Size

Use the slider to select what percentage of your data to reserve for validation:

| Percentage | Description |
|------------|-------------|
| 5-10% | Minimal validation, more data for training |
| 15-25% | Standard range for most applications |
| 30-40% | Large validation set for high-confidence estimates |

**Warning**: Validation sets larger than 40% are generally not recommended.

#### 4.5.3 Selection Algorithms

| Algorithm | Description | Recommended For |
|-----------|-------------|-----------------|
| Kennard-Stone | Maximizes spectral diversity only | When Y distribution doesn't matter |
| SPXY | Balances spectral and target diversity | Most applications (recommended) |
| Random | Simple random selection | Quick testing, baseline comparison |
| Stratified | Ensures balanced target distribution | Classification or highly skewed regression |
| Manual | Select samples from Data Viewer | Expert-guided validation design |

#### 4.5.4 Creating the Validation Set

1. Configure size and algorithm
2. Click "Create Validation Set"
3. Status shows: "Calibration: N samples | Validation: M samples"

**For Manual Selection**:
1. Select algorithm = Manual
2. Go to Data Viewer tab and select rows
3. Return to Validation tab and click "Use Selected Rows from Data Viewer"

#### 4.5.5 Validation Metrics Configuration

| Setting | Description |
|---------|-------------|
| Show validation metrics in results | Display RMSEP and R^2pred columns |
| Calculate for top N models | Limit validation calculation to top performers (50-500) |

---

## Chapter 5: Analysis Progress Tab

The Analysis Progress tab (Tab 5) provides real-time monitoring of running analyses.

### 5.1 Progress Display

```
+------------------------------------------------------------------+
|  Analysis Progress                                                |
|                                                                   |
|  [Animated Running Figure]              Best Model So Far:        |
|                                         PLS snv W=17 (R2=0.9523)  |
|                                                                   |
|  Progress: Testing model 45/120         Est. remaining: 3m 20s    |
|                                                                   |
|  [Pause] [Resume] [Stop]                                          |
|                                                                   |
|  +--------------------------------------------------------------+|
|  | Progress Log:                                                 ||
|  | > Starting analysis...                                        ||
|  | > Testing PLS with raw preprocessing...                       ||
|  | > Completed: PLS raw (R2cv=0.8234, RMSE=1.234)               ||
|  | > Testing PLS with SNV preprocessing...                       ||
|  | ...                                                           ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  Status: Analysis running...                                      |
+------------------------------------------------------------------+
```

### 5.2 Progress Information

| Element | Description |
|---------|-------------|
| Best Model So Far | Updates with the current top performer as analysis progresses |
| Progress Counter | Shows current model number out of total |
| Time Estimate | Estimated remaining time based on elapsed rate |
| Progress Log | Scrollable text area with detailed status messages |

### 5.3 Control Buttons

| Button | Action |
|--------|--------|
| Pause | Suspends analysis after current model completes |
| Resume | Continues paused analysis |
| Stop | Terminates analysis and preserves results collected so far |

### 5.4 Status Messages

The progress log shows:
- Analysis initialization
- Each preprocessing/model combination being tested
- Completion metrics (R^2cv, RMSEcv) for each test
- Variable selection progress (if enabled)
- Ensemble training progress (if enabled)
- Final completion status

**Example Progress Log**:
```
> Starting analysis with 5-fold cross-validation
> Dataset: 150 samples, 500 wavelengths
> Models: PLS, Ridge, RandomForest, LightGBM
> Preprocessing: raw, snv, deriv1 W=17, deriv2 W=17

> Testing PLS with raw preprocessing...
  - Full spectrum: R2cv=0.8523, RMSEcv=2.134
  - Top 50 variables: R2cv=0.8612, RMSEcv=2.089

> Testing PLS with SNV preprocessing...
  - Full spectrum: R2cv=0.9234, RMSEcv=1.523

> Testing RandomForest with raw preprocessing...
...

> Analysis complete! 120 models tested.
> Best model: PLS snv W=17 (R2cv=0.9412, RMSEcv=1.234)
> Results saved to: outputs/analysis_results_20250122_143025.csv
```

---

## Chapter 6: Results Tab

The Results tab (Tab 6) displays all tested model configurations ranked by performance.

### 6.1 Results Table Overview

```
+------------------------------------------------------------------+
|  Model Performance Results                                        |
|  All tested models ranked by performance. Click on any result     |
|  row to load it into the 'Model Development' tab for tuning.      |
|                                                                   |
|  [Legend: Q1 Low Y | Q2 | Q3 | Q4 High Y] [Rank Symbols: * = Expert Choice]|
|                                                                   |
|  +--------------------------------------------------------------+|
|  | Rank | Model | Preprocess | R2   | RMSE | R2cv | RMSEcv |... ||
|  |------|-------|------------|------|------|------|--------|----||
|  | * 1  | PLS   | snv W=17   | 0.95 | 1.12 | 0.93 | 1.34   |... ||
|  | * 2  | PLS   | deriv1 W=17| 0.94 | 1.18 | 0.92 | 1.45   |... ||
|  |   3  | Ridge | snv W=17   | 0.93 | 1.25 | 0.91 | 1.52   |... ||
|  | ...  | ...   | ...        | ...  | ...  | ...  | ...    |... ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  [Export Results to CSV] [Select Top by Region] Top: [3] From: [All]|
|                                                                   |
|  Displaying 120 results. 15 highlighted as top performers by region|
+------------------------------------------------------------------+
```

### 6.2 Results Table Columns

#### 6.2.1 Regression Results Columns

| Column | Description |
|--------|-------------|
| Rank | Overall ranking (1 = best based on composite score) |
| Select | Checkbox for ensemble training selection |
| Model | Algorithm name (PLS, Ridge, RandomForest, etc.) |
| Preprocess | Preprocessing applied (raw, snv, deriv1, deriv2, etc.) |
| Deriv | Derivative order (1, 2, or blank for non-derivative) |
| Window | Savitzky-Golay window size |
| Poly | Polynomial order for derivatives |
| LVs | Latent variables (for PLS models) |
| n_vars | Number of wavelengths used |
| full_vars | Total wavelengths available |
| SubsetTag | Variable subset identifier (e.g., "Top50", "Region_2000-2100") |
| R2 | Calibration R-squared (coefficient of determination) |
| RMSE | Calibration Root Mean Square Error |
| R2cv | Cross-validation R-squared (primary ranking metric) |
| RMSEcv | Cross-validation RMSE |
| MAEcv | Cross-validation Mean Absolute Error |
| RPD | Ratio of Performance to Deviation |
| Bias | Cross-validation Bias (systematic prediction offset) |
| RER | Range Error Ratio |
| RMSE_Q1 | RMSE for Q1 (lowest 25% Y values) |
| RMSE_Q2 | RMSE for Q2 (25-50% Y values) |
| RMSE_Q3 | RMSE for Q3 (50-75% Y values) |
| RMSE_Q4 | RMSE for Q4 (highest 25% Y values) |
| RMSEP | Validation set RMSE (if validation enabled) |
| R2pred | Validation set R-squared (if validation enabled) |
| BestRegion | Y-value region where model excels (Q1-Q4) |
| CompositeScore | Final ranking score (lower is better) |
| ComplexityScore | Unified complexity measure (0-100) |
| top_vars | Most important wavelengths |

##### NIR-Specific Metrics

These metrics are standard in NIR spectroscopy for assessing model quality:

**MAEcv (Mean Absolute Error)**
- Average of absolute prediction errors across CV folds
- Less sensitive to outliers than RMSE
- Same units as the target variable

**RPD (Ratio of Performance to Deviation)**
- Formula: SD(Y) / RMSEcv
- Industry-standard metric for NIR model fitness

| RPD Value | Interpretation |
|-----------|----------------|
| < 1.5 | Poor - not recommended for any application |
| 1.5 - 2.0 | Acceptable for rough screening only |
| 2.0 - 2.5 | Good - suitable for screening and approximate quantification |
| 2.5 - 3.0 | Very good - suitable for quality control |
| > 3.0 | Excellent - suitable for process control and quantification |

**Bias**
- Mean prediction error (average of predicted - actual)
- Positive values indicate systematic overprediction
- Negative values indicate systematic underprediction
- Values near zero indicate no systematic offset

**RER (Range Error Ratio)**
- Formula: Range(Y) / RMSEcv
- Alternative to RPD using data range instead of standard deviation
- Useful when data distribution is non-normal

| RER Value | Interpretation |
|-----------|----------------|
| < 4 | Poor predictive ability |
| 4 - 8 | Acceptable for rough screening |
| 8 - 12 | Good predictive ability |
| 12 - 25 | Very good predictive ability |
| > 25 | Excellent predictive ability |

##### Quartile Performance Columns

The RMSE_Q1 through RMSE_Q4 columns show model performance across different Y-value ranges:

| Column | Y-Value Range | Purpose |
|--------|---------------|---------|
| RMSE_Q1 | Bottom 25% of Y values | Low-range performance |
| RMSE_Q2 | 25% to 50% of Y values | Lower-mid performance |
| RMSE_Q3 | 50% to 75% of Y values | Upper-mid performance |
| RMSE_Q4 | Top 25% of Y values | High-range performance |

**Use Cases**:
- Identify models that excel in specific Y-value ranges
- Detect models that perform poorly at extremes
- Select specialist models for ensemble composition
- Understand prediction reliability across the calibration range

**Example**: A model with low RMSE_Q1 and RMSE_Q4 but high RMSE_Q2/Q3 may have edge bias - performing well at extremes but poorly in the middle range.

#### 6.2.2 Classification Results Columns

**Core Columns**

| Column | Description |
|--------|-------------|
| Rank | Overall ranking |
| Model | Algorithm name |
| Preprocess | Preprocessing applied |
| n_vars | Number of wavelengths used |
| SubsetTag | Variable subset identifier |

**Calibration Metrics** (computed on training data)

| Column | Description |
|--------|-------------|
| Accuracy | Calibration accuracy |
| ROC_AUC | Calibration ROC Area Under Curve |
| F1 | Calibration F1 score |
| Precision | Calibration precision |
| Recall | Calibration recall (sensitivity) |
| Specificity | Calibration specificity (true negative rate) |
| Kappa | Calibration Cohen's Kappa |
| MCC | Calibration Matthews Correlation Coefficient |
| BalancedAcc | Calibration Balanced Accuracy |
| BER | Calibration Balanced Error Rate |
| LogLoss | Calibration Log Loss |

**Cross-Validation Metrics** (computed on held-out folds)

| Column | Description |
|--------|-------------|
| Accuracycv | Cross-validation accuracy (primary ranking metric) |
| ROC_AUCcv | Cross-validation ROC AUC |
| F1cv | Cross-validation F1 score |
| Precisioncv | Cross-validation precision |
| Recallcv | Cross-validation recall |
| Specificitycv | Cross-validation specificity |
| Kappacv | Cross-validation Cohen's Kappa |
| MCCcv | Cross-validation Matthews Correlation Coefficient |
| BalancedAcccv | Cross-validation Balanced Accuracy |
| BERcv | Cross-validation Balanced Error Rate |
| LogLosscv | Cross-validation Log Loss |
| BestClass | Class where model excels |

##### Classification Metric Descriptions

**Specificity (True Negative Rate)**
- Proportion of actual negatives correctly identified
- Formula: TN / (TN + FP)
- Important when false positives are costly

**Cohen's Kappa**
- Measures agreement between predictions and actual values, accounting for chance
- Range: -1 to 1 (1 = perfect, 0 = random chance, negative = worse than chance)

| Kappa Value | Interpretation |
|-------------|----------------|
| < 0.20 | Poor agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.81 - 1.00 | Almost perfect agreement |

**Matthews Correlation Coefficient (MCC)**
- Balanced measure that works well even with imbalanced classes
- Range: -1 to 1 (1 = perfect, 0 = random, -1 = total disagreement)
- Recommended as the primary metric for imbalanced classification

| MCC Value | Interpretation |
|-----------|----------------|
| < 0.30 | Poor classifier |
| 0.30 - 0.50 | Acceptable classifier |
| 0.50 - 0.70 | Good classifier |
| 0.70 - 0.90 | Very good classifier |
| > 0.90 | Excellent classifier |

**Balanced Accuracy**
- Average of recall across all classes
- Formula: (Sensitivity + Specificity) / 2 for binary
- Gives equal weight to each class regardless of size

**Balanced Error Rate (BER)**
- Formula: 1 - Balanced Accuracy
- Lower is better
- Useful for comparing models on imbalanced data

**Log Loss (Cross-Entropy)**
- Measures confidence of probabilistic predictions
- Lower values indicate more confident correct predictions
- Penalizes confident wrong predictions heavily

### 6.3 Composite Scoring Formula

The composite score determines overall ranking. Lower scores are better.

**For Regression (with penalties = 0)**:
```
CompositeScore = -R2cv
```

**For Regression (with penalties > 0)**:
```
PerformanceScore = 0.5 * z_score(RMSEcv) - 0.5 * z_score(R2cv)
VariablePenalty = (variable_penalty/10)^2 * (n_vars/full_vars)
ComplexityPenalty = (complexity_penalty/10)^2 * complexity_fraction

CompositeScore = PerformanceScore + VariablePenalty + ComplexityPenalty
```

**For Classification (with penalties = 0)**:
```
CompositeScore = -Accuracycv - 0.0001 * F1cv
```

**For Classification (with penalties > 0)**:
```
PerformanceScore = -z_score(Accuracycv) - 0.3 * z_score(F1cv)
CompositeScore = PerformanceScore + VariablePenalty + ComplexityPenalty
```

### 6.4 Quartile Highlighting

Results are color-coded based on which Y-value region the model excels in:

| Color | Region | Description |
|-------|--------|-------------|
| Light Blue | Q1 | Low Y values (bottom 25%) |
| Light Green | Q2 | Low-mid Y values (25-50%) |
| Light Orange | Q3 | Mid-high Y values (50-75%) |
| Light Pink | Q4 | High Y values (top 25%) |

**How Quartile Performance is Calculated**:
1. Training data Y values are divided into quartiles
2. For each model, regional RMSE is computed for each quartile
3. Models are ranked within each region
4. Highlighting shows which region(s) each model excels in

### 6.5 Expert's Choice Indicators

Special symbols in the Rank column indicate model quality:

| Symbol | Meaning |
|--------|---------|
| (star) | Expert Choice - Well-generalized, minimal overfitting |
| (bullet) | Good Fit - Acceptable but some overfitting risk |
| Strikethrough | Overfit Warning - Calibration metrics much better than CV |

**Expert Choice Criteria**:
- Model must be in top 3 for its model type
- Calibration R^2 should not exceed CV R^2 by more than threshold
- Cross-validation metrics show good generalization

### 6.6 Sorting and Filtering

**Sorting**:
- Click any column header to sort by that column
- Click again to reverse sort order
- Sort indicators: (up arrow) = ascending, (down arrow) = descending

**Filtering Options**:
- Use "Select Top by Region" to auto-select top N models per quartile
- Configure "Top" spinbox (1-15) and "From" dropdown (All, Q1, Q2, Q3, Q4)

### 6.7 Exporting Results

**Export to CSV/Excel**:
1. Click "Export Results to CSV"
2. Choose save location and filename
3. Select format: CSV (.csv) or Excel (.xlsx)

**Exported File Contains**:
- All columns from results table
- All rows (not just visible/selected)
- Full precision numeric values

### 6.8 Saving Model Files

**From Results Tab**:
1. Double-click a result row to load into Model Development tab
2. In Model Development, click "Save Model (.dasp)"
3. Choose save location

**.dasp File Contents**:
- Trained model object (sklearn compatible)
- Preprocessing configuration
- Wavelength selection
- Training metadata
- Performance metrics

---

## Chapter 7: Ensemble Results Section

The Ensemble Results section appears at the bottom of the Results tab when ensembles are enabled.

### 7.1 Ensemble Results Table

| Column | Description |
|--------|-------------|
| Rank | Ensemble ranking (1 = best) |
| Method | Ensemble type (SimpleAverage, RegionAware, etc.) |
| RMSE | Calibration RMSE |
| R2 | Calibration R-squared |
| RMSEP | Validation RMSE (if validation set used) |
| R2pred | Validation R-squared (if validation set used) |
| RMSEcv | Cross-validation RMSE |
| R2cv | Cross-validation R-squared |
| MAEcv | Cross-validation Mean Absolute Error |
| RPD | Residual Prediction Deviation |
| vs Best Individual | Performance comparison to best single model |

### 7.2 Ensemble Action Buttons

| Button | Function |
|--------|----------|
| Save Selected Ensemble | Export selected ensemble as .dasp file |
| Train Ensemble | Manually train ensemble with custom model selection |
| Show Regional Performance | Display performance breakdown by Y-value region |
| Show Ensemble Weights | View model weights for weighted ensemble methods |
| Show Specialization | See which models are experts for which regions |

### 7.3 Understanding Ensemble Performance

**vs Best Individual Column**:
- Positive value (+0.0234 >) indicates ensemble outperforms best single model
- Negative value (-0.0156 [X]) indicates best single model is better
- "Same" indicates equivalent performance

**RPD (Residual Prediction Deviation)**:
| RPD Value | Interpretation |
|-----------|----------------|
| < 1.5 | Poor predictive ability |
| 1.5 - 2.0 | Acceptable for rough screening |
| 2.0 - 2.5 | Good predictive ability |
| 2.5 - 3.0 | Very good predictive ability |
| > 3.0 | Excellent predictive ability |

---

## Chapter 8: Analysis Workflow Summary

### 8.1 Recommended Analysis Workflow

```
+-------------------+
| 1. Import Data    |
+-------------------+
         |
         v
+-------------------+
| 2. Configure      |
| Basic Settings    |
| - Target variable |
| - CV folds        |
| - Penalties       |
+-------------------+
         |
         v
+-------------------+
| 3. Select         |
| Preprocessing     |
| - Methods         |
| - Window sizes    |
+-------------------+
         |
         v
+-------------------+
| 4. Configure      |
| Variable Selection|
| (optional)        |
+-------------------+
         |
         v
+-------------------+
| 5. Select Models  |
| - Choose tier or  |
| - Custom selection|
+-------------------+
         |
         v
+-------------------+
| 6. Configure      |
| Ensembles         |
| (optional)        |
+-------------------+
         |
         v
+-------------------+
| 7. Set Up         |
| Validation        |
| (recommended)     |
+-------------------+
         |
         v
+-------------------+
| 8. Run Analysis   |
+-------------------+
         |
         v
+-------------------+
| 9. Review Results |
| - Compare models  |
| - Check ensembles |
+-------------------+
         |
         v
+-------------------+
| 10. Export/Save   |
| - Export CSV      |
| - Save best model |
+-------------------+
```

### 8.2 Quick Start Configuration

For a rapid first analysis:

1. **Basic Settings**: 5-fold CV, penalties = 0
2. **Preprocessing**: Raw, SNV, SG1 W=17
3. **Variable Selection**: Disabled (use full spectrum)
4. **Models**: Quick tier (PLS, RandomForest)
5. **Ensembles**: Disabled
6. **Validation**: Optional

**Expected Runtime**: 1-3 minutes depending on dataset size

### 8.3 Comprehensive Analysis Configuration

For thorough method comparison:

1. **Basic Settings**: 10-fold CV, penalties = 2/2
2. **Preprocessing**: All methods, windows 11/17/23
3. **Variable Selection**: Importance + SPA, Top-N (10, 50, 100, 250)
4. **Models**: Comprehensive tier
5. **Ensembles**: Enable all methods, 5 regions
6. **Validation**: 20% SPXY selection

**Expected Runtime**: 30-60 minutes depending on dataset size

### 8.4 Common Configuration Mistakes

| Mistake | Consequence | Solution |
|---------|-------------|----------|
| Too few CV folds | Unreliable performance estimates | Use 5+ folds for >50 samples |
| No validation set | Cannot assess true generalization | Enable 15-25% holdout |
| All models on small data | Overfitting, long runtime | Use Quick or Standard tier |
| Complex preprocessing on simple spectra | Introduces artifacts | Start with Raw/SNV |
| High penalties on exploration | Suppresses good complex models | Start with penalties = 0 |

---

## Chapter 9: Interpreting Results

### 9.1 Calibration vs Cross-Validation Metrics

| Metric Type | What It Shows | Caution |
|-------------|---------------|---------|
| Calibration (R2, RMSE) | Fit to training data | Can be overfit |
| Cross-Validation (R2cv, RMSEcv) | Estimated generalization | More reliable |
| Validation (R2pred, RMSEP) | True independent test | Most reliable |

**Overfitting Indicators**:
- R2 >> R2cv (calibration much better than CV)
- RMSE << RMSEcv (calibration much lower than CV)
- Results table shows strikethrough for overfit models

### 9.2 Comparing Models

**Key Metrics for Regression**:
1. R2cv - Higher is better (target > 0.90 for most applications)
2. RMSEcv - Lower is better (in target units)
3. RPD - Higher is better (target > 2.5 for quantitative applications)

**Key Metrics for Classification**:
1. Accuracycv - Higher is better (target depends on application)
2. F1cv - Higher is better (balances precision and recall)
3. ROC_AUCcv - Higher is better (target > 0.90)

### 9.3 Regional Performance

Understanding quartile-specific performance helps identify:
- Models that excel across all Y-value ranges (universal performers)
- Models that specialize in low or high Y values (specialists)
- Optimal ensemble compositions for different applications

### 9.4 Model Selection Guidelines

| Scenario | Recommended Model |
|----------|-------------------|
| Simple, interpretable model needed | PLS or Ridge |
| Highest accuracy, complexity acceptable | LightGBM or XGBoost |
| Robust to noise and outliers | RandomForest |
| Limited data (<50 samples) | PLS with few components |
| Classification with imbalanced classes | LightGBM with class weights |
| Need prediction uncertainty | RandomForest (can output variance) |

---

## Chapter 10: Troubleshooting Analysis Issues

### 10.1 Analysis Won't Start

| Issue | Cause | Solution |
|-------|-------|----------|
| "No data loaded" | Data not imported | Import data first |
| "No target variable" | Target column not selected | Select target in Basic Settings |
| "No models selected" | All model checkboxes unchecked | Select a tier or check models manually |
| "No preprocessing selected" | All preprocessing unchecked | Check at least Raw |

### 10.2 Poor Results

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| All models R2cv < 0.5 | Weak signal in data | Check data quality, try different wavelength ranges |
| R2 high but R2cv low | Overfitting | Increase CV folds, add regularization |
| Large RMSE differences between models | Inconsistent preprocessing | Use consistent preprocessing across comparison |
| Classification accuracy at chance level | Misaligned labels | Verify target column values |

### 10.3 Slow Analysis

| Cause | Solution |
|-------|----------|
| Too many models | Use Quick or Standard tier |
| Many preprocessing combinations | Reduce window size options |
| Large dataset | Enable wavelength restriction |
| Variable selection enabled | Use faster methods (Importance, SPA) |
| NSGA-II/Bayesian with many trials | Reduce trials/generations |

### 10.4 Memory Issues

| Symptom | Solution |
|---------|----------|
| Application crashes during analysis | Reduce number of concurrent models |
| Out of memory errors | Close other applications, reduce dataset size |
| Slow scrolling in results | Export to CSV and view in spreadsheet application |

---

This concludes Part II: Running Analyses. Continue to Part III for detailed coverage of Model Development, Predictions, and Export functionality.
