# Part VI: Hyperparameter Optimization Reference

This section provides a comprehensive reference for the three hyperparameter optimization methods available in Spectral Predict V1: Grid Search, Bayesian Optimization (TPE), and NSGA-II Multi-Objective Optimization. Understanding the strengths and appropriate use cases for each method is essential for building optimal spectroscopic models.

---

## Table of Contents

1. [Introduction to Hyperparameter Optimization](#1-introduction-to-hyperparameter-optimization)
2. [Grid Search](#2-grid-search)
3. [Bayesian Optimization (TPE)](#3-bayesian-optimization-tpe)
4. [NSGA-II Multi-Objective Optimization](#4-nsga-ii-multi-objective-optimization)
5. [Choosing an Optimization Method](#5-choosing-an-optimization-method)
6. [Computational Considerations](#6-computational-considerations)
7. [Best Practices](#7-best-practices)

---

## 1. Introduction to Hyperparameter Optimization

### 1.1 Why Hyperparameter Optimization Matters

Machine learning models have two types of parameters:

1. **Model Parameters**: Learned from data during training (e.g., regression coefficients, neural network weights)
2. **Hyperparameters**: Set before training and control the learning process (e.g., number of PLS components, regularization strength, tree depth)

Hyperparameter optimization is the process of finding the best combination of hyperparameters that maximizes model performance on unseen data. In spectroscopy, this includes not just model hyperparameters but also:

- **Preprocessing choices**: SNV, derivatives, window sizes
- **Variable selection**: Which wavelengths to include
- **Model architecture**: Number of latent variables, tree depth, learning rate

### 1.2 The Search Space Challenge

For a typical spectroscopic analysis, the search space can be enormous:

```
Example Search Space:
- Preprocessing methods: 6 options (raw, snv, deriv1, deriv2, snv_deriv1, snv_deriv2)
- Window sizes: 7 options (7, 11, 15, 19, 23, 27, 31)
- Model types: 8 options (PLS, Ridge, LightGBM, XGBoost, etc.)
- Model hyperparameters: 10-50 combinations per model
- Variable subsets: 10-20 options

Total combinations: 6 x 7 x 8 x 30 x 15 = 151,200+ configurations
```

Exhaustively evaluating all combinations is computationally prohibitive, which motivates the use of intelligent search strategies.

### 1.3 Overview of Optimization Methods

| Method | Strategy | Best For | Time |
|--------|----------|----------|------|
| **Grid Search** | Exhaustive enumeration | Complete coverage, reproducibility | Medium |
| **Bayesian (TPE)** | Probabilistic modeling | Joint optimization, continuous params | Medium |
| **NSGA-II** | Evolutionary multi-objective | Pareto analysis, parsimony | Long |

---

## 2. Grid Search

### 2.1 Full Name and Abbreviation

**Grid Search** (also known as **Exhaustive Search** or **Parameter Sweep**)

### 2.2 Algorithm Explanation

Grid Search is the simplest hyperparameter optimization strategy. It defines a discrete grid of hyperparameter values and evaluates every possible combination systematically.

```
Algorithm: Grid Search
-----------------------
Input: Parameter grids {P1: [v1, v2, ...], P2: [w1, w2, ...], ...}
Output: Best parameter combination

1. Generate all combinations: C = P1 x P2 x P3 x ...
2. For each combination c in C:
   a. Train model with parameters c
   b. Evaluate via cross-validation
   c. Store score
3. Return combination with best score
```

**Advantages:**
- Guaranteed to find the best combination within the grid
- Highly parallelizable
- Results are reproducible and deterministic
- Easy to understand and interpret

**Disadvantages:**
- Computationally expensive (exponential scaling)
- Cannot explore continuous parameter spaces
- May miss optimal values between grid points
- Wastes computation on poor regions of the space

### 2.3 How Grid Search Works in Spectral Predict

Spectral Predict implements Grid Search with several enhancements:

1. **Tiered Parameter Grids**: Four predefined tiers control search thoroughness
2. **Preprocessing Integration**: Each preprocessing method is tested independently
3. **Variable Selection**: Top-N wavelength subsets based on feature importance
4. **Parallel Execution**: Uses joblib for multi-core parallelization
5. **Progress Tracking**: Real-time updates with best model so far

```
Spectral Predict Grid Search Flow:
==================================

[Data] --> [Preprocessing Methods]
                    |
                    v
           +------------------+
           | For each method: |
           |  - raw           |
           |  - snv           |
           |  - deriv1        |
           |  - deriv2        |
           |  - snv_deriv1    |
           |  - snv_deriv2    |
           +------------------+
                    |
                    v
           [Model Grid Search]
                    |
                    v
    +-------------------------------+
    | For each model type:          |
    |   For each hyperparameter set:|
    |     - Cross-validate          |
    |     - Record metrics          |
    +-------------------------------+
                    |
                    v
           [Variable Selection]
                    |
                    v
    +-------------------------------+
    | For each subset size:         |
    |   - Compute importances       |
    |   - Select top-N wavelengths  |
    |   - Retrain and evaluate      |
    +-------------------------------+
                    |
                    v
           [Ranked Results]
```

### 2.4 The Tier System

Spectral Predict uses a tiered system to balance thoroughness with computational cost.

#### 2.4.1 Quick Tier

**Purpose**: Rapid preliminary analysis, daily QC, quick exploration

**Models Included** (Regression):
- PLS
- RandomForest

**Models Included** (Classification):
- PLS-DA
- RandomForest

**Preprocessing**:
- Window sizes: [11]
- Methods: raw, snv

**Characteristics**:
- Minimal hyperparameter combinations
- Fast execution (minutes)
- Good for initial data exploration

#### 2.4.2 Standard Tier

**Purpose**: Most users, routine analysis, daily work

**Models Included** (Regression):
- PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM

**Models Included** (Classification):
- PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost

**Preprocessing**:
- Window sizes: [7, 19]
- Methods: raw, snv, deriv1, deriv2

**Characteristics**:
- Balanced coverage
- Moderate execution time (10-30 minutes)
- Recommended for most applications

#### 2.4.3 Comprehensive Tier

**Purpose**: Thorough analysis, research, publications

**Models Included** (Regression):
- PLS, Ridge, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, NeuralBoosted

**Models Included** (Classification):
- PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP, NeuralBoosted

**Preprocessing**:
- Window sizes: [7, 15, 25]
- Methods: raw, snv, deriv1, deriv2, snv_deriv1, snv_deriv2

**Characteristics**:
- Extended hyperparameter ranges
- Longer execution (30-60 minutes)
- Suitable for publication-quality models

#### 2.4.4 Experimental Tier

**Purpose**: Research exploration, method comparison, no time constraints

**Models Included** (Regression):
- All available models including SVR, MLP

**Models Included** (Classification):
- All available models

**Preprocessing**:
- All window sizes and methods

**Characteristics**:
- Maximum coverage
- Long execution (hours)
- For comprehensive method comparison

### 2.5 Model-Specific Parameter Grids

#### PLS / PLS-DA

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_components | 1-8 | 1-15 | 1-20 |
| max_iter | [500] | [500] | [500] |
| tol | [1e-6] | [1e-6] | [1e-6] |

#### Ridge

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| alpha | [1.0] | [0.001, 0.01, 0.1, 1.0, 10.0] | Extended range |
| solver | ['auto'] | ['auto'] | ['auto'] |
| tol | [1e-4] | [1e-4] | [1e-4] |

#### Lasso

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| alpha | [0.1] | [0.001, 0.01, 0.1, 1.0] | Extended range |
| selection | ['cyclic'] | ['cyclic'] | ['cyclic'] |

#### ElasticNet

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| alpha | [0.1] | [0.01, 0.1, 1.0] | Extended range |
| l1_ratio | [0.5] | [0.3, 0.5, 0.7] | [0.1, 0.3, 0.5, 0.7, 0.9] |

#### RandomForest

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_estimators | [100] | [100] | [100, 200] |
| max_depth | [None] | [None, 30] | [None, 10, 20, 30] |
| min_samples_split | [2] | [2] | [2, 5] |
| max_features | ['sqrt'] | ['sqrt'] | ['sqrt', 'log2'] |

#### LightGBM

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_estimators | [100] | [100, 200] | [100, 200, 300] |
| learning_rate | [0.1] | [0.05, 0.1] | [0.01, 0.05, 0.1] |
| num_leaves | [31] | [15, 31] | [15, 31, 63] |
| max_depth | [-1] | [5, 10, -1] | [5, 10, 15, -1] |
| reg_alpha | [0.1] | [0.1, 0.5] | [0.1, 0.5, 1.0] |
| reg_lambda | [1.0] | [1.0, 2.0] | [1.0, 2.0, 5.0] |

#### XGBoost

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_estimators | [100] | [100, 200] | [100, 200, 300] |
| learning_rate | [0.1] | [0.05, 0.1] | [0.01, 0.05, 0.1] |
| max_depth | [6] | [3, 6] | [3, 6, 9] |
| subsample | [0.8] | [0.8] | [0.6, 0.8, 1.0] |
| colsample_bytree | [0.8] | [0.6, 0.8] | [0.5, 0.7, 0.9] |
| reg_alpha | [0.1] | [0.1, 0.2] | [0.1, 0.2, 0.5] |
| reg_lambda | [1.0] | [1.0, 2.0] | [1.0, 2.0, 5.0] |

#### CatBoost

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| iterations | [100] | [100, 200] | [100, 200, 300] |
| learning_rate | [0.1] | [0.05, 0.1] | [0.01, 0.05, 0.1] |
| depth | [6] | [4, 5, 6] | [4, 5, 6, 8] |
| l2_leaf_reg | [3.0] | [3.0, 5.0] | [1.0, 3.0, 5.0, 10.0] |

#### SVR / SVM

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| kernel | ['rbf'] | ['rbf', 'linear'] | ['rbf', 'linear', 'poly'] |
| C | [1.0] | [1.0, 10.0] | [0.1, 1.0, 10.0, 100.0] |
| gamma | ['scale'] | ['scale'] | ['scale', 'auto'] |
| epsilon (SVR) | [0.1] | [0.1] | [0.01, 0.1, 0.2] |

#### MLP

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| hidden_layer_sizes | [(64,)] | [(64,), (128, 64)] | [(64,), (128,), (128, 64), (256, 128)] |
| alpha | [0.001] | [0.001] | [0.0001, 0.001, 0.01] |
| learning_rate_init | [0.001] | [0.001] | [0.0001, 0.001, 0.01] |
| activation | ['relu'] | ['relu'] | ['relu', 'tanh'] |

### 2.6 Total Combinations Calculation

The total number of configurations tested in Grid Search is:

$$N_{total} = N_{preprocess} \times N_{models} \times \prod_{i} |P_i| \times N_{subsets}$$

Where:
- $N_{preprocess}$ = Number of preprocessing methods
- $N_{models}$ = Number of model types
- $|P_i|$ = Size of parameter grid for parameter $i$
- $N_{subsets}$ = Number of variable subsets tested

**Example (Standard Tier)**:
- Preprocessing: 4 methods x 2 windows = 8
- Models: 6 types
- Average params per model: ~10 combinations
- Variable subsets: 5 sizes

Total: $8 \times 6 \times 10 \times 5 = 2,400$ configurations

### 2.7 Parallelization via Joblib

Spectral Predict uses joblib for parallel execution:

```python
# Parallel model evaluation
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(evaluate_model)(config)
    for config in configurations
)
```

**Parallelization benefits**:
- `n_jobs=-1`: Uses all available CPU cores
- Near-linear speedup for embarrassingly parallel workloads
- Automatic memory management and task distribution

---

## 3. Bayesian Optimization (TPE)

### 3.1 Full Name and Abbreviation

**Bayesian Optimization** using **Tree-structured Parzen Estimator (TPE)**

### 3.2 Algorithm Explanation

Bayesian Optimization is a sequential model-based optimization strategy that builds a probabilistic model of the objective function and uses it to intelligently select the next points to evaluate.

#### 3.2.1 The TPE Algorithm

TPE is a specific type of Bayesian optimization that models the search differently from traditional approaches:

**Traditional Bayesian Optimization**:
- Models $p(y|x)$ - probability of score given parameters
- Uses Gaussian Process surrogate

**TPE Approach**:
- Models $p(x|y)$ - probability of parameters given score
- Splits observations into "good" (top $\gamma$%) and "bad"
- Models each group separately using kernel density estimation

**Mathematical Formulation**:

Given observations $D = \{(x_1, y_1), ..., (x_n, y_n)\}$, TPE defines:

$$l(x) = p(x | y < y^*)$$
$$g(x) = p(x | y \geq y^*)$$

Where $y^*$ is the $\gamma$-quantile of observed scores (typically $\gamma = 0.25$).

The **Expected Improvement (EI)** acquisition function becomes:

$$EI(x) \propto \frac{l(x)}{g(x)}$$

This ratio is high when:
- $l(x)$ is large: parameters likely to produce good results
- $g(x)$ is small: parameters unlikely to produce bad results

```
TPE Algorithm:
==============

1. Initialize: Evaluate n_startup random configurations
2. For trial t = n_startup to n_trials:
   a. Split observations into good (y < quantile) and bad (y >= quantile)
   b. Fit KDE l(x) to good observations
   c. Fit KDE g(x) to bad observations
   d. For each candidate x:
      - Compute l(x)/g(x) ratio
   e. Select x* with highest ratio
   f. Evaluate f(x*) and add to observations
3. Return best observed configuration
```

#### 3.2.2 Handling Conditional Parameters

TPE naturally handles **tree-structured** (conditional) parameter spaces:

```
Example: Window size only relevant when derivative is enabled

if preprocessing in ['deriv1', 'deriv2']:
    window = suggest_int('window', 5, 51, step=2)
else:
    window = None  # Not applicable
```

This is why TPE is particularly well-suited for spectroscopic optimization where preprocessing choices affect which downstream parameters are relevant.

### 3.3 How Bayesian Optimization Works in Spectral Predict

Spectral Predict's Bayesian optimization (implemented in `unified_bayesian.py`) provides **joint optimization** of:

1. **Preprocessing**: Method selection and parameters
2. **Model Hyperparameters**: All relevant parameters for the chosen model
3. **Variable Selection**: Method and subset size
4. **Regional Subsets**: Dynamically computed on preprocessed data

```
Unified Bayesian Optimization Flow:
===================================

            +-------------------+
            |   TPE Sampler     |
            | (n_startup=20)    |
            +-------------------+
                    |
                    v
    +-------------------------------+
    | Suggest Configuration:        |
    |  - Preprocessing type         |
    |  - Window size (if deriv)     |
    |  - Model hyperparameters      |
    |  - Variable selection method  |
    |  - Subset size                |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Apply Preprocessing           |
    |  raw -> SNV -> Derivative     |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Compute Variable Selection    |
    |  - importance (model-based)   |
    |  - CARS                       |
    |  - regional (dynamic)         |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Cross-Validate Model          |
    |  RMSE (regression)            |
    |  1-Accuracy (classification)  |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Return Score to TPE           |
    |  (subset score, not full)     |
    +-------------------------------+
                    |
                    v
           [Update TPE Model]
                    |
            +-------+-------+
            |               |
        [Good Set]     [Bad Set]
            |               |
            v               v
         l(x) KDE        g(x) KDE
```

### 3.4 Parameters and Settings

#### 3.4.1 Optuna Integration

Spectral Predict uses the **Optuna** library for TPE optimization:

```python
from optuna.samplers import TPESampler

sampler = TPESampler(
    seed=random_state,
    n_startup_trials=20,    # Random exploration first
    n_ei_candidates=32,     # Candidates per EI evaluation
    multivariate=True,      # Model parameter interactions
    consider_endpoints=True # Include boundary values
)
```

#### 3.4.2 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_trials | 300 | Total number of configurations to evaluate |
| n_startup_trials | 20 | Random trials before TPE kicks in |
| cv_folds | 5 | Cross-validation folds |
| n_top_regions | 10 | Regional subsets to explore |

#### 3.4.3 Preprocessing Options

```python
PREPROCESSING_OPTIONS = [
    'raw', 'snv',
    'deriv1', 'deriv2', 'deriv3', 'deriv4',
    'snv_deriv1', 'snv_deriv2', 'snv_deriv3', 'snv_deriv4',
    'deriv1_snv', 'deriv2_snv', 'deriv3_snv', 'deriv4_snv'
]
```

#### 3.4.4 Variable Selection Methods

```python
VAR_METHODS = ['importance', 'cars', 'region']

SUBSET_SIZES = ['full', 10, 20, 50, 100, 250, 500, 1000]
```

### 3.5 Model-Specific Hyperparameter Ranges

Bayesian optimization uses **continuous** ranges rather than discrete grids:

#### PLS

| Parameter | Range | Scale |
|-----------|-------|-------|
| n_components | 2-20 | Integer |

#### Ridge / Lasso

| Parameter | Range | Scale |
|-----------|-------|-------|
| alpha | 1e-4 to 100 | Log |

#### ElasticNet

| Parameter | Range | Scale |
|-----------|-------|-------|
| alpha | 1e-4 to 100 | Log |
| l1_ratio | 0.1 to 0.9 | Linear |

#### LightGBM

| Parameter | Range | Scale |
|-----------|-------|-------|
| n_estimators | 50-500 | Integer |
| learning_rate | 0.01-0.3 | Log |
| num_leaves | 15-127 | Integer |
| max_depth | -1 to 15 | Integer |

#### XGBoost

| Parameter | Range | Scale |
|-----------|-------|-------|
| n_estimators | 50-500 | Integer |
| learning_rate | 0.01-0.3 | Log |
| max_depth | 3-12 | Integer |
| subsample | 0.6-1.0 | Linear |

### 3.6 Why Bayesian Beats Grid Search

| Aspect | Grid Search | Bayesian (TPE) |
|--------|-------------|----------------|
| **Continuous params** | Only discrete values | Any value in range |
| **Example** | alpha in [0.1, 0.2, 0.3] | alpha = 0.127 (optimal) |
| **Joint optimization** | Independent grids | Learns interactions |
| **Exploration** | Exhaustive (wastes time) | Focuses on promising regions |
| **Window sizes** | [7, 19] (2 values) | [5, 7, ..., 51] (24 values) |
| **Search space** | ~3,000 configs | 140,000+ intelligently explored |

### 3.7 When to Use Bayesian Optimization

**Best for**:
- Production modeling where you want the absolute best configuration
- When continuous hyperparameters matter (regularization, learning rates)
- Joint optimization of preprocessing + model + variables
- Limited computational budget (smarter than Grid Search)

**Not ideal for**:
- When you need complete reproducibility of all tested configurations
- When you want to understand the full parameter landscape
- When Pareto analysis of multiple objectives is needed

---

## 4. NSGA-II Multi-Objective Optimization

### 4.1 Full Name and Abbreviation

**NSGA-II**: Non-dominated Sorting Genetic Algorithm II

### 4.2 Algorithm Explanation

NSGA-II is an evolutionary algorithm for multi-objective optimization. Unlike single-objective methods that find one "best" solution, NSGA-II finds the **Pareto front** - a set of solutions representing the optimal tradeoffs between competing objectives.

#### 4.2.1 Multi-Objective Optimization Concepts

**Pareto Dominance**: Solution A dominates solution B if:
1. A is at least as good as B in all objectives
2. A is strictly better than B in at least one objective

**Pareto Front**: The set of all non-dominated solutions (no solution in the front dominates another).

```
Pareto Front Visualization (2 Objectives):
==========================================

Error (minimize)
     ^
     |
  1.0+  x  x
     |      x  x
     |         x  x
     |            x  x
  0.5+              x  x  <- Pareto Front
     |                 x
     |                   x
     |                     x
  0.0+-------------------------> Wavelengths (minimize)
     0    200   400   600   800

Legend:
x = Pareto-optimal solution
  = Dominated solution (not shown)
```

#### 4.2.2 The NSGA-II Algorithm

```
NSGA-II Algorithm:
==================

1. Initialize population P0 of size N
2. For generation g = 0 to G:

   a. CREATE OFFSPRING:
      - Select parents via binary tournament
      - Apply crossover (SBX)
      - Apply mutation (SmartMutation)
      - Create offspring population Q

   b. COMBINE: R = P U Q (size 2N)

   c. NON-DOMINATED SORTING:
      - Identify rank-0 (Pareto front)
      - Remove, repeat for rank-1, rank-2, ...

   d. SELECTION:
      - Fill next generation from lowest ranks
      - Within same rank, prefer diverse solutions
        (crowding distance)

3. Return final Pareto front
```

**Key Components**:

1. **Non-dominated Sorting**: O(MN^2) algorithm to rank solutions
2. **Crowding Distance**: Maintains diversity on the Pareto front
3. **Binary Tournament**: Selects parents based on rank and crowding
4. **Elitism**: Best solutions always survive to next generation

### 4.3 The Three Objectives in Spectral Predict

NSGA-II in Spectral Predict optimizes three competing objectives simultaneously:

#### Objective 1: Prediction Error (Minimize)

**Regression**: Root Mean Square Error (RMSE)
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Classification**: 1 - Accuracy
$$Error = 1 - \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

#### Objective 2: Wavelength Count (Minimize - Parsimony)

Encourages sparse, interpretable models:

$$f_2 = \left(\frac{n_{selected}}{n_{total}}\right)^2$$

The quadratic penalty means:
- 400 wavelengths: penalty = (400/800)^2 = 0.25
- 800 wavelengths: penalty = (800/800)^2 = 1.00

This creates 4x more pressure to reduce from 800 to 400 than from 400 to 200.

#### Objective 3: Model Complexity (Minimize)

Combines model and preprocessing complexity:

$$f_3 = 0.7 \times C_{model} + 0.3 \times C_{preproc}$$

**Model complexity scores**:
| Model | Complexity Range |
|-------|------------------|
| PLS | 0.00 - 0.25 |
| Ridge | 0.00 - 0.20 |
| Lasso | 0.075 |
| ElasticNet | 0.10 |
| LightGBM | 0.15 - 0.40 |
| XGBoost | 0.15 - 0.40 |
| RandomForest | 0.20 - 0.40 |
| CatBoost | 0.175 - 0.40 |
| SVR | 0.25 - 0.40 |
| MLP | 0.25 - 0.45 |

**Preprocessing complexity scores**:
| Preprocessing | Complexity |
|---------------|------------|
| raw | 0.0 |
| snv | 0.2 |
| deriv1 | 0.4 |
| snv_deriv1 | 0.5 |
| deriv2 | 0.6 |
| snv_deriv2 | 0.7 |
| deriv3, deriv4 | 0.8 |

### 4.4 Pareto Front Concept with Diagram

```
3D Pareto Front (3 Objectives):
===============================

                Complexity
                    ^
                   /|
                  / |
                 /  |     * * *
                /   |   *       *
               /    | *           *
              /     |*             *
             /      +----------------> Wavelengths
            /      /
           /      /
          /      / * * *
         /      / *     *
        v      / *       *
      Error   /  *        *
             /

Each point (*) is a Pareto-optimal solution.
No solution dominates another - they represent
different tradeoffs between the objectives.
```

### 4.5 Non-dominated Sorting Algorithm

```
Non-dominated Sorting:
======================

Input: Population P with objective values
Output: Fronts F0, F1, F2, ...

1. For each solution p in P:
   - Sp = solutions dominated by p
   - np = count of solutions dominating p

2. F0 = {p : np = 0}  # First Pareto front

3. i = 0
4. While Fi is not empty:
   - Q = empty set
   - For each p in Fi:
     - For each q in Sp:
       - nq = nq - 1
       - If nq == 0: Q = Q U {q}
   - i = i + 1
   - Fi = Q

Result: F0, F1, F2, ... (non-dominated fronts)
```

### 4.6 Crowding Distance for Diversity

Crowding distance measures how "crowded" a solution is on the Pareto front. Solutions in less crowded regions are preferred to maintain diversity.

$$CD_i = \sum_{m=1}^{M} \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{max} - f_m^{min}}$$

Where:
- $M$ = number of objectives
- $f_m^{i+1}$, $f_m^{i-1}$ = objective values of neighboring solutions
- Boundary solutions receive infinite distance (always preserved)

### 4.7 Selection Strategies

Spectral Predict provides three strategies for selecting from the Pareto front:

#### 4.7.1 Min Error (selection_bias = 0)

**Behavior**: Selects the solution with the absolute lowest prediction error, regardless of wavelength count or complexity.

**Use when**:
- Accuracy is the only priority
- You don't care about model interpretability
- Hardware can handle any model complexity

**Selection**: Searches **all evaluated solutions** (not just Pareto front) for minimum error.

#### 4.7.2 Balanced (selection_bias = 1)

**Behavior**: 50/50 compromise between minimum error and knee point.

**Use when**:
- You want some parsimony but prioritize accuracy
- Willing to trade a small amount of accuracy for fewer wavelengths

**Selection**:
$$score_i = 0.5 \times rank_{error} + 0.5 \times (1 - knee\_score)$$

#### 4.7.3 Knee Point (selection_bias = 2)

**Behavior**: Finds the point of maximum curvature on the Pareto front - where improving one objective starts requiring significant sacrifices in others.

**Use when**:
- You want the optimal tradeoff
- Interpretability and parsimony matter
- Research applications where understanding matters

**Selection Algorithm** (for 2D):
1. Draw line from first to last Pareto point
2. For each point, compute perpendicular distance to line
3. Point with maximum distance is the knee

```
Knee Point Detection:
=====================

Error
  ^
  |
  | *
  |  \
  |   *
  |    \
  |     * <- Knee point (max distance to line)
  |      \
  |       *
  |         \
  +----------*------> Wavelengths
```

### 4.8 SmartMutation: Importance-Biased Selection

NSGA-II in Spectral Predict uses **SmartMutation**, which leverages CARS-Tree importance scores to guide wavelength selection:

```
SmartMutation Logic:
====================

For each wavelength w with importance I(w):

If currently SELECTED (value = 1):
  P(drop) = base_prob x sparsity_bias x (1 - I(w))

  High importance -> Low drop probability
  sparsity_bias = 1.4 increases drop pressure

If currently NOT SELECTED (value = 0):
  P(add) = base_prob x (1/sparsity_bias) x I(w)

  High importance -> High add probability
  1/sparsity_bias reduces add likelihood
```

This creates evolutionary pressure toward:
- Keeping important wavelengths
- Dropping unimportant wavelengths
- Overall parsimony (sparsity_bias > 1)

### 4.9 Chromosome Representation

NSGA-II encodes each solution as an integer chromosome:

```
Chromosome Structure (13 + N_wavelengths genes):
================================================

[0]  : Preprocessing type (0-13)
[1]  : Window size index (0-13)
[2]  : Model type (0-10)
[3]  : Model parameter (0-14)
[4]  : Learning rate gene (0-14)
[5]  : L1 regularization gene (0-14)
[6]  : L2 regularization gene (0-14)
[7]  : ElasticNet l1_ratio gene (0-14)
[8]  : Subsample gene (0-14)
[9]  : Colsample gene (0-14)
[10] : Min samples gene (0-14)
[11] : Gamma gene (0-14)
[12] : Max features gene (0-14)
[13+]: Wavelength selection (0 or 1)

Example (1000 wavelengths):
[7, 6, 2, 8, 10, 12, 14, 7, 10, 10, 2, 0, 7, 0, 1, 1, 0, ...]
 |  |  |  |                                     |  |  |  |
 |  |  |  |                                     wavelengths
 |  |  |  model_param=8 (e.g., 9 PLS components)
 |  |  model=Ridge
 |  window=17
 preproc=snv_deriv2
```

### 4.10 Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| population_size | 60 | Solutions per generation |
| n_generations | 120 | Number of evolutionary cycles |
| cv_folds | 5 | Cross-validation folds |
| min_wavelengths | 10 | Constraint: minimum selected |
| selection_bias | 0.0 | 0=min error, 1=balanced, 2=knee |
| use_guidance | True | Enable CARS-Tree importance guidance |

### 4.11 When to Use NSGA-II

**Best for**:
- Research requiring Pareto analysis
- Understanding tradeoffs between accuracy and parsimony
- Building interpretable models with minimal wavelengths
- Exploring the full spectrum of optimal solutions

**Not ideal for**:
- Quick analysis (long runtime)
- When you only care about accuracy (use Bayesian)
- Simple datasets where Grid Search suffices

---

## 5. Choosing an Optimization Method

### 5.1 Decision Flowchart

```
                    START
                      |
                      v
            +-------------------+
            | Time constraint?  |
            +-------------------+
                  |       |
                < 30 min  > 30 min
                  |       |
                  v       v
          +----------+  +------------------+
          | Quick    |  | Need Pareto      |
          | Grid     |  | analysis?        |
          | Search   |  +------------------+
          +----------+        |       |
                            Yes      No
                              |       |
                              v       v
                      +----------+  +-----------+
                      | NSGA-II  |  | Bayesian  |
                      | (knee/   |  | (TPE)     |
                      | balanced)|  +-----------+
                      +----------+
```

### 5.2 Method Comparison Table

| Criterion | Grid Search | Bayesian (TPE) | NSGA-II |
|-----------|-------------|----------------|---------|
| **Speed** | Medium | Medium | Slow |
| **Coverage** | Exhaustive (grid) | Intelligent sampling | Multi-objective |
| **Reproducibility** | Excellent | Good | Good |
| **Continuous params** | No | Yes | Yes |
| **Joint optimization** | Limited | Excellent | Excellent |
| **Interpretability** | High | Medium | High (Pareto) |
| **Parsimony** | Manual | Automatic | Built-in objective |
| **Best single model** | Good | Excellent | Good |
| **Understanding tradeoffs** | Poor | Poor | Excellent |

### 5.3 Use Case Recommendations

#### Quick Exploration
**Method**: Grid Search (Quick tier)
**Time**: 5-15 minutes
**When**: Initial data exploration, daily QC

#### Production Modeling
**Method**: Bayesian Optimization
**Time**: 20-45 minutes
**When**: Building deployment models, best accuracy needed

#### Research / Publications
**Method**: NSGA-II with Balanced selection
**Time**: 1-3 hours
**When**: Understanding model tradeoffs, interpretability matters

#### Complete Analysis
**Method**: All three in sequence
**Time**: 3-5 hours
**When**: Comprehensive method comparison, publications

### 5.4 Dataset Size Recommendations

| Dataset Size | Recommended Method | Notes |
|--------------|-------------------|-------|
| < 50 samples | Grid Search (Quick) | Limited data, simple models preferred |
| 50-200 samples | Bayesian or Grid (Standard) | Good balance of methods |
| 200-1000 samples | Bayesian | Enough data for complex models |
| > 1000 samples | NSGA-II or Bayesian | Can afford longer optimization |

---

## 6. Computational Considerations

### 6.1 Time Complexity

| Method | Complexity | Typical Time (100 samples, 1000 wavelengths) |
|--------|------------|---------------------------------------------|
| Grid Search (Quick) | O(n_configs) | 5-15 min |
| Grid Search (Standard) | O(n_configs) | 15-45 min |
| Bayesian (300 trials) | O(n_trials x n_cv) | 20-60 min |
| NSGA-II (60 pop, 120 gen) | O(pop x gen x n_cv) | 1-3 hours |

### 6.2 Memory Requirements

| Method | Memory Usage | Notes |
|--------|--------------|-------|
| Grid Search | Low-Medium | Results stored in DataFrame |
| Bayesian | Medium | Optuna study stores trial history |
| NSGA-II | Medium-High | Population + Pareto front storage |

### 6.3 Parallelization

| Method | Parallelization | Efficiency |
|--------|-----------------|------------|
| Grid Search | Excellent (joblib) | Near-linear with cores |
| Bayesian | Limited (sequential) | TPE is inherently sequential |
| NSGA-II | Medium (per-generation) | Population evaluated in parallel |

### 6.4 Early Stopping and Pruning

**Bayesian Optimization**:
- Optuna supports pruning via median stopping rule
- Trials performing worse than median are stopped early

**NSGA-II**:
- No early stopping (evolutionary process requires full generations)
- Controller allows manual cancellation with partial results

---

## 7. Best Practices

### 7.1 Before Optimization

1. **Clean your data**: Remove outliers and verify data quality
2. **Split validation set**: Keep 10-20% for final evaluation
3. **Understand your constraints**: Time, interpretability, deployment requirements

### 7.2 During Optimization

1. **Start simple**: Begin with Grid Search (Quick) to verify pipeline
2. **Scale up gradually**: Move to Standard, then Bayesian/NSGA-II
3. **Monitor progress**: Check for convergence and overfitting

### 7.3 After Optimization

1. **Validate results**: Test top models on held-out validation set
2. **Check for overfitting**: Compare CV scores to validation scores
3. **Consider ensemble**: Combine top-performing models if appropriate

### 7.4 Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Overfitting to CV | Use separate validation set |
| Too many hyperparameters | Start with defaults, tune important ones |
| Ignoring preprocessing | Test multiple preprocessing methods |
| Computational timeout | Start with Quick tier, scale up |
| Selecting only by accuracy | Consider parsimony and interpretability |

### 7.5 Recommended Workflow

```
Recommended Optimization Workflow:
==================================

1. EXPLORE (10 min)
   - Grid Search (Quick tier)
   - Identify promising models and preprocessing

2. OPTIMIZE (30-60 min)
   - Bayesian Optimization (300 trials)
   - Joint preprocessing + hyperparameter tuning

3. ANALYZE (optional, 1-3 hours)
   - NSGA-II with Balanced selection
   - Understand accuracy vs parsimony tradeoffs

4. VALIDATE (10 min)
   - Test top 3-5 models on validation set
   - Select final model based on validation performance

5. DEPLOY
   - Export selected model
   - Document preprocessing pipeline
```

---

## Summary

Spectral Predict V1 provides three complementary hyperparameter optimization methods:

1. **Grid Search**: Exhaustive, reproducible, tiered coverage
2. **Bayesian (TPE)**: Intelligent, continuous, joint optimization
3. **NSGA-II**: Multi-objective, Pareto analysis, parsimony-aware

The choice depends on your constraints:
- **Time-limited**: Grid Search (Quick/Standard)
- **Best accuracy**: Bayesian Optimization
- **Interpretability**: NSGA-II with Balanced selection

For most users, the recommended approach is:
1. Quick Grid Search for exploration
2. Bayesian Optimization for final model selection
3. NSGA-II when parsimony and tradeoff analysis matter

---

*Part VI of the Spectral Predict V1 User Guide*
