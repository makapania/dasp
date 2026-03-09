# Part III: Machine Learning Models Reference

This section provides comprehensive documentation of all machine learning models available in Spectral Predict V1. Each model is documented with its mathematical foundation, algorithm explanation, hyperparameters, and practical guidance for spectroscopy applications.

---

## Table of Contents

1. [Model Overview and Comparison](#model-overview-and-comparison)
2. [PLS (Partial Least Squares Regression)](#1-pls-partial-least-squares-regression)
3. [PLS-DA (Partial Least Squares Discriminant Analysis)](#2-pls-da-partial-least-squares-discriminant-analysis)
4. [Ridge Regression](#3-ridge-regression-l2-regularization)
5. [Lasso Regression](#4-lasso-regression-l1-regularization)
6. [ElasticNet](#5-elasticnet-combined-l1l2-regularization)
7. [Random Forest](#6-random-forest)
8. [XGBoost](#7-xgboost-extreme-gradient-boosting)
9. [LightGBM](#8-lightgbm-light-gradient-boosting-machine)
10. [CatBoost](#9-catboost-categorical-boosting)
11. [SVR/SVC (Support Vector Machines)](#10-svrsvc-support-vector-machines)
12. [MLP (Multi-Layer Perceptron)](#11-mlp-multi-layer-perceptron)
13. [NeuralBoosted](#12-neuralboosted)
14. [Model Selection Guidelines](#model-selection-guidelines)
15. [Hyperparameter Tuning Strategies](#hyperparameter-tuning-strategies)

---

## Model Overview and Comparison

### Model Categories

Spectral Predict organizes models into four categories based on their algorithmic approach:

| Category | Models | Characteristics |
|----------|--------|-----------------|
| **Latent Variable** | PLS, PLS-DA | Dimensionality reduction with maximum covariance |
| **Linear Regularized** | Ridge, Lasso, ElasticNet | Linear regression with penalty terms |
| **Ensemble Tree** | RandomForest, XGBoost, LightGBM, CatBoost | Decision tree ensembles |
| **Neural Network** | MLP, NeuralBoosted | Multilayer perceptrons and boosted networks |
| **Kernel Methods** | SVR, SVC | Support vector machines with kernel transformations |

### Model Tiers in Spectral Predict

Models are organized into tiers for different use cases:

| Tier | Regression Models | Classification Models | Use Case |
|------|-------------------|----------------------|----------|
| **Quick** | PLS, RandomForest | PLS-DA, RandomForest | Rapid testing (3-5 min) |
| **Standard** | PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM | PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost | Daily analysis (10-15 min) |
| **Comprehensive** | PLS, Ridge, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, NeuralBoosted | PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP, NeuralBoosted | Research (20-30 min) |
| **Experimental** | All models | All models | Method exploration (45+ min) |

### Quick Comparison Table

| Model | Type | Interpretable | Handles Collinearity | Feature Selection | Training Speed | Prediction Speed |
|-------|------|---------------|---------------------|-------------------|----------------|------------------|
| PLS | Regression | High | Excellent | Indirect (VIP) | Fast | Very Fast |
| PLS-DA | Classification | High | Excellent | Indirect (VIP) | Fast | Fast |
| Ridge | Regression | High | Good | No | Very Fast | Very Fast |
| Lasso | Regression | High | Good | Yes (Sparse) | Fast | Very Fast |
| ElasticNet | Regression | High | Very Good | Yes (Sparse) | Fast | Very Fast |
| RandomForest | Both | Medium | Good | Yes | Medium | Fast |
| XGBoost | Both | Medium | Good | Yes | Medium | Fast |
| LightGBM | Both | Medium | Good | Yes | Fast | Fast |
| CatBoost | Both | Medium | Good | Yes | Medium | Fast |
| SVR/SVC | Both | Low | Excellent | No | Slow | Medium |
| MLP | Both | Low | Good | No | Medium | Fast |
| NeuralBoosted | Both | Low | Good | Yes | Medium | Medium |

---

## 1. PLS (Partial Least Squares Regression)

### Full Name and Abbreviation

**Partial Least Squares Regression (PLSR)**, also known as **Projection to Latent Structures**

### Mathematical Foundation

PLS is a dimensionality reduction technique that finds latent variables (components) that maximize the covariance between the predictor matrix $X$ and the response $Y$. Unlike Principal Component Analysis (PCA), which only considers variance in $X$, PLS explicitly models the relationship between $X$ and $Y$.

The fundamental decomposition is:

$$X = TP^T + E$$

$$Y = UQ^T + F$$

Where:
- $X$ is the $(n \times p)$ predictor matrix (spectra)
- $Y$ is the $(n \times 1)$ response vector (target property)
- $T$ is the $(n \times A)$ X-scores matrix
- $P$ is the $(p \times A)$ X-loadings matrix
- $U$ is the $(n \times A)$ Y-scores matrix
- $Q$ is the $(1 \times A)$ Y-loadings matrix
- $E$ and $F$ are residual matrices
- $A$ is the number of latent variables (components)

The algorithm maximizes:

$$\max_{w, c} \text{cov}(Xw, Yc)^2$$

subject to $||w|| = ||c|| = 1$

### NIPALS Algorithm

Spectral Predict uses the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm:

1. **Initialize**: Set $u = y$ (first column of $Y$)
2. **Calculate X-weights**: $w = X^T u / (u^T u)$, normalize $||w|| = 1$
3. **Calculate X-scores**: $t = Xw$
4. **Calculate Y-loadings**: $q = Y^T t / (t^T t)$, normalize $||q|| = 1$
5. **Calculate Y-scores**: $u = Yq / (q^T q)$
6. **Check convergence**: If $||t_{new} - t_{old}|| < \text{tol}$, continue to step 7
7. **Deflate matrices**: $X = X - tp^T$ where $p = X^T t / (t^T t)$
8. **Repeat** for next component

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_components` | int | 1-20 | Auto | Number of latent variables to extract |
| `max_iter` | int | 100-1000 | 500 | Maximum NIPALS iterations per component |
| `tol` | float | 1e-8 to 1e-4 | 1e-6 | Convergence tolerance |
| `scale` | bool | True/False | False | Whether to scale X to unit variance |

**Selecting Optimal n_components:**

The number of components is the most critical hyperparameter. Guidelines:

1. **Cross-validation**: Use the value that minimizes RMSECV
2. **Parsimony rule**: If two models have similar RMSECV, prefer fewer components
3. **Practical limits**: Rarely need more than 10-15 components for spectroscopy
4. **Information criterion**: Look for "elbow" in explained variance plot

$$\text{Optimal } A = \arg\min_a \text{RMSECV}(a)$$

### Variable Importance in Projection (VIP)

VIP scores quantify each variable's contribution to the PLS model:

$$VIP_j = \sqrt{p \sum_{a=1}^{A} \frac{w_{aj}^2 \cdot SSY_a}{SSY_{total}}}$$

Where:
- $p$ = number of predictor variables (wavelengths)
- $w_{aj}$ = weight of variable $j$ in component $a$
- $SSY_a$ = sum of squares of $Y$ explained by component $a$
- $SSY_{total}$ = total sum of squares of $Y$

**Interpretation:**
- VIP > 1: Variable is important for the model
- VIP < 0.8: Variable may be a candidate for removal
- VIP = 1: Average contribution

### When to Use PLS

**Strengths:**
- Designed specifically for spectroscopy applications
- Handles severe multicollinearity in spectral data
- Works when $p >> n$ (more wavelengths than samples)
- Provides interpretable latent variables
- Fast training and prediction
- VIP scores enable variable selection

**Weaknesses:**
- Assumes linear relationships between spectra and properties
- May underperform for highly non-linear relationships
- Requires careful selection of n_components
- Sensitive to outliers in training data

**Best for:**
- First-pass modeling of any spectroscopy problem
- When interpretability is important
- Small sample sizes (n < 100)
- Real-time predictions where speed matters

### Spectroscopy-Specific Considerations

1. **Preprocessing**: PLS often works best with SNV or derivatives to remove baseline effects
2. **Number of components**: Start with cross-validation across 1-15 components
3. **Overfitting risk**: More components = better training fit but potential overfitting
4. **Wavelength selection**: Use VIP scores to identify important spectral regions
5. **Outlier sensitivity**: Consider robust PLS variants for contaminated data

### Implementation Details

```python
# Spectral Predict uses sklearn's PLSRegression
from sklearn.cross_decomposition import PLSRegression

model = PLSRegression(
    n_components=n_components,  # Tested 1 to max_n_components
    max_iter=500,               # Default
    tol=1e-6,                   # Default
    scale=False                 # Preprocessing handled separately
)
```

---

## 2. PLS-DA (Partial Least Squares Discriminant Analysis)

### Full Name and Abbreviation

**Partial Least Squares Discriminant Analysis (PLS-DA)**

### Mathematical Foundation

PLS-DA extends PLS for classification by treating class labels as a continuous response variable. The approach uses regression to predict class membership, then applies a decision rule.

For binary classification with classes $\{0, 1\}$:

$$Y = TP^T + E$$

Where $Y$ is encoded as 0 or 1, and the prediction $\hat{y}$ is converted to class labels:

$$\hat{c} = \begin{cases} 1 & \text{if } \hat{y} > \tau \\ 0 & \text{otherwise} \end{cases}$$

The threshold $\tau$ is typically 0.5 but can be optimized for imbalanced classes.

### Algorithm Explanation

Spectral Predict implements PLS-DA as a two-stage pipeline:

**Stage 1: PLS Transformation**
- Extract latent variables that maximize covariance with class labels
- Transform high-dimensional spectra to low-dimensional scores

**Stage 2: Logistic Regression Classification**
- Use PLS scores as input features
- Apply logistic regression for probabilistic classification

$$P(y=1|T) = \frac{1}{1 + e^{-(\beta_0 + \beta^T T)}}$$

This approach provides:
- Probability estimates for each class
- Better handling of multiclass problems
- More robust decision boundaries

### Key Hyperparameters

**PLS Transformation Parameters:**

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_components` | int | 1-20 | Auto | Number of PLS components |
| `max_iter` | int | 100-1000 | 500 | NIPALS iterations |
| `tol` | float | 1e-8 to 1e-4 | 1e-6 | Convergence tolerance |

**Logistic Regression Parameters:**

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `lr_C` | float | 0.001-1000 | 1.0 | Inverse regularization strength |
| `lr_solver` | str | lbfgs, saga | lbfgs | Optimization algorithm |
| `lr_max_iter` | int | 100-5000 | 1000 | Maximum iterations |

### Threshold Setting

For imbalanced classes, optimizing the classification threshold can improve performance:

$$\tau_{opt} = \arg\max_\tau F_1(\tau)$$

Or using Youden's J statistic:

$$\tau_{opt} = \arg\max_\tau (\text{Sensitivity} + \text{Specificity} - 1)$$

### When to Use PLS-DA

**Strengths:**
- Excellent for spectroscopic classification
- Handles multicollinearity naturally
- Works with $p >> n$
- Provides class probabilities
- VIP scores identify discriminating wavelengths

**Weaknesses:**
- Assumes linear class boundaries in latent space
- Sensitive to class imbalance
- May struggle with highly non-linear class separations
- Requires careful component selection

**Best for:**
- Material identification
- Authenticity/adulteration detection
- Quality grade classification
- Any classification where interpretability matters

### Spectroscopy-Specific Considerations

1. **Class balance**: Check class distribution; use stratified sampling
2. **Component selection**: Cross-validate accuracy, not RMSE
3. **Multiclass**: Use one-vs-rest or softmax extension
4. **Preprocessing**: Same considerations as PLS regression
5. **Validation**: Use confusion matrix, not just accuracy

---

## 3. Ridge Regression (L2 Regularization)

### Full Name and Abbreviation

**Ridge Regression**, also known as **Tikhonov Regularization** or **L2-Penalized Regression**

### Mathematical Foundation

Ridge regression adds an L2 penalty to ordinary least squares to control model complexity:

$$\min_w ||Xw - y||_2^2 + \alpha||w||_2^2$$

The closed-form solution is:

$$\hat{w} = (X^T X + \alpha I)^{-1} X^T y$$

Where:
- $X$ is the $(n \times p)$ design matrix
- $y$ is the $(n \times 1)$ response vector
- $w$ is the $(p \times 1)$ coefficient vector
- $\alpha$ is the regularization parameter
- $I$ is the identity matrix

### Algorithm Explanation

The L2 penalty has several effects:

1. **Shrinkage**: Coefficients are shrunk toward zero
2. **Stabilization**: $(X^T X + \alpha I)$ is always invertible
3. **Bias-variance tradeoff**: Adds bias but reduces variance

The amount of shrinkage depends on eigenvalue structure:

$$\hat{w}_j^{ridge} = \frac{d_j^2}{d_j^2 + \alpha} \hat{w}_j^{OLS}$$

Where $d_j$ are singular values of $X$. Small eigenvalues (multicollinearity) are heavily regularized.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `alpha` | float | 0.001-1000 | 1.0 | Regularization strength |
| `solver` | str | auto, svd, cholesky, lsqr, sparse_cg, sag, saga | auto | Optimization algorithm |
| `tol` | float | 1e-6 to 1e-3 | 1e-4 | Convergence tolerance |

**Alpha Selection:**

| Alpha Range | Effect | Best When |
|-------------|--------|-----------|
| 0.001-0.1 | Minimal regularization | Good signal-to-noise |
| 0.1-10 | Moderate regularization | Typical spectroscopy |
| 10-1000 | Heavy regularization | Severe multicollinearity |

### When to Use Ridge Regression

**Strengths:**
- Very fast training (closed-form solution)
- Handles multicollinearity well
- All features retained (no variable selection)
- Stable predictions
- Interpretable coefficients

**Weaknesses:**
- No automatic feature selection
- Coefficients never exactly zero
- Assumes linear relationship
- All wavelengths contribute to prediction

**Best for:**
- When all spectral features are potentially relevant
- Fast initial modeling
- When feature selection is done separately
- Baseline model for comparison

### Spectroscopy-Specific Considerations

1. **Scaling**: Ridge assumes features are on similar scales; standardize if not preprocessing
2. **Wavelength importance**: Use absolute coefficient values as importance proxy
3. **Comparison to PLS**: Ridge often performs similarly but without latent variable interpretation
4. **Large p**: Solver='svd' is numerically stable for high-dimensional data

---

## 4. Lasso Regression (L1 Regularization)

### Full Name and Abbreviation

**Lasso Regression** (Least Absolute Shrinkage and Selection Operator), also known as **L1-Penalized Regression**

### Mathematical Foundation

Lasso uses an L1 penalty that induces sparsity in coefficients:

$$\min_w \frac{1}{2n}||Xw - y||_2^2 + \alpha||w||_1$$

Where:

$$||w||_1 = \sum_{j=1}^{p} |w_j|$$

The L1 penalty creates a non-smooth optimization problem with no closed-form solution.

### Algorithm Explanation

Lasso's key property is **automatic feature selection**:

1. The L1 penalty creates corners in the constraint region
2. Optimal solutions often lie at corners
3. At corners, some coefficients are exactly zero

**Coordinate Descent Solution:**

For each coefficient $w_j$:

$$w_j = S_{\alpha/n}\left(\frac{1}{n}\sum_{i=1}^{n} x_{ij}(y_i - \hat{y}_i^{(-j)})\right)$$

Where $S_\lambda$ is the soft-thresholding operator:

$$S_\lambda(z) = \begin{cases} z - \lambda & \text{if } z > \lambda \\ 0 & \text{if } |z| \leq \lambda \\ z + \lambda & \text{if } z < -\lambda \end{cases}$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `alpha` | float | 0.001-10 | 1.0 | Regularization strength |
| `selection` | str | cyclic, random | cyclic | Coordinate selection method |
| `tol` | float | 1e-6 to 1e-3 | 1e-4 | Convergence tolerance |
| `max_iter` | int | 100-5000 | 500 | Maximum iterations |

**Alpha Effect on Sparsity:**

| Alpha | Typical Sparsity | Effect |
|-------|------------------|--------|
| 0.001 | ~10% zeros | Minimal selection |
| 0.01 | ~30% zeros | Light selection |
| 0.1 | ~60% zeros | Moderate selection |
| 1.0 | ~90% zeros | Heavy selection |

### When to Use Lasso Regression

**Strengths:**
- Automatic wavelength selection
- Sparse, interpretable models
- Identifies most important spectral regions
- Handles multicollinearity (by selection)
- Fast prediction with sparse model

**Weaknesses:**
- Unstable selection with correlated features
- May select only one of correlated wavelengths arbitrarily
- Can over-regularize important features
- No closed-form solution (iterative)

**Best for:**
- When you want to identify key wavelengths
- Building parsimonious models
- Understanding which spectral regions matter
- Reducing model complexity for deployment

### Spectroscopy-Specific Considerations

1. **Correlated wavelengths**: Lasso may arbitrarily select one of correlated bands
2. **Group selection**: Adjacent wavelengths often correlate; consider group lasso
3. **Stability selection**: Run Lasso multiple times on bootstrap samples
4. **Comparison to VIP**: Different selection mechanisms; compare results
5. **Physical interpretation**: Selected wavelengths should match known absorption bands

---

## 5. ElasticNet (Combined L1/L2 Regularization)

### Full Name and Abbreviation

**Elastic Net Regularization**, combining L1 (Lasso) and L2 (Ridge) penalties

### Mathematical Foundation

ElasticNet combines both penalties to achieve the benefits of each:

$$\min_w \frac{1}{2n}||Xw - y||_2^2 + \alpha \rho ||w||_1 + \frac{\alpha(1-\rho)}{2}||w||_2^2$$

Where:
- $\alpha$ controls overall regularization strength
- $\rho$ (l1_ratio) controls the mix: $\rho=1$ is Lasso, $\rho=0$ is Ridge

The naive elastic net solution is:

$$\hat{w}^{enet} = \frac{1}{1 + \alpha(1-\rho)} \hat{w}^{naive}$$

### Algorithm Explanation

ElasticNet provides a "grouping effect" not present in pure Lasso:

1. **Group selection**: Correlated features tend to be selected together
2. **Stability**: More stable than Lasso with correlated predictors
3. **Shrinkage + Selection**: Combines both mechanisms

The optimization uses coordinate descent with modified soft-thresholding:

$$w_j = \frac{S_{\alpha\rho}(z_j)}{1 + \alpha(1-\rho)}$$

Where $z_j$ is the partial residual correlation.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `alpha` | float | 0.01-10 | 1.0 | Overall regularization strength |
| `l1_ratio` | float | 0.0-1.0 | 0.5 | Mix between L1 and L2 |
| `selection` | str | cyclic, random | cyclic | Coordinate selection |
| `tol` | float | 1e-6 to 1e-3 | 1e-4 | Convergence tolerance |

**l1_ratio Guidelines:**

| l1_ratio | Behavior | Best When |
|----------|----------|-----------|
| 0.1-0.3 | Ridge-like, grouped selection | Strong multicollinearity |
| 0.4-0.6 | Balanced | General spectroscopy |
| 0.7-0.9 | Lasso-like, sparse | Feature selection emphasis |

### When to Use ElasticNet

**Strengths:**
- Best of both Ridge and Lasso
- Selects groups of correlated features
- More stable than Lasso
- Handles multicollinearity better than Lasso
- Still provides sparse solutions

**Weaknesses:**
- Two hyperparameters to tune
- Computationally slower than Ridge
- Less interpretable than pure Lasso

**Best for:**
- Spectral data with correlated wavelength groups
- When Lasso is unstable but selection is desired
- Balancing prediction accuracy and interpretability

### Spectroscopy-Specific Considerations

1. **Adjacent wavelengths**: ElasticNet tends to select bands of consecutive wavelengths
2. **Chemical interpretation**: Grouped selection often matches actual absorption bands
3. **Preprocessing interaction**: Derivatives may reduce the advantage over Lasso
4. **Hyperparameter search**: Use 2D grid search over (alpha, l1_ratio)
5. **Default recommendation**: Start with l1_ratio=0.5, alpha=0.1

---

## 6. Random Forest

### Full Name and Abbreviation

**Random Forest (RF)**, an ensemble of decision trees with bagging and feature randomization

### Mathematical Foundation

Random Forest builds an ensemble of decorrelated decision trees:

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Where:
- $B$ = number of trees (n_estimators)
- $T_b(x)$ = prediction of tree $b$

Each tree is trained on:
1. **Bootstrap sample**: Random sample with replacement
2. **Random feature subset**: $m$ features at each split ($m \approx \sqrt{p}$ for classification, $m \approx p/3$ for regression)

### Algorithm Explanation

**Building a Single Tree:**

1. Draw bootstrap sample of size $n$ from training data
2. At each node:
   - Randomly select $m$ features
   - Find best split among these $m$ features
   - Split node into two children
3. Grow tree to maximum depth (no pruning)

**Split Criterion (Regression):**

$$\text{MSE} = \frac{1}{n_L} \sum_{i \in L} (y_i - \bar{y}_L)^2 + \frac{1}{n_R} \sum_{i \in R} (y_i - \bar{y}_R)^2$$

**Feature Importance:**

Mean Decrease in Impurity (MDI):

$$\text{Importance}_j = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \mathbb{1}(j_t = j) \cdot \Delta\text{Impurity}_t$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 100-1000 | 200 | Number of trees |
| `max_depth` | int/None | 5-None | None | Maximum tree depth |
| `max_features` | str/float | sqrt, log2, 0.1-1.0 | sqrt | Features per split |
| `min_samples_split` | int | 2-20 | 2 | Minimum samples to split |
| `min_samples_leaf` | int | 1-10 | 1 | Minimum samples in leaf |
| `bootstrap` | bool | True/False | True | Use bootstrap sampling |
| `max_leaf_nodes` | int/None | 10-None | None | Maximum leaf nodes |

**Effect of Key Parameters:**

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| n_estimators | Faster, less stable | Slower, more stable |
| max_depth | Underfitting | Overfitting |
| max_features | More diversity | Better individual trees |
| min_samples_split | More splits, overfitting | Fewer splits, underfitting |

### When to Use Random Forest

**Strengths:**
- Handles non-linear relationships
- Robust to outliers
- Built-in feature importance
- Minimal hyperparameter tuning needed
- Works well out-of-box
- Parallelizable

**Weaknesses:**
- Less interpretable than linear models
- Can overfit with deep trees
- Memory intensive with many trees
- May struggle with extrapolation
- Feature importance can be biased

**Best for:**
- Non-linear spectral responses
- Quick baseline with good performance
- When interpretability is secondary
- Large datasets with complex patterns

### Spectroscopy-Specific Considerations

1. **max_features**: Use 'sqrt' for classification, 1.0 (all) for regression with spectral data
2. **Feature importance**: Provides wavelength importance, but may be biased toward continuous features
3. **Extrapolation**: RF cannot extrapolate beyond training range; be cautious with predictions outside calibration range
4. **High dimensions**: Works well with thousands of wavelengths
5. **Preprocessing**: Often performs well without preprocessing; experiment with raw vs. preprocessed

---

## 7. XGBoost (Extreme Gradient Boosting)

### Full Name and Abbreviation

**XGBoost (eXtreme Gradient Boosting)**, a regularized gradient boosting framework

### Mathematical Foundation

XGBoost optimizes a regularized objective function:

$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

Where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2 + \alpha||w||_1$$

Where:
- $l(y_i, \hat{y}_i)$ = loss function (MSE for regression, log-loss for classification)
- $T$ = number of leaves in tree
- $w$ = leaf weights
- $\gamma$ = complexity penalty for tree structure
- $\lambda$ = L2 regularization on leaf weights
- $\alpha$ = L1 regularization on leaf weights

### Algorithm Explanation

XGBoost builds trees sequentially, each fitting the gradient of the loss:

**Gradient Boosting Update:**

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$$

Where $\eta$ is the learning rate and $f_t$ is the new tree.

**Newton-Raphson Optimization:**

XGBoost uses second-order gradients for better optimization:

$$g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$$

**Optimal Leaf Weight:**

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Split Gain:**

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 50-500 | 100 | Number of boosting rounds |
| `learning_rate` | float | 0.01-0.3 | 0.1 | Step size shrinkage |
| `max_depth` | int | 3-10 | 6 | Maximum tree depth |
| `subsample` | float | 0.5-1.0 | 0.8 | Row sampling ratio |
| `colsample_bytree` | float | 0.5-1.0 | 0.6 | Column sampling per tree |
| `reg_alpha` | float | 0-1.0 | 0.2 | L1 regularization |
| `reg_lambda` | float | 0.5-2.0 | 1.5 | L2 regularization |
| `min_child_weight` | float | 1-10 | 1 | Minimum sum of instance weight in child |
| `gamma` | float | 0-0.5 | 0 | Minimum loss reduction to split |

**Parameter Interactions:**

| Overfitting | Underfitting |
|-------------|--------------|
| Decrease learning_rate | Increase learning_rate |
| Increase reg_alpha, reg_lambda | Decrease regularization |
| Decrease max_depth | Increase max_depth |
| Increase subsample | Decrease subsample |

### When to Use XGBoost

**Strengths:**
- State-of-the-art accuracy for tabular data
- Built-in regularization
- Handles missing values
- Feature importance measures
- Fast with histogram-based splitting
- Handles imbalanced data

**Weaknesses:**
- Many hyperparameters to tune
- Can overfit without careful tuning
- Less interpretable than linear models
- Sensitive to hyperparameter settings

**Best for:**
- Maximum predictive accuracy
- Complex non-linear relationships
- When regularization is important
- Kaggle-style competitive modeling

### Spectroscopy-Specific Considerations

1. **colsample_bytree**: Lower values (0.6-0.8) help with highly correlated wavelengths
2. **reg_alpha**: L1 regularization helps with feature selection
3. **tree_method**: 'hist' is faster for high-dimensional spectral data
4. **max_depth**: Keep moderate (3-6) to prevent overfitting
5. **Early stopping**: Monitor validation loss to prevent overfitting

---

## 8. LightGBM (Light Gradient Boosting Machine)

### Full Name and Abbreviation

**LightGBM (Light Gradient Boosting Machine)**, a fast gradient boosting implementation by Microsoft

### Mathematical Foundation

LightGBM uses the same gradient boosting framework as XGBoost but with key algorithmic innovations:

$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(x)$$

**Gradient-based One-Side Sampling (GOSS):**

Keeps all instances with large gradients, randomly samples those with small gradients:

$$\text{Weight amplification} = \frac{1 - a}{b}$$

Where $a$ = fraction of large gradient instances, $b$ = fraction of small gradient instances.

### Algorithm Explanation

**Leaf-wise (Best-first) Tree Growth:**

Unlike XGBoost's level-wise growth, LightGBM grows leaf-wise:
- Choose the leaf with maximum delta loss
- Split that leaf
- Continue until num_leaves reached

This is more efficient but requires `num_leaves` control to prevent overfitting.

**Histogram-based Splitting:**

1. Bucket continuous features into discrete bins
2. Compute histograms for gradient statistics
3. Find best split from histograms

Benefits:
- $O(data \times features)$ to $O(data + 2^{bins} \times features)$
- Cache-efficient memory access
- Faster than exact algorithms

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 50-500 | 100 | Number of boosting rounds |
| `learning_rate` | float | 0.01-0.3 | 0.1 | Step size shrinkage |
| `num_leaves` | int | 15-127 | 15 | Maximum number of leaves per tree |
| `max_depth` | int | -1 to 15 | 10 | Maximum tree depth (-1 = unlimited) |
| `min_child_samples` | int | 5-100 | 5 | Minimum samples in leaf |
| `subsample` | float | 0.5-1.0 | 0.8 | Row sampling ratio |
| `colsample_bytree` | float | 0.5-1.0 | 0.8 | Column sampling per tree |
| `reg_alpha` | float | 0-1.0 | 0.3 | L1 regularization |
| `reg_lambda` | float | 0.5-2.0 | 1.5 | L2 regularization |

**num_leaves vs max_depth:**

Unlike XGBoost, LightGBM primarily controls complexity via `num_leaves`:

$$\text{num\_leaves} \approx 2^{\text{max\_depth}}$$

For spectral data, use `num_leaves = 15-31` to prevent overfitting.

### When to Use LightGBM

**Strengths:**
- Fastest gradient boosting implementation
- Memory efficient
- Excellent accuracy
- Handles large datasets well
- Good for high-dimensional data
- Built-in categorical feature handling

**Weaknesses:**
- Can overfit with small datasets
- num_leaves requires careful tuning
- Less mature than XGBoost
- Sensitive to outliers

**Best for:**
- Large spectral datasets
- When training speed matters
- High-dimensional data
- Production deployment

### Spectroscopy-Specific Considerations

1. **num_leaves**: Keep low (15-31) for spectral data to prevent overfitting
2. **min_child_samples**: Set higher (5-20) for small spectral datasets
3. **Speed advantage**: Most pronounced with >1000 samples and >500 wavelengths
4. **Bagging**: Enable subsample with bagging_freq=1 for regularization
5. **verbosity=-1**: Suppress warnings during grid search

---

## 9. CatBoost (Categorical Boosting)

### Full Name and Abbreviation

**CatBoost (Categorical Boosting)**, gradient boosting with ordered boosting by Yandex

### Mathematical Foundation

CatBoost introduces "ordered boosting" to prevent target leakage:

**Ordered Boosting:**

For each sample $i$, residuals are calculated using a model trained only on samples that come before $i$ in a random permutation:

$$r_i^{(t)} = y_i - M^{(t)}_{i-1}(x_i)$$

Where $M^{(t)}_{i-1}$ is trained on samples $\{1, ..., i-1\}$ in the current permutation.

This prevents the gradient bias that can occur in standard gradient boosting.

### Algorithm Explanation

**Symmetric Trees:**

CatBoost uses oblivious (symmetric) decision trees:
- Same splitting criterion at each level
- All nodes at same depth use identical split
- Creates balanced trees with $2^d$ leaves for depth $d$

Benefits:
- Faster inference (SIMD-friendly)
- Natural regularization
- Interpretable structure

**Gradient Estimation:**

Uses ordered statistics to compute unbiased gradient estimates, reducing overfitting.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `iterations` | int | 50-500 | 100 | Number of trees |
| `learning_rate` | float | 0.01-0.3 | 0.1 | Step size shrinkage |
| `depth` | int | 4-10 | 5 | Tree depth |
| `l2_leaf_reg` | float | 1-10 | 4.0 | L2 regularization on leaves |
| `min_data_in_leaf` | int | 1-20 | 1 | Minimum samples in leaf |
| `bootstrap_type` | str | Bayesian, Bernoulli, MVS | Bayesian | Sampling method |
| `bagging_temperature` | float | 0-10 | 1.0 | Bayesian bootstrap parameter |
| `random_strength` | float | 0-10 | 1.0 | Score randomization strength |
| `border_count` | int | 32-255 | 128 | Number of splits for numerical features |

**bootstrap_type Options:**

| Type | Description | Best For |
|------|-------------|----------|
| Bayesian | Weights from exponential distribution | Default, good regularization |
| Bernoulli | Random subset of samples | Large datasets |
| MVS | Minimum variance sampling | Small datasets |

### When to Use CatBoost

**Strengths:**
- Best handling of categorical features
- Ordered boosting reduces overfitting
- Symmetric trees are fast at inference
- Robust to hyperparameter settings
- GPU acceleration available
- Built-in cross-validation

**Weaknesses:**
- Slower training than LightGBM
- Memory intensive
- Less flexible tree structure

**Best for:**
- Mixed numerical/categorical data
- When overfitting is a concern
- Production deployment (fast inference)
- When minimal tuning time available

### Spectroscopy-Specific Considerations

1. **depth**: Use 4-6 for spectral data; deeper trees overfit
2. **Categorical handling**: Useful if metadata (sample type, operator) included
3. **Ordered boosting**: Provides natural regularization for small datasets
4. **l2_leaf_reg**: Increase for more regularization (try 3.0-5.0)
5. **silent mode**: Set verbose=False to suppress training output

---

## 10. SVR/SVC (Support Vector Machines)

### Full Name and Abbreviation

**Support Vector Regression (SVR)** and **Support Vector Classification (SVC)**

### Mathematical Foundation

SVMs find a hyperplane that maximizes the margin while minimizing error:

**Classification (SVC):**

$$\min_{w, b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \xi_i$$

Subject to: $y_i(w^T x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$

**Regression (SVR):**

$$\min_{w, b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} (\xi_i + \xi_i^*)$$

Subject to: $|y_i - (w^T x_i + b)| \leq \epsilon + \xi_i$

### The Kernel Trick

SVMs use kernels to operate in high-dimensional spaces without explicit computation:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

**Common Kernels:**

**Linear:**
$$K(x_i, x_j) = x_i^T x_j$$

**Radial Basis Function (RBF):**
$$K(x_i, x_j) = \exp(-\gamma||x_i - x_j||^2)$$

**Polynomial:**
$$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$$

### Algorithm Explanation

The dual formulation uses Lagrange multipliers:

$$\max_\alpha \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

Subject to: $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

Predictions use support vectors (samples with $\alpha_i > 0$):

$$f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `C` | float | 0.1-1000 | 1.0 | Regularization (inverse) |
| `kernel` | str | rbf, linear, poly | rbf | Kernel function |
| `gamma` | float/str | 0.001-1.0 or 'scale' | 'scale' | RBF kernel width |
| `epsilon` | float | 0.01-0.5 | 0.1 | SVR tube width |
| `degree` | int | 2-5 | 3 | Polynomial degree |
| `coef0` | float | 0-1 | 0 | Polynomial/sigmoid intercept |
| `shrinking` | bool | True/False | True | Use shrinking heuristic |
| `max_iter` | int | 1000-10000 | 5000 | Maximum iterations |

**Kernel Selection:**

| Kernel | Complexity | Best For |
|--------|------------|----------|
| linear | Low | High-dimensional, linearly separable |
| rbf | Medium | General non-linear, default choice |
| poly | High | Polynomial relationships |

### When to Use SVM

**Strengths:**
- Effective in high dimensions
- Memory efficient (uses support vectors only)
- Versatile through kernel selection
- Robust to overfitting with proper C
- Works well with clear margin of separation

**Weaknesses:**
- Slow training for large datasets
- Sensitive to feature scaling
- Kernel selection requires tuning
- No probability estimates without extra computation (classification)
- Poor with noisy data

**Best for:**
- Small to medium datasets (n < 1000)
- Clear class separation
- When memory is limited
- Non-linear relationships with RBF

### Spectroscopy-Specific Considerations

1. **Scaling**: Essential! Use StandardScaler or SNV preprocessing
2. **Kernel choice**: RBF works well for most spectral problems
3. **gamma='scale'**: Good default (1 / (n_features * X.var()))
4. **Training time**: Slow for large spectral datasets; consider subsampling
5. **max_iter**: Increase to 5000+ to ensure convergence

---

## 11. MLP (Multi-Layer Perceptron)

### Full Name and Abbreviation

**Multi-Layer Perceptron (MLP)**, a feedforward artificial neural network

### Mathematical Foundation

An MLP with $L$ hidden layers computes:

**Forward Propagation:**

$$a^{(0)} = x$$
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = g(z^{(l)})$$
$$\hat{y} = a^{(L+1)}$$

Where:
- $W^{(l)}$ = weight matrix for layer $l$
- $b^{(l)}$ = bias vector for layer $l$
- $g(\cdot)$ = activation function

**Loss Function (Regression):**

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \frac{\alpha}{2}\sum_l ||W^{(l)}||_F^2$$

### Algorithm Explanation

**Activation Functions:**

**ReLU (Rectified Linear Unit):**
$$g(z) = \max(0, z)$$

**Tanh:**
$$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Logistic (Sigmoid):**
$$g(z) = \frac{1}{1 + e^{-z}}$$

**Backpropagation:**

Gradients are computed by chain rule:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

Where $\delta^{(l)}$ is the error term for layer $l$.

**Optimization (Adam):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t)$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `hidden_layer_sizes` | tuple | (16,) to (256, 128) | (64,) | Network architecture |
| `alpha` | float | 1e-5 to 0.1 | 0.001 | L2 regularization |
| `learning_rate_init` | float | 1e-4 to 0.1 | 0.001 | Initial learning rate |
| `activation` | str | relu, tanh, logistic | relu | Activation function |
| `solver` | str | adam, sgd, lbfgs | adam | Optimization algorithm |
| `batch_size` | int/str | 32-256 or 'auto' | 'auto' | Mini-batch size |
| `learning_rate` | str | constant, invscaling, adaptive | constant | Learning rate schedule |
| `momentum` | float | 0-0.99 | 0.9 | SGD momentum |
| `max_iter` | int | 100-1000 | 500 | Maximum epochs |
| `early_stopping` | bool | True/False | True | Use validation for stopping |

**Architecture Guidelines:**

| Dataset Size | Recommended Architecture |
|--------------|-------------------------|
| < 100 samples | (16,) or (32,) |
| 100-500 samples | (64,) or (32, 32) |
| 500-1000 samples | (128,) or (64, 32) |
| > 1000 samples | (128, 64) or (256, 128) |

### When to Use MLP

**Strengths:**
- Universal function approximator
- Can learn complex non-linear patterns
- Flexible architecture
- Good for large datasets
- Transfer learning potential

**Weaknesses:**
- Many hyperparameters
- Prone to overfitting
- Requires careful tuning
- Not interpretable
- Sensitive to initialization

**Best for:**
- Large datasets (n > 500)
- Complex non-linear relationships
- When other methods plateau
- When interpretability not needed

### Spectroscopy-Specific Considerations

1. **Input scaling**: Essential! Use StandardScaler or SNV
2. **Architecture**: Start simple (one hidden layer)
3. **early_stopping=True**: Critical to prevent overfitting
4. **alpha regularization**: Use 0.001-0.01 for spectral data
5. **Learning rate**: Adam usually works; try 0.001 first
6. **Validation**: Use 10-20% for early stopping

---

## 12. NeuralBoosted

### Full Name and Abbreviation

**Neural Boosted (NB)**, gradient boosting with neural network weak learners

### Mathematical Foundation

NeuralBoosted combines gradient boosting with small neural networks:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

Where:
- $F_m$ = ensemble prediction after $m$ rounds
- $\nu$ = learning rate (shrinkage)
- $h_m$ = neural network weak learner fitted to residuals

**Regression Loss:**

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (y_i - F_m(x_i))^2$$

Or with Huber loss for robustness:

$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta(|r| - \frac{\delta}{2}) & |r| > \delta \end{cases}$$

### Algorithm Explanation

**Training Procedure:**

1. **Initialize**: $F_0(x) = 0$
2. **For $m = 1$ to $M$ (n_estimators):**
   - Compute residuals: $r_i = y_i - F_{m-1}(x_i)$
   - Fit small MLP $h_m$ to residuals
   - Update ensemble: $F_m = F_{m-1} + \nu \cdot h_m$
   - Check early stopping criterion
3. **Return**: $F_M(x)$

**Weak Learner Architecture:**

Each weak learner is a small MLP:
- Single hidden layer
- 3-5 hidden nodes (intentionally weak)
- tanh activation (smooth gradients)

**Feature Importance:**

Aggregated from input layer weights:

$$\text{Importance}_j = \frac{1}{M}\sum_{m=1}^{M} \frac{1}{H}\sum_{h=1}^{H} |W_{jh}^{(m)}|$$

Where $W_{jh}^{(m)}$ is the weight from feature $j$ to hidden node $h$ in learner $m$.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 50-200 | 100 | Number of weak learners |
| `learning_rate` | float | 0.05-0.3 | 0.1 | Shrinkage parameter |
| `hidden_layer_size` | int | 3-8 | 3 (reg), 5 (clf) | Hidden nodes per weak learner |
| `activation` | str | tanh, relu, identity | tanh | Activation function |
| `alpha` | float | 1e-5 to 0.001 | 0.0001 | L2 regularization |
| `max_iter` | int | 50-200 | 100 | Max iterations per weak learner |
| `early_stopping` | bool | True/False | True | Use validation for stopping |
| `validation_fraction` | float | 0.1-0.2 | 0.15 | Validation set size |
| `n_iter_no_change` | int | 5-20 | 10 | Patience for early stopping |
| `loss` | str | mse, huber | mse | Loss function (regression) |

**Classification-Specific:**

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `early_stopping_metric` | str | accuracy, log_loss | accuracy | Validation metric |
| `class_weight` | str/dict | None, 'balanced' | None | Handle imbalanced classes |

### When to Use NeuralBoosted

**Strengths:**
- Combines boosting stability with neural network flexibility
- Handles non-linear patterns
- Feature importance scores
- More interpretable than deep MLPs
- Robust to overfitting with early stopping

**Weaknesses:**
- Slower than tree-based boosting
- More hyperparameters than RF
- Less established than XGBoost/LightGBM
- Requires more memory

**Best for:**
- Non-linear relationships
- When tree-based methods plateau
- Moderate-sized datasets
- Alternative to pure MLP

### Spectroscopy-Specific Considerations

1. **hidden_layer_size**: Keep small (3-5) for spectral data
2. **activation='tanh'**: Often works best for spectroscopy
3. **learning_rate**: Start with 0.1, decrease if overfitting
4. **early_stopping**: Essential for spectral data
5. **Comparison**: Test against XGBoost and LightGBM

---

## Model Selection Guidelines

### Decision Framework

Use this flowchart to select an appropriate model:

```
START
  |
  v
[Sample size < 100?] --YES--> PLS or PLS-DA
  |
  NO
  |
  v
[Interpretability critical?] --YES--> PLS, Ridge, Lasso, or ElasticNet
  |
  NO
  |
  v
[Training time limited?] --YES--> RandomForest or LightGBM
  |
  NO
  |
  v
[Need maximum accuracy?] --YES--> XGBoost, LightGBM, or CatBoost (ensemble)
  |
  NO
  |
  v
[Complex non-linear patterns?] --YES--> MLP or NeuralBoosted
  |
  NO
  |
  v
Default: Start with PLS, then try RandomForest, then gradient boosting
```

### Model Selection by Problem Type

| Problem Type | Primary Recommendation | Alternative |
|--------------|----------------------|-------------|
| Quantitative analysis | PLS | Ridge, XGBoost |
| Qualitative analysis | PLS-DA | RandomForest, LightGBM |
| Small samples (< 50) | PLS, Ridge | Lasso |
| Large samples (> 1000) | LightGBM, XGBoost | RandomForest |
| Real-time prediction | PLS, Ridge | Sparse Lasso |
| Feature selection | Lasso, ElasticNet | VIP (PLS) |
| Non-linear | XGBoost, LightGBM | MLP, NeuralBoosted |
| Regulatory compliance | PLS | Ridge |

### Model Performance Expectations

**Typical R2 Ranges by Model (well-optimized):**

| Model | Low R2 Data | Medium R2 Data | High R2 Data |
|-------|-------------|----------------|--------------|
| PLS | 0.6-0.7 | 0.8-0.85 | 0.9-0.95 |
| Ridge | 0.55-0.65 | 0.75-0.82 | 0.88-0.93 |
| RandomForest | 0.5-0.6 | 0.75-0.85 | 0.88-0.94 |
| XGBoost | 0.55-0.7 | 0.8-0.88 | 0.9-0.96 |
| LightGBM | 0.55-0.7 | 0.8-0.88 | 0.9-0.96 |

---

## Hyperparameter Tuning Strategies

### General Principles

1. **Start simple**: Use default hyperparameters first
2. **One at a time**: Understand each parameter's effect
3. **Use cross-validation**: Always validate on held-out data
4. **Grid vs. random**: Random search often more efficient
5. **Early stopping**: Prevent overfitting during tuning

### Recommended Search Order

**For PLS:**
1. n_components (most important)
2. Preprocessing method
3. max_iter/tol (rarely needed)

**For Ridge/Lasso/ElasticNet:**
1. alpha (regularization strength)
2. l1_ratio (ElasticNet only)
3. Preprocessing method

**For RandomForest:**
1. n_estimators (more is usually better)
2. max_depth
3. max_features
4. min_samples_split

**For XGBoost/LightGBM:**
1. learning_rate and n_estimators together
2. max_depth / num_leaves
3. subsample and colsample_bytree
4. reg_alpha and reg_lambda

**For MLP/NeuralBoosted:**
1. hidden_layer_sizes / hidden_layer_size
2. learning_rate_init / learning_rate
3. alpha (regularization)
4. Activation function

### Spectral Predict's Tier-Based Defaults

Spectral Predict provides optimized defaults for each tier:

**Standard Tier Defaults:**

| Model | Key Parameters |
|-------|----------------|
| PLS | n_components: 1-8, max_iter: 500 |
| Ridge | alpha: [0.001, 0.01, 0.1, 1.0, 10.0] |
| Lasso | alpha: [0.001, 0.01, 0.1, 1.0] |
| ElasticNet | alpha: [0.01, 0.1, 1.0], l1_ratio: [0.3, 0.5, 0.7] |
| RandomForest | n_estimators: [100], max_depth: [None, 30] |
| XGBoost | n_estimators: [100, 200], learning_rate: [0.05, 0.1], max_depth: [3, 6] |
| LightGBM | n_estimators: [100, 200], num_leaves: [15, 31], max_depth: [5, 10, -1] |
| CatBoost | iterations: [100, 200], depth: [4, 5, 6], l2_leaf_reg: [3.0, 5.0] |

These defaults are optimized for typical spectroscopy datasets and provide a good starting point for most applications.

---

## Appendix: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $X$ | Predictor matrix (n samples x p features) |
| $y$ | Response vector (n samples) |
| $w$ | Weight/coefficient vector |
| $\alpha$ | Regularization parameter |
| $\rho$ | L1/L2 mixing ratio (ElasticNet) |
| $\eta$ | Learning rate |
| $T$ | PLS scores matrix |
| $P$ | PLS loadings matrix |
| $K(\cdot, \cdot)$ | Kernel function |
| $\gamma$ | RBF kernel width / tree complexity penalty |
| $C$ | SVM regularization parameter |
| $\epsilon$ | SVR tube width |

---

## References

1. Wold, S., Sjostrom, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 2017.

5. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS 2018.

6. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Spectral Predict V1 User Guide - Part III*
