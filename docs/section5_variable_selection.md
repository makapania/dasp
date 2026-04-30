# Part V: Variable Selection Reference

## Overview

Variable selection (also called wavelength selection or feature selection) is a critical step in spectroscopic modeling that identifies the most informative spectral variables while removing noise, redundant information, and uninformative regions. Effective variable selection can:

- **Improve prediction performance** by focusing on relevant spectral features
- **Reduce model complexity** and prevent overfitting
- **Enhance interpretability** by identifying key spectral regions
- **Decrease computation time** for model training and prediction
- **Improve calibration transfer** between instruments

Spectral Predict implements a comprehensive suite of variable selection methods, ranging from simple importance-based approaches to sophisticated iterative algorithms.

---

## 1. Importance-Based Selection

Importance-based selection uses model-derived importance scores to rank and select the most informative variables. This approach is intuitive, computationally efficient, and applicable to any model that provides feature importance measures.

### 1.1 Model-Specific Importance Methods

#### 1.1.1 VIP (Variable Importance in Projection) for PLS

**Full Name:** Variable Importance in Projection

VIP scores quantify each variable's contribution to the PLS model by measuring how much the variable participates in explaining both the predictor (X) and response (Y) variance.

**Mathematical Formula:**

For a PLS model with $A$ components, the VIP score for variable $j$ is:

$$VIP_j = \sqrt{\frac{p \sum_{a=1}^{A} SS_{Y,a} \cdot w_{ja}^2}{\sum_{a=1}^{A} SS_{Y,a}}}$$

Where:
- $p$ = total number of variables
- $A$ = number of PLS components
- $SS_{Y,a}$ = sum of squares of $Y$ explained by component $a$
- $w_{ja}$ = weight of variable $j$ for component $a$

**Algorithm Steps:**

```
1. Fit PLS model with A components
2. Extract X-weights matrix W (p x A)
3. Extract X-scores matrix T (n x A)
4. For each component a = 1 to A:
   a. Compute SS_Y,a = sum(T[:,a]^2) * var(y)
5. Compute total SS_Y = sum(SS_Y,a for all a)
6. For each variable j = 1 to p:
   a. Compute weighted sum = sum(w[j,a]^2 * SS_Y,a for all a)
   b. VIP[j] = sqrt(p * weighted_sum / total_SS_Y)
7. Return VIP scores
```

**Interpretation:**
- VIP > 1.0: Variable is highly important (rule of thumb threshold)
- VIP between 0.8-1.0: Moderately important
- VIP < 0.8: Less important, candidate for elimination

**Implementation in Spectral Predict:**

```python
def compute_vip(pls_model, X, y):
    W = pls_model.x_weights_    # (n_features, n_components)
    T = pls_model.x_scores_     # (n_samples, n_components)

    # SSY for each component
    y = np.asarray(y).reshape(-1, 1)
    ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)
    ssy_total = np.sum(ssy_comp)

    # VIP calculation (vectorized)
    n_features = W.shape[0]
    weight = np.sum((W ** 2) * ssy_comp, axis=1)
    vip_scores = np.sqrt(n_features * weight / ssy_total)

    return vip_scores
```

---

#### 1.1.2 MDI (Mean Decrease in Impurity) for Tree-Based Models

**Full Name:** Mean Decrease in Impurity (also called Gini Importance)

MDI measures feature importance by computing the total reduction in node impurity (Gini index or entropy) brought by each feature across all trees in the ensemble.

**Mathematical Formula:**

For feature $j$ in a tree ensemble with $T$ trees:

$$MDI_j = \frac{1}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}_t^j} \frac{n_v}{n} \cdot \Delta I(v)$$

Where:
- $\mathcal{V}_t^j$ = set of nodes in tree $t$ that split on feature $j$
- $n_v$ = number of samples reaching node $v$
- $n$ = total number of samples
- $\Delta I(v)$ = impurity decrease at node $v$

**Impurity Decrease:**

$$\Delta I(v) = I(v) - \frac{n_{v,L}}{n_v} I(v_L) - \frac{n_{v,R}}{n_v} I(v_R)$$

Where $v_L$ and $v_R$ are the left and right child nodes.

**Applicability:**
- RandomForest: Uses `feature_importances_` attribute (MDI by default)
- XGBoost: Uses `feature_importances_` (gain-based by default)
- LightGBM: Uses `feature_importances_` (split-count or gain)
- CatBoost: Uses `feature_importances_`

**Strengths:**
- Computationally efficient (computed during training)
- Available for all tree ensemble models
- Handles non-linear relationships

**Limitations:**
- Biased toward high-cardinality features
- Can be misleading with correlated features
- Does not account for feature interactions directly

---

#### 1.1.3 Coefficient-Based Importance for Linear Models

For linear models (Ridge, Lasso, ElasticNet), feature importance is derived from the absolute value of regression coefficients:

$$Importance_j = |c_j|$$

Where $c_j$ is the fitted coefficient for feature $j$.

**Notes:**
- Features should be standardized for meaningful comparison
- Lasso naturally produces sparse coefficients (built-in selection)
- ElasticNet balances L1 (sparsity) and L2 (stability) regularization

---

#### 1.1.4 Permutation Importance

**Full Name:** Permutation Feature Importance

Permutation importance measures how much model performance decreases when a feature's values are randomly shuffled, breaking its relationship with the target.

**Algorithm:**

```
1. Fit model and compute baseline performance metric M_baseline
2. For each feature j = 1 to p:
   a. Create permuted dataset by shuffling column j
   b. Compute performance M_permuted on permuted data
   c. Importance[j] = M_baseline - M_permuted (or ratio)
3. Optionally repeat K times and average
4. Return importance scores
```

**Strengths:**
- Model-agnostic (works with any model)
- Accounts for feature interactions
- Uses actual model predictions

**Limitations:**
- Computationally expensive (requires multiple model evaluations)
- Can underestimate importance of correlated features
- Sensitive to randomness (multiple repetitions recommended)

---

### 1.2 Top-N Selection

Top-N selection is the simplest variable selection strategy: compute importance scores and keep the N most important variables.

**Standard N Values:**

| N | Use Case |
|---|----------|
| 10 | Very aggressive reduction, key wavelengths only |
| 20 | Aggressive reduction, sparse models |
| 50 | Moderate reduction, balanced |
| 100 | Conservative reduction, retains more information |
| 250 | Mild reduction, baseline for comparison |
| 500 | Light reduction, removes obvious noise |
| 1000 | Minimal reduction, large feature sets |

**Algorithm:**

```
1. Compute importance scores for all p variables
2. Sort variables by importance (descending)
3. Select top N variables with highest importance
4. Return selected indices or boolean mask
```

**Selection by Threshold vs Fixed Count:**

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Fixed Count (Top-N)** | Select exactly N variables | When model complexity must be controlled |
| **Threshold-Based** | Select variables where importance > threshold | When natural grouping exists (e.g., VIP > 1.0) |
| **Percentile-Based** | Select top X% of variables | When relative importance matters |

---

## 2. UVE (Uninformative Variable Elimination)

**Full Name:** Uninformative Variable Elimination

UVE is a statistical method that distinguishes informative variables from noise by comparing real variables against artificial noise variables with known zero information content.

### 2.1 Algorithm

**Detailed Steps:**

```
1. AUGMENTATION
   a. Generate noise matrix N of same size as X
   b. N[i,j] ~ Normal(0, 10^-10) for small noise amplitude
   c. Create augmented matrix X_aug = [X | N] (concatenate)

2. CROSS-VALIDATION MODELING
   For each CV fold k = 1 to K:
   a. Split X_aug into train/test sets
   b. Fit PLS model on augmented training data
   c. Extract regression coefficients c_k for all variables
   d. Store coefficients from each fold

3. RELIABILITY CALCULATION
   For each variable j = 1 to 2p (real + noise):
   a. Compute mean absolute coefficient: mean_j = mean(|c_j|) across folds
   b. Compute standard deviation: std_j = std(c_j) across folds
   c. Reliability score: R_j = mean_j / std_j

4. THRESHOLD DETERMINATION
   a. Extract reliability scores for noise variables: R_noise
   b. Compute threshold: T = max(R_noise) * cutoff_multiplier

5. VARIABLE SELECTION
   a. For real variables (j = 1 to p):
      - If R_j > T: variable is informative
      - If R_j <= T: variable is uninformative
   b. Return importances (reliability scores for real variables)
```

### 2.2 Mathematical Formulas

**Reliability Score:**

$$R_j = \frac{\overline{|c_j|}}{\sigma_{c_j}}$$

Where:
- $\overline{|c_j|}$ = mean of absolute coefficients across CV folds
- $\sigma_{c_j}$ = standard deviation of coefficients across CV folds

**Noise Threshold:**

$$T = \max_{j \in \text{noise}} (R_j) \times \text{cutoff\_multiplier}$$

**Selection Criterion:**

Variable $j$ is selected if: $R_j > T$

### 2.3 Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `cutoff_multiplier` | 1.0 | Multiplier for noise threshold | >1.0: more conservative (keep more); <1.0: more aggressive (eliminate more) |
| `n_components` | auto | Number of PLS components | Higher values capture more variance but risk overfitting |
| `cv_folds` | 5 | Number of CV folds | More folds = more stable but slower |
| `random_state` | 42 | Random seed for noise generation | Ensures reproducibility |

### 2.4 When to Use

**Strengths:**
- Statistically principled approach
- Automatic threshold determination
- No hyperparameter tuning for threshold
- Effective for removing pure noise variables

**Limitations:**
- Computationally expensive (multiple PLS fits)
- Assumes PLS is appropriate for the data
- May not work well with very few samples
- Noise amplitude selection can affect results

**Computational Considerations:**
- Time complexity: O(K * n * p) for K folds
- Memory: Requires storing augmented matrix (2p columns)
- Minimum features: 3 (for meaningful n_components)

**Reference:**
> Centner, V., Massart, D. L., de Noord, O. E., de Jong, S., Vandeginste, B. M., & Sterna, C. (1996). Elimination of uninformative variables for multivariate calibration. *Analytical Chemistry*, 68(21), 3851-3858.

---

## 3. SPA (Successive Projections Algorithm)

**Full Name:** Successive Projections Algorithm

SPA is a forward selection method that minimizes multicollinearity by iteratively selecting variables that are maximally uncorrelated with the already-selected set.

### 3.1 Algorithm

**Detailed Steps:**

```
1. INITIALIZATION
   a. Normalize X to zero mean and unit variance
   b. Compute correlation of each variable with y
   c. Select initial variable: j_0 = argmax(|corr(X[:,j], y)|)
   d. Initialize selected set S = {j_0}

2. ITERATIVE SELECTION
   For iteration i = 1 to n_features - 1:
   a. Extract selected subspace: X_selected = X[:, S]
   b. Compute QR decomposition: Q, R = qr(X_selected)
   c. For each remaining variable j not in S:
      i. Compute projection onto orthogonal complement:
         x_orth = X[:,j] - Q @ (Q^T @ X[:,j])
      ii. Compute projection norm: norm_j = ||x_orth||
   d. Select variable with maximum projection norm:
      j_next = argmax(norm_j) for j not in S
   e. Add to selected set: S = S ∪ {j_next}

3. QUALITY EVALUATION (canonical seed enumeration, Araújo 2001)
   For each candidate first variable k = 1 to J (every variable in turn):
   a. Run forward selection chain from variable k
   b. Build PLS model on selected variables
   c. Compute CV R² score
   d. Track best chain across all J seeds

4. IMPORTANCE SCORING
   a. Variables selected earlier get higher scores
   b. Score = n_features - rank (first selected = highest)
```

### 3.2 Mathematical Formulas

**Projection onto Orthogonal Complement:**

Given selected variables $\mathbf{X}_S$ with QR decomposition $\mathbf{Q}\mathbf{R}$:

$$\mathbf{x}_j^{\perp} = \mathbf{x}_j - \mathbf{Q}\mathbf{Q}^T\mathbf{x}_j$$

**Selection Criterion:**

$$j^* = \arg\max_{j \notin S} ||\mathbf{x}_j^{\perp}||_2$$

**Correlation (for initialization):**

$$\text{corr}(\mathbf{x}_j, \mathbf{y}) = \frac{\mathbf{x}_j^T \mathbf{y}}{||\mathbf{x}_j|| \cdot ||\mathbf{y}||}$$

### 3.3 Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `n_features` | required | Number of features to select | Directly controls output size |
| `cv_folds` | 5 | CV folds for quality evaluation | More folds = more stable scores |

SPA is fully deterministic: every variable is enumerated as a candidate first
variable per Araújo 2001, and the chain with the best CV criterion is returned.
There is no random initialization and no `random_state` parameter.

### 3.4 When to Use

**Strengths:**
- Explicitly minimizes multicollinearity
- Fast algorithm (O(n * p^2) worst case)
- Deterministic (with fixed initialization)
- Works well for spectroscopic data with high collinearity

**Limitations:**
- Does not directly optimize prediction performance
- Sequential nature may miss globally optimal subsets
- Initial variable choice affects final selection
- Assumes linear relationships

**Computational Considerations:**
- Time complexity: O(n_starts * n_features * p^2)
- Memory: O(n * p) for QR decomposition
- Minimum features: 3 for meaningful selection

**Reference:**
> Araujo, M. C. U., et al. (2001). The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73.

---

## 4. UVE-SPA Hybrid

**Full Name:** Uninformative Variable Elimination followed by Successive Projections Algorithm

The UVE-SPA hybrid combines the noise filtering capability of UVE with the collinearity reduction of SPA in a two-stage approach.

### 4.1 Algorithm

**Stage 1: UVE Filtering**
```
1. Apply UVE algorithm to full variable set
2. Compute reliability scores for all variables
3. Filter out variables below noise threshold
4. Keep informative variables only
```

**Stage 2: SPA Selection**
```
1. Apply SPA to UVE-filtered variables
2. Select n_features from the reduced set
3. Minimize collinearity among informative variables
```

**Combined Steps:**

```
1. Run UVE on X, y with cutoff_multiplier
2. Get UVE mask: variables with reliability > threshold
3. Create reduced matrix: X_uve = X[:, uve_mask]
4. Adjust n_features if UVE kept fewer variables
5. Run SPA on X_uve, y with adjusted n_features
6. Map SPA importances back to original indices
7. Return combined importances (0 for eliminated, SPA scores for kept)
```

### 4.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | required | Target number of final features |
| `cutoff_multiplier` | 1.0 | UVE threshold multiplier |
| `uve_n_components` | auto | PLS components for UVE |
| `uve_cv_folds` | 5 | CV folds for UVE |
| `spa_cv_folds` | 5 | CV folds for SPA evaluation |
| `random_state` | 42 | Random seed (UVE only; SPA is deterministic) |

### 4.3 When to Use

**Strengths:**
- Combines noise filtering with collinearity reduction
- Produces stable, interpretable selections
- Two-stage approach prevents noise from influencing SPA

**Limitations:**
- More computationally expensive than either method alone
- UVE may over-filter before SPA can optimize
- Requires tuning parameters for both stages

**Best Practice:**
- Use when data has both noise and high collinearity
- Consider running UVE and SPA separately first to understand data characteristics

---

## 5. iPLS (Interval Partial Least Squares)

**Full Name:** Interval Partial Least Squares

iPLS divides the spectrum into contiguous intervals and evaluates each interval's predictive ability independently, identifying the most informative spectral regions.

### 5.1 Algorithm

**Basic iPLS:**

```
1. INTERVAL CREATION
   a. Divide spectrum into n_intervals equal segments
   b. For n_features variables and n_intervals:
      interval_size = n_features / n_intervals
   c. Create interval boundaries: [(0, size), (size, 2*size), ...]

2. INTERVAL EVALUATION
   For each interval i = 1 to n_intervals:
   a. Extract interval data: X_i = X[:, start_i:end_i]
   b. Auto-select PLS components based on interval size
   c. Build PLS model with cross-validation
   d. Compute RMSECV and R² for interval
   e. Store performance metrics

3. IMPORTANCE ASSIGNMENT
   For each variable j:
   a. Find interval containing variable j
   b. Assign interval's R² score as importance
   c. Better intervals = higher importance

4. RETURN
   a. Best interval (lowest RMSECV)
   b. All interval scores
   c. Variable importances based on interval membership
```

### 5.2 Mathematical Formulas

**Interval Boundaries:**

For interval $i$ (0-indexed):
$$\text{start}_i = i \times \lfloor p / n \rfloor$$
$$\text{end}_i = \begin{cases} (i+1) \times \lfloor p / n \rfloor & i < n-1 \\ p & i = n-1 \end{cases}$$

Where $p$ = total variables, $n$ = number of intervals.

**Cross-Validated RMSE:**

$$RMSECV = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_{i,-k(i)})^2}$$

Where $\hat{y}_{i,-k(i)}$ is the prediction for sample $i$ from the model trained without fold $k(i)$.

### 5.3 Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `n_intervals` | 20 | Number of intervals | 10-40 typical; more intervals = finer resolution |
| `n_components` | auto | PLS components per interval | Auto-adjusted based on interval size |
| `cv_folds` | 5 | CV folds for evaluation | More folds = more stable ranking |
| `random_state` | 42 | Random seed | Ensures reproducibility |

### 5.4 When to Use

**Strengths:**
- Preserves spectral region structure
- Easy to interpret (identifies spectral bands)
- Computationally efficient
- Good for identifying relevant spectral regions

**Limitations:**
- Fixed interval boundaries may miss optimal regions
- Cannot select non-contiguous variables within intervals
- Performance depends on interval count selection

**Reference:**
> Norgaard, L., et al. (2000). Interval partial least-squares regression (iPLS): A comparative chemometric study with an example from near-infrared spectroscopy. *Applied Spectroscopy*, 54(3), 413-419.

---

## 6. iPLS Forward (Forward Interval Selection)

**Full Name:** Forward Interval Partial Least Squares

Forward iPLS starts with an empty selection and iteratively adds the interval that provides the greatest improvement in model performance.

### 6.1 Algorithm

```
1. INITIALIZATION
   a. Divide spectrum into n_intervals
   b. Evaluate each interval independently
   c. Rank intervals by RMSECV (lower = better)
   d. Initialize selected set S = {best_interval}

2. FORWARD SELECTION
   While |S| < max_combine:
   a. For each remaining interval i not in S:
      i. Create combined subset: S_test = S ∪ {i}
      ii. Extract combined variables: X_combined
      iii. Build PLS model and compute RMSECV
   b. Find best addition: i* = argmin(RMSECV over S_test)
   c. If RMSECV(S ∪ {i*}) < RMSECV(S):
      - Add interval: S = S ∪ {i*}
   d. Else:
      - Stop (no improvement)

3. OUTPUT
   a. All individual intervals with scores
   b. Combined subsets from forward selection
   c. Tag each subset with wavelength ranges
```

### 6.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_intervals` | 20 | Number of intervals to divide spectrum |
| `max_combine` | 5 | Maximum intervals to combine |
| `cv_folds` | 5 | CV folds for evaluation |
| `random_state` | 42 | Random seed |

### 6.3 When to Use

**Strengths:**
- Systematically explores interval combinations
- Stops when no improvement (avoids overfitting)
- Produces interpretable multi-region selections

**Limitations:**
- May not find globally optimal combination
- Computational cost increases with n_intervals
- Order of addition affects final selection

---

## 7. iPLS Backward (biPLS)

**Full Name:** Backward Interval Partial Least Squares

Backward iPLS starts with all intervals and iteratively removes the interval whose removal causes the least degradation (or most improvement) in model performance.

### 7.1 Algorithm

```
1. INITIALIZATION
   a. Divide spectrum into n_intervals
   b. Start with all intervals: S = {1, 2, ..., n_intervals}
   c. Evaluate full model as baseline

2. BACKWARD ELIMINATION
   While |S| > min_intervals:
   a. For each interval i in S:
      i. Create test subset: S_test = S \ {i}
      ii. Extract combined variables: X_test
      iii. Build PLS model and compute RMSECV
   b. Find best removal: i* = argmin(RMSECV after removing i)
   c. Remove interval: S = S \ {i*}
   d. Record RMSECV at this step

3. OPTIMAL SELECTION
   a. Find step with lowest RMSECV along elimination path
   b. Mark as optimal subset
   c. Return all steps + optimal

4. OUTPUT
   a. Elimination path with RMSECV at each step
   b. Optimal subset (best RMSECV along path)
   c. Tags indicating retained wavelength ranges
```

### 7.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_intervals` | 20 | Number of initial intervals |
| `min_intervals` | 1 | Minimum intervals to keep |
| `cv_folds` | 5 | CV folds for evaluation |
| `random_state` | 42 | Random seed |

### 7.3 When to Use

**Strengths:**
- Explores full elimination path
- Automatically finds optimal subset size
- Good when most regions are informative

**Limitations:**
- Computationally expensive for many intervals
- May not recover from bad early eliminations
- Assumes global optimum lies along elimination path

**Reference:**
> Leardi, R. & Norgaard, L. (2004). Sequential application of backward interval PLS and genetic algorithms for the selection of relevant spectral regions. *Journal of Chemometrics*, 18(11), 486-497.

---

## 8. CARS (Competitive Adaptive Reweighted Sampling)

**Full Name:** Competitive Adaptive Reweighted Sampling

CARS is a Monte Carlo-based method that uses adaptive reweighted sampling combined with exponential decay to iteratively select optimal variables while balancing exploration and exploitation.

### 8.1 Algorithm

**Detailed Steps:**

```
1. INITIALIZATION
   a. Initialize weights: w_j = 1 for all j = 1 to p
   b. Initialize history arrays for tracking

2. MONTE CARLO ITERATIONS
   For iteration i = 1 to n_iterations:

   a. COMPUTE SAMPLING RATIO
      r_i = 0.8 * exp(-2i / n_iterations)
      n_sample = max(floor(r_i * p * monte_carlo_percent/100),
                     pls_components + 1)

   b. WEIGHTED SAMPLING
      - Normalize weights: prob_j = w_j / sum(w)
      - Sample n_sample variables without replacement
      - Probability proportional to weights

   c. MODEL BUILDING AND EVALUATION
      - Extract sampled variables: X_subset
      - Build PLS model with cross-validation
      - Compute RMSECV for this iteration

   d. WEIGHT UPDATE (Adaptive Reweighting)
      - Fit PLS on full sampled subset
      - Extract coefficients c_j for sampled variables
      - Update weights: w_j = |c_j| / max(|c|)
      - Normalize weights to sum to 1

   e. RECORD HISTORY
      - Store RMSECV, n_selected, selected indices

3. OPTIMAL SELECTION
   a. Find iteration with minimum RMSECV
   b. Extract selected variables from best iteration

4. IMPORTANCE SCORING
   a. Variables in best iteration get accumulated weights
   b. Variables not in best iteration get zero importance
```

### 8.2 Mathematical Formulas

**Exponential Decay Function:**

$$r_i = 0.8 \times \exp\left(-\frac{2i}{n_{iter}}\right)$$

Where $i$ is the current iteration and $n_{iter}$ is total iterations.

**Number of Variables to Sample:**

$$n_{sample} = \max\left(\lfloor r_i \times p \times \frac{mc\%}{100} \rfloor, n_{comp} + 1\right)$$

**Sampling Probability:**

$$P(j) = \frac{w_j}{\sum_{k=1}^{p} w_k}$$

**Weight Update:**

$$w_j^{(i+1)} = \frac{|c_j^{(i)}|}{\max_k |c_k^{(i)}|}$$

Where $c_j^{(i)}$ is the PLS coefficient for variable $j$ at iteration $i$.

### 8.3 Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `n_iterations` | 50 | Monte Carlo iterations | 50-100 |
| `monte_carlo_samples` | 80 | Initial sampling percentage | 70-90 |
| `pls_components` | 5 | PLS components for evaluation | 5-15 |
| `cv_folds` | 5 | CV folds | 5-10 |
| `random_state` | 42 | Random seed | - |
| `model_type` | None | Model for tree-based CARS | - |
| `use_hybrid_importance` | False | Enable CARS-Tree mode | - |
| `hybrid_importance_weight` | 0.5 | Split/gain blend weight | 0.3-0.7 |

### 8.4 When to Use

**Strengths:**
- Balances exploration (random sampling) and exploitation (weight-based)
- Exponential decay forces progressive elimination
- Often produces very compact variable sets
- Cross-validated selection prevents overfitting

**Limitations:**
- Computationally expensive (many model fits)
- Results can vary with random seed
- Requires careful selection of pls_components
- May converge to local optima

**Computational Considerations:**
- Time complexity: O(n_iter * cv_folds * n * n_sample)
- Memory: O(n * p) for data + O(n_iter) for history
- Minimum features: 7 for meaningful selection

**Reference:**
> Li, H. D., et al. (2009). Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration. *Analytica Chimica Acta*, 648(1), 77-84.

---

## 9. CARS-Tree Hybrid

**Full Name:** Competitive Adaptive Reweighted Sampling with Tree-Based Importance

CARS-Tree extends CARS by using LightGBM-based importance instead of PLS coefficients, with a hybrid importance measure that blends split-based and gain-based importance.

### 9.1 Algorithm Modifications

The CARS-Tree variant modifies the standard CARS algorithm in the weight update step:

```
WEIGHT UPDATE (Tree-Based, Hybrid Mode):
1. Fit LightGBM model on sampled variables
2. Extract split importance: split_imp = booster.feature_importance('split')
3. Extract gain importance: gain_imp = booster.feature_importance('gain')
4. Normalize each:
   - split_norm = split_imp / sum(split_imp)
   - gain_norm = gain_imp / sum(gain_imp)
5. Compute hybrid importance:
   - hybrid_imp = weight * split_norm + (1 - weight) * gain_norm
6. Update weights: w_j = hybrid_imp_j
7. Add floor to prevent complete elimination: w_j = max(w_j, 1e-6)
```

### 9.2 Hybrid Importance Formula

$$I_{hybrid,j} = \alpha \times I_{split,j}^{norm} + (1-\alpha) \times I_{gain,j}^{norm}$$

Where:
- $\alpha$ = `hybrid_importance_weight` (default 0.5)
- $I_{split}^{norm}$ = normalized split count importance
- $I_{gain}^{norm}$ = normalized gain importance

### 9.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | required | Must be tree model (RandomForest, XGBoost, LightGBM, CatBoost) |
| `use_hybrid_importance` | True | Enable hybrid split+gain importance |
| `hybrid_importance_weight` | 0.5 | Balance between split (1.0) and gain (0.0) |
| `task_type` | 'regression' | 'regression' or 'classification' |

### 9.4 When to Use

**Strengths:**
- Better suited when final model is tree-based
- Hybrid importance produces denser distributions than split-only
- Handles non-linear relationships naturally

**Limitations:**
- Computationally heavier than PLS-based CARS
- May not be optimal for linear final models
- LightGBM parameters affect results

---

## 10. GA-PLS (Genetic Algorithm PLS)

**Full Name:** Genetic Algorithm for Partial Least Squares Variable Selection

GA-PLS uses genetic algorithm optimization to search for the optimal subset of variables, where each candidate solution (chromosome) is a binary mask indicating which variables are selected.

### 10.1 Algorithm

**Detailed Steps:**

```
1. CHROMOSOME REPRESENTATION
   - Binary string of length p (number of variables)
   - Gene j = 1: variable j is selected
   - Gene j = 0: variable j is excluded

2. POPULATION INITIALIZATION
   For i = 1 to population_size:
   a. Generate random binary chromosome
   b. Ensure minimum wavelengths selected (constraint repair)
   c. Optionally seed some chromosomes with prior importances

3. FITNESS EVALUATION
   For each chromosome:
   a. Extract selected variables: X_subset = X[:, chromosome == 1]
   b. Build PLS model with cross-validation
   c. Compute RMSECV
   d. Fitness = -RMSECV (maximize fitness = minimize error)

4. GENETIC OPERATORS
   a. SELECTION (Tournament)
      - Select tournament_size individuals randomly
      - Winner = individual with best fitness
      - Repeat until new population filled

   b. CROSSOVER (Two-Point)
      - With probability crossover_rate:
        * Select two crossover points
        * Exchange segments between parents
        * Create two children

   c. MUTATION (Bit-Flip)
      - For each gene with probability mutation_rate:
        * Flip bit (0 -> 1 or 1 -> 0)
      - Repair if minimum wavelengths violated

5. ELITISM
   - Keep top 2 individuals unchanged
   - Replace rest with offspring

6. TERMINATION
   - Stop after n_generations
   - Or early stopping if no improvement for early_stopping generations

7. MULTI-RUN AGGREGATION
   - Run GA n_runs times with different seeds
   - Compute selection frequency for each variable
   - Importance = frequency of selection across runs
```

### 10.2 Genetic Operators in Detail

**Tournament Selection:**

```
def tournament_selection(population, fitness, tournament_size=3):
    competitors = random_sample(population, tournament_size)
    winner = competitor with max(fitness)
    return winner
```

**Two-Point Crossover:**

```
def two_point_crossover(parent1, parent2, rate=0.6):
    if random() > rate:
        return parent1.copy(), parent2.copy()

    points = sorted(random_choice(length, 2))
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[points[0]:points[1]] = parent2[points[0]:points[1]]
    child2[points[0]:points[1]] = parent1[points[0]:points[1]]

    return child1, child2
```

**Bit-Flip Mutation:**

```
def bit_flip_mutation(chromosome, rate=0.01, min_wavelengths=5):
    mutated = chromosome.copy()
    mask = random(length) < rate
    mutated[mask] = 1 - mutated[mask]

    # Repair if needed
    if sum(mutated) < min_wavelengths:
        add_random_ones(mutated, min_wavelengths - sum(mutated))

    return mutated
```

### 10.3 Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `population_size` | 64 | Number of individuals | 50-200 |
| `n_generations` | 100 | Maximum generations | 50-200 |
| `crossover_rate` | 0.6 | Crossover probability | 0.4-0.9 |
| `mutation_rate` | 0.01 | Per-gene mutation rate | 0.005-0.05 |
| `n_runs` | 5 | Independent GA runs | 3-10 |
| `selection_threshold` | 0.6 | Min frequency for stable selection | 0.5-0.8 |
| `min_wavelengths` | 5 | Minimum selected variables | 5-20 |
| `early_stopping` | 20 | Generations without improvement | 10-30 |
| `n_components` | 10 | PLS components for fitness | 5-15 |
| `cv` | 5 | CV folds for fitness | 5-10 |

### 10.4 When to Use

**Strengths:**
- Global search capability (explores large solution space)
- No assumptions about variable relationships
- Multi-run aggregation provides stability
- Can escape local optima

**Limitations:**
- Computationally expensive (many fitness evaluations)
- Parameter-sensitive (many hyperparameters to tune)
- No guarantee of global optimum
- Results vary with random seed

**Computational Considerations:**
- Time complexity: O(n_runs * n_gen * pop_size * cv_folds * n * p)
- For 2000 wavelengths: single run ~2-5 minutes, multi-run ~10-25 minutes
- Consider "quick" settings for exploration: pop_size=32, n_gen=50, n_runs=3

**Reference:**
> Leardi, R. & Lupianez Gonzalez, A. (1998). Genetic algorithms applied to feature selection in PLS regression. *Chemometrics and Intelligent Laboratory Systems*, 41(2), 195-207.

---

## 11. VCPA-IRIV

**Full Name:** Variable Combination Population Analysis - Iteratively Retains Informative Variables

VCPA-IRIV is an advanced iterative method that classifies variables into categories based on statistical tests comparing model performance when variables are included versus excluded.

### 11.1 Algorithm

**Detailed Steps:**

```
1. INITIALIZATION
   - Start with all p variables as candidates
   - active_indices = [0, 1, 2, ..., p-1]

2. OUTER ITERATION LOOP
   For iteration = 1 to n_outer_iterations:

   a. BINARY MATRIX GENERATION
      - Create BM with binary_matrix_samples rows
      - Each row: random subset (~50% inclusion probability)
      - Ensure each row has minimum variables for model

   b. INCLUDE/EXCLUDE COMPARISON
      For each row in BM:
      i. Extract selected variables
      ii. Build model (PLS or LightGBM) with CV
      iii. Record RMSECV

      For each variable j:
      - Collect RMSECV_include: errors when j is included
      - Collect RMSECV_exclude: errors when j is excluded

   c. STATISTICAL CLASSIFICATION
      For each variable j:
      i. Compute DMEAN = mean(exclude) - mean(include)
         - Positive DMEAN: including improves performance
         - Negative DMEAN: including worsens performance

      ii. Apply Mann-Whitney U test (p < 0.05 threshold)
          H = 1 if significant, 0 otherwise

      iii. Classify variable:
           - H=1, DMEAN>0: STRONGLY INFORMATIVE (keep)
           - H=0, DMEAN>0: WEAKLY INFORMATIVE (keep)
           - H=0, DMEAN≤0: UNINFORMATIVE (remove)
           - H=1, DMEAN<0: INTERFERING (remove)

   d. VARIABLE ELIMINATION
      - Remove uninformative and interfering variables
      - Update active_indices

   e. CONVERGENCE CHECK
      - Stop if no variables removed
      - Stop if too few variables remain

3. OUTPUT
   - selected_indices: final selected variables
   - variable_categories: classification for each variable
   - importance_scores: 1.0 (strong), 0.5 (weak), 0.25 (other)
   - convergence_history: RMSECV at each iteration
```

### 11.2 Mathematical Formulas

**DMEAN (Difference in Means):**

$$DMEAN_j = \overline{RMSECV}_{exclude,j} - \overline{RMSECV}_{include,j}$$

**Interpretation:**
- $DMEAN_j > 0$: Including variable $j$ reduces error (informative)
- $DMEAN_j \leq 0$: Including variable $j$ increases error (uninformative/interfering)

**Mann-Whitney U Test:**

Tests whether the distribution of RMSECV values when variable $j$ is included differs significantly from when it is excluded.

$$H_0: \text{Distribution}_{include} = \text{Distribution}_{exclude}$$

$$H = \begin{cases} 1 & \text{if } p < 0.05 \\ 0 & \text{otherwise} \end{cases}$$

**Variable Classification:**

| H | DMEAN | Category | Action |
|---|-------|----------|--------|
| 1 | > 0 | Strongly Informative | Keep |
| 0 | > 0 | Weakly Informative | Keep |
| 0 | ≤ 0 | Uninformative | Remove |
| 1 | < 0 | Interfering | Remove |

### 11.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_outer_iterations` | 10 | Maximum elimination rounds |
| `n_inner_iterations` | 50 | Deprecated (use binary_matrix_samples) |
| `binary_matrix_samples` | 100 | Number of random combinations per iteration |
| `pls_components` | 5 | PLS components (when using PLS) |
| `cv_folds` | 5 | CV folds for model evaluation |
| `model_type` | None | 'PLS' (default) or tree model name |
| `importance_threshold` | 0.5 | Deprecated (statistical test used instead) |
| `random_state` | None | Random seed |

### 11.4 When to Use

**Strengths:**
- Statistically rigorous classification
- Identifies interfering variables (unique capability)
- Iterative refinement improves robustness
- No arbitrary threshold selection

**Limitations:**
- Computationally expensive (many model evaluations)
- Requires sufficient samples for reliable statistics
- May converge prematurely with aggressive settings

**Computational Considerations:**
- Time complexity: O(n_outer * binary_samples * cv_folds * n * p)
- Adaptive binary_matrix_samples based on variable count
- Consider model_type='LightGBM' for tree-based final models

**Reference:**
> Yun, Y. H., et al. (2014). A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration. *Analytica Chimica Acta*, 807, 36-43.

---

## Method Comparison Table

| Method | Type | Supervision | Collinearity | Noise Filtering | Computation | Variables Selected |
|--------|------|-------------|--------------|-----------------|-------------|-------------------|
| **VIP/Top-N** | Importance | Yes | No | Partial | Fast | Fixed N |
| **UVE** | Statistical | Yes | No | Yes | Medium | Auto (threshold) |
| **SPA** | Projection | Partial | Yes | No | Fast | Fixed N |
| **UVE-SPA** | Hybrid | Yes | Yes | Yes | Medium | Fixed N |
| **iPLS** | Interval | Yes | Partial | Partial | Medium | Interval-based |
| **iPLS Forward** | Interval | Yes | Partial | Partial | Medium | Multi-interval |
| **iPLS Backward** | Interval | Yes | Partial | Partial | Medium-High | Auto (optimal) |
| **CARS** | Monte Carlo | Yes | No | Partial | High | Auto (best iteration) |
| **CARS-Tree** | Monte Carlo | Yes | No | Partial | High | Auto (best iteration) |
| **GA-PLS** | Evolutionary | Yes | No | No | Very High | Frequency-based |
| **VCPA-IRIV** | Statistical | Yes | No | Yes | Very High | Auto (convergence) |

---

## Selection Guide

### By Data Characteristics

| Situation | Recommended Methods |
|-----------|-------------------|
| High collinearity | SPA, UVE-SPA |
| High noise levels | UVE, VCPA-IRIV |
| Unknown optimal size | CARS, iPLS Backward, VCPA-IRIV |
| Need interpretable regions | iPLS variants |
| Large variable count (>2000) | Top-N first, then refine |
| Small sample size (<50) | UVE, SPA (simpler methods) |
| Tree-based final model | CARS-Tree, VCPA-IRIV with model_type |

### By Computational Budget

| Time Available | Recommended Methods |
|----------------|-------------------|
| < 1 minute | Top-N, SPA |
| 1-5 minutes | UVE, iPLS |
| 5-30 minutes | CARS, UVE-SPA |
| 30+ minutes | GA-PLS, VCPA-IRIV |

### By Goal

| Goal | Recommended Methods |
|------|-------------------|
| Quick screening | Top-N, iPLS |
| Maximum performance | GA-PLS, VCPA-IRIV |
| Identify spectral bands | iPLS variants |
| Remove noise only | UVE |
| Reduce collinearity only | SPA |
| Both noise and collinearity | UVE-SPA |
| Statistical rigor | UVE, VCPA-IRIV |

---

## Best Practices

### 1. Preprocessing Before Selection

- Apply spectral preprocessing (SNV, derivatives) before variable selection
- Variable selection operates on preprocessed data
- Preprocessing affects which variables appear informative

### 2. Cross-Validation Strategy

- Always use cross-validation within selection methods
- External validation set should not influence selection
- Consider stratified CV for classification tasks

### 3. Method Combination

- Start with fast methods (Top-N, SPA) for exploration
- Refine with more sophisticated methods (CARS, GA-PLS)
- Compare multiple methods and look for consensus

### 4. Parameter Tuning

- Use default parameters initially
- Tune based on CV performance, not test performance
- Document parameter choices for reproducibility

### 5. Stability Assessment

- Run stochastic methods (CARS, GA-PLS) multiple times
- Look for variables consistently selected across runs
- Selection frequency indicates variable importance reliability

### 6. Interpretation

- Map selected indices back to wavelengths
- Verify selected regions make spectroscopic sense
- Consult domain knowledge about expected absorption bands

---

## Glossary

| Term | Definition |
|------|------------|
| **Variable** | A single spectral measurement point (wavelength/wavenumber) |
| **Feature** | Synonym for variable in machine learning context |
| **Wavelength** | Spectral position in nm or μm |
| **Wavenumber** | Spectral position in cm^-1 |
| **Collinearity** | High correlation between variables |
| **RMSECV** | Root Mean Square Error of Cross-Validation |
| **CV** | Cross-Validation |
| **PLS** | Partial Least Squares |
| **VIP** | Variable Importance in Projection |
| **MDI** | Mean Decrease in Impurity |
| **Binary Matrix** | Random 0/1 matrix for variable combination testing |
| **Chromosome** | Binary representation of variable selection in GA |
| **Fitness** | Quality measure for GA selection (negative RMSECV) |

---

## References

1. Araújo, M. C. U., et al. (2001). The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73.

2. Centner, V., et al. (1996). Elimination of uninformative variables for multivariate calibration. *Analytical Chemistry*, 68(21), 3851-3858.

3. Li, H. D., et al. (2009). Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration. *Analytica Chimica Acta*, 648(1), 77-84.

4. Leardi, R. & Lupiáñez González, A. (1998). Genetic algorithms applied to feature selection in PLS regression. *Chemometrics and Intelligent Laboratory Systems*, 41(2), 195-207.

5. Nørgaard, L., et al. (2000). Interval partial least-squares regression (iPLS): A comparative chemometric study with an example from near-infrared spectroscopy. *Applied Spectroscopy*, 54(3), 413-419.

6. Leardi, R. & Nørgaard, L. (2004). Sequential application of backward interval PLS and genetic algorithms for the selection of relevant spectral regions. *Journal of Chemometrics*, 18(11), 486-497.

7. Yun, Y. H., et al. (2014). A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration. *Analytica Chimica Acta*, 807, 36-43.

8. Stefansson, P., et al. (2020). Fast method for GA-PLS with simultaneous feature selection and identification of optimal preprocessing technique. *Journal of Chemometrics*, 34(3), e3195.
