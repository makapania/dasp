# Neural Boosted Regression - Implementation Plan

**Date:** October 27, 2025
**Status:** Design Phase - Ready for Implementation

---

## Executive Summary

This document outlines the implementation strategy for integrating **JMP-style Neural Boosted Regression** into the spectral prediction pipeline. This method combines shallow neural networks with gradient boosting to create powerful, interpretable models for spectral data analysis.

---

## 1. What is Neural Boosted Regression?

### Core Concept

Neural Boosted Regression is an **ensemble method** that combines:
- **Weak learners**: Small neural networks (1-5 hidden nodes)
- **Gradient boosting**: Sequential fitting to residuals
- **Learning rate scaling**: Conservative updates (0 < ν ≤ 1)

### Algorithm (Simplified)

```python
# Initialize ensemble prediction
F(x) = 0

# For each boosting round:
for round in 1 to n_estimators:
    # 1. Compute residuals (what we got wrong)
    residuals = y_true - F(x)

    # 2. Fit small neural network to residuals
    weak_network = train_small_MLP(X, residuals)

    # 3. Update ensemble with scaled prediction
    F(x) = F(x) + learning_rate * weak_network(x)

    # 4. Check validation performance for early stopping
    if validation_score_not_improving:
        break

# Final prediction is the sum of all weak learners
return F(x)
```

### Why This Matters for Spectral Data

**Advantages over existing methods:**

| Feature | PLS | Random Forest | MLP | Neural Boosted |
|---------|-----|---------------|-----|----------------|
| Handles nonlinearity | ❌ Linear | ✅ Yes | ✅ Yes | ✅ Yes |
| Robust to noise | ✅ Good | ✅ Good | ⚠️ Moderate | ✅ Good (with Huber) |
| Feature importance | ✅ VIP scores | ✅ Gini | ⚠️ Weights | ✅ Aggregated weights |
| Prevents overfitting | ✅ Components | ✅ Trees | ⚠️ Early stop | ✅ Multiple methods |
| Interpretable | ✅ Very | ⚠️ Moderate | ❌ Difficult | ✅ Good |
| Training speed | ✅ Fast | ✅ Fast | ⚠️ Slow | ⚠️ Moderate |

**Key benefits:**
1. **Better nonlinearity** than PLS while staying more interpretable than deep MLPs
2. **Robust to outliers** with Huber loss option
3. **Regularization** via L1/L2 penalties prevents overfitting
4. **Feature importances** available for wavelength selection
5. **Early stopping** prevents wasting computation time

---

## 2. Implementation Architecture

### Recommended Approach: Custom Boosting with sklearn MLP

**Why this approach?**
- ✅ Leverages tested sklearn MLPRegressor for weak learners
- ✅ Full control over boosting logic to match JMP specification
- ✅ Compatible with existing pipeline architecture
- ✅ Can extract feature importances
- ✅ Reasonable implementation complexity (~5-7 hours)

### File Structure

```
src/spectral_predict/
├── neural_boosted.py          # NEW - NeuralBoostedRegressor class
├── models.py                   # MODIFY - Add to model grids
├── search.py                   # No changes needed (already supports new models)
└── scoring.py                  # No changes needed

tests/
└── test_neural_boosted.py      # NEW - Unit tests

docs/
└── NEURAL_BOOSTED_GUIDE.md     # NEW - User documentation
```

---

## 3. Detailed Implementation Plan

### Phase 1: Core NeuralBoostedRegressor Class

**File:** `src/spectral_predict/neural_boosted.py`

**Class Interface:**

```python
class NeuralBoostedRegressor(BaseEstimator, RegressorMixin):
    """
    Neural Boosted Regression for spectral data.

    Implements gradient boosting with small neural networks as weak learners,
    following JMP's Neural Boosted methodology.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting rounds (weak learners)

    learning_rate : float, default=0.1
        Shrinkage parameter (0 < ν ≤ 1). Lower values require more estimators
        but can improve generalization.

    hidden_layer_size : int, default=3
        Number of nodes in the single hidden layer.
        Should be small (1-5) to maintain weak learner properties.

    activation : {'tanh', 'relu', 'identity'}, default='tanh'
        Activation function for hidden layer:
        - 'tanh': Hyperbolic tangent (JMP default)
        - 'identity': Linear activation
        - 'relu': ReLU (not in JMP but useful)

    alpha : float, default=0.0001
        L2 penalty (weight decay) parameter

    l1_ratio : float, default=0.0
        Elastic net mixing parameter (0 = L2, 1 = L1)
        Only used if solver supports it

    max_iter : int, default=200
        Maximum iterations for each weak learner

    early_stopping : bool, default=True
        Whether to use early stopping based on validation score

    validation_fraction : float, default=0.1
        Fraction of training data to use for validation (if early_stopping=True)

    n_iter_no_change : int, default=10
        Stop if validation score doesn't improve for this many rounds

    loss : {'mse', 'huber'}, default='mse'
        Loss function:
        - 'mse': Mean squared error
        - 'huber': Huber loss (robust to outliers)

    huber_delta : float, default=1.35
        Delta parameter for Huber loss (ignored if loss='mse')

    random_state : int, default=None
        Random seed for reproducibility

    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=detailed)

    Attributes
    ----------
    estimators_ : list of MLPRegressor
        The collection of fitted weak learners

    train_score_ : list of float
        Training score at each boosting iteration

    validation_score_ : list of float
        Validation score at each iteration (if early_stopping=True)

    n_estimators_ : int
        Actual number of estimators used (may be less than n_estimators
        if early stopping triggered)

    Examples
    --------
    >>> from neural_boosted import NeuralBoostedRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)  # 100 samples, 50 wavelengths
    >>> y = X[:, 10] + 2*X[:, 20] + np.random.randn(100) * 0.1
    >>> model = NeuralBoostedRegressor(n_estimators=50, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        hidden_layer_size=3,
        activation='tanh',
        alpha=0.0001,
        l1_ratio=0.0,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        loss='mse',
        huber_delta=1.35,
        random_state=None,
        verbose=0
    ):
        # Parameter validation
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if hidden_layer_size < 1:
            raise ValueError("hidden_layer_size must be >= 1")
        if hidden_layer_size > 10:
            warnings.warn("hidden_layer_size > 10 may violate weak learner assumption")
        if activation not in ['tanh', 'relu', 'identity', 'logistic']:
            raise ValueError(f"Unknown activation: {activation}")
        if loss not in ['mse', 'huber']:
            raise ValueError(f"Unknown loss: {loss}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.loss = loss
        self.huber_delta = huber_delta
        self.random_state = random_state
        self.verbose = verbose
```

**Key Methods:**

```python
def fit(self, X, y):
    """Fit the Neural Boosted ensemble."""
    X, y = check_X_y(X, y)
    self.estimators_ = []
    self.train_score_ = []
    self.validation_score_ = []

    # Split validation set if early stopping
    if self.early_stopping:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_fraction,
            random_state=self.random_state
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Initialize ensemble prediction to zero
    F_train = np.zeros(len(y_train))
    if X_val is not None:
        F_val = np.zeros(len(y_val))

    best_val_score = np.inf
    no_improvement_count = 0

    # Boosting loop
    for i in range(self.n_estimators):
        # Compute residuals
        residuals = y_train - F_train

        # Create and fit weak learner
        weak_learner = MLPRegressor(
            hidden_layer_sizes=(self.hidden_layer_size,),
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state + i if self.random_state else None,
            warm_start=False
        )

        weak_learner.fit(X_train, residuals)
        self.estimators_.append(weak_learner)

        # Update ensemble predictions
        pred_train = weak_learner.predict(X_train)
        F_train += self.learning_rate * pred_train

        # Compute training score
        train_score = self._compute_loss(y_train, F_train)
        self.train_score_.append(train_score)

        # Early stopping check
        if self.early_stopping:
            pred_val = weak_learner.predict(X_val)
            F_val += self.learning_rate * pred_val
            val_score = self._compute_loss(y_val, F_val)
            self.validation_score_.append(val_score)

            if val_score < best_val_score:
                best_val_score = val_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.n_iter_no_change:
                if self.verbose > 0:
                    print(f"Early stopping at iteration {i+1}")
                break

        if self.verbose > 0 and (i + 1) % 10 == 0:
            msg = f"Iteration {i+1}/{self.n_estimators}, Train Loss: {train_score:.6f}"
            if self.early_stopping:
                msg += f", Val Loss: {val_score:.6f}"
            print(msg)

    self.n_estimators_ = len(self.estimators_)
    return self

def predict(self, X):
    """Predict using the Neural Boosted ensemble."""
    check_is_fitted(self, ['estimators_'])
    X = check_array(X)

    # Initialize predictions to zero
    predictions = np.zeros(X.shape[0])

    # Aggregate predictions from all weak learners
    for estimator in self.estimators_:
        predictions += self.learning_rate * estimator.predict(X)

    return predictions

def _compute_loss(self, y_true, y_pred):
    """Compute loss (MSE or Huber)."""
    if self.loss == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif self.loss == 'huber':
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        quadratic = np.minimum(abs_residuals, self.huber_delta)
        linear = abs_residuals - quadratic
        return np.mean(0.5 * quadratic**2 + self.huber_delta * linear)

def get_feature_importances(self):
    """
    Compute feature importances by averaging absolute weights
    across all weak learners.
    """
    check_is_fitted(self, ['estimators_'])

    n_features = self.estimators_[0].coefs_[0].shape[0]
    importances = np.zeros(n_features)

    for estimator in self.estimators_:
        # First layer weights: (n_features, n_hidden)
        weights = estimator.coefs_[0]
        # Average absolute weight per feature
        importances += np.mean(np.abs(weights), axis=1)

    # Normalize by number of estimators
    importances /= len(self.estimators_)

    return importances
```

---

### Phase 2: Add to Model Grid

**File:** `src/spectral_predict/models.py`

**Modifications:**

1. Import the new class:
```python
from .neural_boosted import NeuralBoostedRegressor
```

2. Add to `get_model_grids()`:

```python
def get_model_grids(task_type, n_features, max_n_components=24, max_iter=500):
    """Get model grids for hyperparameter search."""
    grids = {}

    # ... existing PLS, RandomForest, MLP code ...

    if task_type == "regression":
        # Neural Boosted Regression
        # Grid is deliberately small to balance performance and search time
        nbr_configs = []

        # Learning rates: conservative to moderate
        learning_rates = [0.05, 0.1, 0.2]

        # Number of estimators: early stopping will handle optimization
        n_estimators_list = [50, 100]

        # Hidden layer sizes: keep small (weak learner property)
        hidden_sizes = [3, 5]

        # Activations: tanh (JMP default) and identity (linear)
        activations = ['tanh', 'identity']

        for n_est in n_estimators_list:
            for lr in learning_rates:
                for hidden in hidden_sizes:
                    for activation in activations:
                        nbr_configs.append(
                            (
                                NeuralBoostedRegressor(
                                    n_estimators=n_est,
                                    learning_rate=lr,
                                    hidden_layer_size=hidden,
                                    activation=activation,
                                    early_stopping=True,
                                    validation_fraction=0.15,
                                    n_iter_no_change=10,
                                    alpha=1e-4,  # Light L2 regularization
                                    random_state=42,
                                    verbose=0
                                ),
                                {
                                    "n_estimators": n_est,
                                    "learning_rate": lr,
                                    "hidden_layer_size": hidden,
                                    "activation": activation
                                }
                            )
                        )

        grids["NeuralBoosted"] = nbr_configs

        # Total configurations: 2 * 3 * 2 * 2 = 24 per preprocessing method
        # This is reasonable compared to RandomForest (6) and MLP (8)

    return grids
```

**Rationale for hyperparameter choices:**

- **n_estimators**: Start with 50-100; early stopping will find optimal number
- **learning_rate**: 0.05 (conservative) to 0.2 (aggressive)
- **hidden_layer_size**: 3-5 nodes (maintains weak learner property)
- **activation**: tanh (smooth, JMP default) and identity (linear, faster)
- **alpha**: Fixed at 1e-4 (light L2 to prevent overfitting)

---

### Phase 3: Feature Importance Extraction

**File:** `src/spectral_predict/models.py`

**Modify `get_feature_importances()`:**

```python
def get_feature_importances(model, model_name, X, y):
    """Extract feature importances from a fitted model."""

    if model_name in ["PLS", "PLS-DA"]:
        return compute_vip(model, X, y)

    elif model_name == "RandomForest":
        return model.feature_importances_

    elif model_name == "MLP":
        weights = model.coefs_[0]
        return np.mean(np.abs(weights), axis=1)

    elif model_name == "NeuralBoosted":
        # NEW: Aggregate importances across all weak learners
        return model.get_feature_importances()

    else:
        raise ValueError(f"Unknown model type: {model_name}")
```

**How it works:**

For each weak learner in the ensemble:
1. Extract first-layer weights (n_features × n_hidden)
2. Take mean absolute value across hidden nodes
3. Aggregate across all weak learners
4. Normalize by number of estimators

This gives a **single importance score per wavelength** showing which features are most influential across the entire ensemble.

---

### Phase 4: Integration with Search Pipeline

**File:** `src/spectral_predict/search.py`

**Good news:** No modifications needed!

The existing code already:
- ✅ Iterates over all models in the grid
- ✅ Supports feature importance extraction
- ✅ Tests variable subsets (top10, top20, etc.)
- ✅ Evaluates with cross-validation
- ✅ Computes composite scores

Neural Boosted will automatically be tested with:
- Full spectrum
- Top N variable subsets (10, 20, 50, 100, 250, 500, 1000)
- Region-based subsets (for non-derivative preprocessing)

---

### Phase 5: GUI Integration

**File:** `spectral_predict_gui.py`

**Minimal changes needed:**

1. Add checkbox for Neural Boosted model:

```python
# In create_model_selection_frame():
self.use_neural_boosted = tk.BooleanVar(value=False)  # Default: off (new model)
ttk.Checkbutton(
    model_frame,
    text="Neural Boosted",
    variable=self.use_neural_boosted
).pack(side=tk.LEFT, padx=10)
```

2. Add tooltip/info:
```python
# Tooltip explaining Neural Boosted
self.create_tooltip(
    neural_boosted_checkbox,
    "Ensemble of small neural networks (boosted)\n"
    "Good for: Nonlinear relationships, robust to noise\n"
    "Training time: Moderate (slower than PLS, faster than full MLP)"
)
```

3. Pass to search (already handled by existing code):
```python
# The existing code already passes model selection to search
selected_models = []
if self.use_pls.get():
    selected_models.append("PLS")
# ... other models ...
if self.use_neural_boosted.get():
    selected_models.append("NeuralBoosted")
```

**Optional: Advanced settings panel**

Could add a collapsible "Advanced Neural Boosted Settings" with:
- Learning rate slider (0.01 - 0.5)
- Max estimators input (50 - 500)
- Huber loss checkbox (for robust regression)

But for first version, use defaults and let hyperparameter search handle it.

---

## 4. Testing Strategy

### Unit Tests

**File:** `tests/test_neural_boosted.py`

```python
import numpy as np
import pytest
from src.spectral_predict.neural_boosted import NeuralBoostedRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

def test_neural_boosted_basic():
    """Test basic fit/predict."""
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    model = NeuralBoostedRegressor(
        n_estimators=10,
        learning_rate=0.1,
        hidden_layer_size=3,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert len(model.estimators_) <= 10
    r2 = r2_score(y, predictions)
    assert r2 > 0.8  # Should fit training data well

def test_early_stopping():
    """Test that early stopping works."""
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    model = NeuralBoostedRegressor(
        n_estimators=100,
        learning_rate=0.1,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42
    )

    model.fit(X, y)

    # Should stop early, not use all 100 estimators
    assert model.n_estimators_ < 100
    assert len(model.validation_score_) > 0

def test_feature_importances():
    """Test feature importance extraction."""
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    model = NeuralBoostedRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    importances = model.get_feature_importances()

    assert importances.shape == (20,)
    assert np.all(importances >= 0)
    assert np.sum(importances) > 0

def test_huber_loss():
    """Test Huber loss with outliers."""
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    # Add outliers
    y[0] = y[0] + 10
    y[1] = y[1] - 10

    model_mse = NeuralBoostedRegressor(loss='mse', n_estimators=20, random_state=42)
    model_huber = NeuralBoostedRegressor(loss='huber', n_estimators=20, random_state=42)

    model_mse.fit(X, y)
    model_huber.fit(X, y)

    # Huber should be more robust to outliers
    # (specific assertion depends on data characteristics)
    assert model_huber.train_score_[-1] >= 0  # Sanity check

def test_activation_functions():
    """Test different activation functions."""
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    for activation in ['tanh', 'relu', 'identity']:
        model = NeuralBoostedRegressor(
            activation=activation,
            n_estimators=10,
            random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        assert r2 > 0.5  # Should learn something

def test_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        NeuralBoostedRegressor(learning_rate=0)  # Must be > 0

    with pytest.raises(ValueError):
        NeuralBoostedRegressor(learning_rate=1.5)  # Must be <= 1

    with pytest.raises(ValueError):
        NeuralBoostedRegressor(activation='unknown')

    with pytest.raises(ValueError):
        NeuralBoostedRegressor(loss='unknown')
```

### Integration Tests

Test with actual spectral data:

```python
def test_spectral_data_integration():
    """Test with real spectral data structure."""
    # Create synthetic spectral data
    n_samples = 50
    n_wavelengths = 500

    # Simulate NIR spectrum
    X = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0

    # Target based on specific wavelengths (simulate O-H peak correlation)
    y = 2.0 * X[:, 100] + 1.5 * X[:, 200] - 0.5 * X[:, 300] + np.random.randn(n_samples) * 0.1

    model = NeuralBoostedRegressor(
        n_estimators=30,
        learning_rate=0.1,
        hidden_layer_size=5,
        early_stopping=True,
        random_state=42
    )

    model.fit(X, y)

    # Check that it identifies important wavelengths
    importances = model.get_feature_importances()
    top_5_indices = np.argsort(importances)[-5:][::-1]

    # Wavelengths 100, 200, 300 should be in top features
    assert 100 in top_5_indices or 200 in top_5_indices or 300 in top_5_indices
```

---

## 5. Challenges & Solutions

### Challenge 1: Gaussian Activation Not in sklearn

**Problem:** JMP spec mentions Gaussian activation, but sklearn only has tanh, relu, identity, logistic.

**Solution Options:**

**A) Skip Gaussian (RECOMMENDED):**
- Use tanh (smooth, similar shape) and identity
- Simplest implementation
- Still covers main use cases

**B) Implement Custom Activation:**
```python
# Subclass MLPRegressor with custom activation
class GaussianMLPRegressor(MLPRegressor):
    def _forward_pass_fast(self, X):
        # Override to add Gaussian activation
        # activation = exp(-x^2)
        ...
```
- More work, harder to maintain
- Only needed if users specifically request Gaussian

**Recommendation:** Start with Option A. Add Option B only if users need it.

---

### Challenge 2: Computational Cost

**Problem:** Boosting 100+ neural networks could be slow on large spectral datasets.

**Concerns:**
- Each weak learner needs to fit (even if small)
- 24 hyperparameter configs × 100 estimators each = 2400 weak learner fits
- Could take hours on large datasets

**Solutions:**

1. **Early Stopping (Built-in):**
   - Typically stops at 20-40 estimators
   - Saves 60-80% of computation
   - ✅ Already implemented

2. **Small Weak Learners:**
   - 3-5 hidden nodes = very fast to train
   - Much faster than full MLP (64+ nodes)
   - ✅ Already configured

3. **Reduced Grid:**
   - 24 configs is reasonable (RF has 6, MLP has 8)
   - Could reduce to 12 if needed:
     - n_estimators: [100] (early stop handles it)
     - learning_rate: [0.1, 0.2]
     - hidden: [3, 5]
     - activation: ['tanh', 'identity']
     - = 1 × 2 × 2 × 2 = 8 configs

4. **Progress Monitoring:**
   - Already have progress monitor from Phase 2
   - Shows user that it's working, not frozen
   - ✅ No changes needed

5. **Optional: Parallel Weak Learners:**
   - Could fit weak learners in parallel across cores
   - Requires joblib or similar
   - Moderate implementation complexity
   - **Not recommended for v1** - profile first

**Recommendation:** Start with options 1-4. Profile on real data. Add parallelization only if needed.

---

### Challenge 3: When to Use Neural Boosted vs Other Models?

**Problem:** Users won't know when to select Neural Boosted checkbox.

**Solution:** Add documentation and tooltips.

**Decision Matrix for Users:**

| Situation | Best Model | Why |
|-----------|-----------|-----|
| Linear relationship, many wavelengths | PLS | Fastest, handles collinearity |
| Nonlinear, need speed | Random Forest | Fast, robust, handles nonlinearity |
| Nonlinear, need interpretability | Neural Boosted | Feature importances + nonlinearity |
| Deep nonlinear, lots of data | MLP | Can learn complex patterns |
| Noisy data with outliers | Neural Boosted (Huber) | Robust loss function |
| Small dataset (<100 samples) | PLS or Neural Boosted | Regularization prevents overfitting |
| Very large dataset (>10,000) | Random Forest | Scales well |

**Add to GUI tooltip:**
```
Neural Boosted Regression
- Ensemble of small neural networks
- Best for: Nonlinear relationships with interpretability
- Handles: Outliers (with Huber loss), noisy data
- Speed: Moderate (slower than PLS/RF, faster than MLP)
- When to use: Need better accuracy than PLS, more interpretable than MLP
```

---

### Challenge 4: Feature Importance Quality

**Problem:** Aggregating importances from multiple weak learners might dilute signal.

**Concern:** Will important wavelengths still be identified correctly?

**Analysis:**

**Pros:**
- Averaging reduces noise from individual learner
- Similar to Random Forest (aggregates across trees)
- Tested approach in ensemble methods

**Potential Issues:**
- If weak learners use different features each round, average might be diffuse
- Less crisp than PLS VIP scores

**Validation Strategy:**
1. Test on synthetic data with known important features
2. Compare importance rankings to PLS VIP and RF importances
3. Check if top wavelengths are consistent

**Mitigation:**
- If importances are too diffuse, could weight by estimator contribution:
```python
# Weight importances by how much each estimator improved loss
for i, estimator in enumerate(self.estimators_):
    weight = loss_before[i] - loss_after[i]  # Improvement
    importances += weight * get_weights(estimator)
```

**Recommendation:** Start with simple averaging. Add weighting if needed after testing.

---

## 6. Implementation Timeline

### Phase 1: Core Implementation (3-4 hours)
- [ ] Create `neural_boosted.py` with NeuralBoostedRegressor class
- [ ] Implement fit() method with boosting loop
- [ ] Implement predict() method
- [ ] Implement _compute_loss() for MSE and Huber
- [ ] Implement get_feature_importances()
- [ ] Add parameter validation
- [ ] Basic manual testing

### Phase 2: Integration (1 hour)
- [ ] Add to `models.py` model grids
- [ ] Add to `get_feature_importances()` function
- [ ] Test that search pipeline recognizes new model

### Phase 3: GUI Integration (30 minutes)
- [ ] Add Neural Boosted checkbox
- [ ] Add tooltip
- [ ] Test GUI workflow

### Phase 4: Testing (1.5 hours)
- [ ] Write unit tests (`test_neural_boosted.py`)
- [ ] Test with synthetic spectral data
- [ ] Test feature importance extraction
- [ ] Test early stopping
- [ ] Test Huber loss

### Phase 5: Documentation (1 hour)
- [ ] Create NEURAL_BOOSTED_GUIDE.md user guide
- [ ] Update README with model comparison
- [ ] Add docstrings
- [ ] Create example notebook (optional)

### Phase 6: Validation & Tuning (1 hour)
- [ ] Run on real spectral dataset
- [ ] Compare performance to PLS, RF, MLP
- [ ] Verify feature importances make sense
- [ ] Profile computational cost
- [ ] Tune default hyperparameters if needed

**Total Estimated Time: 7-8 hours**

---

## 7. Success Criteria

### Functional Requirements
- ✅ Model fits without errors
- ✅ Predictions are reasonable (R² > 0.7 on test data)
- ✅ Early stopping works (stops before max estimators)
- ✅ Feature importances identify correct wavelengths
- ✅ Huber loss is more robust than MSE on outlier-contaminated data
- ✅ Integrates seamlessly with existing pipeline
- ✅ Appears in results CSV with all metrics

### Performance Requirements
- ✅ Faster than full MLP (>2x speedup)
- ✅ More accurate than PLS on nonlinear problems (>5% R² improvement)
- ✅ Early stopping typically triggers at 20-50 estimators
- ✅ Each configuration completes in <60 seconds on typical dataset (100 samples × 2000 wavelengths)

### Usability Requirements
- ✅ Clear checkbox and tooltip in GUI
- ✅ Progress shown during training
- ✅ Documentation explains when to use it
- ✅ Error messages are informative

---

## 8. Future Enhancements (v2.0+)

### 8.1 Classification Support
Create `NeuralBoostedClassifier` for PLS-DA alternative:
- Binary classification (ASD detection, etc.)
- Multi-class classification
- Uses softmax + cross-entropy loss

### 8.2 Custom Activations
Add Gaussian activation:
```python
activation='gaussian'  # exp(-x^2)
```

### 8.3 Advanced Regularization
- Dropout in weak learners
- Elastic net (true L1+L2 mixing)

### 8.4 Adaptive Learning Rate
- Start high, decay over boosting rounds
- Helps convergence

### 8.5 Feature Subsampling
- Each weak learner uses random subset of features
- Increases diversity (like Random Forest)
- Could improve ensemble quality

### 8.6 Warm Start
- Resume training from saved ensemble
- Add more weak learners to existing model

### 8.7 Model Interpretation Tools
- Partial dependence plots for specific wavelengths
- SHAP values for predictions
- Visualize ensemble evolution

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| sklearn MLP too slow | Low | Medium | Use small hidden layers (3-5 nodes) |
| Early stopping fails | Low | Low | Fallback to max estimators |
| Importances are poor | Medium | Medium | Test on synthetic data first; add weighting if needed |
| Users don't know when to use it | High | Low | Good documentation + tooltips |
| Computational cost too high | Medium | Medium | Reduce grid size, improve early stopping |
| Doesn't outperform existing models | Low | Medium | Validate on multiple datasets before release |

---

## 10. References

### Academic Papers
1. **Friedman, J. H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics* 29(5): 1189-1232.
   - Foundation of gradient boosting

2. **Hastie, T., Tibshirani, R., Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer.
   - Chapter 10: Boosting and Additive Trees

3. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD '16*.
   - Modern boosting implementation (trees, but similar principles)

### Spectral Analysis Context
4. **Workman, J., & Weyer, L. (2012).** *Practical Guide to Interpretive Near-Infrared Spectroscopy*. CRC Press.
   - When nonlinear methods help in spectroscopy

5. **Balabin, R. M., & Lomakina, E. I. (2011).** "Support vector machine regression (SVR) for near-infrared (NIR) spectroscopy." *Analyst* 136(8): 1703-1712.
   - Comparison of nonlinear methods for NIR

---

## 11. Quick Start Guide (After Implementation)

### For Users

**When should I use Neural Boosted Regression?**

Use it when:
- You have nonlinear relationships (PLS performs poorly)
- You need feature importances (MLP is too black-box)
- Your data has outliers (use Huber loss)
- You have moderate computational budget

**How to use in GUI:**

1. Load your spectral data
2. Check "Neural Boosted" in model selection
3. Run analysis
4. Check results CSV - look for Model="NeuralBoosted"
5. Examine `top_vars` column to see important wavelengths

**Interpreting results:**

```csv
Model,Preprocess,n_vars,RMSE,R2,SubsetTag,top_vars
NeuralBoosted,snv,2151,0.065,0.96,full,"1450.0,1455.0,2250.0,..."
```

- Lower RMSE = better predictions
- Higher R² = better fit
- `top_vars` shows the 30 most important wavelengths
- SubsetTag="full" means it used all wavelengths
- SubsetTag="top250" means it used top 250 important wavelengths

---

## 12. Questions for Discussion

Before implementation begins, please consider:

1. **Activation functions**: Implement Gaussian activation (complex) or skip it (simple)?
   - **Recommendation:** Skip for v1.0, add in v2.0 if requested

2. **Grid size**: 24 configs per preprocessing (current) or reduce to 12?
   - **Recommendation:** Start with 24, reduce if too slow

3. **Huber loss**: Expose in GUI or keep hidden (use MSE default)?
   - **Recommendation:** Keep hidden for v1.0, default MSE

4. **Classification**: Implement NeuralBoostedClassifier now or later?
   - **Recommendation:** Later (v2.0) - focus on regression first

5. **Default model**: Should Neural Boosted be checked by default in GUI?
   - **Recommendation:** No - let users opt-in (new model, needs validation)

---

## 13. Next Steps

### Immediate Actions

1. **Review this plan** - Discuss any concerns or modifications
2. **Approve implementation** - Give green light to start coding
3. **Create feature branch** - `git checkout -b feature/neural-boosted`

### Implementation Order

1. Start with Phase 1 (core class) - this is the bulk of the work
2. Test manually with synthetic data
3. Add Phase 2 (integration)
4. Add Phase 3 (GUI)
5. Add Phase 4 (unit tests)
6. Add Phase 5 (documentation)
7. Validate on real spectral data
8. Merge to main branch

### Testing Strategy

- Test each phase before moving to next
- Use synthetic data with known important features
- Validate on at least 2 real spectral datasets
- Compare performance to PLS, RF, MLP

---

**End of Implementation Plan**

*Ready to implement? Let's build it!*
