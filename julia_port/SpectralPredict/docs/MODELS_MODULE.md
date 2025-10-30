# Models Module Documentation

## Overview

The `models.jl` module provides a unified interface for machine learning models used in spectral prediction. It implements wrappers around popular Julia ML libraries to provide consistent APIs for model training, prediction, and feature importance extraction.

## Supported Models

### 1. PLS Regression (Partial Least Squares)

**Best for:** High-dimensional spectral data with collinearity

**Implementation:** Uses `MultivariateStats.jl` CCA (Canonical Correlation Analysis) as PLS equivalent

**Hyperparameters:**
- `n_components`: Number of latent components [1, 2, 3, 5, 7, 10, 15, 20]

**Key Features:**
- Handles multicollinearity well
- Reduces dimensionality while maintaining predictive power
- Provides VIP (Variable Importance in Projection) scores

**Usage:**
```julia
config = Dict("n_components" => 10)
model = build_model("PLS", config, "regression")
fit_model!(model, X_train, y_train)
predictions = predict_model(model, X_test)
```

### 2. Ridge Regression

**Best for:** Linear relationships with regularization

**Implementation:** Uses `GLMNet.jl` with alpha=0 (pure L2 penalty)

**Hyperparameters:**
- `alpha`: Regularization strength [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

**Key Features:**
- Shrinks coefficients to reduce overfitting
- Maintains all features (no feature selection)
- Fast and stable

**Usage:**
```julia
config = Dict("alpha" => 1.0)
model = build_model("Ridge", config, "regression")
fit_model!(model, X_train, y_train)
predictions = predict_model(model, X_test)
```

### 3. Lasso Regression

**Best for:** Feature selection and sparse models

**Implementation:** Uses `GLMNet.jl` with alpha=1 (pure L1 penalty)

**Hyperparameters:**
- `alpha`: Regularization strength [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

**Key Features:**
- Performs automatic feature selection (sets coefficients to zero)
- Produces sparse models
- Good for interpretability

**Usage:**
```julia
config = Dict("alpha" => 0.1)
model = build_model("Lasso", config, "regression")
fit_model!(model, X_train, y_train)
predictions = predict_model(model, X_test)
```

### 4. Elastic Net

**Best for:** Balance between Ridge and Lasso

**Implementation:** Uses `GLMNet.jl` with configurable L1/L2 mix

**Hyperparameters:**
- `alpha`: Overall regularization strength [0.001, 0.01, 0.1, 1.0]
- `l1_ratio`: L1 vs L2 penalty mix [0.1, 0.5, 0.9]
  - 0.0 = Pure Ridge (L2)
  - 1.0 = Pure Lasso (L1)
  - 0.5 = Equal mix

**Key Features:**
- Combines benefits of Ridge and Lasso
- Can handle correlated features better than Lasso
- Performs feature selection with stability

**Usage:**
```julia
config = Dict("alpha" => 1.0, "l1_ratio" => 0.5)
model = build_model("ElasticNet", config, "regression")
fit_model!(model, X_train, y_train)
predictions = predict_model(model, X_test)
```

### 5. Random Forest

**Best for:** Non-linear relationships and robust predictions

**Implementation:** Uses `DecisionTree.jl`

**Hyperparameters:**
- `n_trees`: Number of trees in forest [50, 100, 200]
- `max_features`: Feature sampling strategy ["sqrt", "log2"]
  - "sqrt": sqrt(n_features) features per split
  - "log2": log2(n_features) features per split

**Key Features:**
- Captures non-linear relationships
- Robust to outliers
- Provides feature importances
- No need for feature scaling

**Usage:**
```julia
config = Dict("n_trees" => 100, "max_features" => "sqrt")
model = build_model("RandomForest", config, "regression")
fit_model!(model, X_train, y_train)
predictions = predict_model(model, X_test)
```

### 6. Multi-Layer Perceptron (MLP)

**Best for:** Complex non-linear patterns

**Implementation:** Uses `Flux.jl` neural networks

**Hyperparameters:**
- `hidden_layers`: Network architecture [(50,), (100,), (50, 50)]
  - (50,): Single hidden layer with 50 neurons
  - (50, 50): Two hidden layers with 50 neurons each
- `learning_rate`: Optimizer learning rate [0.001, 0.01]

**Key Features:**
- Powerful function approximation
- Automatic feature learning
- Early stopping to prevent overfitting
- Data normalization built-in

**Usage:**
```julia
config = Dict("hidden_layers" => (50, 50), "learning_rate" => 0.01)
model = build_model("MLP", config, "regression")
fit_model!(model, X_train, y_train)
predictions = predict_model(model, X_test)
```

## API Reference

### Configuration Functions

#### `get_model_configs(model_name::String)::Vector{Dict{String, Any}}`

Returns all hyperparameter configurations to search for a given model.

**Arguments:**
- `model_name`: One of "PLS", "Ridge", "Lasso", "ElasticNet", "RandomForest", "MLP"

**Returns:**
- Vector of configuration dictionaries

**Example:**
```julia
configs = get_model_configs("PLS")
# Returns: [
#   Dict("n_components" => 1),
#   Dict("n_components" => 2),
#   ...
# ]
```

### Model Building

#### `build_model(model_name::String, config::Dict{String, Any}, task_type::String)`

Creates a new model instance with specified configuration.

**Arguments:**
- `model_name`: Model type name
- `config`: Hyperparameter dictionary (from `get_model_configs`)
- `task_type`: "regression" (classification not yet implemented)

**Returns:**
- Model instance (PLSModel, RidgeModel, etc.)

**Example:**
```julia
config = Dict("n_components" => 10)
model = build_model("PLS", config, "regression")
```

### Training

#### `fit_model!(model, X::Matrix{Float64}, y::Vector{Float64})`

Trains the model on provided data. Modifies the model in-place.

**Arguments:**
- `model`: Model instance from `build_model`
- `X`: Training features (n_samples × n_features)
- `y`: Training targets (n_samples,)

**Returns:**
- The fitted model (for chaining)

**Side Effects:**
- Modifies model in-place
- Stores normalization parameters if needed

**Example:**
```julia
fit_model!(model, X_train, y_train)
```

### Prediction

#### `predict_model(model, X::Matrix{Float64})::Vector{Float64}`

Generates predictions using a fitted model.

**Arguments:**
- `model`: Fitted model instance
- `X`: Features for prediction (n_samples × n_features)

**Returns:**
- Vector of predictions (n_samples,)

**Throws:**
- `ArgumentError` if model is not fitted

**Example:**
```julia
predictions = predict_model(model, X_test)
```

### Feature Importance

#### `get_feature_importances(model, model_name::String, X::Matrix{Float64}, y::Vector{Float64})::Vector{Float64}`

Extracts feature importance scores from a fitted model.

**Arguments:**
- `model`: Fitted model instance
- `model_name`: Model type name
- `X`: Training features (needed for some methods)
- `y`: Training targets (needed for VIP scores)

**Returns:**
- Vector of importance scores (n_features,) - higher values = more important

**Methods by Model Type:**

| Model | Method | Description |
|-------|--------|-------------|
| PLS | VIP Scores | Variable Importance in Projection - accounts for explained variance |
| Ridge/Lasso/ElasticNet | Absolute Coefficients | Magnitude of linear coefficients |
| RandomForest | Split-based | Importance based on split quality improvement |
| MLP | Weight Magnitudes | Mean absolute weight from first layer |

**Example:**
```julia
importances = get_feature_importances(model, "PLS", X_train, y_train)
top_10 = sortperm(importances, rev=true)[1:10]
println("Top 10 features: ", top_10)
```

## Model Comparison

| Model | Linearity | Feature Selection | Multicollinearity | Interpretability | Speed |
|-------|-----------|-------------------|-------------------|------------------|-------|
| PLS | Linear | No | Excellent | Good | Fast |
| Ridge | Linear | No | Good | Good | Very Fast |
| Lasso | Linear | Yes | Poor | Excellent | Fast |
| ElasticNet | Linear | Yes | Good | Good | Fast |
| RandomForest | Non-linear | Implicit | Excellent | Fair | Medium |
| MLP | Non-linear | No | Good | Poor | Slow |

## Workflow Examples

### Basic Workflow

```julia
# 1. Get configurations
configs = get_model_configs("PLS")

# 2. Select configuration
config = configs[5]  # Or Dict("n_components" => 10)

# 3. Build model
model = build_model("PLS", config, "regression")

# 4. Train model
fit_model!(model, X_train, y_train)

# 5. Make predictions
y_pred = predict_model(model, X_test)

# 6. Evaluate
rmse = sqrt(mean((y_test .- y_pred).^2))
r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

# 7. Get important features
importances = get_feature_importances(model, "PLS", X_train, y_train)
```

### Hyperparameter Search

```julia
# Search over all PLS configurations
best_r2 = -Inf
best_config = nothing
best_model = nothing

for config in get_model_configs("PLS")
    model = build_model("PLS", config, "regression")
    fit_model!(model, X_train, y_train)

    y_pred = predict_model(model, X_val)
    r2 = 1 - sum((y_val .- y_pred).^2) / sum((y_val .- mean(y_val)).^2)

    if r2 > best_r2
        best_r2 = r2
        best_config = config
        best_model = model
    end
end

println("Best config: ", best_config)
println("Best R²: ", best_r2)
```

### Multi-Model Comparison

```julia
models_to_test = ["PLS", "Ridge", "RandomForest", "MLP"]
results = Dict()

for model_name in models_to_test
    # Get first config for quick comparison
    config = get_model_configs(model_name)[1]

    model = build_model(model_name, config, "regression")
    fit_model!(model, X_train, y_train)

    y_pred = predict_model(model, X_test)
    rmse = sqrt(mean((y_test .- y_pred).^2))
    r2 = 1 - sum((y_test .- y_pred).^2) / sum((y_test .- mean(y_test)).^2)

    results[model_name] = Dict("rmse" => rmse, "r2" => r2)
end

# Find best model
best_model = argmax(k -> results[k]["r2"], keys(results))
println("Best model: ", best_model)
```

## Implementation Notes

### PLS via CCA

The Julia implementation uses `MultivariateStats.CCA` (Canonical Correlation Analysis) as an equivalent to PLS regression. While not identical to sklearn's PLSRegression, it provides similar dimensionality reduction and prediction capabilities.

### GLMNet Integration

Ridge, Lasso, and ElasticNet all use `GLMNet.jl`. Note the parameter mapping:
- Our `alpha` = GLMNet's `lambda` (regularization strength)
- Our `l1_ratio` = GLMNet's `alpha` (L1 vs L2 mix)

This follows sklearn's convention rather than GLMNet's native convention.

### Random Forest Feature Importance

The feature importance calculation for Random Forest is simplified compared to sklearn. It counts the number of splits on each feature across all trees, weighted by the importance of those splits.

### MLP Training Details

- Uses Adam optimizer
- 20% validation split for early stopping
- Patience of 20 epochs (checks every 10 epochs)
- Maximum 1000 epochs
- Data is z-score normalized (mean=0, std=1)
- ReLU activation for hidden layers, linear for output

## Error Handling

All functions include comprehensive error checking:

```julia
# Model not fitted
model = build_model("PLS", Dict("n_components" => 5), "regression")
predict_model(model, X_test)  # Throws ArgumentError

# Invalid model name
get_model_configs("InvalidModel")  # Throws ArgumentError

# Unsupported task type
build_model("PLS", config, "classification")  # Throws ArgumentError
```

## Performance Tips

1. **PLS**: Start with n_components around sqrt(n_features), then search
2. **Ridge/Lasso**: Try logarithmic grid of alphas first
3. **RandomForest**: More trees = better but slower; start with 100
4. **MLP**: Smaller networks train faster; try (50,) before (100, 100)

## Future Enhancements

- Classification support (PLS-DA, RF Classifier, MLP Classifier)
- Neural Boosted models
- Cross-validation utilities
- Ensemble methods
- Model serialization (save/load)
- GPU acceleration for MLP

## See Also

- `examples/models_example.jl` - Complete usage examples
- `test/test_models.jl` - Test suite and edge cases
- Python reference: `src/spectral_predict/models.py`
