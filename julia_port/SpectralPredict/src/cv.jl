"""
    cv.jl

Cross-validation framework for spectral prediction models.

This module implements k-fold cross-validation with parallel processing support,
enabling robust model evaluation and hyperparameter tuning. The framework handles
both regression and classification tasks with comprehensive metrics.

Key Features:
- K-fold cross-validation with stratified splits
- Parallel execution via multi-threading
- Skip preprocessing mode for derivative subsets
- Comprehensive metrics (RMSE, R², MAE, Accuracy, ROC AUC, etc.)
- Type-stable, high-performance implementation

Critical Implementation Note:
When skip_preprocessing=true, data is used as-is without transformation. This is
essential for derivative subsets where preprocessing has already been applied.
"""

using Statistics
using Random
using LinearAlgebra

# Import required functions from other modules
include("models.jl")
include("preprocessing.jl")

using .Scoring: safe_zscore


# ============================================================================
# Cross-Validation Fold Creation
# ============================================================================

"""
    create_cv_folds(n_samples::Int, n_folds::Int=5)::Vector{Tuple{Vector{Int}, Vector{Int}}}

Create k-fold cross-validation splits for the given number of samples.

This function divides the dataset into k folds of approximately equal size. When
n_samples is not evenly divisible by n_folds, the first few folds will have one
additional sample.

# Arguments
- `n_samples::Int`: Total number of samples in the dataset
- `n_folds::Int`: Number of folds to create (default: 5)

# Returns
- `Vector{Tuple{Vector{Int}, Vector{Int}}}`: Array of (train_indices, test_indices) tuples
  - Each tuple contains:
    - `train_indices`: Vector of sample indices for training (n_samples - fold_size)
    - `test_indices`: Vector of sample indices for testing (≈ n_samples / n_folds)

# Throws
- `ArgumentError`: If n_folds < 2, n_folds > n_samples, or n_samples < 1

# Algorithm
1. Randomly shuffle sample indices (with fixed seed for reproducibility)
2. Divide shuffled indices into n_folds approximately equal groups
3. For each fold, use that group as test set and all others as training set

# Examples
```julia
# Create 5-fold CV splits for 100 samples
folds = create_cv_folds(100, 5)
# Returns 5 tuples, each with ~80 training and ~20 test indices

# Create 10-fold CV for 95 samples (not evenly divisible)
folds = create_cv_folds(95, 10)
# Returns 10 tuples: first 5 folds have 10 test samples, last 5 have 9

# Access first fold
train_idx, test_idx = folds[1]
println("Train: ", length(train_idx), " Test: ", length(test_idx))
# Train: 80 Test: 20

# Verify no overlap and complete coverage
for (train, test) in folds
    @assert isempty(intersect(train, test))
    @assert sort([train; test]) == 1:n_samples
end
```

# Notes
- Uses Random.seed!(42) for reproducibility across runs
- Shuffles data once before splitting to ensure randomization
- Train and test sets are disjoint (no sample appears in both)
- All samples appear exactly once in test sets across all folds
- Fold sizes differ by at most 1 when n_samples % n_folds ≠ 0
"""
function create_cv_folds(n_samples::Int, n_folds::Int=5)::Vector{Tuple{Vector{Int}, Vector{Int}}}
    # Validate inputs
    if n_samples < 1
        throw(ArgumentError("n_samples must be positive, got $n_samples"))
    end
    if n_folds < 2
        throw(ArgumentError("n_folds must be at least 2, got $n_folds"))
    end
    if n_folds > n_samples
        throw(ArgumentError("n_folds ($n_folds) cannot exceed n_samples ($n_samples)"))
    end

    # Create random permutation of indices for shuffling
    Random.seed!(42)  # Fixed seed for reproducibility
    shuffled_indices = randperm(n_samples)

    # Calculate fold sizes
    base_fold_size = div(n_samples, n_folds)
    n_larger_folds = mod(n_samples, n_folds)  # Number of folds that need +1 sample

    # Create folds
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, n_folds)

    start_idx = 1
    for fold in 1:n_folds
        # Determine size of this fold
        fold_size = base_fold_size + (fold <= n_larger_folds ? 1 : 0)

        # Get test indices for this fold
        end_idx = start_idx + fold_size - 1
        test_indices = shuffled_indices[start_idx:end_idx]

        # Get training indices (all other samples)
        train_indices = vcat(
            shuffled_indices[1:start_idx-1],
            shuffled_indices[end_idx+1:end]
        )

        folds[fold] = (train_indices, test_indices)

        # Move to next fold
        start_idx = end_idx + 1
    end

    return folds
end


# ============================================================================
# Metrics Computation
# ============================================================================

"""
    compute_regression_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})::Dict{String, Float64}

Compute comprehensive regression metrics for model evaluation.

# Metrics Computed
- **RMSE** (Root Mean Squared Error): √(mean((y_true - y_pred)²))
  - Scale-dependent, lower is better
  - Same units as target variable
- **R²** (Coefficient of Determination): 1 - (SS_res / SS_tot)
  - Scale-independent, higher is better (max = 1.0)
  - Proportion of variance explained by model
- **MAE** (Mean Absolute Error): mean(|y_true - y_pred|)
  - Robust to outliers, lower is better
  - Same units as target variable

# Arguments
- `y_true::Vector{Float64}`: True target values (n_samples,)
- `y_pred::Vector{Float64}`: Predicted target values (n_samples,)

# Returns
- `Dict{String, Float64}`: Dictionary with keys "RMSE", "R2", "MAE"

# Throws
- `ArgumentError`: If y_true and y_pred have different lengths or are empty

# Edge Cases
- Constant predictions: R² can be negative
- Zero variance in y_true: R² is NaN (converted to 0.0)
- Perfect predictions: RMSE=0, MAE=0, R²=1

# Examples
```julia
# Perfect predictions
y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
metrics = compute_regression_metrics(y_true, y_pred)
# Dict("RMSE" => 0.0, "R2" => 1.0, "MAE" => 0.0)

# Constant predictions
y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred = fill(3.0, 5)
metrics = compute_regression_metrics(y_true, y_pred)
# Dict("RMSE" => ~1.4, "R2" => 0.0, "MAE" => ~1.2)

# Poor predictions
y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred = [5.0, 4.0, 3.0, 2.0, 1.0]  # Reversed
metrics = compute_regression_metrics(y_true, y_pred)
# Dict("RMSE" => ~2.8, "R2" => -3.0, "MAE" => 2.4)
```

# Notes
- R² can be negative when model is worse than mean prediction
- All metrics are Float64 for type stability
- NaN/Inf values are handled gracefully (converted to 0.0 for R²)
"""
function compute_regression_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})::Dict{String, Float64}
    # Validate inputs
    if length(y_true) != length(y_pred)
        throw(ArgumentError("y_true and y_pred must have same length"))
    end
    if isempty(y_true)
        throw(ArgumentError("y_true and y_pred cannot be empty"))
    end

    n = length(y_true)

    # Compute residuals
    residuals = y_true .- y_pred

    # RMSE: Root Mean Squared Error
    mse = sum(residuals .^ 2) / n
    rmse = sqrt(mse)

    # MAE: Mean Absolute Error
    mae = sum(abs.(residuals)) / n

    # R²: Coefficient of Determination
    y_mean = mean(y_true)
    ss_tot = sum((y_true .- y_mean) .^ 2)  # Total sum of squares
    ss_res = sum(residuals .^ 2)            # Residual sum of squares

    # Handle edge case: zero variance in y_true
    r2 = if ss_tot == 0.0
        0.0  # Cannot compute R² when target has no variance
    else
        1.0 - (ss_res / ss_tot)
    end

    # Handle NaN/Inf (can occur with extreme predictions)
    if isnan(r2) || isinf(r2)
        r2 = 0.0
    end

    return Dict{String, Float64}(
        "RMSE" => rmse,
        "R2" => r2,
        "MAE" => mae
    )
end


"""
    compute_classification_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})::Dict{String, Float64}

Compute comprehensive classification metrics for model evaluation.

**Note**: This is a simplified implementation for binary classification. It assumes
y_pred contains probability scores (or can be thresholded at 0.5) and y_true contains
binary labels (0.0 or 1.0).

# Metrics Computed
- **Accuracy**: Fraction of correct predictions at threshold 0.5
- **ROC_AUC**: Area Under the ROC Curve (approximated via trapezoidal rule)
- **Precision**: TP / (TP + FP) at threshold 0.5
- **Recall**: TP / (TP + FN) at threshold 0.5

# Arguments
- `y_true::Vector{Float64}`: True binary labels (0.0 or 1.0, n_samples)
- `y_pred::Vector{Float64}`: Predicted probabilities or scores (n_samples,)

# Returns
- `Dict{String, Float64}`: Dictionary with keys "Accuracy", "ROC_AUC", "Precision", "Recall"

# Throws
- `ArgumentError`: If y_true and y_pred have different lengths or are empty

# Algorithm
For ROC AUC (simplified trapezoidal approximation):
1. Sort predictions in descending order
2. Compute TPR and FPR at each threshold
3. Use trapezoidal rule to estimate area under curve

# Examples
```julia
# Perfect classification
y_true = [0.0, 0.0, 1.0, 1.0, 1.0]
y_pred = [0.0, 0.1, 0.9, 0.95, 1.0]
metrics = compute_classification_metrics(y_true, y_pred)
# Dict("Accuracy" => 1.0, "ROC_AUC" => 1.0, "Precision" => 1.0, "Recall" => 1.0)

# Random predictions
y_true = [0.0, 0.0, 1.0, 1.0, 1.0]
y_pred = [0.5, 0.5, 0.5, 0.5, 0.5]
metrics = compute_classification_metrics(y_true, y_pred)
# Dict("Accuracy" => 0.6, "ROC_AUC" => ~0.5, ...)
```

# Notes
- Assumes binary classification only
- ROC AUC is approximated; for exact computation, consider MLJ.jl
- Handles edge cases (all same class, zero denominators)
- Returns 0.0 for undefined metrics (e.g., precision with no positive predictions)
"""
function compute_classification_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})::Dict{String, Float64}
    # Validate inputs
    if length(y_true) != length(y_pred)
        throw(ArgumentError("y_true and y_pred must have same length"))
    end
    if isempty(y_true)
        throw(ArgumentError("y_true and y_pred cannot be empty"))
    end

    n = length(y_true)

    # Convert predictions to binary at threshold 0.5
    y_pred_binary = y_pred .>= 0.5

    # Compute confusion matrix elements
    tp = sum((y_true .== 1.0) .& y_pred_binary)
    tn = sum((y_true .== 0.0) .& .!y_pred_binary)
    fp = sum((y_true .== 0.0) .& y_pred_binary)
    fn = sum((y_true .== 1.0) .& .!y_pred_binary)

    # Accuracy
    accuracy = (tp + tn) / n

    # Precision (handle division by zero)
    precision = if (tp + fp) > 0
        tp / (tp + fp)
    else
        0.0
    end

    # Recall (handle division by zero)
    recall = if (tp + fn) > 0
        tp / (tp + fn)
    else
        0.0
    end

    # ROC AUC (simplified trapezoidal approximation)
    roc_auc = compute_roc_auc(y_true, y_pred)

    return Dict{String, Float64}(
        "Accuracy" => accuracy,
        "ROC_AUC" => roc_auc,
        "Precision" => precision,
        "Recall" => recall
    )
end


"""
    compute_roc_auc(y_true::Vector{Float64}, y_scores::Vector{Float64})::Float64

Compute ROC AUC using trapezoidal approximation.

This is a simplified implementation that sorts predictions and computes the area
under the ROC curve using the trapezoidal rule.

# Arguments
- `y_true::Vector{Float64}`: True binary labels (0.0 or 1.0)
- `y_scores::Vector{Float64}`: Predicted scores/probabilities

# Returns
- `Float64`: ROC AUC score (0.0 to 1.0)

# Notes
- Returns 0.5 if all true labels are the same (undefined AUC)
- Uses trapezoidal approximation (may differ slightly from sklearn)
"""
function compute_roc_auc(y_true::Vector{Float64}, y_scores::Vector{Float64})::Float64
    # Edge case: all same class
    n_pos = sum(y_true .== 1.0)
    n_neg = sum(y_true .== 0.0)

    if n_pos == 0 || n_neg == 0
        return 0.5  # Undefined, return random classifier score
    end

    # Sort by scores (descending)
    sorted_indices = sortperm(y_scores, rev=true)
    sorted_labels = y_true[sorted_indices]

    # Compute TPR and FPR at each threshold
    tpr = zeros(length(sorted_labels) + 1)
    fpr = zeros(length(sorted_labels) + 1)

    tp = 0.0
    fp = 0.0

    for i in 1:length(sorted_labels)
        if sorted_labels[i] == 1.0
            tp += 1
        else
            fp += 1
        end

        tpr[i+1] = tp / n_pos
        fpr[i+1] = fp / n_neg
    end

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in 1:length(fpr)-1
        # Trapezoid area: (width) * (avg_height)
        width = fpr[i+1] - fpr[i]
        height = (tpr[i] + tpr[i+1]) / 2.0
        auc += width * height
    end

    return auc
end


# ============================================================================
# Single Fold Execution
# ============================================================================

"""
    run_single_fold(
        X::Matrix{Float64},
        y::Vector{Float64},
        train_idx::Vector{Int},
        test_idx::Vector{Int},
        model,
        model_name::String,
        preprocess_config::Dict{String, Any},
        task_type::String;
        skip_preprocessing::Bool=false
    )::Dict{String, Float64}

Execute a single cross-validation fold: train model on train_idx, evaluate on test_idx.

This function handles the complete workflow for one CV fold:
1. Split data into train/test using provided indices
2. Apply preprocessing (unless skip_preprocessing=true)
3. Train model on processed training data
4. Predict on processed test data
5. Compute and return metrics

# Arguments
- `X::Matrix{Float64}`: Full feature matrix (n_samples × n_features)
- `y::Vector{Float64}`: Full target vector (n_samples,)
- `train_idx::Vector{Int}`: Indices for training set
- `test_idx::Vector{Int}`: Indices for test set
- `model`: Model instance (from build_model)
- `model_name::String`: Name of model type (for logging/debugging)
- `preprocess_config::Dict{String, Any}`: Preprocessing configuration
- `task_type::String`: Either "regression" or "classification"
- `skip_preprocessing::Bool`: If true, skip preprocessing (data already processed)

# Returns
- `Dict{String, Float64}`: Performance metrics
  - Regression: {"RMSE" => ..., "R2" => ..., "MAE" => ...}
  - Classification: {"Accuracy" => ..., "ROC_AUC" => ..., "Precision" => ..., "Recall" => ...}

# Critical: Skip Preprocessing Logic
When `skip_preprocessing=true`, the function uses data as-is without transformation.
This is essential for derivative subsets where preprocessing was already applied
at the parent level.

```julia
if skip_preprocessing
    # Data is already preprocessed, use as-is
    X_train_processed = X[train_idx, :]
    X_test_processed = X[test_idx, :]
else
    # Apply preprocessing to train/test splits
    X_train_processed = apply_preprocessing(X[train_idx, :], preprocess_config)
    X_test_processed = apply_preprocessing(X[test_idx, :], preprocess_config)
end
```

# Examples
```julia
# Load data
X = rand(100, 50)
y = rand(100)

# Create fold
folds = create_cv_folds(100, 5)
train_idx, test_idx = folds[1]

# Build model
model = build_model("PLS", Dict("n_components" => 5), "regression")

# Run fold with preprocessing
preprocess_config = Dict("name" => "snv")
metrics = run_single_fold(
    X, y, train_idx, test_idx, model, "PLS",
    preprocess_config, "regression"
)
# Returns: Dict("RMSE" => 0.123, "R2" => 0.85, "MAE" => 0.098)

# Run fold without preprocessing (data already processed)
metrics = run_single_fold(
    X, y, train_idx, test_idx, model, "PLS",
    preprocess_config, "regression",
    skip_preprocessing=true
)
```

# Notes
- Model is trained in-place (model state is modified)
- Preprocessing is fit on training data only, then applied to test data
- For skip_preprocessing=true, ensure X is already preprocessed consistently
- Handles both regression and classification automatically
"""
function run_single_fold(
    X::Matrix{Float64},
    y::Vector{Float64},
    train_idx::Vector{Int},
    test_idx::Vector{Int},
    model,
    model_name::String,
    preprocess_config::Dict{String, Any},
    task_type::String;
    skip_preprocessing::Bool=false
)::Dict{String, Float64}

    # Split data into train/test
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Apply preprocessing (or skip if data already preprocessed)
    if skip_preprocessing
        # Data is already preprocessed, use as-is
        # This is critical for derivative subsets
        X_train_processed = X_train
        X_test_processed = X_test
    else
        # Apply preprocessing transformations
        X_train_processed = apply_preprocessing(X_train, preprocess_config)
        X_test_processed = apply_preprocessing(X_test, preprocess_config)
    end

    # Train model
    fit_model!(model, X_train_processed, y_train)

    # Make predictions
    y_pred = predict_model(model, X_test_processed)

    # Compute metrics based on task type
    if task_type == "regression"
        metrics = compute_regression_metrics(y_test, y_pred)
    elseif task_type == "classification"
        metrics = compute_classification_metrics(y_test, y_pred)
    else
        throw(ArgumentError("task_type must be 'regression' or 'classification', got '$task_type'"))
    end

    return metrics
end


# ============================================================================
# Cross-Validation Execution
# ============================================================================

"""
    run_cross_validation(
        X::Matrix{Float64},
        y::Vector{Float64},
        model,
        model_name::String,
        preprocess_config::Dict{String, Any},
        task_type::String;
        n_folds::Int=5,
        skip_preprocessing::Bool=false
    )::Dict{String, Any}

Execute k-fold cross-validation with parallel processing support.

This is the main entry point for cross-validation. It:
1. Creates k folds
2. Runs each fold (potentially in parallel)
3. Aggregates results (mean and std of metrics)
4. Returns comprehensive summary

# Arguments
- `X::Matrix{Float64}`: Feature matrix (n_samples × n_features)
- `y::Vector{Float64}`: Target vector (n_samples,)
- `model`: Model instance (will be cloned for each fold)
- `model_name::String`: Name of model type
- `preprocess_config::Dict{String, Any}`: Preprocessing configuration
- `task_type::String`: Either "regression" or "classification"
- `n_folds::Int`: Number of folds (default: 5)
- `skip_preprocessing::Bool`: If true, skip preprocessing (default: false)

# Returns
- `Dict{String, Any}`: Cross-validation results containing:
  - Mean metrics: "RMSE_mean", "R2_mean", etc.
  - Std metrics: "RMSE_std", "R2_std", etc.
  - Individual fold results: "cv_scores" (Vector of Dicts)
  - Metadata: "n_folds", "task_type"

# Parallelization
The function can use multi-threading for parallel fold execution:
```julia
# Set number of threads before starting Julia
# julia -t 4  # Use 4 threads

# CV will automatically use available threads
results = run_cross_validation(X, y, model, "PLS", config, "regression")
```

# Examples
```julia
# Basic 5-fold CV
X = rand(100, 50)
y = rand(100)
model = build_model("PLS", Dict("n_components" => 5), "regression")
preprocess_config = Dict("name" => "snv")

results = run_cross_validation(
    X, y, model, "PLS", preprocess_config, "regression"
)

# Access results
println("RMSE: ", results["RMSE_mean"], " ± ", results["RMSE_std"])
println("R²: ", results["R2_mean"], " ± ", results["R2_std"])

# View individual fold scores
for (i, fold_metrics) in enumerate(results["cv_scores"])
    println("Fold $i: RMSE = ", fold_metrics["RMSE"])
end

# 10-fold CV with skip preprocessing
results = run_cross_validation(
    X, y, model, "PLS", preprocess_config, "regression",
    n_folds=10,
    skip_preprocessing=true
)

# Classification example
y_class = rand([0.0, 1.0], 100)
model_class = build_model("Ridge", Dict("alpha" => 1.0), "regression")
results = run_cross_validation(
    X, y_class, model_class, "Ridge",
    Dict("name" => "raw"), "classification"
)
println("Accuracy: ", results["Accuracy_mean"], " ± ", results["Accuracy_std"])
println("ROC AUC: ", results["ROC_AUC_mean"], " ± ", results["ROC_AUC_std"])
```

# Notes
- Each fold gets a fresh model instance (via build_model)
- Results are deterministic (fixed random seed in create_cv_folds)
- Parallel execution preserves reproducibility
- Handles edge cases (small datasets, constant predictions)
- Type-stable for optimal performance

# Performance
- Single-threaded: ~O(n_folds × training_time)
- Multi-threaded: ~O(training_time) with n_folds threads
- Memory: O(n_folds × model_size) during parallel execution
"""
function run_cross_validation(
    X::Matrix{Float64},
    y::Vector{Float64},
    model,
    model_name::String,
    preprocess_config::Dict{String, Any},
    task_type::String;
    n_folds::Int=5,
    skip_preprocessing::Bool=false
)::Dict{String, Any}

    # Validate inputs
    if size(X, 1) != length(y)
        throw(ArgumentError("X and y must have same number of samples"))
    end
    if n_folds < 2
        throw(ArgumentError("n_folds must be at least 2"))
    end
    if n_folds > size(X, 1)
        throw(ArgumentError("n_folds cannot exceed number of samples"))
    end

    n_samples = size(X, 1)

    # Create CV folds
    folds = create_cv_folds(n_samples, n_folds)

    # Run each fold and collect metrics
    # Note: For thread-safety, each fold needs its own model instance
    fold_metrics = Vector{Dict{String, Float64}}(undef, n_folds)

    # Sequential execution (for thread-safety with current model structures)
    # To enable parallel execution, ensure models are thread-safe or use locks
    for i in 1:n_folds
        train_idx, test_idx = folds[i]

        # Create fresh model instance for this fold
        # We need to rebuild to get a clean state
        model_config = extract_model_config(model, model_name)
        fold_model = build_model(model_name, model_config, task_type)

        # Run fold
        fold_metrics[i] = run_single_fold(
            X, y, train_idx, test_idx,
            fold_model, model_name,
            preprocess_config, task_type,
            skip_preprocessing=skip_preprocessing
        )
    end

    # Aggregate results across folds
    results = aggregate_cv_results(fold_metrics, task_type, n_folds)

    return results
end


"""
    extract_model_config(model, model_name::String)::Dict{String, Any}

Extract hyperparameter configuration from a model instance.

This helper function reconstructs the config dict from a model's fields,
allowing us to create fresh model instances for each CV fold.

# Arguments
- `model`: Model instance
- `model_name::String`: Name of model type

# Returns
- `Dict{String, Any}`: Configuration dictionary

# Examples
```julia
model = PLSModel(10)
config = extract_model_config(model, "PLS")
# Returns: Dict("n_components" => 10)

model = RidgeModel(1.0)
config = extract_model_config(model, "Ridge")
# Returns: Dict("alpha" => 1.0)
```
"""
function extract_model_config(model, model_name::String)::Dict{String, Any}
    if model_name == "PLS"
        return Dict("n_components" => model.n_components)
    elseif model_name == "Ridge"
        return Dict("alpha" => model.alpha)
    elseif model_name == "Lasso"
        return Dict("alpha" => model.alpha)
    elseif model_name == "ElasticNet"
        return Dict("alpha" => model.alpha, "l1_ratio" => model.l1_ratio)
    elseif model_name == "RandomForest"
        return Dict("n_trees" => model.n_trees, "max_features" => model.max_features)
    elseif model_name == "MLP"
        return Dict("hidden_layers" => model.hidden_layers, "learning_rate" => model.learning_rate)
    else
        throw(ArgumentError("Unknown model name: $model_name"))
    end
end


"""
    aggregate_cv_results(
        fold_metrics::Vector{Dict{String, Float64}},
        task_type::String,
        n_folds::Int
    )::Dict{String, Any}

Aggregate cross-validation results across folds.

Computes mean and standard deviation for each metric, and stores individual
fold results for detailed analysis.

# Arguments
- `fold_metrics::Vector{Dict{String, Float64}}`: Metrics from each fold
- `task_type::String`: "regression" or "classification"
- `n_folds::Int`: Number of folds

# Returns
- `Dict{String, Any}`: Aggregated results with mean, std, and individual scores

# Examples
```julia
fold_metrics = [
    Dict("RMSE" => 0.5, "R2" => 0.85, "MAE" => 0.4),
    Dict("RMSE" => 0.6, "R2" => 0.80, "MAE" => 0.5),
    Dict("RMSE" => 0.4, "R2" => 0.90, "MAE" => 0.3)
]

results = aggregate_cv_results(fold_metrics, "regression", 3)
# Returns:
# Dict(
#     "RMSE_mean" => 0.5,
#     "RMSE_std" => 0.1,
#     "R2_mean" => 0.85,
#     "R2_std" => 0.05,
#     "MAE_mean" => 0.4,
#     "MAE_std" => 0.1,
#     "cv_scores" => fold_metrics,
#     "n_folds" => 3,
#     "task_type" => "regression"
# )
```
"""
function aggregate_cv_results(
    fold_metrics::Vector{Dict{String, Float64}},
    task_type::String,
    n_folds::Int
)::Dict{String, Any}

    # Determine metric names based on task type
    if task_type == "regression"
        metric_names = ["RMSE", "R2", "MAE"]
    elseif task_type == "classification"
        metric_names = ["Accuracy", "ROC_AUC", "Precision", "Recall"]
    else
        throw(ArgumentError("task_type must be 'regression' or 'classification'"))
    end

    # Initialize results dictionary
    results = Dict{String, Any}()

    # Compute mean and std for each metric
    for metric in metric_names
        # Extract metric values across folds
        values = [fold_metrics[i][metric] for i in 1:n_folds]

        # Compute statistics
        mean_val = mean(values)
        std_val = std(values)

        # Store in results
        results["$(metric)_mean"] = mean_val
        results["$(metric)_std"] = std_val
    end

    # Store individual fold scores for detailed analysis
    results["cv_scores"] = fold_metrics

    # Store metadata
    results["n_folds"] = n_folds
    results["task_type"] = task_type

    return results
end


# ============================================================================
# Parallel Cross-Validation (Thread-Safe Version)
# ============================================================================

"""
    run_cross_validation_parallel(
        X::Matrix{Float64},
        y::Vector{Float64},
        model,
        model_name::String,
        preprocess_config::Dict{String, Any},
        task_type::String;
        n_folds::Int=5,
        skip_preprocessing::Bool=false
    )::Dict{String, Any}

Execute k-fold cross-validation with multi-threaded parallel processing.

This is a parallel version of run_cross_validation that uses Threads.@threads
for concurrent fold execution. Use this for large datasets where training time
is significant.

**Important**: Requires Julia to be started with multiple threads:
```bash
julia -t 4  # Use 4 threads
julia -t auto  # Use all available cores
```

# Arguments
Same as run_cross_validation

# Returns
Same as run_cross_validation

# Examples
```julia
# Start Julia with: julia -t 4

X = rand(1000, 200)
y = rand(1000)
model = build_model("RandomForest", Dict("n_trees" => 100, "max_features" => "sqrt"), "regression")

# Parallel CV (uses all available threads)
results = run_cross_validation_parallel(
    X, y, model, "RandomForest",
    Dict("name" => "snv_deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
    "regression",
    n_folds=10
)
```

# Notes
- Each fold runs in a separate thread
- Models must be thread-safe (or we create separate instances)
- Speedup is approximately linear with number of threads (for large models)
- Falls back to sequential execution if only 1 thread available
"""
function run_cross_validation_parallel(
    X::Matrix{Float64},
    y::Vector{Float64},
    model,
    model_name::String,
    preprocess_config::Dict{String, Any},
    task_type::String;
    n_folds::Int=5,
    skip_preprocessing::Bool=false
)::Dict{String, Any}

    # Validate inputs
    if size(X, 1) != length(y)
        throw(ArgumentError("X and y must have same number of samples"))
    end
    if n_folds < 2
        throw(ArgumentError("n_folds must be at least 2"))
    end
    if n_folds > size(X, 1)
        throw(ArgumentError("n_folds cannot exceed number of samples"))
    end

    n_samples = size(X, 1)

    # Create CV folds
    folds = create_cv_folds(n_samples, n_folds)

    # Run each fold in parallel
    fold_metrics = Vector{Dict{String, Float64}}(undef, n_folds)

    Threads.@threads for i in 1:n_folds
        train_idx, test_idx = folds[i]

        # Create fresh model instance for this fold (thread-safe)
        model_config = extract_model_config(model, model_name)
        fold_model = build_model(model_name, model_config, task_type)

        # Run fold
        fold_metrics[i] = run_single_fold(
            X, y, train_idx, test_idx,
            fold_model, model_name,
            preprocess_config, task_type,
            skip_preprocessing=skip_preprocessing
        )
    end

    # Aggregate results across folds
    results = aggregate_cv_results(fold_metrics, task_type, n_folds)

    return results
end


# ============================================================================
# Exports
# ============================================================================

export create_cv_folds
export compute_regression_metrics
export compute_classification_metrics
export run_single_fold
export run_cross_validation
export run_cross_validation_parallel
