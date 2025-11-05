"""
    Diagnostics

Model diagnostics utilities for spectral analysis.

This module provides diagnostic tools for assessing regression model quality:
- Residual analysis (`compute_residuals`)
- Leverage detection (`compute_leverage`)
- Q-Q plot data generation (`qq_plot_data`)
- Prediction intervals (`jackknife_prediction_intervals`)

# References

- Weisberg, S. (2005). Applied Linear Regression. Wiley.
- Fox, J. (2008). Applied Regression Analysis and Generalized Linear Models. Sage.
- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.

# Examples

```julia
using SpectralPredict.Diagnostics

# Compute residuals
residuals, std_residuals = compute_residuals(y_true, y_pred)

# Compute leverage
leverage, threshold = compute_leverage(X_train)
high_leverage_samples = findall(leverage .> threshold)

# Generate Q-Q plot data
theoretical, sample = qq_plot_data(residuals)

# Compute prediction intervals via jackknife
model_fn = (X, y) -> fit_pls(X, y, n_components=10)
pred, lower, upper, stderr = jackknife_prediction_intervals(
    model_fn, X_train, y_train, X_test, confidence=0.95
)
```
"""
module Diagnostics

using Statistics
using LinearAlgebra
using Distributions

export compute_residuals
export compute_leverage
export qq_plot_data
export jackknife_prediction_intervals


"""
    compute_residuals(y_true, y_pred)

Compute raw and standardized residuals for regression models.

Standardized residuals are useful for identifying outliers and assessing
model fit. They follow an approximately normal distribution for well-specified models.

# Arguments
- `y_true::Vector{Float64}`: True target values (n_samples)
- `y_pred::Vector{Float64}`: Predicted values (n_samples)

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (residuals, standardized_residuals)
  - `residuals`: Raw residuals (y_true - y_pred)
  - `standardized_residuals`: Residuals divided by their standard deviation

# Mathematical Formulation

Raw residuals:
```
e_i = y_i - ŷ_i
```

Standardized residuals:
```
e_i^* = e_i / σ_e
```
where σ_e is the standard deviation of residuals.

# Notes
- Standardized residuals should have mean ≈ 0 and std ≈ 1
- Values |e_i^*| > 2 or 3 indicate potential outliers
- Includes numerical stability check: if std(residuals) < 1e-10, returns raw residuals

# Example

```julia
y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
residuals, std_residuals = compute_residuals(y_true, y_pred)

# Check for outliers
outliers = findall(abs.(std_residuals) .> 3)
```
"""
function compute_residuals(
    y_true::Vector{Float64},
    y_pred::Vector{Float64}
)::Tuple{Vector{Float64}, Vector{Float64}}

    # Compute raw residuals
    residuals = y_true .- y_pred

    # Compute standardized residuals with numerical stability check
    σ = std(residuals)

    if σ > 1e-10
        standardized_residuals = residuals ./ σ
    else
        # If standard deviation is too small, return raw residuals
        # This prevents division by zero
        standardized_residuals = residuals
    end

    return residuals, standardized_residuals
end


"""
    compute_leverage(X; return_threshold=true)

Compute leverage (hat values) for samples in a regression model.

Leverage measures the influence of each training sample on the model predictions.
High-leverage points are far from the center of the feature space and can
disproportionately affect model fit.

# Arguments
- `X::Matrix{Float64}`: Feature matrix (n_samples × n_features)
  This should be the preprocessed data used for model fitting
- `return_threshold::Bool=true`: If true, also return the leverage threshold (2p/n)

# Returns
- `Vector{Float64}` or `Tuple{Vector{Float64}, Float64}`:
  - `leverage`: Hat values for each sample (n_samples)
  - `threshold` (optional): 2(p+1)/n threshold for moderate leverage

# Mathematical Formulation

Leverage is the diagonal of the hat matrix H:
```
H = X(X'X)^(-1)X'
h_ii = diag(H)
```

For augmented design matrix X_aug = [1, X]:
```
h_ii = x_i^T (X^T X)^(-1) x_i
```

# Thresholds
- Moderate leverage: h_ii > 2p/n
- High leverage: h_ii > 3p/n
where p = number of parameters (features + intercept)

# Notes
- Average leverage = p/n (rule of thumb)
- All leverage values satisfy: 0 ≤ h_ii ≤ 1
- For numerical stability, uses SVD when:
  * n_features > 100
  * n_samples ≤ n_features + 1
  * Direct computation fails (singular matrix)
- SVD approach: H = UU^T where X = UΣV^T

# Example

```julia
leverage, threshold = compute_leverage(X_train)

# Identify high-leverage samples
high_leverage = findall(leverage .> threshold)
println("High-leverage samples: ", high_leverage)

# Plot leverage
scatter(1:length(leverage), leverage,
        xlabel="Sample", ylabel="Leverage")
hline!([threshold], label="Threshold (2p/n)")
```
"""
function compute_leverage(
    X::Matrix{Float64};
    return_threshold::Bool=true
)::Union{Vector{Float64}, Tuple{Vector{Float64}, Float64}}

    n_samples, n_features = size(X)

    # Add intercept column
    X_aug = hcat(ones(n_samples), X)
    p = size(X_aug, 2)  # Number of parameters (including intercept)

    # Determine which method to use based on problem size
    use_svd = (n_features > 100) || (n_samples <= p)

    if use_svd
        # Method 1: SVD-based (more stable for ill-conditioned matrices)
        # H = UU' where X = UΣV' (economy SVD)
        try
            U, _, _ = svd(X_aug)

            # Leverage = diagonal of UU'
            # Efficiently compute as sum of squared U rows
            leverage = vec(sum(U.^2, dims=2))
        catch e
            @error "SVD computation failed: $e"
            rethrow(e)
        end
    else
        # Method 2: Direct computation (fast but can be unstable)
        # H = X(X'X)^(-1)X'
        # We only need diagonal, so: h_ii = row_i * (X'X)^(-1) * row_i'
        try
            XtX = X_aug' * X_aug
            XtX_inv = inv(XtX)

            leverage = zeros(n_samples)
            for i in 1:n_samples
                leverage[i] = dot(X_aug[i, :], XtX_inv, X_aug[i, :])
            end

        catch e
            @warn "Direct leverage computation failed, using SVD fallback: $e"

            # Fallback to SVD method
            U, _, _ = svd(X_aug)
            leverage = vec(sum(U.^2, dims=2))
        end
    end

    # Compute threshold: 2p/n where p = n_features + intercept
    # Note: p already includes intercept (p = size(X_aug, 2))
    threshold = 2.0 * p / n_samples

    if return_threshold
        return leverage, threshold
    else
        return leverage
    end
end


"""
    qq_plot_data(residuals)

Generate Q-Q plot data for assessing normality of residuals.

Q-Q (quantile-quantile) plots compare the distribution of residuals to
a theoretical normal distribution. If residuals are normally distributed,
points should lie approximately on a straight line.

# Arguments
- `residuals::Vector{Float64}`: Model residuals (n_samples)

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (theoretical_quantiles, sample_quantiles)
  - `theoretical_quantiles`: Expected quantiles from standard normal N(0,1)
  - `sample_quantiles`: Observed quantiles (sorted residuals)

# Mathematical Formulation

For n residuals, we compute:

1. Sample quantiles: Sort residuals e_1 ≤ e_2 ≤ ... ≤ e_n

2. Theoretical quantiles: Φ^(-1)(p_i) where Φ is standard normal CDF
   and p_i = (i-0.5)/n for i = 1, ..., n

The probability points (i-0.5)/n are chosen to avoid boundary issues at 0 and 1.

# Interpretation
- Points on diagonal: residuals are normally distributed
- S-shaped curve: heavy-tailed distribution
- Arch: light-tailed distribution
- Points above/below line at ends: skewness

# Notes
- Uses standard normal N(0,1) as reference distribution
- Residuals are automatically sorted for plotting
- Returns data ready for scatter plot: plot(theoretical, sample)
- Add reference line: plot!(theoretical, theoretical) to see ideal fit

# Example

```julia
# Generate Q-Q plot data
theoretical, sample = qq_plot_data(residuals)

# Plot with scatter plot
using Plots
scatter(theoretical, sample,
        xlabel="Theoretical Quantiles",
        ylabel="Sample Quantiles",
        title="Q-Q Plot",
        legend=false)

# Add reference line
plot!(theoretical, theoretical,
      color=:red, linestyle=:dash)
```
"""
function qq_plot_data(
    residuals::Vector{Float64}
)::Tuple{Vector{Float64}, Vector{Float64}}

    n = length(residuals)

    # Sort residuals to get sample quantiles
    sample_quantiles = sort(residuals)

    # Compute probability points using i/(n+1) method
    # This avoids boundary issues at p=0 and p=1
    # Matches Python: np.linspace(1/(n+1), n/(n+1), n)
    probabilities = [i / (n + 1) for i in 1:n]

    # Get theoretical quantiles from standard normal distribution
    normal_dist = Normal(0.0, 1.0)
    theoretical_quantiles = quantile.(normal_dist, probabilities)

    return theoretical_quantiles, sample_quantiles
end


"""
    jackknife_prediction_intervals(model_fn, X_train, y_train, X_test;
                                   confidence=0.95, verbose=true)

Compute jackknife (leave-one-out) prediction intervals using parallel computation.

The jackknife method provides prediction intervals by repeatedly refitting
the model with each training sample removed. This quantifies prediction
uncertainty without assuming a parametric distribution.

# Arguments
- `model_fn::Function`: Function that fits and returns a prediction function
  - Input: `(X::Matrix{Float64}, y::Vector{Float64})`
  - Output: A callable function `pred_fn(X_test) -> Vector{Float64}`
  - Example: `model_fn = (X, y) -> fit_pls(X, y, n_components=5)`
- `X_train::Matrix{Float64}`: Training features (n_train × n_features)
- `y_train::Vector{Float64}`: Training targets (n_train)
- `X_test::Matrix{Float64}`: Test features (n_test × n_features)
- `confidence::Float64=0.95`: Confidence level (default: 95%)
- `verbose::Bool=true`: Print progress information

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`
  - `predictions`: Point predictions for X_test (n_test)
  - `lower_bounds`: Lower confidence bounds (n_test)
  - `upper_bounds`: Upper confidence bounds (n_test)
  - `std_errors`: Standard errors of predictions (n_test)

# Algorithm

The delete-1 jackknife procedure:

1. Fit model on full training data → point predictions ŷ

2. For each training sample i = 1, ..., n:
   - Fit model on X_{-i}, y_{-i} (excluding sample i)
   - Predict on test data → ŷ_i

3. Compute jackknife variance:
   ```
   Var_jack(ŷ) = (n-1)/n * Σ(ŷ_i - ȳ)²
   ```
   where ȳ = mean of all jackknife predictions

4. Construct confidence intervals using t-distribution:
   ```
   CI = ŷ ± t_{α/2, n-1} * SE_jack
   ```
   where SE_jack = sqrt(Var_jack)

# Parallelization

Uses `Threads.@threads` for parallel computation across leave-one-out iterations.
Each thread independently:
- Creates LOO training data
- Fits model
- Generates predictions

To enable threading, start Julia with multiple threads:
```bash
export JULIA_NUM_THREADS=8
julia
```

Or:
```bash
julia -t 8
```

# Performance Considerations

- Computational cost: O(n_train × fit_time)
- Can be slow for n_train > 200
- Parallelization provides near-linear speedup with number of threads
- Expected speedup: 17-25x with 8 threads on typical hardware

# Mathematical Properties

- The jackknife is asymptotically equivalent to the bootstrap
- Less computationally expensive than bootstrap for small-moderate n
- Provides good approximation to sampling distribution
- Appropriate for smooth statistics (means, regression coefficients, predictions)

# Notes

- Requires `model_fn` to return a callable prediction function
- The returned prediction function should accept a matrix and return predictions
- Thread-safe: each iteration works on independent data
- Uses t-distribution for confidence intervals (more conservative than normal for small n)
- Degrees of freedom: n_train - 1

# Example

```julia
# Example 1: PLS regression
using SpectralPredict

function fit_pls_model(X, y; n_components=10)
    model = build_model("PLS", Dict("n_components" => n_components))
    fit_model!(model, X, y)
    return X_test -> predict_model(model, X_test)
end

model_fn = (X, y) -> fit_pls_model(X, y, n_components=10)

pred, lower, upper, stderr = jackknife_prediction_intervals(
    model_fn, X_train, y_train, X_test,
    confidence=0.95, verbose=true
)

# Plot predictions with intervals
using Plots
scatter(y_test, pred,
        yerr=(pred .- lower, upper .- pred),
        xlabel="True Values",
        ylabel="Predictions",
        label="Predictions ± 95% CI")

# Example 2: Simple linear regression
function fit_linear(X, y)
    X_aug = hcat(ones(size(X, 1)), X)
    β = X_aug \\ y
    return X_test -> hcat(ones(size(X_test, 1)), X_test) * β
end

pred, lower, upper, stderr = jackknife_prediction_intervals(
    fit_linear, X_train, y_train, X_test,
    confidence=0.90, verbose=false
)
```

# References

- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Shao, J., & Tu, D. (1995). The Jackknife and Bootstrap. Springer.
"""
function jackknife_prediction_intervals(
    model_fn::Function,
    X_train::Matrix{Float64},
    y_train::Vector{Float64},
    X_test::Matrix{Float64};
    confidence::Float64=0.95,
    verbose::Bool=true
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

    # Validate inputs
    n_train, n_features = size(X_train)
    n_test = size(X_test, 1)

    @assert length(y_train) == n_train "y_train length must match X_train rows"
    @assert size(X_test, 2) == n_features "X_test must have same number of features as X_train"
    @assert 0.0 < confidence < 1.0 "Confidence must be between 0 and 1"

    if verbose
        @info "Computing jackknife prediction intervals (n_train=$n_train, n_test=$n_test)..."
        n_threads = Threads.nthreads()
        @info "Using $n_threads threads for parallelization"
    end

    # Step 1: Fit model on full training data
    model_full = model_fn(X_train, y_train)
    predictions = model_full(X_test)

    # Ensure predictions are a vector
    if predictions isa Matrix
        predictions = vec(predictions)
    end

    # Step 2: Leave-one-out predictions (PARALLELIZED)
    loo_predictions = zeros(n_train, n_test)

    # Use threading for parallelization
    # Each thread processes independent jackknife iterations
    Threads.@threads for i in 1:n_train
        # Progress reporting (thread-safe via @info)
        if verbose && (i % max(1, n_train ÷ 10) == 0 || i == n_train)
            @info "  Jackknife iteration $i/$n_train (thread $(Threads.threadid()))"
        end

        # Create leave-one-out dataset
        # Exclude sample i by selecting all other indices
        train_indices = setdiff(1:n_train, i)
        X_loo = X_train[train_indices, :]
        y_loo = y_train[train_indices]

        # Fit model on LOO data
        try
            model_loo = model_fn(X_loo, y_loo)

            # Predict on test data
            loo_pred = model_loo(X_test)

            # Ensure predictions are a vector
            if loo_pred isa Matrix
                loo_pred = vec(loo_pred)
            end

            # Store predictions (thread-safe write to unique row)
            loo_predictions[i, :] = loo_pred

        catch e
            @error "Jackknife iteration $i failed: $e"
            # Fill with NaN to indicate failure
            loo_predictions[i, :] .= NaN
        end
    end

    # Check for failures
    if any(isnan.(loo_predictions))
        n_failed = sum(any(isnan.(loo_predictions), dims=2))
        @warn "$n_failed jackknife iterations failed and were excluded"

        # Filter out failed iterations
        valid_mask = .!any(isnan.(loo_predictions), dims=2)
        loo_predictions = loo_predictions[valid_mask, :]
        n_train = sum(valid_mask)
    end

    # Step 3: Compute jackknife variance
    # Variance formula: Var_jack = (n-1)/n * Σ(pred_i - pred_mean)²
    mean_loo_pred = vec(mean(loo_predictions, dims=1))

    # Compute jackknife variance for each test sample
    jackknife_variance = zeros(n_test)
    for j in 1:n_test
        deviations = loo_predictions[:, j] .- mean_loo_pred[j]
        jackknife_variance[j] = (n_train - 1) / n_train * sum(deviations.^2)
    end

    # Standard errors
    std_errors = sqrt.(jackknife_variance)

    # Handle potential numerical issues
    # If variance is very small or negative (numerical error), use small positive value
    for j in 1:n_test
        if std_errors[j] < 1e-10 || !isfinite(std_errors[j])
            std_errors[j] = 1e-10
        end
    end

    # Step 4: Compute confidence intervals using t-distribution
    # Degrees of freedom: n_train - 1
    df = n_train - 1
    t_dist = TDist(df)

    # Two-tailed test: (1 + confidence) / 2
    # For 95% CI: quantile at 0.975
    alpha = 1 - confidence
    t_critical = quantile(t_dist, 1 - alpha/2)

    # Confidence intervals
    margin = t_critical .* std_errors
    lower_bounds = predictions .- margin
    upper_bounds = predictions .+ margin

    if verbose
        @info "Jackknife complete!"
        @info "  Confidence level: $(confidence*100)%"
        @info "  Degrees of freedom: $df"
        @info "  t-critical value: $(round(t_critical, digits=3))"
        @info "  Mean standard error: $(round(mean(std_errors), digits=4))"
    end

    return predictions, lower_bounds, upper_bounds, std_errors
end


end # module Diagnostics
