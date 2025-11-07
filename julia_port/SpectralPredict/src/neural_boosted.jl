"""
    Neural Boosted Regression

Gradient boosting with neural network weak learners.
Similar to sklearn's GradientBoostingRegressor but uses MLPs as base learners.

Key differences from Python implementation:
- Uses Flux.jl instead of sklearn.neural_network.MLPRegressor
- Manual early stopping implementation
- More efficient memory management

Author: Spectral Predict Team
Date: November 2025
"""

module NeuralBoosted

using Flux
using LinearAlgebra
using Statistics
using Random
using Optim
using Zygote

export NeuralBoostedRegressor, fit!, predict, get_feature_importances

"""
    NeuralBoostedRegressor

Gradient boosting with MLP weak learners following JMP's Neural Boosted methodology.

# Fields

## Hyperparameters
- `n_estimators::Int`: Maximum number of boosting stages (default: 100)
- `learning_rate::Float64`: Shrinkage parameter (0 < ν ≤ 1, default: 0.1)
- `hidden_layer_size::Int`: Neurons in single hidden layer (default: 3, keep 1-5 for weak learners)
- `activation::String`: Activation function - "tanh" (JMP default), "relu", "sigmoid", "identity" (default: "tanh")
- `alpha::Float64`: L2 regularization (weight decay) parameter (default: 0.0001)
- `max_iter::Int`: Maximum iterations per weak learner training (default: 100)
- `early_stopping::Bool`: Use validation set for early stopping (default: true)
- `validation_fraction::Float64`: Fraction of data for validation (default: 0.15)
- `n_iter_no_change::Int`: Stop if no improvement for this many iterations (default: 10)
- `loss::String`: Loss function - "mse" or "huber" (default: "mse")
- `huber_delta::Float64`: Delta parameter for Huber loss (default: 1.35)
- `random_state::Int`: Random seed for reproducibility (default: 42)
- `verbose::Int`: Verbosity level - 0 (silent), 1 (progress), 2 (detailed) (default: 0)

## Fitted Attributes
- `estimators_::Vector{Any}`: Vector of fitted Flux Chain models (weak learners)
- `train_score_::Vector{Float64}`: Training loss at each boosting iteration
- `validation_score_::Vector{Float64}`: Validation loss at each iteration (if early_stopping=true)
- `n_estimators_::Int`: Actual number of estimators fitted (may be < n_estimators if early stopped)

# Examples

```julia
using SpectralPredict.NeuralBoosted

# Create synthetic data
X = randn(100, 50)  # 100 samples, 50 wavelengths
y = X[:, 10] .+ 2 .* X[:, 20] .+ randn(100) .* 0.1

# Fit model
model = NeuralBoostedRegressor(n_estimators=50, learning_rate=0.1, verbose=1)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)

# Get feature importances
importances = feature_importances(model)
```

# Notes

For best results:
- Keep hidden_layer_size small (3-5 nodes) to maintain weak learner properties
- Use learning_rate in range 0.05-0.2
- Enable early_stopping to prevent overfitting
- Use Huber loss if data contains outliers

# References

Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine."
Annals of Statistics 29(5): 1189-1232.
"""
mutable struct NeuralBoostedRegressor
    # Hyperparameters
    n_estimators::Int
    learning_rate::Float64
    hidden_layer_size::Int
    activation::String
    alpha::Float64
    max_iter::Int
    early_stopping::Bool
    validation_fraction::Float64
    n_iter_no_change::Int
    loss::String
    huber_delta::Float64
    random_state::Int
    verbose::Int

    # Fitted attributes
    estimators_::Vector{Any}
    train_score_::Vector{Float64}
    validation_score_::Vector{Float64}
    n_estimators_::Int

    # Constructor
    function NeuralBoostedRegressor(;
        n_estimators::Int=100,
        learning_rate::Float64=0.1,
        hidden_layer_size::Int=3,
        activation::String="tanh",
        alpha::Float64=0.0001,
        max_iter::Int=100,
        early_stopping::Bool=true,
        validation_fraction::Float64=0.15,
        n_iter_no_change::Int=10,
        loss::String="mse",
        huber_delta::Float64=1.35,
        random_state::Int=42,
        verbose::Int=0
    )
        # Parameter validation
        if !(0 < learning_rate <= 1)
            error("learning_rate must be in (0, 1], got $learning_rate")
        end
        if hidden_layer_size < 1
            error("hidden_layer_size must be >= 1, got $hidden_layer_size")
        end
        if hidden_layer_size > 10
            @warn "hidden_layer_size=$hidden_layer_size is large for weak learner. Consider using 3-5 nodes."
        end
        if !(activation in ["tanh", "relu", "sigmoid", "identity"])
            error("Unknown activation: $activation. Must be 'tanh', 'relu', 'sigmoid', or 'identity'")
        end
        if !(loss in ["mse", "huber"])
            error("Unknown loss: $loss. Must be 'mse' or 'huber'")
        end
        if !(0 < validation_fraction < 1)
            error("validation_fraction must be in (0, 1), got $validation_fraction")
        end

        new(
            n_estimators, learning_rate, hidden_layer_size,
            activation, alpha, max_iter, early_stopping,
            validation_fraction, n_iter_no_change, loss,
            huber_delta, random_state, verbose,
            [], Float64[], Float64[], 0  # Fitted attributes initialized
        )
    end
end

"""
    build_mlp(n_input, hidden_size, activation_str)

Build a simple MLP with one hidden layer for use as weak learner.

# Architecture
- Input layer: n_input neurons
- Hidden layer: hidden_size neurons with specified activation
- Output layer: 1 neuron (regression, linear activation)

# Arguments
- `n_input::Int`: Number of input features
- `hidden_size::Int`: Number of neurons in hidden layer
- `activation_str::String`: Activation function name ("tanh", "relu", "sigmoid", "identity")

# Returns
- Flux Chain model representing the MLP
"""
function build_mlp(
    n_input::Int,
    hidden_size::Int,
    activation_str::String
)
    # Select activation function
    if activation_str == "relu"
        activation = relu
    elseif activation_str == "tanh"
        activation = tanh
    elseif activation_str == "sigmoid"
        activation = σ
    elseif activation_str == "identity"
        activation = identity
    else
        @warn "Unknown activation '$activation_str', using tanh"
        activation = tanh
    end

    # Build network: input -> hidden (with activation) -> output (linear)
    model = Chain(
        Dense(n_input => hidden_size, activation),
        Dense(hidden_size => 1)  # Linear output for regression
    )

    return model
end

"""
    mse_loss(y_true, y_pred)

Compute mean squared error loss.

# Arguments
- `y_true::AbstractVector`: True target values
- `y_pred::AbstractVector`: Predicted values

# Returns
- Mean squared error (scalar)
"""
function mse_loss(y_true::AbstractVector, y_pred::AbstractVector)::Float64
    return mean((y_true .- y_pred).^2)
end

"""
    huber_loss(y_true, y_pred, delta)

Compute Huber loss (robust to outliers).

Huber loss is quadratic for small errors and linear for large errors:
- L(r) = 0.5 * r^2                    if |r| <= delta
- L(r) = delta * (|r| - 0.5 * delta)  if |r| > delta

# Arguments
- `y_true::AbstractVector`: True target values
- `y_pred::AbstractVector`: Predicted values
- `delta::Float64`: Threshold parameter (default: 1.35)

# Returns
- Mean Huber loss (scalar)
"""
function huber_loss(y_true::AbstractVector, y_pred::AbstractVector, delta::Float64)::Float64
    residuals = y_true .- y_pred
    abs_residuals = abs.(residuals)

    # Quadratic part: min(|r|, delta)
    quadratic = min.(abs_residuals, delta)
    # Linear part: |r| - quadratic
    linear = abs_residuals .- quadratic

    # Loss = 0.5 * quadratic^2 + delta * linear
    loss = 0.5 .* quadratic.^2 .+ delta .* linear

    return mean(loss)
end

# New LBFGS-based train_weak_learner! function

"""
    train_weak_learner!(model, X, y, max_iter, alpha, verbose)

Train weak learner using LBFGS optimizer (matches sklearn MLPRegressor with solver='lbfgs').

# Arguments
- `model`: Flux Chain to train
- `X::Matrix{Float64}`: Training data (n_samples × n_features)
- `y::Vector{Float64}`: Target values (n_samples,)
- `max_iter::Int`: Maximum LBFGS iterations (default: 100)
- `alpha::Float64`: L2 regularization parameter (default: 0.0001)
- `verbose::Int`: Verbosity level (0=silent, 1=warnings, 2=debug)

# Returns
- `success::Bool`: true if training succeeded, false otherwise
"""
function train_weak_learner!(
    model,
    X::Matrix{Float64},
    y::Vector{Float64},
    max_iter::Int,
    alpha::Float64,
    verbose::Int
)::Bool
    try
        # Prepare data (Flux expects features × samples)
        X_t = Float64.(X')
        y_t = Float64.(reshape(y, 1, :))

        # Flatten model parameters for LBFGS
        ps, re = Flux.destructure(model)

        # CRITICAL FIX: Convert to Float64 (Flux defaults to Float32)
        ps = Float64.(ps)

        # Define loss function for LBFGS
        function loss_fn(params::Vector{Float64})
            # Reconstruct model from flattened parameters
            m = re(Float32.(params))  # Convert back to Float32 for Flux

            # Forward pass
            pred = m(X_t)

            # MSE loss + L2 regularization
            mse = Flux.mse(pred, y_t)
            l2_penalty = sum(params .^ 2)

            total_loss = mse + alpha * l2_penalty

            # Check for NaN/Inf
            if !isfinite(total_loss)
                return Inf
            end

            return total_loss
        end

        # Define gradient function
        function gradient_fn!(G::Vector{Float64}, params::Vector{Float64})
            # Compute gradient using automatic differentiation
            grad = Zygote.gradient(loss_fn, params)[1]

            if grad === nothing || any(!isfinite, grad)
                fill!(G, 0.0)
                return
            end

            G .= grad
        end

        # Run LBFGS optimization
        result = optimize(
            loss_fn,
            gradient_fn!,
            ps,
            LBFGS(),
            Optim.Options(
                iterations=max_iter,
                f_tol=5e-4,  # Match sklearn tolerance (relaxed from 1e-4)
                g_tol=1e-5,
                show_trace=(verbose >= 2),
                store_trace=false,
                extended_trace=false
            )
        )

        # Check convergence
        if !Optim.converged(result)
            if verbose >= 1
                @warn "LBFGS did not converge" iterations=Optim.iterations(result) f_minimum=Optim.minimum(result)
            end
            # Continue anyway - partial convergence may be acceptable
        end

        # Update model with optimized parameters
        optimal_params = Optim.minimizer(result)

        # Convert Float64 optimal params back to Float32 for Flux model
        # and reconstruct the model in-place using re
        # The reconstructed model IS the updated model
        updated_model = re(Float32.(optimal_params))

        # Copy parameters from updated model back to original model
        # We need to do this layer by layer
        for (orig_layer, new_layer) in zip(model, updated_model)
            if hasfield(typeof(orig_layer), :weight)
                orig_layer.weight .= new_layer.weight
            end
            if hasfield(typeof(orig_layer), :bias)
                orig_layer.bias .= new_layer.bias
            end
        end

        # Validate predictions
        pred = model(X_t)
        if any(!isfinite, pred)
            if verbose >= 1
                @warn "Model produces non-finite predictions after training"
            end
            return false
        end

        if verbose >= 2
            final_loss = Optim.minimum(result)
            println("  ✓ LBFGS converged: $(Optim.iterations(result)) iterations, loss=$(final_loss)")
        end

        return true

    catch e
        if verbose >= 1
            @warn "Training failed with exception" exception=e
        end
        return false
    end
end

"""
    fit!(model::NeuralBoostedRegressor, X, y)

Fit the Neural Boosted Regressor using gradient boosting.

Implements stagewise additive modeling:
1. Split train/validation if early_stopping=true
2. Initialize ensemble prediction F(x) = 0
3. For m = 1 to n_estimators:
   a. Compute residuals: r = y - F(x)
   b. Fit weak learner h_m(x) to residuals
   c. Update: F(x) = F(x) + ν * h_m(x)  (ν = learning_rate)
   d. Check early stopping on validation set
4. Return fitted model

# Arguments
- `model::NeuralBoostedRegressor`: Model to fit (modified in place)
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training targets (n_samples)

# Returns
- `model`: Fitted model (same object, modified in place)
"""
function fit!(
    model::NeuralBoostedRegressor,
    X::Matrix{Float64},
    y::Vector{Float64}
)
    Random.seed!(model.random_state)

    n_samples, n_features = size(X)

    if model.verbose >= 1
        println("Fitting NeuralBoostedRegressor:")
        println("  Samples: $n_samples, Features: $n_features")
        println("  n_estimators: $(model.n_estimators)")
        println("  learning_rate: $(model.learning_rate)")
        println("  hidden_layer_size: $(model.hidden_layer_size)")
        println("  activation: $(model.activation)")
    end

    # Step 0: Validate minimum sample size
    # Need enough samples to train weak learners (at least hidden_layer_size + 2 for network)
    min_required_for_training = model.hidden_layer_size + 2
    if n_samples < min_required_for_training
        error("NeuralBoostedRegressor requires at least $(min_required_for_training) samples " *
              "for hidden_layer_size=$(model.hidden_layer_size), got $n_samples. " *
              "Try reducing hidden_layer_size or use a different model.")
    end

    # Step 1: Train/validation split (if early stopping)
    if model.early_stopping
        # Calculate validation set size
        n_val = Int(floor(n_samples * model.validation_fraction))

        # CRITICAL FIX: Ensure validation set has at least 1 sample
        # and training set has enough samples for weak learner training
        min_train_samples = max(10, min_required_for_training)

        if n_val < 1 || (n_samples - n_val) < min_train_samples
            if model.verbose >= 1
                println("  WARNING: Dataset too small for early stopping validation split.")
                println("           (n_samples=$n_samples requires at least $(min_train_samples + 1) for early stopping)")
                println("           Disabling early stopping for this fit.")
            end
            # Disable early stopping for this fit
            X_train, y_train = X, y
            X_val, y_val = nothing, nothing
            early_stopping_active = false
        else
            # Validation set will be non-empty, proceed with split
            n_train = n_samples - n_val

            # Random shuffle and split
            indices = randperm(n_samples)
            train_idx = indices[1:n_train]
            val_idx = indices[n_train+1:end]

            X_train, y_train = X[train_idx, :], y[train_idx]
            X_val, y_val = X[val_idx, :], y[val_idx]
            early_stopping_active = true

            if model.verbose >= 1
                println("  Training samples: $n_train")
                println("  Validation samples: $n_val")
            end
        end
    else
        X_train, y_train = X, y
        X_val, y_val = nothing, nothing
        early_stopping_active = false
    end

    # Initialize ensemble
    model.estimators_ = []
    model.train_score_ = Float64[]
    model.validation_score_ = Float64[]

    # Initialize predictions to zero (F_0(x) = 0)
    F_train = zeros(size(X_train, 1))
    F_val = early_stopping_active ? zeros(size(X_val, 1)) : nothing

    # Early stopping tracking
    best_val_score = Inf
    no_improvement_count = 0

    # Track weak learner failures for diagnostics
    n_failed_learners = 0

    # Step 2: Boosting loop
    for m in 1:model.n_estimators
        # PHASE 1 FIX: Set unique random seed for each weak learner (diversity)
        Random.seed!(model.random_state + m)

        if model.verbose >= 1
            println("  Stage $m/$(model.n_estimators)...")
        end

        # Compute residuals: what the ensemble got wrong
        residuals = y_train .- F_train

        # Build weak learner (small MLP)
        weak_learner = build_mlp(
            n_features,
            model.hidden_layer_size,
            model.activation
        )

        # Train weak learner on residuals
        try
            train_weak_learner!(
                weak_learner,
                X_train,
                residuals,
                model.max_iter,
                model.alpha,
                model.verbose
            )
        catch e
            n_failed_learners += 1
            if model.verbose >= 1
                @warn "Weak learner $m failed to converge: $e. Skipping."
            end
            continue  # Skip this learner if training fails
        end

        # Get predictions from weak learner
        X_train_t = Float64.(X_train')
        h_m_train = vec(weak_learner(X_train_t))

        # PHASE 1 FIX: Validate predictions before updating ensemble
        if any(isnan.(h_m_train)) || any(isinf.(h_m_train))
            n_failed_learners += 1
            if model.verbose >= 1
                @warn "Weak learner $m produced invalid predictions (NaN/Inf). Skipping."
            end
            continue
        end

        # Update ensemble predictions: F_m(x) = F_{m-1}(x) + ν * h_m(x)
        F_train .+= model.learning_rate .* h_m_train

        # Compute training loss
        if model.loss == "mse"
            train_loss = mse_loss(y_train, F_train)
        elseif model.loss == "huber"
            train_loss = huber_loss(y_train, F_train, model.huber_delta)
        end
        push!(model.train_score_, train_loss)

        # Save estimator
        push!(model.estimators_, weak_learner)

        # Early stopping check (only if validation set exists)
        if early_stopping_active
            X_val_t = Float64.(X_val')
            h_m_val = vec(weak_learner(X_val_t))

            # PHASE 1 FIX: Validate validation predictions
            if any(isnan.(h_m_val)) || any(isinf.(h_m_val))
                if model.verbose >= 1
                    @warn "Weak learner $m produced invalid validation predictions (NaN/Inf)."
                end
                # Remove already-added estimator
                pop!(model.estimators_)
                pop!(model.train_score_)
                n_failed_learners += 1
                continue
            end

            F_val .+= model.learning_rate .* h_m_val

            if model.loss == "mse"
                val_loss = mse_loss(y_val, F_val)
            elseif model.loss == "huber"
                val_loss = huber_loss(y_val, F_val, model.huber_delta)
            end
            push!(model.validation_score_, val_loss)

            # Track best validation score
            if val_loss < best_val_score
                best_val_score = val_loss
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end

            # Stop if no improvement
            if no_improvement_count >= model.n_iter_no_change
                if model.verbose >= 1
                    println("  Early stopping at stage $m (no improvement for $(model.n_iter_no_change) iterations)")
                    println("  Best validation score: $(best_val_score)")
                end
                break
            end

            # Progress reporting
            if model.verbose >= 1 && m % 10 == 0
                println("  Train Loss: $(train_loss), Val Loss: $(val_loss)")
            end
        else
            # Progress reporting (no validation)
            if model.verbose >= 1 && m % 10 == 0
                println("  Train Loss: $(train_loss)")
            end
        end
    end

    model.n_estimators_ = length(model.estimators_)

    # Critical validation: ensure at least one estimator was successfully trained
    if isempty(model.estimators_)
        error("NeuralBoosted training failed: No weak learners were successfully trained. " *
              "All $(n_failed_learners) weak learners failed during training. " *
              "This may be due to:\n" *
              "  1. Dataset too small (n=$(size(X_train, 1)) samples, try early_stopping=false for small datasets)\n" *
              "  2. Numerical instability (check for NaN/Inf values in your data)\n" *
              "  3. Weak learner convergence issues (try increasing max_iter or adjusting learning_rate)\n" *
              "Set verbose=1 to see individual weak learner failures.")
    end

    # Warn if a significant portion of learners failed
    if n_failed_learners > 0 && model.verbose >= 1
        failure_rate = n_failed_learners / (model.n_estimators_ + n_failed_learners)
        if failure_rate > 0.5
            @warn "$(n_failed_learners) out of $(model.n_estimators_ + n_failed_learners) weak learners failed ($(round(failure_rate*100, digits=1))% failure rate). Model may be unstable."
        elseif n_failed_learners > 5
            println("  Note: $(n_failed_learners) weak learners failed but $(model.n_estimators_) succeeded.")
        end
    end

    if model.verbose >= 1
        println("Fitting complete! $(model.n_estimators_) weak learners trained.")
        println("  Final train loss: $(model.train_score_[end])")
        if early_stopping_active && !isempty(model.validation_score_)
            println("  Final val loss: $(model.validation_score_[end])")
        end
    end

    return model
end

"""
    predict(model::NeuralBoostedRegressor, X)

Predict using fitted Neural Boosted Regressor.

Aggregates predictions from all weak learners:
F(x) = Σ_{m=1}^M ν * h_m(x)

where ν is the learning_rate and h_m are the weak learners.

# Arguments
- `model::NeuralBoostedRegressor`: Fitted model
- `X::Matrix{Float64}`: Features (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predictions (n_samples)

# Throws
- Error if model is not fitted yet
"""
function predict(
    model::NeuralBoostedRegressor,
    X::Matrix{Float64}
)::Vector{Float64}

    if isempty(model.estimators_)
        error("Model not fitted yet. Call fit!() first.")
    end

    n_samples = size(X, 1)
    predictions = zeros(n_samples)

    # Transpose for Flux (features × samples)
    X_t = Float64.(X')

    # Aggregate predictions from all weak learners
    for weak_learner in model.estimators_
        h_m = vec(weak_learner(X_t))
        predictions .+= model.learning_rate .* h_m
    end

    return predictions
end

"""
    feature_importances(model::NeuralBoostedRegressor)

Compute feature importances by averaging absolute first-layer weights.

Feature importance for variable i is computed as:
importance_i = (1/N) * Σ_n mean(|W_n[i,:]|)

Where:
- N = number of weak learners
- W_n = first-layer weight matrix of weak learner n (hidden_size × n_features)
- W_n[:,i] = weights from feature i to all hidden nodes

# Arguments
- `model::NeuralBoostedRegressor`: Fitted model

# Returns
- `Vector{Float64}`: Feature importance scores (length = n_features)
  All values are non-negative and sum to 1.0

# Notes
Features with consistently high weights across many learners will have higher
importance scores. This provides insight into which wavelengths/features
contribute most to predictions.

# Throws
- Error if model is not fitted yet
"""
function get_feature_importances(
    model::NeuralBoostedRegressor
)::Vector{Float64}

    if isempty(model.estimators_)
        error("Model not fitted yet. Call fit!() first.")
    end

    # Get number of features from first estimator
    # The first layer is model[1], which is a Dense layer
    first_layer = model.estimators_[1][1]
    # Weight matrix is (hidden_size × n_features) in Flux
    n_features = size(first_layer.weight, 2)

    importances = zeros(n_features)

    # Average absolute weights across all estimators
    for weak_learner in model.estimators_
        # First layer: Dense(n_features => hidden_size)
        # Weight matrix: (hidden_size × n_features)
        first_layer_weights = weak_learner[1].weight

        # Average absolute weights for each input feature (across hidden nodes)
        # Mean over dimension 1 (hidden nodes) -> gives importance per feature
        feature_weights = vec(mean(abs.(first_layer_weights), dims=1))

        importances .+= feature_weights
    end

    # Normalize by number of estimators
    importances ./= length(model.estimators_)

    # Convert to relative importances (sum to 1)
    importances ./= sum(importances)

    return importances
end

end  # module NeuralBoosted
