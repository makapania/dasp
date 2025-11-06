"""
Model definitions and wrapper functions for ML models.

This module provides a unified interface for different machine learning models
used in spectral prediction, including:
- Partial Least Squares (PLS) Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Random Forest
- Multi-Layer Perceptron (MLP)
- Neural Boosted Regression (Gradient Boosting with Neural Network weak learners)

Each model type has specific hyperparameter grids and methods for training,
prediction, and feature importance extraction.
"""

using LinearAlgebra
using Statistics
using MultivariateStats
using GLMNet
using DecisionTree
using Flux
using Random
using StatsBase

# Import NeuralBoosted module (included in parent SpectralPredict module)
using .NeuralBoosted

# ============================================================================
# Model Configuration Generator
# ============================================================================

"""
    get_model_configs(model_name::String)::Vector{Dict{String, Any}}

Generate hyperparameter configuration grids for a specified model type.

# Arguments
- `model_name::String`: Name of the model ("PLS", "Ridge", "Lasso", "ElasticNet",
                        "RandomForest", "MLP", "NeuralBoosted")

# Returns
- `Vector{Dict{String, Any}}`: Array of hyperparameter dictionaries to search over

# Example
```julia
configs = get_model_configs("PLS")
# Returns: [{n_components: 1}, {n_components: 2}, ...]
```

# Supported Models and Hyperparameters

## PLS (Partial Least Squares)
- `n_components`: [1, 2, 3, 5, 7, 10, 15, 20]

## Ridge Regression
- `alpha`: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

## Lasso Regression
- `alpha`: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

## ElasticNet
- `alpha`: [0.001, 0.01, 0.1, 1.0]
- `l1_ratio`: [0.1, 0.5, 0.9]

## RandomForest
- `n_trees`: [50, 100, 200]
- `max_features`: ["sqrt", "log2"]

## MLP (Multi-Layer Perceptron)
- `hidden_layers`: [(50,), (100,), (50, 50)]
- `learning_rate`: [0.001, 0.01]

## NeuralBoosted
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.05, 0.1, 0.2]
- `hidden_layer_size`: [3, 5]
- `activation`: ["tanh", "relu"]
"""
function get_model_configs(model_name::String)::Vector{Dict{String, Any}}
    if model_name == "PLS"
        n_components_list = [1, 2, 3, 5, 7, 10, 15, 20]
        return [Dict("n_components" => nc) for nc in n_components_list]

    elseif model_name == "Ridge"
        alpha_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        return [Dict("alpha" => a) for a in alpha_list]

    elseif model_name == "Lasso"
        alpha_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        return [Dict("alpha" => a) for a in alpha_list]

    elseif model_name == "ElasticNet"
        alpha_list = [0.001, 0.01, 0.1, 1.0]
        l1_ratio_list = [0.1, 0.5, 0.9]
        configs = Dict{String, Any}[]
        for alpha in alpha_list
            for l1_ratio in l1_ratio_list
                push!(configs, Dict("alpha" => alpha, "l1_ratio" => l1_ratio))
            end
        end
        return configs

    elseif model_name == "RandomForest"
        n_trees_list = [50, 100, 200]
        max_features_list = ["sqrt", "log2"]
        configs = Dict{String, Any}[]
        for n_trees in n_trees_list
            for max_features in max_features_list
                push!(configs, Dict("n_trees" => n_trees, "max_features" => max_features))
            end
        end
        return configs

    elseif model_name == "MLP"
        hidden_layers_list = [(50,), (100,), (50, 50)]
        learning_rate_list = [0.001, 0.01]
        configs = Dict{String, Any}[]
        for hidden_layers in hidden_layers_list
            for lr in learning_rate_list
                push!(configs, Dict("hidden_layers" => hidden_layers, "learning_rate" => lr))
            end
        end
        return configs

    elseif model_name == "NeuralBoosted"
        n_estimators_list = [50, 100, 200]
        learning_rate_list = [0.05, 0.1, 0.2]
        hidden_layer_size_list = [3, 5]
        activation_list = ["tanh", "relu"]
        configs = Dict{String, Any}[]
        for n_est in n_estimators_list
            for lr in learning_rate_list
                for hidden_size in hidden_layer_size_list
                    for act in activation_list
                        push!(configs, Dict(
                            "n_estimators" => n_est,
                            "learning_rate" => lr,
                            "hidden_layer_size" => hidden_size,
                            "activation" => act
                        ))
                    end
                end
            end
        end
        return configs

    else
        throw(ArgumentError("Unknown model name: $model_name. Supported models: PLS, Ridge, Lasso, ElasticNet, RandomForest, MLP, NeuralBoosted"))
    end
end


# ============================================================================
# Model Building Functions
# ============================================================================

"""
    PLSModel

Wrapper for Partial Least Squares regression model.

# Fields
- `model`: The fitted PLS model from MultivariateStats
- `n_components::Int`: Number of PLS components
- `mean_X::Vector{Float64}`: Mean of training features (for centering)
- `mean_y::Float64`: Mean of training target (for centering)
"""
mutable struct PLSModel
    model::Union{Nothing, MultivariateStats.CCA}
    n_components::Int
    mean_X::Union{Nothing, Vector{Float64}}
    mean_y::Union{Nothing, Float64}
end

PLSModel(n_components::Int) = PLSModel(nothing, n_components, nothing, nothing)


"""
    RidgeModel

Wrapper for Ridge regression model using GLMNet.

# Fields
- `model`: The fitted GLMNet model
- `alpha::Float64`: Regularization strength
"""
mutable struct RidgeModel
    model::Union{Nothing, Any}
    alpha::Float64
end

RidgeModel(alpha::Float64) = RidgeModel(nothing, alpha)


"""
    LassoModel

Wrapper for Lasso regression model using GLMNet.

# Fields
- `model`: The fitted GLMNet model
- `alpha::Float64`: Regularization strength
"""
mutable struct LassoModel
    model::Union{Nothing, Any}
    alpha::Float64
end

LassoModel(alpha::Float64) = LassoModel(nothing, alpha)


"""
    ElasticNetModel

Wrapper for Elastic Net regression model using GLMNet.

# Fields
- `model`: The fitted GLMNet model
- `alpha::Float64`: Overall regularization strength
- `l1_ratio::Float64`: Mix of L1 vs L2 penalty (0=Ridge, 1=Lasso)
"""
mutable struct ElasticNetModel
    model::Union{Nothing, Any}
    alpha::Float64
    l1_ratio::Float64
end

ElasticNetModel(alpha::Float64, l1_ratio::Float64) = ElasticNetModel(nothing, alpha, l1_ratio)


"""
    RandomForestModel

Wrapper for Random Forest regression model.

# Fields
- `forest`: The fitted ensemble of decision trees
- `n_trees::Int`: Number of trees in the forest
- `max_features::String`: Strategy for selecting features at each split
"""
mutable struct RandomForestModel
    forest::Union{Nothing, Any}
    n_trees::Int
    max_features::String
end

RandomForestModel(n_trees::Int, max_features::String) = RandomForestModel(nothing, n_trees, max_features)


"""
    MLPModel

Wrapper for Multi-Layer Perceptron (neural network) regression model.

# Fields
- `model`: The Flux neural network
- `hidden_layers::Tuple`: Architecture of hidden layers
- `learning_rate::Float64`: Learning rate for training
- `mean_X::Vector{Float64}`: Mean of training features (for normalization)
- `std_X::Vector{Float64}`: Std dev of training features (for normalization)
- `mean_y::Float64`: Mean of training target (for normalization)
- `std_y::Float64`: Std dev of training target (for normalization)
"""
mutable struct MLPModel
    model::Union{Nothing, Any}
    hidden_layers::Tuple
    learning_rate::Float64
    mean_X::Union{Nothing, Vector{Float64}}
    std_X::Union{Nothing, Vector{Float64}}
    mean_y::Union{Nothing, Float64}
    std_y::Union{Nothing, Float64}
end

MLPModel(hidden_layers::Tuple, learning_rate::Float64) =
    MLPModel(nothing, hidden_layers, learning_rate, nothing, nothing, nothing, nothing)


"""
    NeuralBoostedModel

Wrapper for Neural Boosted Regressor (gradient boosting with neural network weak learners).

# Fields
- `model::Union{Nothing, NeuralBoostedRegressor}`: The fitted NeuralBoosted model
- `n_estimators::Int`: Maximum number of boosting stages
- `learning_rate::Float64`: Shrinkage parameter for boosting
- `hidden_layer_size::Int`: Number of neurons in weak learner hidden layer
- `activation::String`: Activation function for weak learners
- `alpha::Float64`: L2 regularization parameter (default: 0.0001)
- `max_iter::Int`: Maximum iterations per weak learner (default: 100)
- `early_stopping::Bool`: Use validation-based early stopping (default: true)
- `verbose::Int`: Verbosity level (default: 0)
"""
mutable struct NeuralBoostedModel
    model::Union{Nothing, NeuralBoostedRegressor}
    n_estimators::Int
    learning_rate::Float64
    hidden_layer_size::Int
    activation::String
    alpha::Float64
    max_iter::Int
    early_stopping::Bool
    verbose::Int
end

NeuralBoostedModel(
    n_estimators::Int,
    learning_rate::Float64,
    hidden_layer_size::Int,
    activation::String;
    alpha::Float64=0.0001,
    max_iter::Int=100,
    early_stopping::Bool=true,
    verbose::Int=0
) = NeuralBoostedModel(nothing, n_estimators, learning_rate, hidden_layer_size, activation, alpha, max_iter, early_stopping, verbose)


"""
    build_model(model_name::String, config::Dict{String, Any}, task_type::String)

Create and return a model instance based on the specified configuration.

# Arguments
- `model_name::String`: Name of the model type
- `config::Dict{String, Any}`: Hyperparameter configuration dictionary
- `task_type::String`: Task type ("regression" or "classification")

# Returns
- Model instance ready for training

# Example
```julia
config = Dict("n_components" => 5)
model = build_model("PLS", config, "regression")
```

# Notes
- Classification support currently limited; most models are regression-focused
- Models are initialized but not trained until fit_model! is called
"""
function build_model(model_name::String, config::Dict{String, Any}, task_type::String)
    if task_type != "regression"
        throw(ArgumentError("Currently only regression tasks are supported"))
    end

    if model_name == "PLS"
        n_components = config["n_components"]
        return PLSModel(n_components)

    elseif model_name == "Ridge"
        alpha = config["alpha"]
        return RidgeModel(alpha)

    elseif model_name == "Lasso"
        alpha = config["alpha"]
        return LassoModel(alpha)

    elseif model_name == "ElasticNet"
        alpha = config["alpha"]
        l1_ratio = config["l1_ratio"]
        return ElasticNetModel(alpha, l1_ratio)

    elseif model_name == "RandomForest"
        n_trees = config["n_trees"]
        max_features = config["max_features"]
        return RandomForestModel(n_trees, max_features)

    elseif model_name == "MLP"
        hidden_layers = config["hidden_layers"]
        learning_rate = config["learning_rate"]
        return MLPModel(hidden_layers, learning_rate)

    elseif model_name == "NeuralBoosted"
        n_estimators = config["n_estimators"]
        learning_rate = config["learning_rate"]
        hidden_layer_size = config["hidden_layer_size"]
        activation = config["activation"]
        # Optional parameters with defaults
        alpha = get(config, "alpha", 0.0001)
        max_iter = get(config, "max_iter", 100)
        # Changed default to false because early_stopping=true causes issues with small datasets
        # (after CV split, training sets can be very small, e.g. n=17 samples)
        early_stopping = get(config, "early_stopping", false)
        verbose = get(config, "verbose", 0)
        return NeuralBoostedModel(n_estimators, learning_rate, hidden_layer_size, activation,
                                  alpha=alpha, max_iter=max_iter, early_stopping=early_stopping, verbose=verbose)

    else
        throw(ArgumentError("Unknown model name: $model_name"))
    end
end


# ============================================================================
# Model Fitting Functions
# ============================================================================

"""
    fit_model!(model::PLSModel, X::Matrix{Float64}, y::Vector{Float64})

Train a PLS regression model.

# Arguments
- `model::PLSModel`: PLS model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Data is centered before fitting (mean removed)
- Uses MultivariateStats.llsq for PLS computation
"""
function fit_model!(model::PLSModel, X::Matrix{Float64}, y::Vector{Float64})
    # Center the data
    model.mean_X = vec(mean(X, dims=1))
    model.mean_y = mean(y)

    X_centered = X .- model.mean_X'
    y_centered = y .- model.mean_y

    # Fit PLS using CCA (Canonical Correlation Analysis) as equivalent to PLS
    # Note: MultivariateStats doesn't have direct PLS, but CCA with proper setup works
    n_features = size(X, 2)

    # Convert y to matrix for CCA
    Y_mat = reshape(y_centered, :, 1)

    # For CCA, max components is limited by min dimension of both X and Y
    # Since Y is univariate (1 feature), we can only extract 1 component
    n_components = min(model.n_components, n_features, size(X, 1), size(Y_mat, 2))

    # Fit using the simpls algorithm equivalent
    model.model = fit(CCA, X_centered', Y_mat'; outdim=n_components)

    return model
end


"""
    fit_model!(model::RidgeModel, X::Matrix{Float64}, y::Vector{Float64})

Train a Ridge regression model.

# Arguments
- `model::RidgeModel`: Ridge model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Uses GLMNet with alpha=0 (pure L2 penalty)
- Lambda value is scaled by alpha parameter
"""
function fit_model!(model::RidgeModel, X::Matrix{Float64}, y::Vector{Float64})
    # GLMNet uses alpha for elastic net mixing (0=ridge, 1=lasso)
    # Our alpha is the regularization strength (lambda in GLMNet terms)
    # Set alpha=0 for pure ridge, and use our alpha as lambda
    model.model = glmnet(X, y, alpha=0.0, lambda=[model.alpha])
    return model
end


"""
    fit_model!(model::LassoModel, X::Matrix{Float64}, y::Vector{Float64})

Train a Lasso regression model.

# Arguments
- `model::LassoModel`: Lasso model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Uses GLMNet with alpha=1 (pure L1 penalty)
- Lambda value is scaled by alpha parameter
"""
function fit_model!(model::LassoModel, X::Matrix{Float64}, y::Vector{Float64})
    # Set alpha=1 for pure lasso, and use our alpha as lambda
    model.model = glmnet(X, y, alpha=1.0, lambda=[model.alpha])
    return model
end


"""
    fit_model!(model::ElasticNetModel, X::Matrix{Float64}, y::Vector{Float64})

Train an Elastic Net regression model.

# Arguments
- `model::ElasticNetModel`: Elastic Net model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Uses GLMNet with specified l1_ratio for L1/L2 mixing
- Lambda value is scaled by alpha parameter
"""
function fit_model!(model::ElasticNetModel, X::Matrix{Float64}, y::Vector{Float64})
    # Use l1_ratio for elastic net mixing
    model.model = glmnet(X, y, alpha=model.l1_ratio, lambda=[model.alpha])
    return model
end


"""
    fit_model!(model::RandomForestModel, X::Matrix{Float64}, y::Vector{Float64})

Train a Random Forest regression model.

# Arguments
- `model::RandomForestModel`: Random Forest model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Uses DecisionTree.jl's RandomForestRegressor
- max_features determines number of features considered at each split
"""
function fit_model!(model::RandomForestModel, X::Matrix{Float64}, y::Vector{Float64})
    n_features = size(X, 2)

    # Determine number of features to consider at each split
    if model.max_features == "sqrt"
        n_subfeatures = Int(floor(sqrt(n_features)))
    elseif model.max_features == "log2"
        n_subfeatures = Int(floor(log2(n_features)))
    else
        n_subfeatures = n_features
    end

    # Build random forest
    # DecisionTree.jl uses positional arguments:
    # build_forest(labels, features, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase; rng, impurity_importance)
    model.forest = build_forest(y, X,
                                n_subfeatures,           # positional arg 1
                                model.n_trees,           # positional arg 2
                                0.7,                     # partial_sampling (positional arg 3)
                                -1,                      # max_depth, -1 = no limit (positional arg 4)
                                5,                       # min_samples_leaf (positional arg 5)
                                2,                       # min_samples_split (positional arg 6)
                                0.0;                     # min_purity_increase (positional arg 7)
                                rng=Random.MersenneTwister(42))
    return model
end


"""
    fit_model!(model::MLPModel, X::Matrix{Float64}, y::Vector{Float64})

Train a Multi-Layer Perceptron regression model.

# Arguments
- `model::MLPModel`: MLP model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Data is standardized (z-score normalization) before training
- Uses Flux.jl for neural network implementation
- Trained with Adam optimizer for 1000 epochs (with early stopping)
- Uses 20% of data for validation-based early stopping
"""
function fit_model!(model::MLPModel, X::Matrix{Float64}, y::Vector{Float64})
    # Normalize data
    model.mean_X = vec(mean(X, dims=1))
    model.std_X = vec(std(X, dims=1))
    model.std_X[model.std_X .== 0.0] .= 1.0  # Avoid division by zero

    model.mean_y = mean(y)
    model.std_y = std(y)
    if model.std_y == 0.0
        model.std_y = 1.0
    end

    X_norm = (X .- model.mean_X') ./ model.std_X'
    y_norm = (y .- model.mean_y) ./ model.std_y

    # Build neural network architecture
    n_features = size(X, 2)
    layers = []

    # Input to first hidden layer
    push!(layers, Dense(n_features, model.hidden_layers[1], relu))

    # Additional hidden layers
    for i in 2:length(model.hidden_layers)
        push!(layers, Dense(model.hidden_layers[i-1], model.hidden_layers[i], relu))
    end

    # Output layer (linear activation for regression)
    push!(layers, Dense(model.hidden_layers[end], 1))

    # Create chain
    model.model = Chain(layers...)

    # Prepare data
    X_train = X_norm'  # Flux expects features × samples
    y_train = reshape(y_norm, 1, :)  # 1 × samples for single output

    # Training setup (NEW FLUX API)
    opt = Adam(model.learning_rate)
    opt_state = Flux.setup(opt, model.model)

    # Simple early stopping: track validation loss
    n_samples = size(X, 1)
    n_val = Int(floor(0.2 * n_samples))
    n_train = n_samples - n_val

    # Split into train/validation
    perm = randperm(Random.MersenneTwister(42), n_samples)
    train_idx = perm[1:n_train]
    val_idx = perm[n_train+1:end]

    X_train_split = X_train[:, train_idx]
    y_train_split = y_train[:, train_idx]
    X_val = X_train[:, val_idx]
    y_val = y_train[:, val_idx]

    # Training loop with early stopping
    best_val_loss = Inf
    patience = 20
    patience_counter = 0
    max_epochs = 1000

    for epoch in 1:max_epochs
        # Training (NEW FLUX API)
        grads = Flux.gradient(model.model) do m
            ŷ = m(X_train_split)
            Flux.mse(ŷ, y_train_split)
        end

        Flux.update!(opt_state, model.model, grads[1])

        # Validation
        if epoch % 10 == 0
            val_pred = model.model(X_val)
            val_loss = Flux.mse(val_pred, y_val)

            if val_loss < best_val_loss
                best_val_loss = val_loss
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    # Early stopping triggered
                    break
                end
            end
        end
    end

    return model
end


"""
    fit_model!(model::NeuralBoostedModel, X::Matrix{Float64}, y::Vector{Float64})

Train a Neural Boosted Regressor model.

# Arguments
- `model::NeuralBoostedModel`: Neural Boosted model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Notes
- Uses gradient boosting with neural network weak learners
- Implements early stopping on validation set by default
- Weak learners are small MLPs (typically 3-5 hidden neurons)
- Learning rate shrinks each weak learner contribution
"""
function fit_model!(model::NeuralBoostedModel, X::Matrix{Float64}, y::Vector{Float64})
    # Create NeuralBoostedRegressor with configured hyperparameters
    model.model = NeuralBoostedRegressor(
        n_estimators=model.n_estimators,
        learning_rate=model.learning_rate,
        hidden_layer_size=model.hidden_layer_size,
        activation=model.activation,
        alpha=model.alpha,
        max_iter=model.max_iter,
        early_stopping=model.early_stopping,
        verbose=model.verbose
    )

    # Fit the model (uses NeuralBoosted.fit!)
    NeuralBoosted.fit!(model.model, X, y)

    return model
end


"""
    fit_model!(model, X::Matrix{Float64}, y::Vector{Float64})

Generic fit function that dispatches to specific model implementations.

# Arguments
- `model`: Model instance to train
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)
"""
function fit_model!(model, X::Matrix{Float64}, y::Vector{Float64})
    throw(ArgumentError("fit_model! not implemented for model type $(typeof(model))"))
end


# ============================================================================
# Prediction Functions
# ============================================================================

"""
    predict_model(model::PLSModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted PLS model.

# Arguments
- `model::PLSModel`: Fitted PLS model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::PLSModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    # Center using training means
    X_centered = X .- model.mean_X'

    # Project X into latent space using xproj
    T = X_centered * model.model.xproj  # Get scores (n_samples × n_components)

    # Project from latent space to y using yproj
    # For PLS regression: y_pred = T * yproj'
    y_pred = vec(T * model.model.yproj')

    # Add back mean
    return y_pred .+ model.mean_y
end


"""
    predict_model(model::RidgeModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted Ridge model.

# Arguments
- `model::RidgeModel`: Fitted Ridge model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::RidgeModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    # GLMNet predict
    return vec(GLMNet.predict(model.model, X))
end


"""
    predict_model(model::LassoModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted Lasso model.

# Arguments
- `model::LassoModel`: Fitted Lasso model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::LassoModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    return vec(GLMNet.predict(model.model, X))
end


"""
    predict_model(model::ElasticNetModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted Elastic Net model.

# Arguments
- `model::ElasticNetModel`: Fitted Elastic Net model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::ElasticNetModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    return vec(GLMNet.predict(model.model, X))
end


"""
    predict_model(model::RandomForestModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted Random Forest model.

# Arguments
- `model::RandomForestModel`: Fitted Random Forest model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::RandomForestModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.forest)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    # Apply forest to each sample
    n_samples = size(X, 1)
    predictions = zeros(n_samples)

    for i in 1:n_samples
        predictions[i] = apply_forest(model.forest, X[i, :])
    end

    return predictions
end


"""
    predict_model(model::MLPModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted MLP model.

# Arguments
- `model::MLPModel`: Fitted MLP model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::MLPModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    # Normalize using training statistics
    X_norm = (X .- model.mean_X') ./ model.std_X'

    # Predict
    y_norm = model.model(X_norm')  # Flux expects features × samples

    # Denormalize
    y_pred = vec(y_norm) .* model.std_y .+ model.mean_y

    return y_pred
end


"""
    predict_model(model::NeuralBoostedModel, X::Matrix{Float64})::Vector{Float64}

Generate predictions using a fitted Neural Boosted model.

# Arguments
- `model::NeuralBoostedModel`: Fitted Neural Boosted model
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model::NeuralBoostedModel, X::Matrix{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    # Use NeuralBoosted.predict
    return NeuralBoosted.predict(model.model, X)
end


"""
    predict_model(model, X::Matrix{Float64})::Vector{Float64}

Generic prediction function that dispatches to specific model implementations.

# Arguments
- `model`: Fitted model instance
- `X::Matrix{Float64}`: Features for prediction (n_samples × n_features)

# Returns
- `Vector{Float64}`: Predicted values (n_samples,)
"""
function predict_model(model, X::Matrix{Float64})::Vector{Float64}
    throw(ArgumentError("predict_model not implemented for model type $(typeof(model))"))
end


# ============================================================================
# Feature Importance Functions
# ============================================================================

"""
    compute_vip_scores(model::PLSModel, X::Matrix{Float64}, y::Vector{Float64})::Vector{Float64}

Compute Variable Importance in Projection (VIP) scores for a fitted PLS model.

VIP scores measure the importance of each variable in the PLS model, taking into
account the amount of explained Y-variance in each dimension.

# Arguments
- `model::PLSModel`: Fitted PLS model
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Returns
- `Vector{Float64}`: VIP score for each feature (n_features,)

# Notes
- VIP scores > 1 are generally considered important
- Formula: VIP_j = sqrt(p * Σ(w²_jk * SSY_k) / Σ(SSY_k))
  where p is number of features, w are weights, SSY is explained variance
"""
function compute_vip_scores(model::PLSModel, X::Matrix{Float64}, y::Vector{Float64})::Vector{Float64}
    if isnothing(model.model)
        throw(ArgumentError("Model has not been fitted yet"))
    end

    # Get PLS weights and scores
    # Note: MultivariateStats CCA stores these differently than sklearn
    # We need to extract the projection matrices

    # Center the data
    X_centered = X .- model.mean_X'
    y_centered = y .- model.mean_y

    # Get the projection/weight matrix
    W = MultivariateStats.projection(model.model, :x)  # This is the X-weights matrix

    # Get scores (transformed X)
    T = MultivariateStats.predict(model.model, X_centered', :x)  # n_components × n_samples
    T = T'  # Convert to n_samples × n_components

    # Compute explained variance by each component
    # SSY: sum of squares of y explained by each component
    n_components = size(T, 2)
    ssy_comp = zeros(n_components)

    for k in 1:n_components
        # Variance of y explained by component k
        ssy_comp[k] = sum(T[:, k].^2) * var(y_centered)
    end

    ssy_total = sum(ssy_comp)

    # VIP calculation
    n_features = size(X, 2)
    vip_scores = zeros(n_features)

    for i in 1:n_features
        # Sum of weighted squared loadings across components
        weight_sum = 0.0
        for k in 1:n_components
            weight_sum += (W[i, k]^2) * ssy_comp[k]
        end
        vip_scores[i] = sqrt(n_features * weight_sum / ssy_total)
    end

    return vip_scores
end


"""
    get_feature_importances(
        model,
        model_name::String,
        X::Matrix{Float64},
        y::Vector{Float64}
    )::Vector{Float64}

Extract feature importance scores from a fitted model.

Different models use different methods for computing feature importance:
- **PLS**: VIP (Variable Importance in Projection) scores
- **Ridge/Lasso/ElasticNet**: Absolute values of coefficients
- **RandomForest**: Built-in feature importances (mean decrease in impurity)
- **MLP**: Mean absolute weights from first layer

# Arguments
- `model`: Fitted model instance
- `model_name::String`: Name of the model type
- `X::Matrix{Float64}`: Training features (n_samples × n_features)
- `y::Vector{Float64}`: Training target values (n_samples,)

# Returns
- `Vector{Float64}`: Feature importance scores (n_features,) - higher is more important

# Example
```julia
importances = get_feature_importances(model, "PLS", X_train, y_train)
top_features = sortperm(importances, rev=true)[1:10]  # Top 10 features
```
"""
function get_feature_importances(
    model,
    model_name::String,
    X::Matrix{Float64},
    y::Vector{Float64}
)::Vector{Float64}

    if model_name == "PLS"
        return compute_vip_scores(model, X, y)

    elseif model_name in ["Ridge", "Lasso", "ElasticNet"]
        # Use absolute values of coefficients
        if isnothing(model.model)
            throw(ArgumentError("Model has not been fitted yet"))
        end

        # GLMNet stores coefficients in a matrix (n_features × n_lambda)
        # We used a single lambda, so take the first column
        coefs = model.model.betas[:, 1]
        return abs.(coefs)

    elseif model_name == "RandomForest"
        # DecisionTree.jl doesn't have built-in feature importances
        # We need to compute them manually by aggregating across trees
        if isnothing(model.forest)
            throw(ArgumentError("Model has not been fitted yet"))
        end

        n_features = size(X, 2)
        importances = zeros(n_features)

        # For each tree, compute feature importance based on split criteria improvement
        # This is a simplified version - computing mean decrease in variance
        for tree in model.forest.trees
            tree_importances = compute_tree_importances(tree, n_features)
            importances .+= tree_importances
        end

        # Normalize by number of trees
        importances ./= length(model.forest.trees)

        # Normalize to sum to 1
        if sum(importances) > 0
            importances ./= sum(importances)
        end

        return importances

    elseif model_name == "MLP"
        # Use mean absolute weights from first layer
        if isnothing(model.model)
            throw(ArgumentError("Model has not been fitted yet"))
        end

        # Get first layer weights
        first_layer = model.model.layers[1]
        weights = first_layer.weight  # output_dim × input_dim

        # Mean absolute weight for each input feature
        importances = vec(mean(abs.(weights), dims=1))

        return importances

    elseif model_name == "NeuralBoosted"
        # Use NeuralBoosted.get_feature_importances
        if isnothing(model.model)
            throw(ArgumentError("Model has not been fitted yet"))
        end

        return NeuralBoosted.get_feature_importances(model.model)

    else
        throw(ArgumentError("Feature importance not implemented for model: $model_name"))
    end
end


"""
    compute_tree_importances(tree, n_features::Int)::Vector{Float64}

Compute feature importances for a single decision tree.

# Arguments
- `tree`: Decision tree node
- `n_features::Int`: Total number of features

# Returns
- `Vector{Float64}`: Feature importance scores for this tree
"""
function compute_tree_importances(tree, n_features::Int)::Vector{Float64}
    importances = zeros(n_features)

    # Recursively traverse tree and accumulate importance
    function traverse_node(node, importances)
        if isa(node, DecisionTree.Leaf)
            return
        end

        # This is a split node
        # Importance is based on the weighted impurity decrease
        # For regression: use variance reduction

        # Feature used for this split
        feature_idx = node.featid

        # Accumulate importance (simplified: just count splits)
        # More sophisticated: weight by number of samples and impurity decrease
        importances[feature_idx] += 1.0

        # Recurse to children
        if !isa(node.left, Nothing)
            traverse_node(node.left, importances)
        end
        if !isa(node.right, Nothing)
            traverse_node(node.right, importances)
        end
    end

    traverse_node(tree, importances)

    return importances
end


# ============================================================================
# Exports
# ============================================================================

export get_model_configs
export build_model
export fit_model!
export predict_model
export get_feature_importances
export compute_vip_scores

# Export model types
export PLSModel
export RidgeModel
export LassoModel
export ElasticNetModel
export RandomForestModel
export MLPModel
export NeuralBoostedModel
