# ⚠️ SUPERSEDED - See JULIA_PORT_HANDOFF.md

**This document is outdated. Please read:**
# 👉 **JULIA_PORT_HANDOFF.md** 👈

The new document includes:
- All bug fixes validated and working
- Phase 1 (core) vs Phase 2 (GUI) strategy
- 6-8 week Phase 1 roadmap
- Complete algorithm details with today's fixes

---

# Julia Port Implementation Guide (OLD)

**Goal**: 30-100x speedup through full Julia rewrite
**Timeline**: 4-8 weeks
**Risk**: Medium-High (new codebase in different language)
**Recommendation**: Only pursue if 2-10x speedups from other methods aren't sufficient

---

## Executive Summary

### Why Julia?

Julia is uniquely suited for this project because:
- ✅ **JIT-compiled to native code** (LLVM) - comparable to C/C++ performance
- ✅ **90% as fast as C**, 10-100x faster than Python
- ✅ **Excellent scientific computing ecosystem** - mature ML and signal processing libraries
- ✅ **Python-like syntax** - low learning curve for Python developers
- ✅ **Can call Python libraries** if needed (PyCall.jl)
- ✅ **Multiple dispatch** - elegant code organization
- ✅ **Built-in parallelization** - easy to leverage multiple cores
- ✅ **GPU support** - can accelerate further if needed

### Expected Performance Gains

| Component | Python (baseline) | Julia (expected) | Speedup |
|-----------|------------------|------------------|---------|
| Preprocessing | 3.8s | 0.1-0.2s | 20-40x |
| PLS model | 5-10s | 0.3-0.5s | 15-30x |
| Random Forest | 20-30s | 1-2s | 15-20x |
| Neural networks | 30-60s | 2-5s | 10-20x |
| **Total pipeline** | **164s** | **5-10s** | **15-30x** |

With further optimization and GPU acceleration: **50-100x** total speedup possible.

---

## Julia Ecosystem - Library Mapping

### Core Scientific Computing

| Python | Julia Equivalent | Maturity | Notes |
|--------|-----------------|----------|-------|
| numpy | Native arrays | ⭐⭐⭐⭐⭐ | Built-in, BLAS/LAPACK support |
| scipy | Multiple packages | ⭐⭐⭐⭐ | Distributed across ecosystem |
| pandas | DataFrames.jl | ⭐⭐⭐⭐⭐ | Feature-complete, fast |

### Machine Learning

| Python | Julia | Maturity | Performance vs Python |
|--------|-------|----------|----------------------|
| scikit-learn (PLS) | PartialLeastSquares.jl | ⭐⭐⭐⭐ | 20-50x faster |
| scikit-learn (RF) | DecisionTree.jl | ⭐⭐⭐⭐ | 10-20x faster |
| scikit-learn (MLP) | Flux.jl | ⭐⭐⭐⭐⭐ | 10-30x faster, GPU support |
| scikit-learn (general) | MLJ.jl | ⭐⭐⭐⭐ | Unified ML framework |
| LightGBM | LightGBM.jl | ⭐⭐⭐ | Wrapper, native speed |
| XGBoost | XGBoost.jl | ⭐⭐⭐⭐ | Wrapper, native speed |

### Signal Processing

| Python | Julia | Maturity | Notes |
|--------|-------|----------|-------|
| scipy.signal | DSP.jl | ⭐⭐⭐⭐ | Has Savitzky-Golay filter |
| scipy.signal.butter | DSP.jl | ⭐⭐⭐⭐ | Butterworth filters |

### Visualization

| Python | Julia | Maturity | Notes |
|--------|-------|----------|-------|
| matplotlib | Plots.jl | ⭐⭐⭐⭐ | Multiple backends (GR, PyPlot, Plotly) |
| seaborn | StatsPlots.jl | ⭐⭐⭐⭐ | Statistical plotting |

### File I/O

| Python | Julia | Maturity | Notes |
|--------|-------|----------|-------|
| csv | CSV.jl | ⭐⭐⭐⭐⭐ | Very fast CSV parser |
| json | JSON.jl | ⭐⭐⭐⭐⭐ | Full JSON support |
| Custom (ASD) | Custom | N/A | Need to port ASD reader |

---

## Implementation Timeline

### Week 1-2: Foundation & Setup

**Goal**: Environment setup, basic data structures, I/O

#### Tasks

1. **Install Julia and setup project** (1-2 hours)
   ```bash
   # Install Julia
   brew install julia  # macOS
   # or download from https://julialang.org/downloads/

   # Create project
   mkdir julia_port
   cd julia_port
   julia
   ```

   ```julia
   # In Julia REPL
   ]  # Enter package mode
   generate SpectralPredict
   activate .
   ```

2. **Add dependencies** (30 minutes)
   ```julia
   # In Julia package mode
   add CSV
   add DataFrames
   add Statistics
   add LinearAlgebra
   add DSP
   add MLJ
   add Flux
   add DecisionTree
   add PartialLeastSquares
   add LightGBM
   add Plots
   ```

3. **Port ASD file reader** (4-6 hours)
   ```julia
   # File: src/io.jl
   module IO

   using DataFrames

   """
   Read ASD spectral file.

   ASD files have binary format with header and spectral data.
   """
   function read_asd(filepath::String)
       # Open file in binary mode
       io = open(filepath, "r")

       # Read header (first 484 bytes)
       header = read(io, 484)

       # Parse wavelength range from header
       # (specific to ASD format - port from Python)
       n_wavelengths = parse_header_nwav(header)
       start_wavelength = parse_header_start(header)
       end_wavelength = parse_header_end(header)

       # Read spectral data (Float32)
       spectra = Vector{Float32}(undef, n_wavelengths)
       read!(io, spectra)

       close(io)

       # Generate wavelength array
       wavelengths = range(start_wavelength, end_wavelength, length=n_wavelengths)

       return wavelengths, spectra
   end

   """
   Load all ASD files from directory.
   """
   function load_asd_directory(dirpath::String)
       files = filter(f -> endswith(f, ".asd"), readdir(dirpath, join=true))

       # Parallel loading
       results = @async map(read_asd, files)

       return fetch(results)
   end

   """
   Read CSV reference data.
   """
   function read_reference(filepath::String)
       return CSV.read(filepath, DataFrame)
   end

   end  # module
   ```

4. **Create basic data structures** (2-3 hours)
   ```julia
   # File: src/types.jl
   module Types

   using DataFrames

   """
   Spectral dataset container.
   """
   struct SpectralData
       X::Matrix{Float64}              # (n_samples, n_wavelengths)
       y::Vector{Float64}              # (n_samples,)
       wavelengths::Vector{Float64}    # (n_wavelengths,)
       sample_ids::Vector{String}      # (n_samples,)
       metadata::DataFrame
   end

   """
   Model configuration.
   """
   struct ModelConfig
       name::String
       params::Dict{Symbol, Any}
   end

   """
   Search results.
   """
   struct SearchResult
       model_name::String
       preprocessing::String
       n_vars::Int
       rmse::Float64
       r2::Float64
       cv_scores::Vector{Float64}
       feature_importances::Vector{Float64}
       best_params::Dict{Symbol, Any}
   end

   end  # module
   ```

**Week 1-2 Deliverables**:
- ✅ Julia environment setup
- ✅ All dependencies installed
- ✅ ASD file reader working
- ✅ CSV reference reader working
- ✅ Basic data structures defined

---

### Week 3-4: Preprocessing & Feature Engineering

**Goal**: Port all preprocessing functions (SNV, derivatives, regions)

#### Tasks

1. **SNV Transform** (1-2 hours)
   ```julia
   # File: src/preprocessing.jl
   module Preprocessing

   using Statistics

   """
   Standard Normal Variate transformation.

   This is JIT-compiled automatically by Julia.
   """
   function snv_transform(X::Matrix{Float64})
       n_samples, n_wavelengths = size(X)
       X_snv = similar(X)

       # Loop is fast in Julia (JIT compiled)
       @inbounds for i in 1:n_samples
           μ = mean(view(X, i, :))
           σ = std(view(X, i, :))

           if σ > 1e-10
               X_snv[i, :] = (X[i, :] .- μ) ./ σ
           else
               X_snv[i, :] = X[i, :] .- μ
           end
       end

       return X_snv
   end

   # Benchmark: Should be 20-50x faster than Python
   ```

2. **Savitzky-Golay Derivatives** (2-3 hours)
   ```julia
   using DSP

   """
   Apply Savitzky-Golay derivative filter.
   """
   function savitzky_golay_derivative(X::Matrix{Float64};
                                      window_length::Int=5,
                                      polyorder::Int=2,
                                      deriv::Int=1)
       n_samples, n_wavelengths = size(X)
       X_deriv = similar(X)

       # Pre-compute SG filter coefficients
       coeffs = sg_coefficients(window_length, polyorder, deriv)

       # Apply filter to each spectrum
       @inbounds for i in 1:n_samples
           X_deriv[i, :] = conv(X[i, :], coeffs)
       end

       return X_deriv
   end

   """
   Compute Savitzky-Golay filter coefficients.
   """
   function sg_coefficients(window_length::Int, polyorder::Int, deriv::Int)
       # Use DSP.jl savitzky_golay function
       # or implement manually using Vandermonde matrix
       half_window = (window_length - 1) ÷ 2

       # Create Vandermonde matrix
       x = -half_window:half_window
       V = [x[i]^j for i in 1:length(x), j in 0:polyorder]

       # Compute coefficients
       # (specific math - port from scipy implementation)
       y = zeros(polyorder + 1)
       y[deriv + 1] = factorial(deriv)

       coeffs = V \ y

       return coeffs
   end
   ```

3. **Region Analysis** (2-3 hours)
   ```julia
   """
   Define spectral regions based on wavelength ranges.
   """
   function define_regions(wavelengths::Vector{Float64},
                          region_defs::Vector{Tuple{Float64, Float64}})
       regions = []
       for (start_wl, end_wl) in region_defs
           idx = findall(start_wl .<= wavelengths .<= end_wl)
           push!(regions, idx)
       end
       return regions
   end

   """
   Compute region importances (correlation with target).
   """
   function compute_region_importances(X::Matrix{Float64},
                                       y::Vector{Float64},
                                       regions::Vector{Vector{Int}})
       importances = zeros(length(regions))

       for (i, region_idx) in enumerate(regions)
           # Mean absolute correlation in region
           corrs = [cor(X[:, j], y) for j in region_idx]
           importances[i] = mean(abs.(corrs))
       end

       return importances
   end
   ```

4. **Preprocessing Pipeline** (1-2 hours)
   ```julia
   """
   Apply preprocessing pipeline.
   """
   function preprocess(X::Matrix{Float64}, method::Symbol)
       if method == :snv
           return snv_transform(X)
       elseif method == :derivative1
           return savitzky_golay_derivative(X, deriv=1)
       elseif method == :derivative2
           return savitzky_golay_derivative(X, deriv=2)
       elseif method == :snv_derivative1
           X_snv = snv_transform(X)
           return savitzky_golay_derivative(X_snv, deriv=1)
       elseif method == :snv_derivative2
           X_snv = snv_transform(X)
           return savitzky_golay_derivative(X_snv, deriv=2)
       else
           error("Unknown preprocessing method: $method")
       end
   end
   ```

**Week 3-4 Deliverables**:
- ✅ SNV transform (20-50x faster than Python)
- ✅ Savitzky-Golay derivatives (10-30x faster)
- ✅ Region analysis
- ✅ Preprocessing pipeline
- ✅ Unit tests for all preprocessing functions

---

### Week 5-6: Machine Learning Models

**Goal**: Port all model types (PLS, Random Forest, MLP, Neural Boosted)

#### Tasks

1. **PLS Regression** (3-4 hours)
   ```julia
   # File: src/models/pls.jl
   using PartialLeastSquares
   using Statistics

   """
   PLS Regression wrapper.
   """
   struct PLSModel
       model::PLSRegression
       n_components::Int
   end

   function fit_pls(X::Matrix{Float64}, y::Vector{Float64}, n_components::Int)
       model = PLSRegression(n_components=n_components)
       fit!(model, X, y)
       return PLSModel(model, n_components)
   end

   function predict(pls::PLSModel, X::Matrix{Float64})
       return predict(pls.model, X)
   end

   """
   Compute VIP scores for PLS model.
   """
   function compute_vip(pls::PLSModel, X::Matrix{Float64}, y::Vector{Float64})
       W = pls.model.weights  # (n_features, n_components)
       T = pls.model.scores   # (n_samples, n_components)

       # Explained variance per component
       SSY = [var(T[:, j] * pls.model.loadings[j]) for j in 1:pls.n_components]

       # VIP scores
       n_features = size(X, 2)
       vip = zeros(n_features)

       for i in 1:n_features
           vip[i] = sqrt(n_features * sum((W[i, :]' .^ 2) .* SSY) / sum(SSY))
       end

       return vip
   end
   ```

2. **Random Forest** (2-3 hours)
   ```julia
   # File: src/models/random_forest.jl
   using DecisionTree

   """
   Random Forest wrapper.
   """
   function fit_random_forest(X::Matrix{Float64}, y::Vector{Float64};
                              n_trees::Int=100,
                              max_depth::Int=-1,
                              min_samples_split::Int=2)
       model = build_forest(y, X,
                           n_trees,
                           max_depth=max_depth,
                           min_samples_split=min_samples_split)
       return model
   end

   function predict(rf, X::Matrix{Float64})
       return apply_forest(rf, X)
   end

   """
   Get feature importances from Random Forest.
   """
   function feature_importances(rf)
       # DecisionTree.jl provides impurity-based importances
       return impurity_importance(rf)
   end
   ```

3. **MLP (Neural Network)** (4-6 hours)
   ```julia
   # File: src/models/mlp.jl
   using Flux
   using Statistics

   """
   Multi-layer Perceptron for regression.
   """
   function build_mlp(n_features::Int, hidden_sizes::Vector{Int})
       layers = []

       # Input layer
       push!(layers, Dense(n_features, hidden_sizes[1], tanh))

       # Hidden layers
       for i in 2:length(hidden_sizes)
           push!(layers, Dense(hidden_sizes[i-1], hidden_sizes[i], tanh))
       end

       # Output layer (linear)
       push!(layers, Dense(hidden_sizes[end], 1))

       return Chain(layers...)
   end

   """
   Train MLP with early stopping.
   """
   function fit_mlp(X::Matrix{Float64}, y::Vector{Float64};
                    hidden_sizes::Vector{Int}=[64, 32],
                    max_epochs::Int=500,
                    learning_rate::Float64=0.001,
                    batch_size::Int=32)

       n_samples, n_features = size(X)
       model = build_mlp(n_features, hidden_sizes)

       # Prepare data
       data = Flux.Data.DataLoader((X', y'), batchsize=batch_size, shuffle=true)

       # Optimizer
       opt = ADAM(learning_rate)

       # Loss function
       loss(x, y) = Flux.mse(model(x), y)

       # Training loop with early stopping
       best_loss = Inf
       patience = 10
       patience_counter = 0

       for epoch in 1:max_epochs
           for (x_batch, y_batch) in data
               grads = gradient(() -> loss(x_batch, y_batch), Flux.params(model))
               Flux.update!(opt, Flux.params(model), grads)
           end

           # Validation loss (on full dataset for simplicity)
           current_loss = loss(X', y')

           if current_loss < best_loss
               best_loss = current_loss
               patience_counter = 0
           else
               patience_counter += 1
           end

           if patience_counter >= patience
               @info "Early stopping at epoch $epoch"
               break
           end
       end

       return model
   end

   function predict(model, X::Matrix{Float64})
       return vec(model(X'))
   end

   """
   Get feature importances (mean absolute weight from input layer).
   """
   function feature_importances(model)
       # Extract first layer weights
       W1 = Flux.params(model)[1]  # (hidden_size, n_features)
       return vec(mean(abs.(W1), dims=1))
   end
   ```

4. **Neural Boosted Regression** (6-8 hours)
   ```julia
   # File: src/models/neural_boosted.jl
   using Flux
   using Statistics

   """
   Neural Boosted Regression - gradient boosting with neural network weak learners.
   """
   mutable struct NeuralBoostedRegressor
       n_estimators::Int
       learning_rate::Float64
       hidden_size::Int
       estimators::Vector
       train_scores::Vector{Float64}
       validation_scores::Vector{Float64}

       function NeuralBoostedRegressor(;
                                       n_estimators::Int=100,
                                       learning_rate::Float64=0.1,
                                       hidden_size::Int=3)
           new(n_estimators, learning_rate, hidden_size, [], [], [])
       end
   end

   """
   Build a tiny weak learner network.
   """
   function build_weak_learner(n_features::Int, hidden_size::Int)
       return Chain(
           Dense(n_features, hidden_size, tanh),
           Dense(hidden_size, 1)
       )
   end

   """
   Fit Neural Boosted Regressor.
   """
   function fit!(nbr::NeuralBoostedRegressor,
                X::Matrix{Float64}, y::Vector{Float64})

       n_samples, n_features = size(X)

       # Initialize ensemble prediction
       F = zeros(n_samples)

       # Boosting loop
       for i in 1:nbr.n_estimators
           # Compute residuals
           residuals = y - F

           # Build and train weak learner
           weak_learner = build_weak_learner(n_features, nbr.hidden_size)

           # Quick training (fewer epochs for weak learner)
           opt = ADAM(0.01)
           loss(x, y) = Flux.mse(weak_learner(x), y)

           # Train for 50 iterations (fast for weak learner)
           for epoch in 1:50
               grads = gradient(() -> loss(X', residuals'), Flux.params(weak_learner))
               Flux.update!(opt, Flux.params(weak_learner), grads)
           end

           # Store weak learner
           push!(nbr.estimators, weak_learner)

           # Update ensemble predictions
           pred = vec(weak_learner(X'))
           F .+= nbr.learning_rate .* pred

           # Track training score
           train_score = mean((y - F) .^ 2)
           push!(nbr.train_scores, train_score)

           # Simple early stopping (optional)
           if i > 20 && train_score > nbr.train_scores[end-10]
               @info "Early stopping at iteration $i"
               break
           end
       end

       return nbr
   end

   """
   Predict using Neural Boosted ensemble.
   """
   function predict(nbr::NeuralBoostedRegressor, X::Matrix{Float64})
       predictions = zeros(size(X, 1))

       for estimator in nbr.estimators
           predictions .+= nbr.learning_rate .* vec(estimator(X'))
       end

       return predictions
   end

   """
   Get feature importances from Neural Boosted model.
   """
   function feature_importances(nbr::NeuralBoostedRegressor)
       n_features = size(Flux.params(nbr.estimators[1])[1], 2)
       importances = zeros(n_features)

       # Aggregate importances across all weak learners
       for estimator in nbr.estimators
           W1 = Flux.params(estimator)[1]  # First layer weights
           importances .+= vec(mean(abs.(W1), dims=1))
       end

       # Normalize
       importances ./= length(nbr.estimators)

       return importances
   end
   ```

**Week 5-6 Deliverables**:
- ✅ PLS regression with VIP scores (20-50x faster)
- ✅ Random Forest (10-20x faster)
- ✅ MLP (10-30x faster)
- ✅ Neural Boosted (10-30x faster)
- ✅ Feature importance extraction for all models
- ✅ Unit tests for all models

---

### Week 7: Model Selection & Search

**Goal**: Port hyperparameter search and cross-validation

#### Tasks

1. **Cross-Validation** (2-3 hours)
   ```julia
   # File: src/search.jl
   using MLJ

   """
   K-Fold cross-validation.
   """
   function cross_validate(model, X::Matrix{Float64}, y::Vector{Float64};
                          n_folds::Int=5)
       n_samples = size(X, 1)
       fold_size = n_samples ÷ n_folds

       scores = zeros(n_folds)

       for fold in 1:n_folds
           # Split data
           val_start = (fold - 1) * fold_size + 1
           val_end = fold * fold_size
           val_idx = val_start:val_end
           train_idx = setdiff(1:n_samples, val_idx)

           X_train, y_train = X[train_idx, :], y[train_idx]
           X_val, y_val = X[val_idx, :], y[val_idx]

           # Fit and predict
           fit!(model, X_train, y_train)
           y_pred = predict(model, X_val)

           # Compute score (R²)
           scores[fold] = r2_score(y_val, y_pred)
       end

       return scores
   end

   """
   R² score.
   """
   function r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})
       ss_res = sum((y_true - y_pred) .^ 2)
       ss_tot = sum((y_true .- mean(y_true)) .^ 2)
       return 1 - ss_res / ss_tot
   end
   ```

2. **Model Grid Search** (3-4 hours)
   ```julia
   """
   Run grid search over model configurations.
   """
   function run_search(X::Matrix{Float64}, y::Vector{Float64},
                      preprocessing_methods::Vector{Symbol},
                      model_types::Vector{Symbol})

       results = []

       # Iterate over preprocessing
       for preprocess_method in preprocessing_methods
           X_prep = preprocess(X, preprocess_method)

           # Iterate over models
           for model_type in model_types
               configs = get_model_configs(model_type)

               for config in configs
                   # Build model
                   model = build_model(model_type, config)

                   # Cross-validate
                   cv_scores = cross_validate(model, X_prep, y)

                   # Fit on full data
                   fit!(model, X_prep, y)

                   # Get predictions
                   y_pred = predict(model, X_prep)

                   # Compute metrics
                   rmse = sqrt(mean((y - y_pred) .^ 2))
                   r2 = r2_score(y, y_pred)

                   # Get feature importances
                   importances = feature_importances(model)

                   # Store result
                   result = SearchResult(
                       string(model_type),
                       string(preprocess_method),
                       size(X_prep, 2),
                       rmse,
                       r2,
                       cv_scores,
                       importances,
                       config
                   )

                   push!(results, result)

                   @info "$(model_type) + $(preprocess_method): R²=$(r2), RMSE=$(rmse)"
               end
           end
       end

       return results
   end

   """
   Get model configurations for grid search.
   """
   function get_model_configs(model_type::Symbol)
       if model_type == :PLS
           return [Dict(:n_components => n) for n in [5, 10, 15, 20, 24]]

       elseif model_type == :RandomForest
           return [
               Dict(:n_trees => 100, :max_depth => -1),
               Dict(:n_trees => 200, :max_depth => 10),
               Dict(:n_trees => 300, :max_depth => -1)
           ]

       elseif model_type == :MLP
           return [
               Dict(:hidden_sizes => [64]),
               Dict(:hidden_sizes => [64, 32]),
               Dict(:hidden_sizes => [128, 64])
           ]

       elseif model_type == :NeuralBoosted
           return [
               Dict(:n_estimators => 100, :learning_rate => 0.1, :hidden_size => 3),
               Dict(:n_estimators => 100, :learning_rate => 0.2, :hidden_size => 5)
           ]

       else
           error("Unknown model type: $model_type")
       end
   end
   ```

3. **Parallel Search** (1-2 hours)
   ```julia
   using Distributed

   """
   Run search in parallel across multiple cores.
   """
   function run_search_parallel(X::Matrix{Float64}, y::Vector{Float64},
                               preprocessing_methods::Vector{Symbol},
                               model_types::Vector{Symbol};
                               n_workers::Int=4)

       # Add worker processes
       addprocs(n_workers)

       # Load code on workers
       @everywhere include("src/SpectralPredict.jl")

       # Create task list
       tasks = []
       for preprocess_method in preprocessing_methods
           for model_type in model_types
               push!(tasks, (preprocess_method, model_type))
           end
       end

       # Run tasks in parallel
       results = pmap(tasks) do (preprocess_method, model_type)
           run_single_search(X, y, preprocess_method, model_type)
       end

       return vcat(results...)
   end
   ```

**Week 7 Deliverables**:
- ✅ Cross-validation framework
- ✅ Grid search implementation
- ✅ Parallel search (multi-core)
- ✅ Results aggregation and ranking

---

### Week 8: Integration & Python Bridge

**Goal**: Create Python interface, CLI, and final integration

#### Tasks

1. **Command-Line Interface** (2-3 hours)
   ```julia
   # File: src/cli.jl
   using ArgParse

   function parse_commandline()
       s = ArgParseSettings()

       @add_arg_table! s begin
           "--asd-dir"
               help = "Directory containing ASD files"
               required = true
           "--reference"
               help = "CSV file with reference values"
               required = true
           "--id-column"
               help = "Column name for sample IDs"
               required = true
           "--target"
               help = "Column name for target variable"
               required = true
           "--models"
               help = "Models to use (comma-separated)"
               default = "PLS,RandomForest,MLP,NeuralBoosted"
           "--output"
               help = "Output directory"
               default = "results"
       end

       return parse_args(s)
   end

   function main()
       args = parse_commandline()

       @info "Loading data from $(args["asd-dir"])"
       X, wavelengths, sample_ids = load_asd_directory(args["asd-dir"])

       @info "Loading reference data from $(args["reference"])"
       reference = read_reference(args["reference"])

       # Match samples
       y = reference[!, Symbol(args["target"])]

       # Parse models
       models = Symbol.(split(args["models"], ","))

       # Run search
       @info "Running model search..."
       results = run_search(X, y,
                           [:snv, :snv_derivative1, :snv_derivative2],
                           models)

       # Save results
       save_results(results, args["output"])

       @info "Done! Results saved to $(args["output"])"
   end
   ```

2. **Python Bridge (PyCall)** (3-4 hours)
   ```python
   # File: python_bridge/spectral_predict_julia.py
   """
   Python wrapper for Julia implementation.

   Usage:
       from spectral_predict_julia import JuliaSpectralPredict

       model = JuliaSpectralPredict()
       results = model.run_search(asd_dir, reference_file, target_column)
   """

   from julia import Main as Julia
   import pandas as pd

   class JuliaSpectralPredict:
       def __init__(self):
           # Load Julia code
           Julia.include("julia_port/src/SpectralPredict.jl")
           self.julia_module = Julia.SpectralPredict

       def run_search(self, asd_dir, reference_file, id_column, target_column,
                     models=None, output_dir="results"):
           """
           Run spectral prediction search using Julia backend.

           Parameters
           ----------
           asd_dir : str
               Directory containing ASD files
           reference_file : str
               Path to CSV with reference values
           id_column : str
               Column name for sample IDs
           target_column : str
               Column name for target variable
           models : list of str, optional
               Models to use (default: all)
           output_dir : str, optional
               Output directory for results

           Returns
           -------
           results : pd.DataFrame
               Search results with scores and parameters
           """
           if models is None:
               models = ["PLS", "RandomForest", "MLP", "NeuralBoosted"]

           # Call Julia function
           results = self.julia_module.run_search_from_files(
               asd_dir, reference_file, id_column, target_column,
               models, output_dir
           )

           # Convert to pandas DataFrame
           return self._julia_to_pandas(results)

       def _julia_to_pandas(self, julia_results):
           """Convert Julia results to pandas DataFrame."""
           # Implementation depends on Julia output format
           pass
   ```

3. **Testing & Validation** (4-6 hours)
   ```julia
   # File: test/runtests.jl
   using Test
   using SpectralPredict

   @testset "SpectralPredict.jl" begin
       @testset "Preprocessing" begin
           X = randn(10, 100)

           @testset "SNV Transform" begin
               X_snv = snv_transform(X)
               @test size(X_snv) == size(X)
               @test all(isfinite.(X_snv))
               # Mean should be ~0, std should be ~1 for each sample
               @test all(abs.(mean(X_snv, dims=2)) .< 0.1)
           end

           @testset "SG Derivative" begin
               X_deriv = savitzky_golay_derivative(X, deriv=1)
               @test size(X_deriv) == size(X)
               @test all(isfinite.(X_deriv))
           end
       end

       @testset "Models" begin
           X = randn(50, 100)
           y = randn(50)

           @testset "PLS" begin
               model = fit_pls(X, y, 5)
               y_pred = predict(model, X)
               @test length(y_pred) == length(y)
               @test all(isfinite.(y_pred))
           end

           @testset "Random Forest" begin
               model = fit_random_forest(X, y, n_trees=10)
               y_pred = predict(model, X)
               @test length(y_pred) == length(y)
           end

           @testset "Neural Boosted" begin
               nbr = NeuralBoostedRegressor(n_estimators=10)
               fit!(nbr, X, y)
               y_pred = predict(nbr, X)
               @test length(y_pred) == length(y)
           end
       end

       @testset "Cross-Validation" begin
           X = randn(50, 100)
           y = randn(50)
           model = fit_pls(X, y, 5)

           scores = cross_validate(model, X, y, n_folds=5)
           @test length(scores) == 5
           @test all(isfinite.(scores))
       end
   end
   ```

**Week 8 Deliverables**:
- ✅ Command-line interface
- ✅ Python bridge (optional but recommended)
- ✅ Comprehensive test suite
- ✅ Documentation
- ✅ Benchmarks comparing Julia vs Python

---

## Performance Benchmarking

### Create Benchmark Suite

```julia
# File: benchmarks/benchmark.jl
using BenchmarkTools
using SpectralPredict

# Create realistic test data
X = randn(100, 2151)  # 100 samples, 2151 wavelengths
y = randn(100)

@info "Benchmarking Julia implementation..."

println("\n=== Preprocessing ===")
@btime snv_transform($X)
@btime savitzky_golay_derivative($X, deriv=1)

println("\n=== Models ===")
@btime fit_pls($X, $y, 10)

pls_model = fit_pls(X, y, 10)
@btime predict($pls_model, $X)

@btime fit_random_forest($X, $y, n_trees=100)

nbr = NeuralBoostedRegressor(n_estimators=30)
@btime fit!($nbr, $X, $y)

println("\n=== Full Pipeline ===")
@btime run_search($X, $y, [:snv], [:PLS])
```

### Compare with Python

```bash
# Python benchmark (original)
time .venv/bin/spectral-predict --asd-dir example/ \
    --reference example/BoneCollagen.csv \
    --id-column "File Number" \
    --target "%Collagen"

# Julia benchmark (new)
time julia julia_port/src/cli.jl --asd-dir example/ \
    --reference example/BoneCollagen.csv \
    --id-column "File Number" \
    --target "%Collagen"

# Expected: Julia should be 15-30x faster
```

---

## Deployment Strategy

### Option 1: Standalone Julia CLI

Users install Julia and run directly:

```bash
# Install Julia
brew install julia

# Install package
cd julia_port
julia
]  # Enter package mode
activate .
instantiate

# Run
julia src/cli.jl --asd-dir data/ --reference ref.csv ...
```

**Pros**: Maximum performance
**Cons**: Users need Julia installed

### Option 2: Python Wrapper

Keep Python CLI, call Julia backend:

```python
# In existing spectral_predict/cli.py
try:
    from .julia_backend import run_search_julia
    USE_JULIA = True
except ImportError:
    USE_JULIA = False

def main():
    if USE_JULIA:
        results = run_search_julia(...)  # 30x faster
    else:
        results = run_search_python(...)  # Fallback
```

**Pros**: Transparent to users, backwards compatible
**Cons**: Need both Python and Julia, complexity

### Option 3: Compiled Binary (PackageCompiler.jl)

Create standalone executable:

```julia
using PackageCompiler

create_app("julia_port", "spectral_predict_binary",
           precompile_execution_file="benchmarks/precompile.jl")
```

**Pros**: No Julia installation needed, single executable
**Cons**: Large binary (~500MB), compile time

---

## Maintenance Considerations

### Code Organization

```
spectral-predict/
├── src/                          # Python implementation (keep)
│   └── spectral_predict/
├── julia_port/                   # Julia implementation
│   ├── src/
│   │   ├── SpectralPredict.jl   # Main module
│   │   ├── io.jl
│   │   ├── preprocessing.jl
│   │   ├── models/
│   │   │   ├── pls.jl
│   │   │   ├── random_forest.jl
│   │   │   ├── mlp.jl
│   │   │   └── neural_boosted.jl
│   │   ├── search.jl
│   │   └── cli.jl
│   ├── test/
│   │   └── runtests.jl
│   ├── benchmarks/
│   │   └── benchmark.jl
│   ├── Project.toml             # Dependencies
│   └── Manifest.toml
├── python_bridge/               # Python-Julia bridge (optional)
│   └── spectral_predict_julia.py
└── docs/
    ├── JULIA_INSTALLATION.md
    └── JULIA_BENCHMARK_RESULTS.md
```

### Testing Strategy

- **Unit tests**: Both Python and Julia versions
- **Integration tests**: Verify Julia gives same results as Python
- **Performance tests**: Track speedups over time
- **Continuous integration**: Test both implementations

### Feature Parity

- New features should be added to both Python and Julia versions
- Or decide to deprecate Python version if Julia proves stable
- Document differences clearly

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Julia ecosystem gaps | High | Research libraries before starting |
| Learning curve | Medium | Start with simple components, build up |
| Maintenance burden | High | Consider using Julia as optional fast backend only |
| Deployment complexity | Medium | Use Python wrapper or compiled binary |
| Results don't match Python | High | Comprehensive validation tests |
| Performance not as expected | Medium | Profile and optimize incrementally |

---

## Decision Framework

### When to Port to Julia

✅ **Good reasons**:
- Need 30-100x speedup
- Processing very large datasets regularly
- Other optimizations (Python-based) insufficient
- Team willing to learn Julia
- Time available for 4-8 week project

❌ **Bad reasons**:
- Just want faster code (try Numba/Cython first)
- Small datasets (won't see much benefit)
- One-off analysis (not worth the investment)
- No time for testing and validation

### Alternatives to Full Port

Before committing to full Julia port, consider:

1. **Partial port**: Just preprocessing and PLS in Julia (1-2 weeks)
2. **Numba optimization**: JIT compile Python code (1-2 days, 2-5x speedup)
3. **Cython**: Compile to C (1 week, 5-10x speedup)
4. **PyTorch**: Use for neural models (2-3 days, 5-10x speedup)
5. **C++ extension**: For critical bottlenecks only (2 weeks, 20-50x)

Julia port makes sense if:
- Need maximum performance (30-100x)
- Want clean, maintainable fast code
- Willing to invest 4-8 weeks
- Plan to use long-term

---

## Success Metrics

After Julia port, verify:

1. **Performance**: 15-30x speedup minimum (target: 30-100x)
2. **Accuracy**: Results within 0.1% of Python version
3. **Feature parity**: All Python features working
4. **Ease of use**: Simple installation and CLI
5. **Maintainability**: Clean, documented code
6. **Testing**: >90% code coverage

---

## Next Steps

### Before Starting

1. **Validate need**: Is 30-100x speedup necessary?
2. **Try alternatives**: Numba, Cython, PyTorch first
3. **Prototype**: Port just SNV transform and PLS (2 days) to validate approach
4. **Get buy-in**: Team agreement on maintenance burden

### Quick Prototype (2 days)

```julia
# Test Julia viability with minimal port
module TestPort

using Statistics

function snv_transform(X::Matrix{Float64})
    # ... implement SNV ...
end

# Test on real data
X = randn(100, 2151)
@time snv_transform(X)  # Should be 20-50x faster than Python
end
```

If prototype shows good results → proceed with full port
If not → stick with Python optimizations

---

## Summary

**Julia Port Overview**:
- **Timeline**: 4-8 weeks
- **Expected Speedup**: 30-100x
- **Risk**: Medium-High
- **Effort**: High
- **Maintenance**: Ongoing

**Recommended Approach**:
1. Try quick optimizations first (Numba, neural boosting fixes)
2. If still need more speed, do 2-day Julia prototype
3. If prototype successful, commit to 8-week full port
4. Deploy as Python wrapper for backwards compatibility

**When to Skip**:
- If 2-10x speedup is sufficient (use Numba/Cython instead)
- If working with small datasets (<1000 samples)
- If no long-term maintenance plan

---

**Julia port is powerful but should be a last resort after exhausting easier optimizations.**
