"""
    SpectralPredict

Julia port of the DASP Spectral Prediction system for machine learning-based spectral analysis.

This package provides comprehensive tools for:
- Spectral data preprocessing (SNV, Savitzky-Golay derivatives)
- Multiple ML models (PLS, Ridge, Lasso, ElasticNet, RandomForest, MLP)
- Automated hyperparameter search with cross-validation
- Feature subset selection (variable and region-based)
- Model ranking and comparison

# Quick Start

```julia
using SpectralPredict

# Load data
X, y, wavelengths, sample_ids = load_spectral_dataset(
    "data/spectra",
    "data/reference.csv",
    "sample_id",
    "protein_pct"
)

# Run comprehensive search
results = run_search(
    X, y, wavelengths,
    task_type="regression",
    models=["PLS", "Ridge", "RandomForest"],
    preprocessing=["snv", "deriv"],
    enable_variable_subsets=true,
    enable_region_subsets=true
)

# View top models
top_10 = first(results, 10)
println(top_10)

# Save results
save_results(results, "predictions.csv")
```

# Main Functions

- `run_search()` - Main hyperparameter search function
- `load_spectral_dataset()` - Load spectral data and reference values
- `save_results()` - Save results to CSV
- `apply_preprocessing()` - Apply preprocessing transformations
- `create_region_subsets()` - Identify important spectral regions

# Module Structure

- `preprocessing.jl` - SNV, derivatives, pipeline
- `models.jl` - ML model wrappers (PLS, RF, MLP, etc.)
- `cv.jl` - Cross-validation framework
- `regions.jl` - Spectral region analysis
- `scoring.jl` - Model ranking system
- `search.jl` - Main search orchestration
- `io.jl` - File I/O (CSV, SPC)

# Citation

If you use this package in your research, please cite:
DASP Spectral Prediction System
Julia Port - October 2025

# Documentation

See README.md and docs/ folder for complete documentation.
"""
module SpectralPredict

# Standard library imports
using Statistics
using LinearAlgebra
using Random
using Distributed

# External package imports
using DataFrames
using CSV
using ProgressMeter
using StatsBase
using MultivariateStats
using GLMNet
using DecisionTree
using Flux
using SavitzkyGolay
using DSP

# Include all module files
include("preprocessing.jl")
include("models.jl")
include("cv.jl")
include("regions.jl")
include("scoring.jl")
include("search.jl")
include("io.jl")

# Export main functions

## Search and analysis
export run_search
export run_single_config
export generate_preprocessing_configs

## Preprocessing
export apply_preprocessing
export apply_snv
export apply_derivative
export build_preprocessing_pipeline

## Models
export build_model
export fit_model!
export predict_model
export get_feature_importances
export get_model_configs
export PLSModel, RidgeModel, LassoModel, ElasticNetModel, RandomForestModel, MLPModel

## Cross-validation
export run_cross_validation
export run_cross_validation_parallel
export create_cv_folds
export run_single_fold
export compute_regression_metrics
export compute_classification_metrics

## Regions
export compute_region_correlations
export create_region_subsets
export combine_region_indices

## Scoring
export compute_composite_score
export rank_results!

## I/O
export load_spectral_dataset
export read_csv
export read_spc
export read_reference_csv
export save_results
export align_xy

# Version information
const VERSION = v"0.1.0"

"""
    version()

Display SpectralPredict version information.
"""
function version()
    println("SpectralPredict.jl v$(VERSION)")
    println("Julia port of DASP Spectral Prediction System")
    println("October 2025")
end

export version

end # module SpectralPredict
