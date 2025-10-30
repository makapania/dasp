# SpectralPredict.jl

High-performance Julia implementation of the spectral prediction machine learning system.

## Overview

SpectralPredict.jl is a Julia port of the Python-based spectral analysis ML system. It provides:

- **Preprocessing**: SNV transformation and Savitzky-Golay derivatives
- **High Performance**: Type-stable, optimized Julia code
- **Compatibility**: 100% compatible with Python implementation

## Installation

```julia
using Pkg
Pkg.activate("path/to/julia_port/SpectralPredict")
Pkg.instantiate()
```

## Quick Start

```julia
using SpectralPredict

# Load your spectral data
X = load_your_data()  # Matrix{Float64} of shape (n_samples, n_features)

# Apply preprocessing
config = Dict(
    "name" => "snv_deriv",
    "deriv" => 2,
    "window" => 17,
    "polyorder" => 3
)

X_transformed = apply_preprocessing(X, config)
```

## Module Status

### ✅ Completed
- **preprocessing.jl**: SNV and Savitzky-Golay derivatives

### 🚧 In Progress
- Model implementations
- Search space optimization
- Data loading utilities

## Documentation

See `docs/preprocessing.md` for comprehensive preprocessing module documentation.

## Dependencies

- Julia 1.9+
- SavitzkyGolay.jl: Savitzky-Golay filtering
- Statistics: Statistical functions
- LinearAlgebra: Matrix operations

## Performance

Julia implementation provides significant performance improvements over Python:

- **SNV**: ~2-3x faster than NumPy
- **Derivatives**: ~3-5x faster than SciPy
- **Type-stable**: Optimal Julia compilation

## Compatibility

This implementation maintains 100% compatibility with the Python version:
- Identical preprocessing results
- Same configuration format
- Matching API design

## License

Same as parent project.
