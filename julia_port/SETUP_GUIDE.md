# Julia Port Setup Guide

**Date:** October 29, 2025
**Project:** DASP Spectral Prediction - Julia Port (Phase 1)

---

## Step 1: Install Julia

### Windows Installation

1. **Download Julia:**
   - Visit: https://julialang.org/downloads/
   - Download: "Windows x86 64-bit (installer)" - Current stable release (1.10.x recommended)
   - Or direct link: https://julialang-s3.julialang.org/bin/winnt/x64/1.10/julia-1.10.0-win64.exe

2. **Run installer:**
   - Accept defaults
   - ✅ Check "Add Julia to PATH" (important!)
   - Install location: `C:\Users\sponheim\AppData\Local\Programs\Julia-1.10.0\`

3. **Verify installation:**
   ```bash
   julia --version
   # Should output: julia version 1.10.x
   ```

---

## Step 2: Create Julia Project

```bash
cd C:\Users\sponheim\git\dasp\julia_port
julia
```

In Julia REPL:
```julia
# Enter package mode with ]
]

# Generate project
generate SpectralPredict
activate SpectralPredict

# Exit package mode with backspace
# Exit Julia
exit()
```

---

## Step 3: Install Dependencies

```bash
cd SpectralPredict
julia
```

In Julia REPL (enter package mode with `]`):
```julia
# Core ML & Stats
add MLJ
add MultivariateStats
add GLMNet
add DecisionTree
add Flux

# Data Processing
add DataFrames
add CSV
add Tables

# Numerical & Stats
add StatsBase
add LinearAlgebra
add Statistics

# Signal Processing (for derivatives)
add DSP

# Utilities
add ArgParse
add ProgressMeter

# Testing
add Test

# Exit package mode
# (press backspace)

# Precompile packages
using MLJ, DataFrames, CSV, MultivariateStats, GLMNet
using DecisionTree, Flux, StatsBase, DSP

# Exit Julia
exit()
```

**Note:** First installation will take 10-20 minutes to download and precompile.

---

## Step 4: Verify Setup

Create test file: `test_setup.jl`
```julia
using DataFrames
using Statistics
using LinearAlgebra

# Test basic operations
X = randn(10, 5)
println("Created matrix: ", size(X))

# Test SNV
X_snv = (X .- mean(X, dims=2)) ./ std(X, dims=2)
println("SNV applied: ", size(X_snv))

println("\n✅ Julia setup complete!")
```

Run test:
```bash
julia test_setup.jl
```

---

## Step 5: Development Environment (Optional but Recommended)

### Option 1: VS Code + Julia Extension
1. Install VS Code: https://code.visualstudio.com/
2. Install Julia extension: `julia-vscode.language-julia`
3. Open `SpectralPredict` folder in VS Code
4. Enjoy syntax highlighting, REPL integration, debugging

### Option 2: Jupyter + IJulia
```julia
# In Julia REPL (package mode)
add IJulia

# Start Jupyter
using IJulia
notebook()
```

---

## Project Structure (Created Automatically)

```
julia_port/
├── SpectralPredict/
│   ├── Project.toml         # Dependencies
│   ├── Manifest.toml        # Locked versions (auto-generated)
│   ├── src/
│   │   └── SpectralPredict.jl  # Main module
│   ├── test/
│   │   └── runtests.jl     # Tests
│   └── examples/
│       └── basic_analysis.jl
└── SETUP_GUIDE.md (this file)
```

---

## Next Steps After Setup

1. Implement preprocessing (Week 1)
2. Implement models (Week 2)
3. Implement CV framework (Week 3)
4. Continue with roadmap...

---

## Troubleshooting

### "Julia not found" after installation
- Restart terminal/VS Code
- Manually add to PATH: `C:\Users\sponheim\AppData\Local\Programs\Julia-1.10.0\bin`

### Package installation fails
- Check internet connection
- Try: `] up` to update package registry
- Try: `] gc` to garbage collect old packages

### Slow precompilation
- Normal on first install (10-20 min)
- Subsequent startups will be fast (<5 seconds)

---

## Useful Julia Commands

```julia
# Package mode (press ])
add PackageName    # Install package
rm PackageName     # Remove package
up                 # Update packages
status             # List installed packages
gc                 # Clean up old versions

# Help mode (press ?)
?functionname      # Get help on function

# Shell mode (press ;)
ls                 # Run shell commands
```

---

## Performance Tips for Julia

1. **Type stability** - Declare types where possible
2. **Preallocate arrays** - Use `zeros()`, `ones()`, `Vector{T}(undef, n)`
3. **Views over copies** - Use `@view X[:, indices]`
4. **Avoid global variables** - Wrap code in functions
5. **Profile first** - Use `@time`, `@btime` from BenchmarkTools

---

## Ready to Begin!

Once Julia is installed and dependencies are added, you're ready to start implementing the core algorithms.

**Estimated Phase 1 timeline:** 6-8 weeks
**Target speedup:** 2-5x faster than Python

🚀 **Let's build this!**
