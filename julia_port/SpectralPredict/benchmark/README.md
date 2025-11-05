# SpectralPredict.jl Performance Benchmarks

Comprehensive performance benchmarking suite for measuring Julia vs Python speedups.

## Overview

This directory contains benchmarks for all performance-critical components of the Julia port:

- **Variable Selection** (`bench_variable_selection.jl`) - UVE, SPA, iPLS, UVE-SPA
- **Diagnostics** (`bench_diagnostics.jl`) - Leverage, residuals, jackknife intervals
- **Neural Boosted** (`bench_neural_boosted.jl`) - Gradient boosting with MLP weak learners
- **MSC Preprocessing** (`bench_msc.jl`) - Multiplicative scatter correction
- **Comprehensive** (`bench_comprehensive.jl`) - Runs all benchmarks and generates report

## Quick Start

### Run All Benchmarks

```bash
cd julia_port/SpectralPredict
julia --threads=auto benchmark/bench_comprehensive.jl
```

### Run Individual Benchmarks

```bash
# Variable selection
julia --threads=auto benchmark/bench_variable_selection.jl

# Diagnostics (parallelization benefits)
julia --threads=auto benchmark/bench_diagnostics.jl

# Neural boosted regressor
julia --threads=auto benchmark/bench_neural_boosted.jl

# MSC preprocessing
julia --threads=auto benchmark/bench_msc.jl
```

## Expected Speedups

Based on the Julia Porting Implementation Plan:

| Module | Operation | Target Speedup | Notes |
|--------|-----------|----------------|-------|
| **Variable Selection** |
| | SPA selection | 10-20x | Parallelized |
| | UVE selection | 6-10x | Vectorized |
| | iPLS selection | 8-12x | Efficient CV |
| | UVE-SPA | 8-15x | Combined |
| **Diagnostics** |
| | Leverage | 5-8x | Efficient LA |
| | Residuals | 3-5x | Vectorized |
| | Jackknife | 17-25x | Parallelized |
| **Neural Boosted** |
| | Training | 2-3x | Flux.jl |
| | Prediction | 3-5x | Efficient |
| **Preprocessing** |
| | MSC | 8-12x | BLAS |
| **Overall Pipeline** | | **5-15x** | Full workflow |

## Test Data Scales

Benchmarks use realistic spectroscopy data sizes:

### Variable Selection & MSC
- **Small**: 100 samples × 500 wavelengths
- **Medium**: 300 samples × 1500 wavelengths (typical NIR)
- **Large**: 1000 samples × 2151 wavelengths (full resolution)

### Diagnostics & Neural Boosted
- **Small**: 100 samples × 50 features
- **Medium**: 300 samples × 150 features (after variable selection)
- **Large**: 1000 samples × 300 features

## Parallelization Testing

### Thread Configuration

```bash
# Automatic (recommended)
julia --threads=auto benchmark/bench_diagnostics.jl

# Specific thread count
julia --threads=4 benchmark/bench_diagnostics.jl
julia --threads=8 benchmark/bench_diagnostics.jl

# Check thread count in Julia
julia> Threads.nthreads()
```

### Parallelized Operations

1. **SPA Selection** - Parallelizes candidate variable evaluation
2. **Jackknife Intervals** - Parallelizes leave-one-out cross-validation
3. **iPLS** - Parallelizes interval evaluation (future enhancement)

### Testing Speedup Scaling

Run diagnostics benchmark with different thread counts:

```bash
julia --threads=1 benchmark/bench_diagnostics.jl > results_1thread.txt
julia --threads=2 benchmark/bench_diagnostics.jl > results_2threads.txt
julia --threads=4 benchmark/bench_diagnostics.jl > results_4threads.txt
julia --threads=8 benchmark/bench_diagnostics.jl > results_8threads.txt
```

Expected scaling for jackknife:
- 1 thread: baseline
- 2 threads: ~1.8x speedup
- 4 threads: ~3.5x speedup
- 8 threads: ~6.5x speedup

## Benchmark Methodology

### Warmup Runs
- Each benchmark includes warmup runs to trigger JIT compilation
- Warmup times are **excluded** from reported statistics
- Ensures accurate measurement of steady-state performance

### Statistics Reported
- **Mean**: Average execution time across runs
- **Std Dev**: Standard deviation (consistency indicator)
- **Min/Max**: Range of observed times
- **Memory**: Estimated memory allocation

### Iteration Counts
- Fast operations: 5-10 runs
- Medium operations: 3-5 runs
- Slow operations (neural nets): 3 runs

## Output Files

### Console Output
- Detailed timing statistics for each test
- Progress indicators during execution
- Summary tables comparing scales

### JSON Report
`benchmark_report.json` - Machine-readable results:
```json
{
  "timestamp": "2025-11-05T...",
  "julia_version": "1.11.1",
  "threads": 8,
  "target_speedups": {...},
  "system": {...}
}
```

## Comparing with Python

### 1. Run Python Benchmarks

Create equivalent Python benchmarks (see `BENCHMARK_REPORT_TEMPLATE.md`):

```python
# bench_python.py
import time
from spectral_predict.variable_selection import uve_selection

# Generate test data
X, y = generate_data(300, 1500)

# Time UVE
start = time.time()
result = uve_selection(X, y, n_components=10)
elapsed = time.time() - start

print(f"Python UVE: {elapsed:.4f} s")
```

### 2. Calculate Speedup

```
Speedup = Python_time / Julia_time

Example:
  Python UVE: 2.450 s
  Julia UVE:  0.245 s
  Speedup: 10.0x ✓
```

### 3. Verify Against Targets

Check that calculated speedups meet or exceed targets:
- UVE: 6-10x → Achieved 10x ✓
- Overall: 5-15x → Measure on full pipeline

## Troubleshooting

### Low Speedups

**Problem**: Speedups lower than expected

**Solutions**:
1. Check thread count: `julia> Threads.nthreads()`
2. Enable threading: `julia --threads=auto`
3. Check BLAS threads: `LinearAlgebra.BLAS.get_num_threads()`
4. Verify packages installed: `] status`

### High Variance

**Problem**: Large standard deviation in timings

**Solutions**:
1. Close other applications
2. Run on dedicated system
3. Increase iteration count
4. Check for thermal throttling

### Out of Memory

**Problem**: Benchmark crashes with OOM error

**Solutions**:
1. Reduce data size in benchmark script
2. Use `--heap-size-hint=XG` flag
3. Close other applications
4. Run smaller scale tests only

### Compilation Overhead

**Problem**: First run much slower than subsequent runs

**Expected**: This is normal! JIT compilation happens on first run.
- Warmup runs trigger compilation
- Only timed runs are measured
- Use PackageCompiler.jl for precompilation

## Advanced Usage

### Custom Data Sizes

Edit benchmark scripts to test specific sizes:

```julia
# bench_variable_selection.jl (line ~180)
test_scales = [
    ("Custom", 500, 1000, 40),  # Add your size
    # ... existing scales
]
```

### Custom Configurations

Test different model parameters:

```julia
# bench_neural_boosted.jl
configs = [
    (n_estimators=200, hidden_size=5),  # Your config
    # ... existing configs
]
```

### Memory Profiling

Add detailed memory tracking:

```julia
using Profile

# Profile memory allocations
@time result = uve_selection(X, y, n_components=10)
@allocated result = uve_selection(X, y, n_components=10)
```

### GPU Benchmarking (Future)

When GPU support is added:

```julia
using CUDA

# Move data to GPU
X_gpu = CuArray(X)
y_gpu = CuArray(y)

# Time GPU version
@time result = uve_selection_gpu(X_gpu, y_gpu)
```

## Continuous Benchmarking

### Automated Testing

Add to CI/CD pipeline:

```yaml
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: |
    julia --threads=auto benchmark/bench_comprehensive.jl
    # Compare against baseline
    python scripts/compare_benchmarks.py
```

### Regression Detection

Track performance over commits:

```bash
# Save baseline
julia benchmark/bench_comprehensive.jl > baseline.txt

# After changes, compare
julia benchmark/bench_comprehensive.jl > current.txt
diff baseline.txt current.txt
```

## References

- **BenchmarkTools.jl**: Advanced benchmarking (optional dependency)
- **PkgBenchmark.jl**: Package-level benchmarking
- **ProfileView.jl**: Profiling and optimization
- **Julia Performance Tips**: https://docs.julialang.org/en/v1/manual/performance-tips/

## Support

For issues with benchmarks:

1. Check Julia version: `julia --version` (need 1.9+)
2. Verify packages: `] status` in Julia REPL
3. Review `TROUBLESHOOTING.md` in SpectralPredict/
4. Check implementation plan for expected behavior

## License

Same as SpectralPredict.jl main package.
