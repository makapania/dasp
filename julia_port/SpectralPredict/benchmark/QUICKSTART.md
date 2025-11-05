# Benchmark Quick Start Guide

Run performance benchmarks in 5 minutes.

## Prerequisites

1. **Julia 1.9+** installed
2. **Python 3.8+** installed (for comparison)
3. SpectralPredict packages installed in both

## Step 1: Install Julia Dependencies

```bash
cd julia_port/SpectralPredict

# Enter Julia REPL
julia

# Activate project and install
] activate .
] instantiate
```

Exit Julia (Ctrl+D).

## Step 2: Run Julia Benchmarks

```bash
# Run all benchmarks (recommended)
julia --threads=auto benchmark/bench_comprehensive.jl

# Or run individual benchmarks
julia --threads=auto benchmark/bench_variable_selection.jl
julia --threads=auto benchmark/bench_diagnostics.jl
julia --threads=auto benchmark/bench_neural_boosted.jl
julia --threads=auto benchmark/bench_msc.jl
```

**Expected time:** 10-20 minutes for comprehensive suite

## Step 3: Run Python Benchmarks (Optional)

For direct comparison:

```bash
# From repository root
cd julia_port/SpectralPredict
python benchmark/run_python_comparison.py
```

This generates `python_benchmark_results.json`.

## Step 4: Interpret Results

### Look for Key Metrics

The benchmark output includes:

```
Variable Selection Performance Benchmarks
==========================================

Scale: Medium (300 × 1500)
==========================================

Medium (300 × 1500) - UVE Selection
============================================================

  Warming up UVE...
  Running UVE (5 iterations)...

UVE Selection:
  Mean:   0.2450 ± 0.0123 s    <-- Average time
  Min:    0.2301 s              <-- Best time
  Max:    0.2612 s              <-- Worst time
  Memory: 45.23 MB              <-- Memory used
  Target speedup: 6-10x vs Python
```

### Calculate Speedup

If Python took 2.450s and Julia took 0.245s:

```
Speedup = 2.450 / 0.245 = 10.0x ✓
```

Compare against target (6-10x) → **Success!**

## Step 5: Review Summary

At the end of comprehensive benchmark:

```
SUMMARY REPORT
==============================================================

Mean Execution Times (seconds):
--------------------------------------------------------------
Scale                     UVE          SPA          iPLS
--------------------------------------------------------------
Small (100 × 500)         0.0231       0.0145       0.1234
Medium (300 × 1500)       0.2450       0.1823       1.2341
Large (1000 × 2151)       2.3451       1.4567       8.9012

Target Speedups (vs Python):
  - UVE: 6-10x faster
  - SPA: 10-20x faster (with parallelization)
  - iPLS: 8-12x faster
  - UVE-SPA: 8-15x faster

Overall Pipeline Target: 5-15x faster than Python
```

## Troubleshooting

### Issue: Only 1 thread available

```julia
julia> Threads.nthreads()
1
```

**Solution:**

Linux/Mac:
```bash
export JULIA_NUM_THREADS=auto
julia --threads=auto benchmark/bench_comprehensive.jl
```

Windows:
```bash
set JULIA_NUM_THREADS=auto
julia --threads=auto benchmark/bench_comprehensive.jl
```

### Issue: Package not found

```
ERROR: ArgumentError: Package SavitzkyGolay not found
```

**Solution:**
```bash
julia
] activate .
] add SavitzkyGolay
] instantiate
```

### Issue: Out of memory

**Solution:** Run smaller scale tests only:

Edit the benchmark file and comment out large scales:

```julia
test_scales = [
    ("Small (100 × 500)", 100, 500, 20),
    # ("Large (1000 × 2151)", 1000, 2151, 100)  # Comment out
]
```

### Issue: Python comparison fails

Make sure SpectralPredict is installed:

```bash
cd /path/to/dasp
pip install -e .
```

## Next Steps

1. **Review full results** in console output
2. **Check JSON report** at `benchmark/benchmark_report.json`
3. **Compare with Python** by running `run_python_comparison.py`
4. **Fill out report template** at `BENCHMARK_REPORT_TEMPLATE.md`
5. **Test parallelization** by varying thread counts:
   ```bash
   julia --threads=1 benchmark/bench_diagnostics.jl
   julia --threads=4 benchmark/bench_diagnostics.jl
   julia --threads=8 benchmark/bench_diagnostics.jl
   ```

## Expected Results

If all is working correctly, you should see:

- ✅ **Variable Selection**: 6-20x speedup (varies by method)
- ✅ **Diagnostics**: 3-25x speedup (jackknife highest)
- ✅ **Neural Boosted**: 2-5x speedup
- ✅ **MSC**: 8-12x speedup
- ✅ **Overall Pipeline**: 5-15x speedup

## Help

For detailed information, see:
- `README.md` - Full documentation
- `BENCHMARK_REPORT_TEMPLATE.md` - Report template
- `../TROUBLESHOOTING.md` - General troubleshooting
- `../documentation/JULIA_PORTING_IMPLEMENTATION_PLAN.md` - Implementation details

## Quick Reference

### Commands

```bash
# All benchmarks
julia --threads=auto benchmark/bench_comprehensive.jl

# Individual benchmarks
julia --threads=auto benchmark/bench_variable_selection.jl
julia --threads=auto benchmark/bench_diagnostics.jl
julia --threads=auto benchmark/bench_neural_boosted.jl
julia --threads=auto benchmark/bench_msc.jl

# Python comparison
python benchmark/run_python_comparison.py
```

### Files Generated

- `benchmark_report.json` - Machine-readable results
- `python_benchmark_results.json` - Python timings (if run)

### Key Metrics

- **Mean time**: Average across runs
- **Speedup**: Python_time / Julia_time
- **Target**: Expected speedup from implementation plan
- **Status**: ✅ (meets target) | ⚠️ (close) | ❌ (below target)

---

**Time to run:** ~15 minutes for full suite

**Ready to start?** Run: `julia --threads=auto benchmark/bench_comprehensive.jl`
