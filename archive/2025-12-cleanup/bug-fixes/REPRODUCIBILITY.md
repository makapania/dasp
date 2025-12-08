# Reproducibility Guide

This document explains how to use the reproducibility toggle mechanism in DASP for scientific research.

## TL;DR - Quick Start

```python
# 🚀 Fast mode (exploration) - DEFAULT
results, _ = run_search(X, y, task_type='regression')

# 🔬 Reproducible mode (publications)
results, _ = run_search(X, y, task_type='regression', reproducible=True)
```

That's it! Settings automatically toggle on and off.

---

## The Problem

Machine learning pipelines have many sources of non-determinism:
- **Parallel processing**: CV folds execute in random order
- **BLAS threading**: Multi-threaded linear algebra operations
- **Random number generation**: Model initialization, CV splits
- **Ranking instability**: Tiny metric differences cause different rankings

This makes results **non-reproducible** - running the same code twice gives different answers!

---

## The Solution

DASP now has a **single reproducible mode** that eliminates all sources of non-determinism:

### Fast Mode (Default) - For Exploration
```python
results, _ = run_search(
    X, y,
    task_type='regression',
    folds=5,
    max_n_components=20,
    reproducible=False  # ← DEFAULT (can omit this line)
)
```

**What happens:**
- ✅ Uses all CPU cores (n_jobs=-1)
- ✅ Fast execution (~100% speed)
- ❌ Results may vary slightly between runs
- ❌ **NOT suitable for publications**

**Use when:**
- Exploring different models
- Testing preprocessing methods
- Iterating on hyperparameters
- Quick prototyping

---

### Reproducible Mode - For Publications
```python
results, _ = run_search(
    X, y,
    task_type='regression',
    folds=5,
    max_n_components=20,
    reproducible=True,     # ← Enable reproducibility
    random_state=42        # ← Control randomness
)
```

**What happens:**
```
================================================================================
REPRODUCIBLE MODE ENABLED
================================================================================
Settings:
  - BLAS threads: 1 (deterministic linear algebra)
  - CV execution: Serial (n_jobs=1)
  - Model parallelism: Disabled (n_jobs=1)
  - Random seed: 42
WARNING: Reproducible mode is ~3-5x slower than parallel execution.
After this run completes, BLAS settings will be restored automatically.
================================================================================

[... search runs ...]

================================================================================
BLAS thread settings restored to original values
================================================================================
```

**What's controlled:**
- ✅ BLAS/LAPACK threads set to 1 (deterministic linear algebra)
- ✅ CV folds execute serially (no race conditions)
- ✅ All models use n_jobs=1 (RandomForest, XGBoost, etc.)
- ✅ All RNG operations use fixed random_state
- ✅ Rankings stable even for near-ties (no z-score amplification when penalties=0)
- ✅ Settings automatically restored after run

**Guarantees:**
- ✅ **Bit-identical results** across runs
- ✅ Same input → Same output, every time
- ✅ Suitable for scientific publications
- ✅ Regulatory compliance ready

**Use when:**
- Finalizing results for papers
- Generating publication figures
- Regulatory submissions
- Need to reproduce exact results later

**Performance:**
- ⚠️ ~3-5x slower than fast mode
- Example: 10 minute search → 30-50 minutes

---

## Automatic Toggle - How It Works

### Before (Old Approach - BROKEN ❌)
```python
# Set BLAS threads
set_blas_threads(1)

# Run analysis
results = run_search(...)

# PROBLEM: Threads stay at 1 forever!
# Next run is still slow even without reproducible=True

# Manual fix required:
restore_default_threads()  # Easy to forget!
```

### After (New Approach - WORKS ✅)
```python
# Run 1: Reproducible
results1 = run_search(..., reproducible=True)
# → BLAS=1 during run
# → BLAS automatically restored after run

# Run 2: Fast (immediately after)
results2 = run_search(..., reproducible=False)
# → BLAS uses all cores
# → No manual cleanup needed!

# Run 3: Reproducible again
results3 = run_search(..., reproducible=True)
# → BLAS=1 during run
# → BLAS automatically restored again
```

**Key Point:** You can switch between modes as often as you want!

---

## Typical Workflow

### Phase 1: Exploration (Fast Mode)
```python
# Try different preprocessing methods (fast)
for method in ['raw', 'snv', 'deriv']:
    results, _ = run_search(
        X, y,
        task_type='regression',
        preprocessing_methods={method: True},
        reproducible=False  # Fast!
    )
    print(f"{method}: R² = {results.iloc[0]['R2']:.3f}")
```

### Phase 2: Finalization (Reproducible Mode)
```python
# Once you've decided on the best approach, finalize with reproducible mode
final_results, encoder = run_search(
    X, y,
    task_type='regression',
    preprocessing_methods={'snv': True},  # Best from exploration
    reproducible=True,  # Publication-ready
    random_state=42
)

# Save for paper
final_results.to_csv('publication_results.csv')
```

### Phase 3: Verification (Reproducible Mode)
```python
# Weeks/months later, verify you can reproduce exact results
verify_results, _ = run_search(
    X, y,
    task_type='regression',
    preprocessing_methods={'snv': True},
    reproducible=True,
    random_state=42  # Same seed
)

# Should be EXACTLY identical
assert final_results.equals(verify_results)  # ✅ Passes!
```

---

## Advanced: Different Random Seeds

```python
# Get different (but reproducible) CV splits
results_seed_42, _ = run_search(..., reproducible=True, random_state=42)
results_seed_123, _ = run_search(..., reproducible=True, random_state=123)

# Different seeds → Different CV splits → Different metrics
assert not results_seed_42.equals(results_seed_123)

# But each seed is individually reproducible
results_seed_42_again, _ = run_search(..., reproducible=True, random_state=42)
assert results_seed_42.equals(results_seed_42_again)  # ✅ Exact match
```

---

## Advanced: Context Manager (For Power Users)

If you need fine-grained control over reproducibility:

```python
from spectral_predict.reproducibility import reproducible_context

# Option 1: Wrap specific code blocks
with reproducible_context(n_threads=1, random_state=42):
    # Only this code runs with BLAS=1
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
# BLAS restored automatically here

# Option 2: Multiple analyses with different settings
for seed in [42, 123, 456]:
    with reproducible_context(random_state=seed):
        results = run_search(X, y, task_type='regression')
        # Each iteration uses different seed but is reproducible
```

---

## Troubleshooting

### Q: How do I know if reproducibility is working?
**A:** Run your analysis twice and compare results:
```python
results1, _ = run_search(..., reproducible=True, random_state=42)
results2, _ = run_search(..., reproducible=True, random_state=42)
pd.testing.assert_frame_equal(results1, results2)  # Should pass!
```

### Q: My reproducible run is slow. Is this normal?
**A:** Yes! Reproducible mode is 3-5x slower because:
- CV folds run serially instead of in parallel
- BLAS uses 1 thread instead of all cores
- Models use n_jobs=1 instead of n_jobs=-1

This is the trade-off for exact reproducibility.

### Q: Can I manually restore BLAS settings?
**A:** Usually not needed (automatic), but if you want:
```python
from spectral_predict.reproducibility import restore_default_threads
restore_default_threads()
```

### Q: What if I get an error during a reproducible run?
**A:** Settings are currently NOT restored on errors. If needed:
```python
try:
    results = run_search(..., reproducible=True)
except Exception as e:
    # Settings may still be at BLAS=1
    restore_default_threads()  # Manual restore
    raise
```

We may add try/finally blocks in the future for automatic restoration on errors.

### Q: How can I check current BLAS settings?
**A:**
```python
from spectral_predict.reproducibility import check_reproducibility_status
status = check_reproducibility_status()
print(status['blas_threads_env'])
```

---

## Testing Reproducibility

Run the test suite:
```bash
pytest tests/test_reproducibility.py -v
```

Run the demo:
```bash
python examples/reproducibility_demo.py
```

---

## What's Reproducible vs What's Not

### ✅ Reproducible with `reproducible=True`:
- CV fold splits
- CV fold execution order
- BLAS/LAPACK operations (PLS, Ridge, etc.)
- RandomForest tree building
- XGBoost/LightGBM boosting
- Variable selection (UVE noise generation, etc.)
- Model rankings (even for near-ties)

### ❌ NOT controlled (but doesn't affect results):
- Python hash randomization (dict ordering is stable since Python 3.7)
- Operating system thread scheduling (doesn't matter with n_jobs=1)
- GPU operations (none in this codebase)

---

## References

**Implementation Files:**
- `src/spectral_predict/reproducibility.py` - BLAS control utilities
- `src/spectral_predict/search.py` - Main search with reproducible mode
- `src/spectral_predict/scoring.py` - Stable ranking for near-ties
- `src/spectral_predict/variable_selection.py` - Seeded RNG for all methods
- `src/spectral_predict/models.py` - Configurable n_jobs for all models

**Tests:**
- `tests/test_reproducibility.py` - Comprehensive test suite
- `examples/reproducibility_demo.py` - Interactive demonstration

---

## Best Practices

1. **Use fast mode during development**
   - Iterate quickly on preprocessing and models
   - Explore different hyperparameters

2. **Switch to reproducible mode for final results**
   - Before writing your paper
   - Before submitting to a journal
   - Before sharing results with collaborators

3. **Document your random_state**
   - Include in methods section: "Analysis used random_state=42"
   - This allows others to reproduce your exact results

4. **Version control your code AND random_state**
   - Git commit your analysis script
   - Note the random_state used
   - Future you will thank present you!

5. **Test reproducibility before publication**
   - Run your analysis twice
   - Verify bit-identical results
   - This catches issues before peer review

---

## Summary

| Aspect | Fast Mode | Reproducible Mode |
|--------|-----------|-------------------|
| **Speed** | 100% (baseline) | ~20-33% (3-5x slower) |
| **Parallel CV** | ✅ Yes (n_jobs=-1) | ❌ No (n_jobs=1) |
| **BLAS threads** | All cores | 1 thread |
| **Model parallelism** | ✅ Yes (n_jobs=-1) | ❌ No (n_jobs=1) |
| **Reproducibility** | ❌ Not guaranteed | ✅ Bit-identical |
| **Use case** | Exploration | Publications |
| **Auto-restore** | N/A | ✅ Yes |

**Remember:** Toggle as often as you want - settings automatically restore after each run!

---

**Questions?** Check `examples/reproducibility_demo.py` for a working example.
