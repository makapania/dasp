# Adaptive Performance Strategy: Graceful Degradation

**Date**: 2025-11-18
**Status**: 🎯 **CRITICAL INSIGHT** - User experience first!

---

## 💡 User's Key Insight

> "Would this also be easier for the user to state if they maybe don't have a great computer, so that they can just fall back on python as implemented now and which is easier on resources?"

**This is BRILLIANT!** We need to support ALL users, not just power users with GPUs.

---

## 🎯 Design Principle: Adaptive Performance

### The Problem with "GPU Required"
If we hard-code GPU or heavy parallelization:
- ❌ Laptop users (no GPU) can't run it
- ❌ Users with 2-4 cores get no benefit from 16-core optimizations
- ❌ Memory-constrained systems crash
- ❌ Complexity barrier for casual users

### The Right Approach: Graceful Degradation
**Auto-detect hardware and adapt**:
- ✅ GPU available? Use it (fast)
- ✅ No GPU? Fall back to CPU (still works)
- ✅ 16 cores? Use all of them (fast)
- ✅ 2 cores? Still parallel, just less speedup
- ✅ Limited RAM? Use sequential mode

---

## 🏗️ Architecture: Three Performance Tiers

```
┌─────────────────────────────────────────────────┐
│  TIER 3: POWER MODE (GPU + Full Parallel)      │
│  - GPU available                                 │
│  - 8+ CPU cores                                  │
│  - 16+ GB RAM                                    │
│  → 8-20x speedup                                 │
└─────────────────────────────────────────────────┘
                    ↓ (auto-fallback if no GPU)
┌─────────────────────────────────────────────────┐
│  TIER 2: PARALLEL MODE (CPU Multiprocessing)    │
│  - No GPU (or GPU disabled)                      │
│  - 4+ CPU cores                                  │
│  - 8+ GB RAM                                     │
│  → 4-8x speedup                                  │
└─────────────────────────────────────────────────┘
                    ↓ (auto-fallback if low cores/RAM)
┌─────────────────────────────────────────────────┐
│  TIER 1: STANDARD MODE (Current Python)         │
│  - Any hardware                                  │
│  - Works on laptops, old PCs                     │
│  - Low memory usage                              │
│  → Baseline (but reliable!)                      │
└─────────────────────────────────────────────────┘
```

**Key**: Same codebase, automatic adaptation!

---

## 💻 Implementation: Auto-Detection

### Step 1: Detect Hardware Capabilities

```python
import multiprocessing
import psutil

def detect_hardware_capabilities():
    """
    Detect available hardware and return optimal configuration.

    Returns
    -------
    config : dict
        Optimal settings for this hardware
    """
    config = {
        'tier': 1,  # Default: standard mode
        'gpu_available': False,
        'n_workers': 1,
        'use_parallel': False,
        'use_gpu': False,
        'memory_gb': 0
    }

    # Check CPU cores
    n_cores = multiprocessing.cpu_count()
    config['n_workers'] = max(1, n_cores - 1)  # Leave 1 core for OS

    # Check RAM
    config['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # Check for GPU (XGBoost)
    try:
        import xgboost as xgb
        # Try to create GPU model
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
        import numpy as np
        test_model.fit(np.random.randn(10, 10), np.random.randn(10))
        config['gpu_available'] = True
        config['use_gpu'] = True
        print("✓ GPU detected and working")
    except Exception as e:
        config['gpu_available'] = False
        config['use_gpu'] = False
        print("ℹ No GPU available (will use CPU)")

    # Determine tier based on capabilities
    if config['gpu_available'] and n_cores >= 8 and config['memory_gb'] >= 16:
        config['tier'] = 3  # Power mode
        config['use_parallel'] = True
        print(f"→ POWER MODE: GPU + {n_cores} cores")
    elif n_cores >= 4 and config['memory_gb'] >= 8:
        config['tier'] = 2  # Parallel mode
        config['use_parallel'] = True
        config['use_gpu'] = False  # No GPU or not enough resources
        print(f"→ PARALLEL MODE: {n_cores} cores (no GPU)")
    else:
        config['tier'] = 1  # Standard mode
        config['use_parallel'] = False
        config['use_gpu'] = False
        print(f"→ STANDARD MODE: {n_cores} cores (limited resources)")

    return config
```

---

### Step 2: Adaptive Model Training

```python
def train_model_adaptive(X, y, model_name, params, hw_config):
    """
    Train model using best available method for this hardware.

    Parameters
    ----------
    hw_config : dict
        Hardware configuration from detect_hardware_capabilities()
    """

    if model_name in ['XGBoost', 'LightGBM']:
        # Boosting models - GPU helps most

        if model_name == 'XGBoost':
            if hw_config['use_gpu']:
                # TIER 3: GPU mode
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                print("  [GPU] XGBoost training...")
            else:
                # TIER 1/2: CPU mode
                params['tree_method'] = 'hist'  # Still fast on CPU
                print("  [CPU] XGBoost training...")

            import xgboost as xgb
            model = xgb.XGBRegressor(**params)

        elif model_name == 'LightGBM':
            if hw_config['use_gpu']:
                # TIER 3: GPU mode
                params['device'] = 'gpu'
                params['gpu_platform_id'] = 0
                params['gpu_device_id'] = 0
                print("  [GPU] LightGBM training...")
            else:
                # TIER 1/2: CPU mode
                params['device'] = 'cpu'
                print("  [CPU] LightGBM training...")

            import lightgbm as lgb
            model = lgb.LGBMRegressor(**params)

    else:
        # PLS, Ridge, etc. - standard sklearn (already fast)
        from sklearn.linear_model import Ridge
        from sklearn.cross_decomposition import PLSRegression

        if model_name == 'Ridge':
            model = Ridge(**params)
        elif model_name == 'PLS':
            model = PLSRegression(**params)

        print(f"  [CPU] {model_name} training...")

    # Train
    model.fit(X, y)
    return model
```

---

### Step 3: Adaptive Grid Search

```python
def run_search_adaptive(X, y, models, preprocessing, varsel, hw_config):
    """
    Run grid search with adaptive parallelization.

    Automatically uses best strategy for available hardware.
    """

    # Generate all combinations
    combinations = [
        (m, p, v) for m in models
        for p in preprocessing
        for v in varsel
    ]

    print(f"\nGrid search: {len(combinations)} combinations")
    print(f"Hardware tier: {hw_config['tier']}")

    if hw_config['use_parallel'] and len(combinations) > 4:
        # TIER 2 or 3: Use multiprocessing
        print(f"Using parallel search ({hw_config['n_workers']} workers)")

        from multiprocessing import Pool
        from functools import partial

        train_func = partial(
            train_single_combination,
            X=X, y=y, hw_config=hw_config
        )

        with Pool(hw_config['n_workers']) as pool:
            results = pool.map(train_func, combinations)

    else:
        # TIER 1: Sequential (but still works!)
        print("Using sequential search (single core)")

        results = []
        for i, combo in enumerate(combinations):
            print(f"  [{i+1}/{len(combinations)}] {combo}")
            result = train_single_combination(combo, X, y, hw_config)
            results.append(result)

    return results
```

---

## 🎛️ User Interface: Performance Settings

### Option 1: Auto-Detect (Recommended Default)

```python
# In GUI or config
performance_mode = "auto"  # Let system decide

if performance_mode == "auto":
    hw_config = detect_hardware_capabilities()
else:
    hw_config = get_manual_config(performance_mode)
```

### Option 2: Manual Override

Add to GUI settings:

```
┌─────────────────────────────────────────┐
│  Performance Settings                   │
├─────────────────────────────────────────┤
│                                         │
│  ○ Auto (Recommended)                   │
│    → Detects and uses best settings    │
│                                         │
│  ○ Power (GPU + All Cores)              │
│    → Fastest, requires GPU              │
│                                         │
│  ○ Balanced (CPU Parallel)              │
│    → Fast, no GPU needed                │
│                                         │
│  ○ Standard (Single Core)               │
│    → Slower, works on any PC            │
│                                         │
│  Advanced:                              │
│  ☐ Use GPU (if available)               │
│  Workers: [8] cores                     │
│                                         │
└─────────────────────────────────────────┘
```

---

## 📊 User Experience by Hardware

### Scenario 1: Power User (You!)
- **Hardware**: Desktop, RTX 3080, 16 cores, 32GB RAM
- **Auto-detected**: Tier 3 (Power Mode)
- **Experience**: "Analysis completed in 22 minutes" (8-12x speedup)
- **GPU usage**: 90%+ during XGBoost/LightGBM

### Scenario 2: Office Desktop
- **Hardware**: Dell Optiplex, no GPU, 8 cores, 16GB RAM
- **Auto-detected**: Tier 2 (Parallel Mode)
- **Experience**: "Analysis completed in 1.2 hours" (4x speedup)
- **CPU usage**: Distributed across 8 cores

### Scenario 3: Laptop User
- **Hardware**: Laptop, no GPU, 4 cores, 8GB RAM
- **Auto-detected**: Tier 1 (Standard Mode) or Tier 2 (if 4 cores is enough)
- **Experience**: "Analysis completed in 3.5 hours" (baseline or 2-3x speedup)
- **Note**: Battery-friendly, doesn't overheat

### Scenario 4: Old PC
- **Hardware**: 2013 desktop, 2 cores, 4GB RAM
- **Auto-detected**: Tier 1 (Standard Mode)
- **Experience**: "Analysis completed in 4.5 hours" (same as now)
- **Important**: Still works! Doesn't crash!

---

## 🛡️ Safety Features

### Memory Protection

```python
def check_memory_safe(X, y, hw_config):
    """
    Check if dataset fits in available memory with safety margin.
    """
    import sys

    # Estimate memory needed
    data_size_gb = (X.nbytes + y.nbytes) / (1024**3)

    if hw_config['use_parallel']:
        # Parallel mode duplicates data across workers
        estimated_usage = data_size_gb * (hw_config['n_workers'] + 2)
    else:
        estimated_usage = data_size_gb * 2  # Safety margin

    available_gb = hw_config['memory_gb'] * 0.7  # Leave 30% for OS

    if estimated_usage > available_gb:
        print(f"⚠ Warning: Dataset ({data_size_gb:.1f}GB) may use too much memory")
        print(f"   Available: {available_gb:.1f}GB, Estimated: {estimated_usage:.1f}GB")
        print(f"   → Falling back to sequential mode")

        # Disable parallel
        hw_config['use_parallel'] = False
        hw_config['n_workers'] = 1
        hw_config['tier'] = 1

    return hw_config
```

### Graceful GPU Failures

```python
def train_with_gpu_fallback(X, y, model_params):
    """
    Try GPU first, fall back to CPU if fails.
    """
    try:
        # Try GPU
        model_params['tree_method'] = 'gpu_hist'
        model = xgb.XGBRegressor(**model_params)
        model.fit(X, y)
        return model, "GPU"

    except Exception as e:
        # GPU failed (OOM, driver issue, etc.)
        print(f"  GPU failed ({str(e)[:50]}...), using CPU")

        # Fall back to CPU
        model_params['tree_method'] = 'hist'
        model = xgb.XGBRegressor(**model_params)
        model.fit(X, y)
        return model, "CPU"
```

---

## 📈 Performance Expectations by Tier

| Hardware Tier | Example Hardware | Speedup | Typical Time |
|---------------|------------------|---------|--------------|
| **Tier 3** (Power) | Desktop + RTX 3080, 16 cores | 8-20x | 15-30 min |
| **Tier 2** (Parallel) | Desktop, no GPU, 8 cores | 4-8x | 30-75 min |
| **Tier 1** (Standard) | Laptop, 4 cores | 1-2x | 2-4 hours |
| **Minimal** | Old PC, 2 cores | 1x | 4-5 hours (baseline) |

**Key**: Everyone can run it, performance scales with hardware!

---

## 🎯 Implementation Priority

### Phase 1: Add Auto-Detection (1 day)
1. Write `detect_hardware_capabilities()`
2. Add to GUI initialization
3. Display detected tier in UI
4. Test on various machines

### Phase 2: Adaptive Grid Search (1 day)
1. Write `run_search_adaptive()`
2. Integrate with existing search.py
3. Add memory safety checks
4. Test sequential and parallel modes

### Phase 3: Adaptive Model Training (1 day)
1. Write `train_model_adaptive()`
2. Add GPU fallback logic
3. Integrate with grid search
4. Test GPU and CPU paths

### Phase 4: GUI Settings (1 day)
1. Add performance settings panel
2. Allow manual override
3. Save preferences
4. Show current mode during analysis

**Total**: 4 days to full adaptive implementation

---

## 💡 Additional Benefits

### For Development
- ✅ Can test on laptop (no GPU needed)
- ✅ CI/CD works (GitHub Actions = CPU only)
- ✅ Easier debugging (sequential mode)

### For Users
- ✅ Works out of the box (auto-detect)
- ✅ No configuration needed for beginners
- ✅ Power users can override if desired
- ✅ Clear feedback ("Running in Power Mode")

### For Distribution
- ✅ Single installer works for everyone
- ✅ No "GPU version" vs "CPU version"
- ✅ Future-proof (new hardware auto-detected)

---

## 🎓 Julia Fit in This Architecture

If you later add Julia for even more speed:

```python
def run_search_adaptive(X, y, models, preprocessing, varsel, hw_config):
    # Existing tiers...

    # TIER 4: JULIA MODE (optional, future)
    if hw_config.get('julia_available') and hw_config['tier'] >= 3:
        print("Using Julia backend (maximum performance)")
        from julia import Main as jl
        jl.include("julia_backend/search.jl")
        results = jl.run_search(X, y, ...)
        return results

    # Else: fall back to Python tiers 1-3
    ...
```

**Perfect**: Julia becomes another optional tier, not a requirement!

---

## 🎯 Revised Recommendation

### NEW Strategy: Adaptive Performance
1. ✅ Implement auto-detection (1 day)
2. ✅ Add adaptive grid search (1 day)
3. ✅ Add adaptive model training (1 day)
4. ✅ Add GUI settings (1 day)

**Result**:
- Power users get 8-20x speedup (GPU + parallel)
- Regular users get 4-8x speedup (parallel only)
- Laptop users get 1-2x speedup (or same, but still works!)
- **Everyone happy!**

### Julia Becomes Optional Future Enhancement
- Not needed now (Python adaptive solves it)
- Can add later as "Tier 4" for extreme performance
- No risk to existing users

---

## 📊 Cost-Benefit Analysis (Updated)

| Approach | User Coverage | Max Speedup | Effort | Risk |
|----------|---------------|-------------|--------|------|
| **Adaptive Python** | 100% | 8-20x | 4 days | Very Low ✅ |
| GPU-Required | 30% | 10-20x | 3 days | High ❌ |
| Julia Migration | 100%* | 20-50x | 12 weeks | Medium ❌ |

*Julia would need same adaptive logic

**Clear winner**: Adaptive Python!

---

## 🎯 Final Recommendation

**Implement Adaptive Performance in Python**:

```python
# User runs analysis
results = run_search(X, y, models, preprocessing, varsel)

# Behind the scenes:
# 1. Auto-detect hardware
# 2. Use GPU if available (8-20x faster)
# 3. Use parallel if enough cores (4-8x faster)
# 4. Use sequential if needed (still works!)
# 5. User doesn't need to configure anything!
```

**Benefits**:
- ✅ Works for all users (laptops to workstations)
- ✅ Automatic optimization (no user configuration)
- ✅ Graceful degradation (never crashes due to hardware)
- ✅ Fast to implement (4 days vs 12 weeks)
- ✅ Low risk (Python, tested libraries)

**This is the RIGHT solution!**

---

Should I implement the adaptive performance system?
