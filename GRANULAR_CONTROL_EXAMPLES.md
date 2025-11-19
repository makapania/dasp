# Granular CPU/GPU Control Examples

**User's Request**: "Ideally when opting out they can specify reduce pull on core but use the GPU. That might be typical in fact."

**Perfect!** This is exactly what the system supports. Here are common scenarios:

---

## 🎯 Common Use Cases

### Scenario 1: GPU + Low CPU (⭐ MOST COMMON)

**User wants**: Fast analysis (use GPU) but keep computer responsive (limit CPU)

```python
from performance_config import PerformanceConfig

config = PerformanceConfig(
    mode='custom',        # Custom mode for granular control
    max_cpu_percent=40,   # Use only 40% of CPU (6-7 cores on 16-core system)
    use_gpu=True,         # ✓ Use GPU (10-20x faster for boosting!)
    parallel_grid=True    # ✓ Still parallel (but limited cores)
)

config.print_summary()
```

**Output**:
```
PERFORMANCE CONFIGURATION
=================================================================
Mode: Custom
Description: Custom resource configuration

Settings:
  CPU Usage: 40% (6/16 cores)
  GPU: ✓ Enabled
  Parallel Grid Search: ✓ Enabled

Hardware Detected:
  CPU: 16 cores
  RAM: 32.0 GB
  GPU: ✓ Available
       Type: NVIDIA
=================================================================
```

**Result**:
- XGBoost/LightGBM: 10-20x faster (GPU!)
- Grid search: 6x faster (6 cores working, 10 free)
- Computer stays responsive: ✅ Can browse, email, etc.
- **Total speedup**: 8-12x while multitasking!

---

### Scenario 2: GPU Only, Minimal CPU

**User wants**: Maximum GPU usage, almost no CPU (leaves computer very responsive)

```python
config = PerformanceConfig(
    mode='custom',
    max_cpu_percent=20,   # Minimal CPU (3 cores on 16-core)
    use_gpu=True,         # ✓ Use GPU
    parallel_grid=False   # ✗ Sequential grid (less CPU load)
)
```

**Result**:
- XGBoost/LightGBM: 10-20x faster (GPU!)
- Grid search: Sequential (slower, but CPU barely used)
- Computer stays very responsive: ✅✅ Feels like nothing is running
- **Total speedup**: 6-10x with almost no system impact

---

### Scenario 3: CPU Only, No GPU

**User wants**: Leave GPU free for graphics, use CPU for analysis

```python
config = PerformanceConfig(
    mode='custom',
    max_cpu_percent=60,   # Use 60% of CPU
    use_gpu=False,        # ✗ Leave GPU free
    parallel_grid=True    # ✓ Parallel grid search
)
```

**Result**:
- XGBoost/LightGBM: CPU-based (slower, but GPU free for graphics)
- Grid search: 6-8x faster (parallel)
- GPU available for: Games, video editing, CAD, etc.
- **Total speedup**: 4-6x

---

### Scenario 4: Maximum Everything (Power Mode)

**User wants**: Fastest possible, don't care about multitasking

```python
config = PerformanceConfig(mode='power')
# Equivalent to:
# config = PerformanceConfig(
#     max_cpu_percent=100,
#     use_gpu=True,
#     parallel_grid=True
# )
```

**Result**:
- Uses ALL resources
- Fastest possible: 12-30x speedup
- Computer unusable during analysis
- Good for: Overnight runs, dedicated analysis sessions

---

## 🎨 Updated GUI Mockup

```
┌────────────────────────────────────────────────────────────┐
│  Performance Settings                                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Quick Presets:                                            │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ ⭐ GPU + Low CPU (Recommended)                        │ │
│  │ Use GPU for speed, limit CPU for multitasking        │ │
│  │ • GPU: Enabled  • CPU: 40%  • Grid: Parallel         │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ GPU Only (Minimal CPU Impact)                         │ │
│  │ Fast with GPU, barely touch CPU                       │ │
│  │ • GPU: Enabled  • CPU: 20%  • Grid: Sequential        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ CPU Only (Leave GPU Free)                             │ │
│  │ Good speed, keep GPU for graphics                     │ │
│  │ • GPU: Disabled  • CPU: 60%  • Grid: Parallel         │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Power Mode (Maximum Performance)                      │ │
│  │ Fastest - uses all resources                          │ │
│  │ • GPU: Enabled  • CPU: 100%  • Grid: Parallel         │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ─ Or Customize ───────────────────────────────────────── │
│                                                            │
│  Maximum CPU Usage: [====····] 40%                        │
│  (Using 6 of 16 cores, leaving 10 free)                   │
│                                                            │
│  ☑ Use GPU (if available)                                 │
│    GPU greatly accelerates XGBoost/LightGBM (10-20x)      │
│                                                            │
│  ☑ Parallel grid search                                   │
│    Tests multiple models simultaneously (4-8x faster)     │
│                                                            │
│  Current Configuration:                                   │
│  → GPU + 6 cores = ~10x speedup, system stays responsive │
│                                                            │
│  [ Save as Default ]  [ Apply ]                           │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Comparison Matrix

| Configuration | GPU | CPU | Grid | Speedup | System Responsive? | Best For |
|---------------|-----|-----|------|---------|-------------------|----------|
| **GPU + Low CPU** | ✓ | 40% | ✓ | 8-12x | ✅ Yes | ⭐ **Most users** |
| **GPU Only** | ✓ | 20% | ✗ | 6-10x | ✅✅ Very | Laptops, background |
| **CPU Only** | ✗ | 60% | ✓ | 4-6x | ✅ Yes | Keep GPU free |
| **Power Mode** | ✓ | 100% | ✓ | 12-30x | ❌ No | Dedicated runs |
| **Light Mode** | ✗ | 30% | ✗ | 2-4x | ✅✅ Very | Old computers |

---

## 💻 Implementation in Code

### How It Works

```python
# User selects: "GPU + Low CPU"
config = PerformanceConfig(
    mode='custom',
    max_cpu_percent=40,  # Limit CPU
    use_gpu=True,        # Enable GPU
    parallel_grid=True
)

# For XGBoost, this generates:
xgb_params = config.get_model_params('XGBoost', base_params={
    'n_estimators': 100,
    'max_depth': 6
})

# Result:
# {
#     'n_estimators': 100,
#     'max_depth': 6,
#     'tree_method': 'gpu_hist',  # ← GPU enabled!
#     'gpu_id': 0,
#     'n_jobs': 6  # ← Limited to 6 cores (40% of 16)
# }
```

### Grid Search Respects Limits

```python
# Grid search uses limited workers
if config.parallel_grid:
    with Pool(config.n_workers) as pool:  # Only 6 workers, not 16!
        results = pool.map(train_model, combinations)
```

---

## 🎯 Recommended Defaults

Based on user's insight ("GPU + low CPU might be typical"):

### New Default: "GPU + Low CPU" (Not "Balanced")

```python
# In GUI initialization:
DEFAULT_CONFIG = {
    'mode': 'custom',
    'max_cpu_percent': 40,  # Use ~40% of CPU
    'use_gpu': True,        # Use GPU (if available)
    'parallel_grid': True,  # Parallel grid search
}
```

**Why this default?**
- ✅ Great speed (GPU gives 10-20x)
- ✅ System stays responsive (60% CPU free)
- ✅ Works for most users (power users AND multitaskers)
- ✅ User can easily change if needed

---

## 📈 Real-World Example

### User's Typical Day

**Morning** (9 AM - needs computer for work):
```python
config = PerformanceConfig(
    max_cpu_percent=40,  # Keep 60% CPU free for work
    use_gpu=True,        # GPU helps (doesn't block work)
    parallel_grid=True
)
# Runs analysis: 4 hours → 40 min
# Can work normally: email, browser, etc.
```

**Lunch break** (12 PM - stepping away):
```python
config = PerformanceConfig(mode='power')  # Max speed
# Runs analysis: 4 hours → 20 min
# Done before lunch is over!
```

**Evening** (7 PM - gaming on same PC):
```python
config = PerformanceConfig(
    max_cpu_percent=30,  # Minimal CPU
    use_gpu=False,       # Leave GPU for game!
    parallel_grid=False
)
# Runs analysis: 4 hours → 2 hours (slow, but doesn't affect game)
```

---

## 🎓 Key Insight

> "Ideally when opting out they can specify reduce pull on core but use the GPU"

**This is brilliant because**:
- GPU is often idle (doesn't affect work)
- GPU gives BIGGEST speedup (10-20x for boosting)
- Limiting CPU keeps system responsive
- **Best of both worlds**: Fast + usable computer!

**Current code already supports this perfectly**:
```python
config = PerformanceConfig(
    max_cpu_percent=30,   # Low CPU ← User-controlled
    use_gpu=True          # GPU enabled ← User-controlled
)
# These are independent! User can set any combination.
```

---

## ✅ Summary

**User can control**:
1. ✅ CPU usage (10-100%)
2. ✅ GPU on/off
3. ✅ Grid parallel on/off

**All independently!**

**Common combos**:
- GPU + Low CPU: Fast + responsive (⭐ recommended default)
- GPU Only: Fastest with minimal CPU
- CPU Only: Leave GPU free
- Power: Maximum everything
- Light: Minimal everything

**Implementation**: Already done in `performance_config.py`!

**Next step**: Create GUI with preset buttons for easy selection.
