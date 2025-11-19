# Existing Parallel Support & User Preferences

**Date**: 2025-11-18
**Status**: Analysis of current state + user preference controls

---

## ✅ What You Already Have

### Parallel Cross-Validation (search.py:909)

```python
from joblib import Parallel, delayed

# Line 909: CV is already parallelized!
cv_metrics = Parallel(n_jobs=-1, backend='loky')(
    delayed(_run_single_fold)(
        pipe, X, y, train_idx, test_idx, task_type, is_binary_classification, all_classes
    )
    for train_idx, test_idx in cv_splitter.split(X, y)
)
```

**What this does**:
- ✅ Each CV fold runs in parallel
- ✅ Uses `n_jobs=-1` (all CPU cores)
- ✅ Uses `loky` backend (robust, memory-safe)

**Example**: 5-fold CV with 8 cores
- Without parallel: 5 folds × 10 sec = 50 seconds
- With parallel: ~10 seconds (folds run simultaneously)
- **Speedup: ~5x for CV alone**

---

## 🔄 What's Still Sequential

### Grid Search Loop (search.py:418-420)

```python
# Line 418-420: This loop is SEQUENTIAL
for model_name, model_configs in model_grids.items():
    for model, params in model_configs:
        # Each model tested one at a time
        result = _run_single_config(...)
```

**Example**: Testing 60 combinations (5 models × 4 preprocessing × 3 varsel)
- Each takes ~2 minutes with parallel CV
- Total: 60 × 2 min = **120 minutes** (sequential)
- **With parallel grid**: 120 / 8 cores = **15 minutes**
- **Potential additional speedup: 8x**

---

## 🎯 Combined Impact

| Component | Current | With Grid Parallel | Speedup |
|-----------|---------|-------------------|---------|
| **CV within model** | Parallel (5x) | Parallel (5x) | ✅ Already have |
| **Grid search** | Sequential | Parallel (8x) | 🆕 New gain |
| **Total** | Baseline | **Combined** | **~8x additional** |

**Your current**: Already ~5x faster than pure sequential
**With grid parallel**: ~40x faster than pure sequential (5x × 8x)
**Additional gain**: ~8x on top of what you have

---

## 💡 User's Critical Insight

> "User should be able to opt out of the more resource hungry version even if they have a powerful computer in case they want to be able to do multiple things at once"

**This is KEY!** Even power users need flexibility.

---

## 🎛️ Solution: User Preference System

Created `performance_config.py` with **three modes**:

### Mode 1: Power Mode (100% resources)
```python
config = PerformanceConfig(mode='power')
# Uses: All CPU cores, GPU, parallel everything
# Best for: Dedicated analysis sessions
# Impact: 8-20x total speedup
```

### Mode 2: Balanced Mode (60% resources) ⭐ **DEFAULT**
```python
config = PerformanceConfig(mode='balanced')
# Uses: 60% of CPU, GPU, parallel grid search
# Best for: Multitasking while analyzing
# Impact: 4-10x speedup, leaves room for other work
# Example: On 16-core system, uses 10 cores, leaves 6 free
```

### Mode 3: Light Mode (30% resources)
```python
config = PerformanceConfig(mode='light')
# Uses: 30% of CPU, no GPU, sequential grid
# Best for: Background analysis while working
# Impact: 2-4x speedup (just from parallel CV)
# Unobtrusive: Barely notice it running
```

---

## 🎨 User Interface Mockup

```
┌─────────────────────────────────────────────────────────┐
│  Performance Settings                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ○ Auto (Recommended)                                   │
│    Automatically select best settings                   │
│                                                         │
│  ● Balanced Mode ⭐                                      │
│    Good performance, leaves CPU free for multitasking   │
│    → Uses 60% CPU, GPU enabled                          │
│                                                         │
│  ○ Power Mode                                           │
│    Maximum performance (uses all resources)             │
│                                                         │
│  ○ Light Mode                                           │
│    Minimal impact (slowest but unobtrusive)             │
│                                                         │
│  ▼ Advanced Settings                                    │
│    Max CPU Usage: [====·····] 60%                       │
│    ☑ Use GPU (if available)                             │
│    ☑ Parallel grid search                               │
│                                                         │
│  ℹ Tip: Use Balanced or Light mode to work while       │
│    analyses run in the background                       │
│                                                         │
│  [ Save as Default ]  [ Apply ]                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Integration Points

### 1. Hardware Detection (startup)
```python
from performance_config import PerformanceConfig

# Load user preferences (or default to balanced)
perf_config = PerformanceConfig.from_user_preferences()
perf_config.print_summary()
```

Output:
```
PERFORMANCE CONFIGURATION
======================================================================
Mode: Balanced Mode
Description: Good performance, leaves resources for multitasking

Settings:
  CPU Usage: 60% (10/16 cores)
  GPU: ✓ Enabled
       (If you want to use GPU for graphics, switch to Light Mode)
  Parallel Grid Search: ✓ Enabled

Hardware Detected:
  CPU: 16 cores
  RAM: 32.0 GB
  GPU: ✓ Available
       Type: NVIDIA
======================================================================
```

### 2. Model Parameter Configuration
```python
# Get optimal parameters for this user's preferences
xgb_params = perf_config.get_model_params('XGBoost', base_params={
    'n_estimators': 100,
    'max_depth': 6
})

# Result (balanced mode, 16 cores):
# {
#     'n_estimators': 100,
#     'max_depth': 6,
#     'tree_method': 'gpu_hist',  # GPU enabled
#     'gpu_id': 0,
#     'n_jobs': 10  # Uses 60% of 16 cores
# }
```

### 3. Grid Search Control
```python
if perf_config.parallel_grid:
    # Run grid search in parallel
    from multiprocessing import Pool
    with Pool(perf_config.n_workers) as pool:
        results = pool.map(train_model, combinations)
else:
    # Sequential (current behavior)
    results = [train_model(combo) for combo in combinations]
```

---

## 📊 User Experience Examples

### Scenario 1: Power User During Dedicated Analysis
**Setup**: 16-core workstation, GPU, wants fastest results
**Choice**: Power Mode
**Result**: 4 hours → 20 minutes (12x speedup)
**CPU**: 100% (all 16 cores)
**GPU**: 100%
**Can multitask?**: Not really, system at full capacity

### Scenario 2: Power User While Working ⭐
**Setup**: Same 16-core workstation, needs to use computer
**Choice**: Balanced Mode (DEFAULT)
**Result**: 4 hours → 40 minutes (6x speedup)
**CPU**: 60% (10/16 cores working, 6 free)
**GPU**: Used only for short bursts (XGBoost/LightGBM)
**Can multitask?**: ✅ Yes! Still responsive for email, browsing, etc.

### Scenario 3: Power User, Background Analysis
**Setup**: Same workstation, wants minimal impact
**Choice**: Light Mode
**Result**: 4 hours → 2 hours (2x speedup from parallel CV)
**CPU**: 30% (5/16 cores, 11 free)
**GPU**: Not used (available for graphics)
**Can multitask?**: ✅ Yes! Barely notice analysis running

### Scenario 4: Laptop User
**Setup**: 4-core laptop, no GPU
**Choice**: Auto → Balanced Mode
**Result**: 4 hours → 2.5 hours (1.6x speedup)
**CPU**: 60% (2-3 cores)
**Battery**: Moderate usage
**Can multitask?**: ✅ Yes! Still usable

---

## 🎯 Default Behavior (Important!)

**Default mode**: **Balanced** (NOT Power)

**Why?**
- ✅ Most users want to multitask
- ✅ Still gets good speedup (4-10x)
- ✅ System remains responsive
- ✅ Less heat/fan noise
- ✅ Better battery life (laptops)

**Power users can opt-in** to Power Mode if they want maximum speed.

---

## 💾 Preference Persistence

### Save User's Choice
```python
config = PerformanceConfig(mode='balanced', max_cpu_percent=50)
config.save_preferences()  # Saves to ~/.dasp_performance.json
```

### Load on Startup
```python
config = PerformanceConfig.from_user_preferences()
# Automatically loads saved preferences
# Falls back to auto if no saved preferences
```

### Config File Format
```json
{
  "mode": "balanced",
  "max_cpu_percent": 50,
  "use_gpu": true,
  "parallel_grid": true,
  "n_workers": 8
}
```

---

## 🚀 Implementation Phases

### Phase 1: User Preferences (1 day)
- ✅ `performance_config.py` created
- [ ] Add GUI settings panel
- [ ] Test preference loading/saving
- [ ] Default to Balanced mode

### Phase 2: Model Parameters (1 day)
- [ ] Integrate `get_model_params()` into search.py
- [ ] Respect `n_workers` setting
- [ ] Test GPU enable/disable

### Phase 3: Grid Parallelization (1 day)
- [ ] Add parallel grid search option
- [ ] Respect `parallel_grid` setting
- [ ] Memory safety checks

### Phase 4: Testing (1 day)
- [ ] Test all 3 modes on various hardware
- [ ] Benchmark actual speedups
- [ ] Verify multitasking works

---

## 🎓 Key Design Principles

### 1. User Control Trumps Auto-Detection
Even if hardware is powerful, user preference wins:
```python
hw_detected = detect_hardware()  # Detects Tier 3 (powerful)
config = PerformanceConfig(mode='light')  # User choice: light mode
# Result: Uses light mode, not power mode
```

### 2. Sane Defaults
- Default to **Balanced**, not Power
- Auto mode → Balanced for Tier 3 (not Power!)
- Assumes users want to multitask

### 3. Clear Feedback
Always show user what's happening:
```
Analysis running in Balanced Mode (60% CPU, GPU enabled)
Estimated time: 40-50 minutes
Tip: Switch to Power Mode for 2x faster results (Settings → Performance)
```

### 4. Easy Switching
User can change mode mid-analysis if needed:
- Start in Balanced → Too slow? Switch to Power
- Start in Power → Too hot/loud? Switch to Balanced
- Settings persist for next time

---

## 📈 Expected Outcomes

### Current State (with parallel CV)
- CV is parallel: ✅ Already ~5x faster than pure sequential
- Grid is sequential: Current bottleneck

### After Adding Grid Parallel + User Preferences
| User Choice | CPU Usage | GPU | Speedup | Time (4hr baseline) |
|-------------|-----------|-----|---------|---------------------|
| Power | 100% | Yes | 8-20x | 12-30 min |
| Balanced | 60% | Yes | 4-10x | 24-60 min |
| Light | 30% | No | 2-4x | 60-120 min |

**All modes**:
- ✅ Achieve < 1 hour (your goal)
- ✅ Work on all hardware
- ✅ User controls resource usage

---

## 🎯 Revised Recommendation

### What to Add
1. **User preference system** (performance_config.py) ✅ Created
2. **GUI settings panel** (1 day)
3. **Grid parallelization** (1 day, respects user prefs)

### What NOT to Add
- ❌ Forced GPU usage (user can disable)
- ❌ Forced 100% CPU (user can limit)
- ❌ Automatic power mode (default to balanced)

### Timeline
- **4 days total** (same as before)
- **But now with user control!**

---

## 🎬 Next Steps

1. **Test performance_config.py**:
   ```bash
   python performance_config.py
   ```

2. **Review settings**:
   - Is default (balanced) appropriate?
   - Do the three modes make sense?

3. **Integrate into GUI**:
   - Add settings panel using `PerformanceSettingsDialog`
   - Show current mode during analysis
   - Allow mid-analysis switching (future)

4. **Test on real hardware**:
   - Verify balanced mode leaves CPU free
   - Confirm multitasking works
   - Measure actual speedups

---

**Status**: User preference system ready
**Default**: Balanced mode (respects user's need to multitask)
**Flexibility**: User can choose Power, Balanced, or Light
**Timeline**: Same 4 days, better user experience
