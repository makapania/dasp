# Subset Analysis Debug Guide

**Date:** 2025-10-29
**Issue:** Subset ranges selected but not running
**Status:** Debug logging added

---

## What Was Added

I've added comprehensive debug logging to `src/spectral_predict/search.py` to help diagnose why subset analysis isn't running when selected.

### Debug Output Locations

#### 1. Initial Configuration Display (Line 156-162)
When analysis starts, you'll now see:
```
Running regression search with 5-fold CV...
Models: ['PLS', 'RandomForest', 'MLP', 'NeuralBoosted']
Preprocessing configs: 4
Enable variable subsets: True          ← CHECK THIS
Variable counts: [10, 20, 50, 100]     ← CHECK THIS
Enable region subsets: True
```

**What to look for:**
- `Enable variable subsets:` should be `True` if you checked the box
- `Variable counts:` should show the numbers you selected (e.g., `[10, 20, 50, 100, 250]`)
- If `Variable counts:` is `None` or `[]`, no subsets will run

#### 2. Per-Model Subset Analysis (Line 265-268)
For each model that supports subsets, you'll see either:

**If subsets are enabled:**
```
  → Computing feature importances for PLS subset analysis...
```

**If subsets are disabled:**
```
  ⊗ Skipping subset analysis for PLS (variable subsets disabled)
```

#### 3. Variable Count Validation (Line 312-316)
When computing subsets, you'll see:
```
  → User variable counts: [10, 20, 50, 100, 250]
  → Valid variable counts (< 2000 features): [10, 20, 50, 100, 250]
```

**What to look for:**
- If "Valid variable counts" is empty `[]`, it means all your selected counts are >= the number of features
- Example: If you have 50 wavelengths but selected counts of [100, 250, 500], all will be filtered out

**Warning message if all counts invalid:**
```
  ⚠ Warning: No valid variable counts to test (all selected counts >= 50 features)
```

#### 4. Individual Subset Tests (Line 320)
For each valid count, you'll see:
```
  → Testing top-10 variable subset...
  → Testing top-20 variable subset...
  → Testing top-50 variable subset...
```

#### 5. Region-Based Subsets (Line 346)
If region subsets are enabled:
```
  → Testing 5 region-based subsets...
```

---

## Common Issues and Solutions

### Issue 1: "Enable variable subsets: False"
**Problem:** The checkbox is not checked or the value isn't being passed correctly.

**Solution:**
1. In the GUI, go to Tab 2: Analysis Configuration
2. Scroll to "Subset Analysis" section
3. Verify "✓ Enable Top-N Variable Analysis" is CHECKED
4. Try running a minimal test with just one model

### Issue 2: "Variable counts: None" or "Variable counts: []"
**Problem:** No variable counts are selected, or they're not being passed from GUI.

**Solutions:**

**A. Nothing selected in GUI:**
1. Go to Tab 2: Analysis Configuration
2. Scroll to "Top-N Variable Counts"
3. Check at least one box (e.g., "N=10 ⭐")

**B. GUI not passing values (bug):**
Check the GUI code in `spectral_predict_gui_optimized.py` around line 893-908:
```python
# Collect top-N variable counts
variable_counts = []
if self.var_10.get():
    variable_counts.append(10)
if self.var_20.get():
    variable_counts.append(20)
# ... etc
```

Add debug print BEFORE calling run_search:
```python
print(f"DEBUG: variable_counts = {variable_counts}")
print(f"DEBUG: enable_variable_subsets = {enable_variable_subsets}")
```

### Issue 3: "Valid variable counts (< N features): []"
**Problem:** All selected counts are larger than your number of wavelengths.

**Example:**
- You have 50 wavelengths after filtering
- You selected counts: [100, 250, 500, 1000]
- All counts > 50, so all are filtered out

**Solution:**
1. Check how many wavelengths you have (shown in Tab 1 status)
2. Select counts LESS than your wavelength count
3. For small datasets, use only: N=10, N=20

### Issue 4: Model doesn't support subsets
**Problem:** Subset analysis only works for specific models.

**Supported models:**
- ✅ PLS
- ✅ PLS-DA
- ✅ RandomForest
- ✅ MLP
- ✅ NeuralBoosted

**Not supported:**
- ❌ Other models (if you add custom ones)

**Solution:**
Make sure you have at least one of the supported models selected.

### Issue 5: "Skipping subset analysis (variable subsets disabled)"
**Problem:** You see this message even though you checked the box.

**Debug steps:**
1. Add print statement in GUI before calling run_search:
```python
print(f"GUI: enable_variable_subsets = {self.enable_variable_subsets.get()}")
```

2. Check if it prints `True`
3. If it prints `False`, the checkbox isn't working
4. If it prints `True` but search.py shows `False`, there's a parameter passing issue

---

## How to Debug Your Specific Issue

### Step 1: Run Analysis and Capture Output
1. Run your analysis
2. Look at the Progress tab (Tab 3)
3. Look for the debug messages listed above
4. Copy the relevant output

### Step 2: Identify the Problem
Match what you see to the patterns above:

**Pattern A: Subsets not enabled**
```
Enable variable subsets: False
  ⊗ Skipping subset analysis for PLS (variable subsets disabled)
```
→ Go to Solution for Issue 1

**Pattern B: No counts selected**
```
Enable variable subsets: True
Variable counts: None
```
→ Go to Solution for Issue 2

**Pattern C: Counts too large**
```
Variable counts: [100, 250, 500]
Valid variable counts (< 50 features): []
⚠ Warning: No valid variable counts to test (all selected counts >= 50 features)
```
→ Go to Solution for Issue 3

**Pattern D: Wrong models**
```
Models: ['SomeOtherModel']
```
→ Go to Solution for Issue 4

### Step 3: Apply Fix and Re-run
After applying the fix, re-run and verify you see:
```
Enable variable subsets: True
Variable counts: [10, 20, 50]
  → Computing feature importances for PLS subset analysis...
  → User variable counts: [10, 20, 50]
  → Valid variable counts (< 2000 features): [10, 20, 50]
  → Testing top-10 variable subset...
  → Testing top-20 variable subset...
  → Testing top-50 variable subset...
```

---

## Expected Output for Working Subset Analysis

Here's what you should see if everything is working correctly:

```
Running regression search with 5-fold CV...
Models: ['PLS', 'RandomForest', 'MLP', 'NeuralBoosted']
Preprocessing configs: 2
Enable variable subsets: True
Variable counts: [10, 20, 50, 100, 250]
Enable region subsets: True

[1/8] Testing PLS with raw preprocessing
  → Computing feature importances for PLS subset analysis...
  → User variable counts: [10, 20, 50, 100, 250]
  → Valid variable counts (< 2000 features): [10, 20, 50, 100, 250]
  → Testing top-10 variable subset...
  → Testing top-20 variable subset...
  → Testing top-50 variable subset...
  → Testing top-100 variable subset...
  → Testing top-250 variable subset...
  → Testing 5 region-based subsets...

[2/8] Testing PLS with snv preprocessing
  → Computing feature importances for PLS subset analysis...
  → User variable counts: [10, 20, 50, 100, 250]
  → Valid variable counts (< 2000 features): [10, 20, 50, 100, 250]
  → Testing top-10 variable subset...
  [... etc ...]
```

---

## Still Not Working?

If you've checked everything above and subsets still aren't running, here's what to check next:

### Add Diagnostic Prints to GUI

Edit `spectral_predict_gui_optimized.py`, find the `_run_analysis_thread` method (around line 866), and add these prints:

```python
def _run_analysis_thread(self, selected_models):
    try:
        from spectral_predict.search import run_search
        from spectral_predict.report import write_markdown_report

        # ... existing code ...

        # Collect subset analysis settings
        enable_variable_subsets = self.enable_variable_subsets.get()
        enable_region_subsets = self.enable_region_subsets.get()

        # Collect top-N variable counts
        variable_counts = []
        if self.var_10.get():
            variable_counts.append(10)
        if self.var_20.get():
            variable_counts.append(20)
        if self.var_50.get():
            variable_counts.append(50)
        if self.var_100.get():
            variable_counts.append(100)
        if self.var_250.get():
            variable_counts.append(250)
        if self.var_500.get():
            variable_counts.append(500)
        if self.var_1000.get():
            variable_counts.append(1000)

        # ADD THESE DEBUG PRINTS
        print("=" * 60)
        print("GUI DEBUG: Parameters being passed to run_search:")
        print(f"  enable_variable_subsets: {enable_variable_subsets} (type: {type(enable_variable_subsets)})")
        print(f"  variable_counts: {variable_counts} (type: {type(variable_counts)})")
        print(f"  enable_region_subsets: {enable_region_subsets} (type: {type(enable_region_subsets)})")
        print("=" * 60)

        # ... rest of the code ...
```

This will show you EXACTLY what values are being passed from the GUI.

### Check for Multiple Versions

Make sure you're running the updated code:
```bash
cd C:\Users\sponheim\git\dasp
python -c "import src.spectral_predict.search as s; print(s.__file__)"
```

This will show you which version of search.py is being loaded.

### Check Python Cache

Sometimes Python caches old versions. Clear the cache:
```bash
cd C:\Users\sponheim\git\dasp
find . -type d -name "__pycache__" -exec rm -rf {} +
# or on Windows:
del /s /q __pycache__
```

Then restart the GUI and try again.

---

## Summary Checklist

Before running your next analysis, verify:

- [ ] "Enable Top-N Variable Analysis" checkbox is CHECKED
- [ ] At least one N value is checked (10, 20, 50, 100, 250, 500, or 1000)
- [ ] At least one model selected that supports subsets (PLS, RF, MLP, or NeuralBoosted)
- [ ] Selected N values are LESS than your number of wavelengths
- [ ] You're running the updated code with debug logging

After running, check the output for:
- [ ] `Enable variable subsets: True`
- [ ] `Variable counts:` shows your selected values
- [ ] `Valid variable counts:` is not empty
- [ ] You see "→ Testing top-N variable subset..." messages

---

## Contact/Next Steps

If you've gone through all of the above and subsets still aren't running, please provide:

1. **Screenshot of Tab 2** showing your subset analysis settings
2. **Full console output** from the start of the analysis (showing the debug messages)
3. **Number of wavelengths** in your dataset (from Tab 1 status)
4. **Models selected** for analysis

This will help pinpoint the exact issue.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Related Files:** `src/spectral_predict/search.py`, `spectral_predict_gui_optimized.py`
