# Corrected Workflow - Phase 2 Progress Monitor

**Issue Fixed:** Progress monitor was appearing BEFORE the interactive GUI, skipping the data review step.

---

## ✅ Correct Workflow (Now Implemented)

### Step-by-Step Process:

```
1. User clicks "Run Analysis" in Main GUI
   ↓
2. Status: "Loading data..."
   - Loads spectral data (ASD or CSV)
   - Loads reference CSV
   - Aligns X and y
   ↓
3. IF "Show interactive data preview (GUI)" is CHECKED:
   ↓
   Status: "Opening interactive preview..."
   ↓
   ┌──────────────────────────────────────────────────┐
   │   INTERACTIVE DATA PREVIEW WINDOW OPENS          │
   │                                                  │
   │  • View raw spectra                              │
   │  • View 1st/2nd derivatives                      │
   │  • View correlation with target                  │
   │  • Convert to absorbance if needed               │
   │  • Review data statistics                        │
   │                                                  │
   │        [Continue to Model Search →]              │
   └──────────────────────────────────────────────────┘
   ↓
4. User clicks "Continue to Model Search →"
   ↓
5. Interactive GUI closes
   ↓
6. Status: "Starting model search..."
   ↓
7. IF "Show live progress monitor" is CHECKED:
   ↓
   ┌──────────────────────────────────────────────────┐
   │      PROGRESS MONITOR WINDOW OPENS               │
   │                                                  │
   │  Progress bar: [████████░░░░░] 42.5%            │
   │  Model 150 of 350                                │
   │  Elapsed: 00:05:32   ETA: 7m 15s                 │
   │                                                  │
   │  Current: Testing RandomForest with SNV...       │
   │                                                  │
   │  Best Model:                                     │
   │    Model: PLS                                    │
   │    RMSE: 0.0823 | R²: 0.9542                    │
   │                                                  │
   │        [Cancel]    [Minimize]                    │
   └──────────────────────────────────────────────────┘
   ↓
8. Models run in background thread
   - Progress monitor updates in real-time
   - Main GUI stays responsive
   ↓
9. Analysis complete!
   - Progress monitor shows "Complete"
   - Results saved to outputs/results.csv
   - Report saved to reports/*.md
```

---

## 🎯 Key Points

### Two Separate GUIs, Correct Order:

1. **Interactive Data Preview GUI** (FIRST)
   - Purpose: Review and transform data
   - When: Before model search
   - Control: Checkbox "Show interactive data preview (GUI)"
   - User action: Click "Continue to Model Search →"

2. **Progress Monitor GUI** (SECOND)
   - Purpose: Track model search progress
   - When: During model search
   - Control: Checkbox "Show live progress monitor"
   - User action: Watch progress, optionally cancel or minimize

### Both Can Be Disabled:

- ☐ "Show interactive data preview" → Skips data review, goes straight to model search
- ☐ "Show live progress monitor" → No progress window, runs in background like Phase 1

---

## 🔧 What Was Changed

### Before (WRONG):
```python
def _run_analysis_with_progress():
    # Created progress monitor immediately
    progress_monitor = ProgressMonitor()

    # Then tried to load data in background thread
    # This skipped the interactive GUI!
```

### After (CORRECT):
```python
def _run_analysis_with_progress():
    # STEP 1: Load data (main thread)
    X = load_spectral_data()
    y = load_reference()
    X_aligned, y_aligned = align_xy(X, y)

    # STEP 2: Show interactive GUI if enabled (main thread)
    if self.use_gui.get():
        result = run_interactive_loading_gui(X_aligned, y_aligned)
        if not result['user_continue']:
            return  # User cancelled
        X_aligned = result['X']  # Use potentially transformed data

    # STEP 3: Create progress monitor (main thread)
    progress_monitor = ProgressMonitor()

    # STEP 4: Run model search in background thread
    thread = Thread(target=lambda: run_search(...))
    thread.start()
```

---

## 📊 Visual Flow Diagram

```
Main GUI
  │
  ├─ User enters file paths
  ├─ User sets options:
  │   ☑ Show interactive data preview (GUI)
  │   ☑ Show live progress monitor
  │
  └─ User clicks "Run Analysis"
      │
      ├─ [Main Thread] Load data files
      │
      ├─ [Main Thread] Show Interactive Preview GUI
      │   │
      │   ├─ User reviews spectra
      │   ├─ User converts to absorbance (optional)
      │   ├─ User reviews correlations
      │   │
      │   └─ User clicks "Continue to Model Search"
      │
      ├─ Interactive GUI closes
      │
      ├─ [Main Thread] Create Progress Monitor
      │   │
      │   └─ Progress Monitor window opens
      │
      ├─ [Background Thread] Run model search
      │   │
      │   ├─ For each model:
      │   │   ├─ Train & evaluate
      │   │   └─ Call progress_callback()
      │   │       └─ Update Progress Monitor
      │   │
      │   └─ Save results
      │
      └─ [Main Thread] Show completion message
          │
          └─ Progress Monitor shows "Complete!"
```

---

## ✅ Testing Checklist

To verify the workflow is correct:

- [ ] Click "Run Analysis" with both checkboxes ON
  - [ ] Interactive preview opens FIRST
  - [ ] Can view spectra, derivatives, correlations
  - [ ] Can convert to absorbance
  - [ ] Click "Continue to Model Search"
  - [ ] Interactive GUI closes
  - [ ] Progress monitor opens SECOND
  - [ ] Models run, progress updates
  - [ ] Analysis completes

- [ ] Click "Run Analysis" with interactive preview OFF, progress monitor ON
  - [ ] No interactive GUI appears
  - [ ] Progress monitor opens immediately
  - [ ] Models run

- [ ] Click "Run Analysis" with both checkboxes OFF
  - [ ] No GUIs appear
  - [ ] Runs in background (like original)

---

## 🎓 Why This Matters

### Original Problem:
Users said: "The progress bar appeared in the wrong place. It should be AFTER the screen where I review the spectra and decide to transform to absorbance."

### Why It Was Wrong:
The code was:
1. Creating progress monitor
2. Loading data in background thread
3. Never showing interactive GUI

This meant users couldn't review their data before the analysis started!

### Why The Fix Works:
Now the flow is:
1. Load data (quick, main thread)
2. Show interactive GUI (user reviews and approves)
3. Show progress monitor (during actual model search)

This gives users full control over the workflow.

---

## 📝 Updated User Instructions

### How to Use (Correct Workflow):

1. **Launch GUI:**
   ```bash
   python spectral_predict_gui.py
   ```

2. **Select your data files:**
   - ASD directory OR CSV spectra file
   - Reference CSV
   - Column names

3. **Configure options:**
   - ☑ **"Show interactive data preview (GUI)"** - Check this to review your data first
   - ☑ **"Show live progress monitor"** - Check this to see real-time progress
   - Set CV folds, penalty, etc.

4. **Click "Run Analysis"**

5. **FIRST: Interactive Preview Window Opens**
   - **Review your spectra** in the "Raw Spectra" tab
   - **Check derivatives** if interested
   - **View correlations** with your target variable
   - **Convert to absorbance** if needed (click button)
   - When satisfied, click **"Continue to Model Search →"**

6. **SECOND: Progress Monitor Window Opens**
   - Watch models being tested in real-time
   - See best model found so far
   - Check ETA
   - Minimize if you want to multitask
   - Cancel if needed

7. **Analysis Completes**
   - Progress monitor shows "Complete!"
   - Results saved to `outputs/results.csv`
   - Report saved to `reports/YourTarget.md`

---

## 🔄 Comparison

### Phase 1 (Original):
```
Main GUI → Subprocess → Console output → Done
```
No visibility into progress, no data review.

### Phase 2 (Initial - WRONG):
```
Main GUI → Progress Monitor → (Interactive GUI skipped) → Done
```
Progress visible but data review skipped!

### Phase 2 (Fixed - CORRECT):
```
Main GUI → Interactive Preview → Progress Monitor → Done
```
Full workflow: Review data THEN track progress. ✅

---

## 🎉 Summary

**Issue:** Progress monitor appeared too early, skipping data review.

**Solution:** Reorganized workflow to:
1. Load data
2. Show interactive preview (if enabled)
3. Wait for user to click "Continue"
4. THEN show progress monitor during model search

**Result:** Users get the correct workflow they expected!

---

**Date:** October 27, 2025
**Status:** FIXED ✅
**Ready for Testing:** Yes
