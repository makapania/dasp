# AGENT 5: Diagnostic Plot Suite - Comprehensive Testing Guide

**Date:** 2025-11-07
**Purpose:** Complete testing checklist for Tab 7 diagnostic plots
**Prerequisites:** Implementation code applied to `spectral_predict_gui_optimized.py`

---

## Overview

This testing guide covers:
1. **UI Verification** - Layout, responsiveness, visual consistency
2. **Functional Testing** - Plot generation for different scenarios
3. **Edge Cases** - Unusual inputs, error handling
4. **Performance Testing** - Speed, memory usage
5. **Integration Testing** - Interaction with other tabs
6. **User Experience** - Real-world workflows

---

## Pre-Testing Setup

### Required Test Data

1. **Validation Set:**
   - Go to Tab 1 (Data Upload)
   - Load spectral data (ASD/SPC directory or CSV)
   - Create validation set (e.g., 20% with Kennard-Stone)
   - Note validation set size for reference

2. **Saved Models:**
   - Have 3-5 saved .dasp model files ready
   - Mix of model types (PLS, Ridge, RandomForest, etc.)
   - Mix of preprocessing methods (raw, SNV, SG1, etc.)
   - Ensure models were trained on same dataset

3. **New Data (Optional):**
   - CSV file with spectral data (no y values)
   - ASD directory with new spectra

---

## Test Suite 1: UI Verification

### Test 1.1: Layout Check

**Steps:**
1. Launch GUI and navigate to Tab 7
2. Scroll to Step 5 (Diagnostic Plots section)

**Verify:**
- [ ] Section title: "Step 5: Diagnostic Plots (Validation Set)"
- [ ] Info text: "Diagnostic plots are shown when using validation set..."
- [ ] Three plot frames visible side-by-side
- [ ] Frame labels: "Predictions", "Residuals", "Model Comparison"
- [ ] Placeholder text visible in all frames
- [ ] No layout overlaps or cut-off elements

**Expected Result:** Clean, professional layout with placeholders

---

### Test 1.2: Responsive Layout

**Steps:**
1. Resize main window to various sizes:
   - Maximize window
   - Half-screen width
   - Minimum allowed size

**Verify:**
- [ ] Plot frames resize proportionally
- [ ] No overlap between frames
- [ ] Scrollbar appears when needed
- [ ] Text remains readable at all sizes
- [ ] Plots don't become distorted

**Expected Result:** Layout adapts gracefully to window size

---

### Test 1.3: Visual Consistency

**Steps:**
1. Compare Tab 7 plot styling with Tab 6 plots
2. Check font sizes, colors, borders

**Verify:**
- [ ] Similar plot styling (colors, fonts, grids)
- [ ] Consistent LabelFrame appearance
- [ ] Matching caption text style
- [ ] Professional, cohesive look

**Expected Result:** Visual harmony across tabs

---

## Test Suite 2: Functional Testing

### Test 2.1: Validation Set - Single Model

**Setup:**
- Load 1 model
- Load validation set
- Run predictions

**Verify Plot 1 (Predictions):**
- [ ] Scatter plot displays (observed vs predicted)
- [ ] 1:1 red dashed line present
- [ ] Statistics box shows: R², RMSE, MAE, Bias, n
- [ ] Values are reasonable (R² between -1 and 1, n matches validation size)
- [ ] Title shows model name
- [ ] Axis labels clear ("Observed Values", "Predicted Values")
- [ ] Grid present and subtle
- [ ] Legend shows "1:1 Line"

**Verify Plot 2 (Residuals):**
- [ ] Four subplots present (2x2 grid)
- [ ] Top-left: Residuals vs Fitted (scatter + zero line)
- [ ] Top-right: Residuals vs Index (scatter + zero line + large residuals highlighted)
- [ ] Bottom-left: Q-Q Plot (scatter + reference line)
- [ ] Bottom-right: Histogram (bars + zero line)
- [ ] All plots have titles, labels, grids
- [ ] Orange trend line visible in Residuals vs Fitted
- [ ] Red X markers for large residuals in Residuals vs Index

**Verify Plot 3 (Model Comparison):**
- [ ] Placeholder message: "Load multiple models for comparison"
- [ ] Message is centered and readable

**Expected Result:** Plots 1 & 2 show detailed diagnostics, Plot 3 shows placeholder

---

### Test 2.2: Validation Set - Multiple Models

**Setup:**
- Load 3+ models
- Load validation set
- Run predictions

**Verify Plot 1 & 2:**
- [ ] Same as Test 2.1 (using first model)
- [ ] First model name in title

**Verify Plot 3 (Model Comparison):**
- [ ] Bar chart displays
- [ ] One bar per model
- [ ] Bars sorted by R² (best first)
- [ ] R² value labeled on top of each bar
- [ ] RMSE value inside each bar (if space)
- [ ] Bars color-coded:
  - Green: top tier (within 5% of best)
  - Yellow: good (within 15% of best)
  - Red: needs improvement (<85% of best)
- [ ] X-axis labels show model names (rotated 45°)
- [ ] Y-axis labeled "R² Score"
- [ ] Title: "Model Comparison (Validation Set Performance)"
- [ ] Legend explains color coding
- [ ] Grid on y-axis only

**Expected Result:** All 3 plots show meaningful diagnostics

---

### Test 2.3: New Data (No Validation)

**Setup:**
- Load models
- Load CSV or directory data (NOT validation set)
- Run predictions

**Verify:**
- [ ] All 3 plots show placeholder:
  - "Diagnostic plots available only for validation set"
- [ ] Placeholder text is centered
- [ ] Placeholder text is gray/muted color
- [ ] No error messages

**Expected Result:** Graceful handling - placeholders instead of plots

---

### Test 2.4: No Models Loaded

**Setup:**
- Clear all models
- Attempt to run predictions

**Verify:**
- [ ] Error dialog: "Please load at least one model first"
- [ ] Plots remain cleared or show placeholders
- [ ] No crash or traceback

**Expected Result:** User-friendly error message

---

### Test 2.5: No Data Loaded

**Setup:**
- Load models
- Clear data
- Attempt to run predictions

**Verify:**
- [ ] Error dialog: "Please load prediction data first"
- [ ] Plots remain cleared or show placeholders
- [ ] No crash or traceback

**Expected Result:** User-friendly error message

---

## Test Suite 3: Edge Cases

### Test 3.1: Perfect Predictions (R² = 1.0)

**Setup:**
- Create a model that predicts perfectly
- OR modify predictions_df manually for testing

**Verify:**
- [ ] Plot 1: All points on 1:1 line
- [ ] Statistics: R² = 1.0000, RMSE ≈ 0, MAE ≈ 0, Bias ≈ 0
- [ ] Plot 2: All residuals ≈ 0 (flat line at zero)
- [ ] Q-Q plot: perfect diagonal
- [ ] Histogram: single spike at zero
- [ ] Plot 3: Bar extends to y=1.0
- [ ] No errors or warnings

**Expected Result:** Perfect predictions display correctly

---

### Test 3.2: Terrible Predictions (R² < 0)

**Setup:**
- Apply a model to mismatched data
- OR modify predictions_df to have very poor predictions

**Verify:**
- [ ] Plot 1: Points far from 1:1 line
- [ ] Statistics: R² < 0, high RMSE, high MAE
- [ ] Plot 2: Residuals show clear pattern/bias
- [ ] Q-Q plot: deviation from diagonal
- [ ] Histogram: wide distribution
- [ ] Plot 3: Negative R² bars (below zero)
- [ ] Y-axis extends below zero
- [ ] No errors or warnings

**Expected Result:** Poor predictions display correctly with negative R²

---

### Test 3.3: Very Small Validation Set (n < 10)

**Setup:**
- Create validation set with only 5-8 samples
- Load model and run predictions

**Verify:**
- [ ] All plots render without error
- [ ] Statistics show correct n value
- [ ] Histogram has appropriate number of bins (not 20)
- [ ] Q-Q plot works with few points
- [ ] No "insufficient data" errors

**Expected Result:** Plots adapt to small sample size

---

### Test 3.4: Large Validation Set (n > 500)

**Setup:**
- Create large validation set (if available)
- Load model and run predictions

**Verify:**
- [ ] Plots render in reasonable time (< 5 seconds)
- [ ] Scatter plots not too cluttered (opacity helps)
- [ ] Smoothing line in residuals plot works
- [ ] No memory errors
- [ ] GUI remains responsive

**Expected Result:** Efficient handling of large datasets

---

### Test 3.5: Model Names Too Long

**Setup:**
- Load models with very long filenames
- Run predictions

**Verify:**
- [ ] Plot 1 title truncates long names (with "...")
- [ ] Plot 3 x-axis labels truncate long names
- [ ] No text overflow outside plot area
- [ ] Labels remain readable

**Expected Result:** Long names handled gracefully

---

### Test 3.6: Single Sample Prediction

**Setup:**
- Load validation set with 1 sample
- Run predictions

**Verify:**
- [ ] Plots render without crash
- [ ] Statistics show n=1
- [ ] Residuals plot shows single point
- [ ] Histogram may not be meaningful but doesn't error
- [ ] Q-Q plot shows single point

**Expected Result:** Single sample handled without crash

---

### Test 3.7: Model Prediction Failure

**Setup:**
- Load model trained on different wavelengths
- Run predictions (should fail for that model)

**Verify:**
- [ ] Error logged to console
- [ ] Other models continue to process
- [ ] Statistics show only successful models
- [ ] Plots show successful model data
- [ ] Status message: "Complete with warnings: X succeeded, Y failed"

**Expected Result:** Graceful degradation - partial success

---

## Test Suite 4: Performance Testing

### Test 4.1: Plot Generation Speed

**Setup:**
- Validation set with 100 samples
- 3 models loaded

**Measure:**
- Time from "Run All Models" click to plots displayed

**Verify:**
- [ ] Plots render in < 3 seconds (after predictions complete)
- [ ] No noticeable GUI freeze
- [ ] Progress bar updates smoothly

**Expected Result:** Fast, responsive plotting

---

### Test 4.2: Memory Usage

**Setup:**
- Large validation set (if available)
- Multiple models

**Monitor:**
- Task Manager / Activity Monitor during prediction

**Verify:**
- [ ] Memory usage remains reasonable
- [ ] No memory leak (usage returns to baseline after)
- [ ] No accumulation of old plots in memory

**Expected Result:** Efficient memory management

---

### Test 4.3: Repeated Predictions

**Setup:**
- Run predictions 5 times in a row without restarting

**Verify:**
- [ ] Each run clears previous plots properly
- [ ] No slowdown on repeated runs
- [ ] No accumulation of widgets
- [ ] Memory stable across runs

**Expected Result:** Consistent performance on repeated use

---

## Test Suite 5: Integration Testing

### Test 5.1: Tab Switching During Plot Generation

**Steps:**
1. Load validation set and models
2. Click "Run All Models"
3. Immediately switch to another tab
4. Switch back to Tab 7

**Verify:**
- [ ] Plots complete in background
- [ ] Plots visible when returning to Tab 7
- [ ] No errors or crashes

**Expected Result:** Thread-safe plot generation

---

### Test 5.2: Validation Set Changes

**Steps:**
1. Create validation set (20%)
2. Load models, run predictions, view plots
3. Return to Tab 1, change validation set to 30%
4. Return to Tab 7, run predictions again

**Verify:**
- [ ] New plots reflect new validation set size
- [ ] Statistics show updated n value
- [ ] Previous plots properly cleared
- [ ] Metrics recalculated correctly

**Expected Result:** Dynamic response to data changes

---

### Test 5.3: Model Reload

**Steps:**
1. Load models, run predictions, view plots
2. Clear models
3. Reload same models
4. Run predictions again

**Verify:**
- [ ] Plots regenerate correctly
- [ ] Results identical to first run
- [ ] No stale data from previous load

**Expected Result:** Clean state management

---

## Test Suite 6: User Experience

### Test 6.1: Typical Workflow

**Scenario:** User wants to validate a saved PLS model

**Steps:**
1. Navigate to Tab 7
2. Load PLS model file
3. Select "Use Pre-Selected Validation Set"
4. Click "Load Data" (validation set loads)
5. Click "Run All Models"
6. Examine plots and statistics

**Verify:**
- [ ] Workflow is intuitive
- [ ] All steps clearly labeled
- [ ] Feedback at each step (status messages)
- [ ] Results easy to interpret
- [ ] Plots provide actionable insights

**Expected Result:** Smooth, professional user experience

---

### Test 6.2: Model Selection Workflow

**Scenario:** User has 5 models and wants to pick the best

**Steps:**
1. Load all 5 models
2. Load validation set
3. Run predictions
4. Examine Plot 3 (Model Comparison)

**Verify:**
- [ ] Best model immediately apparent (green bar, highest)
- [ ] R² and RMSE clearly displayed
- [ ] Color coding helps quick decision
- [ ] Model names identifiable
- [ ] Easy to export results for documentation

**Expected Result:** Efficient model selection process

---

### Test 6.3: Diagnostic Interpretation

**Scenario:** User sees poor R² and wants to understand why

**Steps:**
1. Run predictions with a model showing R² < 0.5
2. Examine Plot 2 (Residuals)

**Verify:**
- [ ] Residuals vs Fitted shows heteroscedasticity pattern
- [ ] Residuals vs Index shows outliers highlighted
- [ ] Q-Q plot shows non-normality
- [ ] Histogram confirms distribution issues
- [ ] User can identify specific problems

**Expected Result:** Plots provide diagnostic value

---

## Test Suite 7: Error Handling

### Test 7.1: Matplotlib Not Available

**Setup:**
- Temporarily rename matplotlib import
- OR set HAS_MATPLOTLIB = False

**Verify:**
- [ ] No crash when navigating to Tab 7
- [ ] Plot frames remain empty
- [ ] No error dialogs spam user
- [ ] Rest of tab functions normally

**Expected Result:** Graceful degradation without matplotlib

---

### Test 7.2: Corrupted Prediction Data

**Setup:**
- Manually corrupt self.predictions_df
- Trigger plot generation

**Verify:**
- [ ] Error logged to console
- [ ] Error message in plot frames
- [ ] GUI remains responsive
- [ ] User can recover by re-running predictions

**Expected Result:** Error handling prevents crash

---

### Test 7.3: Mismatched Validation Indices

**Setup:**
- Manually modify validation indices to not match data
- Run predictions

**Verify:**
- [ ] Error caught and logged
- [ ] Helpful error message
- [ ] No index out of bounds crash
- [ ] User directed to reload data

**Expected Result:** Robust error handling

---

## Acceptance Criteria

### Must Pass (Critical):
- [ ] All functional tests (2.1-2.5) pass
- [ ] No crashes on edge cases (3.1-3.7)
- [ ] Performance acceptable (< 3 sec plot generation)
- [ ] User workflows (6.1-6.3) are smooth

### Should Pass (Important):
- [ ] UI is responsive (1.2)
- [ ] Visual consistency (1.3)
- [ ] Integration tests (5.1-5.3) pass
- [ ] Error handling (7.1-7.3) works

### Nice to Have (Optional):
- [ ] Perfect predictions display (3.1)
- [ ] Very large datasets (3.4)
- [ ] Advanced diagnostics interpretation (6.3)

---

## Regression Testing

After any code changes, re-run:
1. Test 2.1 (Validation Set - Single Model)
2. Test 2.2 (Validation Set - Multiple Models)
3. Test 2.3 (New Data)
4. Test 3.2 (Terrible Predictions)
5. Test 6.1 (Typical Workflow)

**Time Required:** ~10 minutes for regression suite

---

## Troubleshooting Common Issues

### Issue: Plots don't appear

**Check:**
- Is HAS_MATPLOTLIB = True?
- Are plot frames defined (hasattr(self, 'tab7_plot1_frame'))?
- Is _tab7_generate_plots() being called?
- Check console for error messages

**Fix:** Review integration code in _update_prediction_statistics()

---

### Issue: Plots show wrong data

**Check:**
- Is is_validation correctly detecting validation set?
- Are indices aligned (self.validation_y.loc[...] correct)?
- Is predictions_df properly populated?

**Fix:** Add print statements to trace data flow

---

### Issue: Layout is broken

**Check:**
- Grid row/column weights set correctly?
- Frame.pack() vs .grid() conflicts?
- Window too small?

**Fix:** Review UI code in _add_plot_frames_to_tab7()

---

### Issue: Plots are slow

**Check:**
- Validation set size (n)?
- Number of models?
- Are plots being regenerated unnecessarily?

**Fix:** Profile with cProfile, optimize bottlenecks

---

## Test Log Template

Use this template to document test results:

```
Date: 2025-11-07
Tester: [Your Name]
Version: [Commit Hash]

Test Suite 1: UI Verification
  Test 1.1: Layout Check ..................... [ PASS / FAIL ]
  Test 1.2: Responsive Layout ................ [ PASS / FAIL ]
  Test 1.3: Visual Consistency ............... [ PASS / FAIL ]

Test Suite 2: Functional Testing
  Test 2.1: Validation Set - Single Model .... [ PASS / FAIL ]
  Test 2.2: Validation Set - Multiple Models . [ PASS / FAIL ]
  Test 2.3: New Data (No Validation) ......... [ PASS / FAIL ]
  Test 2.4: No Models Loaded ................. [ PASS / FAIL ]
  Test 2.5: No Data Loaded ................... [ PASS / FAIL ]

[Continue for all test suites...]

Summary:
  Total Tests: 35
  Passed: XX
  Failed: XX
  Skipped: XX

Critical Issues Found:
  1. [Description]
  2. [Description]

Recommendations:
  - [Action items]
```

---

## Continuous Integration

For automated testing (future):
1. Selenium/PyAutoGUI for GUI testing
2. Screenshot comparison for visual regression
3. Performance benchmarks tracked over time
4. Automated test runs on every commit

---

**End of Testing Guide**

For questions or issues, contact: Agent 5 (Diagnostic Plot Suite)
