# Quick Start Guide - Phase 2 Progress Monitor

## 🚀 What's New in Phase 2?

Phase 2 adds a **live progress monitor** that shows real-time updates during analysis!

### Features:
- ✅ **Progress bar** showing completion percentage
- ✅ **Real-time ETA** (Estimated Time Remaining)
- ✅ **Best model tracker** - see the best model found so far
- ✅ **Cancel button** - stop analysis if needed
- ✅ **Elapsed time** counter
- ✅ **Current task** display

---

## 📖 How to Use

### Method 1: GUI (Easiest)

1. **Launch the GUI:**
   ```bash
   python spectral_predict_gui.py
   ```

2. **Select your data files** (as usual)

3. **Make sure "Show live progress monitor" is CHECKED** ✓ (it's on by default)

4. **Click "Run Analysis"**

5. **Watch the magic happen!**
   - A progress window opens automatically
   - Shows real-time updates as models are tested
   - Displays best model found so far
   - Updates ETA based on processing speed

6. **Optional: Click "Minimize"** to work on other tasks while analysis runs

7. **Click "Cancel Analysis"** if you need to stop early

---

### Method 2: Try the Demo First

Want to see what the progress monitor looks like before running a real analysis?

```bash
python test_progress_monitor.py
```

**Choose:**
- `[1]` - Regression demo (shows RMSE, R² metrics)
- `[2]` - Classification demo (shows ROC AUC, Accuracy)

The demo simulates a 150-model analysis in about 10 seconds so you can see all features in action!

---

## 🎯 What You'll See

### Progress Monitor Window:

```
╔═══════════════════════════════════════════════════════╗
║            Analysis in Progress                       ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  ┌─ Overall Progress ──────────────────────────┐     ║
║  │                                              │     ║
║  │  [████████████░░░░░░░░░] 42.5%              │     ║
║  │                                              │     ║
║  │  Model 150 of 350                           │     ║
║  │                                              │     ║
║  │  Elapsed: 00:05:32   Est. Remaining: 7m 15s │     ║
║  └──────────────────────────────────────────────┘     ║
║                                                       ║
║  ┌─ Current Task ───────────────────────────────┐     ║
║  │  Stage: Testing model configurations        │     ║
║  │                                              │     ║
║  │  Testing RandomForest with SNV preprocessing│     ║
║  └──────────────────────────────────────────────┘     ║
║                                                       ║
║  ┌─ Best Model So Far ──────────────────────────┐     ║
║  │  Model: PLS                                  │     ║
║  │  Preprocessing: d1_sg7                       │     ║
║  │  Variables: 250 (top250)                     │     ║
║  │  Performance: RMSE: 0.0823 | R²: 0.9542     │     ║
║  └──────────────────────────────────────────────┘     ║
║                                                       ║
║         [Cancel Analysis]    [Minimize]              ║
║                                                       ║
║  Running... 150/350 models tested                    ║
╚═══════════════════════════════════════════════════════╝
```

---

## ⏱️ Typical Timeline

For a standard analysis with ~350 model configurations:

| Time | What's Happening | Progress |
|------|------------------|----------|
| 0:00 | Loading data, analyzing regions | 0-2% |
| 0:30 | Testing first batch of models | 5-15% |
| 2:00 | Finding good models, ETA stabilizes | 20-30% |
| 5:00 | About halfway through | 50% |
| 10:00 | Most models tested | 80% |
| 12:00 | Final models completing | 95% |
| 13:00 | **Complete!** Results saved | 100% ✅ |

**Total time:** Typically 10-20 minutes depending on:
- Number of samples (more samples = slower)
- Number of features (more features = slower for some models)
- Model types (RandomForest slower than PLS)

---

## 🎨 Understanding the Display

### Progress Bar Colors:
- **Blue filled:** Completed models
- **Gray unfilled:** Remaining models

### Stage Labels:
- **Purple "Analyzing spectral regions"**: Finding important wavelength regions (quick, <1 min)
- **Blue "Testing model configurations"**: Running cross-validation on each model (bulk of time)
- **Green "Analysis Complete!"**: Done! ✅
- **Red "Analysis Failed"**: Error occurred ❌

### Best Model Updates:
- Updates **only when a better model is found**
- Shows the model configuration and performance
- Helps you know if good models are being discovered

### ETA (Estimated Time Remaining):
- Starts as "Calculating..." (needs ~20 models to estimate)
- Becomes accurate after ~50 models tested
- **Adapts to processing speed** (some models are slower)
- Format: `5m 23s` or `1h 15m` (human-readable)

---

## 💡 Tips & Tricks

### ✅ Best Practices:

1. **Let it run**
   - The first 5-10% might feel slow (loading data, region analysis)
   - Speed picks up once model testing starts
   - ETA becomes accurate after ~30 models

2. **Minimize if needed**
   - Click "Minimize" to hide the window
   - Continue working on other tasks
   - Window stays in taskbar, can restore anytime

3. **Watch the best model**
   - If you see good metrics early (high R², low RMSE), that's great!
   - The system will keep searching for even better configurations

4. **Cancel if wrong data**
   - If you realize you selected wrong target or data
   - Click "Cancel Analysis"
   - It will finish the current model gracefully, then stop

### ❌ Things to Avoid:

1. **Don't close the progress window by clicking X**
   - Use "Cancel Analysis" button instead
   - Closing window won't stop the analysis (it runs in background)

2. **Don't restart if ETA seems long**
   - ETA starts pessimistic, improves as analysis progresses
   - Trust the ETA after ~50 models tested

3. **Don't run multiple analyses at once**
   - Wait for current analysis to complete
   - Running multiple will slow down your computer

---

## 🔧 Troubleshooting

### Progress window doesn't appear?
✓ **Check:** Is "Show live progress monitor" checked in GUI?
✓ **Fix:** Check the box and run again

### Window freezes or becomes unresponsive?
✓ **Cause:** Normal - Windows might mark it as "Not Responding" during heavy computation
✓ **Fix:** Just wait, it will update. The background thread is still working!

### ETA keeps changing?
✓ **Cause:** Normal - ETA adapts to actual processing speed
✓ **Fix:** Nothing needed. It stabilizes after ~50 models.

### Best model shows "No models tested yet" for a long time?
✓ **Cause:** Still in region analysis phase or loading data
✓ **Fix:** Wait for model testing phase to start (usually <1 minute)

### Analysis is slower than expected?
✓ **Check:** How many samples? (More samples = slower CV)
✓ **Check:** Running other heavy programs? (Close to free up CPU)
✓ **Typical:** 10-20 minutes for 20-50 samples is normal

---

## 🎓 Understanding Metrics

### For Regression Tasks:
- **RMSE (Root Mean Square Error):** Lower is better
  - Typical good value: <0.10 (depends on target scale)
- **R² (R-squared):** Higher is better
  - Typical good value: >0.85

### For Classification Tasks:
- **ROC AUC:** Higher is better
  - Typical good value: >0.90
- **Accuracy:** Higher is better
  - Typical good value: >0.85

**Example of a GREAT regression model:**
```
RMSE: 0.0823 | R²: 0.9542
(Low error, explains 95.4% of variance - excellent!)
```

**Example of a GOOD but not great model:**
```
RMSE: 0.1456 | R²: 0.8123
(Moderate error, explains 81% of variance - usable)
```

---

## 🚦 Next Steps After Completion

When analysis completes (100%), you'll see:

1. **Progress window shows:** "Analysis Complete! Results saved to..."
2. **Main GUI shows:** "✓ Analysis complete! Check outputs/ directory"
3. **Success popup:** Shows paths to results files

### View Results:

```bash
# Results CSV with all models ranked
outputs/results.csv

# Detailed markdown report
reports/YourTargetVariable.md
```

### Use Best Model:

1. Open `outputs/results.csv`
2. Look at **row 1** (rank #1 = best model)
3. Note the model type, preprocessing, and variable count
4. Retrain on your full dataset using those exact settings

---

## 📊 What Changed from Phase 1?

### Phase 1 (IMPROVEMENTS_PHASE1.md):
✅ Improved model ranking (better scoring formula)
✅ Expanded variable selection (10, 20, 50, 100, 250, 500, 1000)
✅ Spectral region analysis
✅ Console progress messages

### Phase 2 (NEW - IMPROVEMENTS_PHASE2.md):
✅ **Live progress monitor window**
✅ **Real-time ETA calculation**
✅ **Best model tracking**
✅ **Cancel/Minimize controls**
✅ **Threaded execution (GUI stays responsive)**

**Both phases work together!** Phase 1 improved what models are tested and how they're ranked. Phase 2 lets you watch it happen in real-time.

---

## ❓ FAQ

**Q: Can I run analysis without the progress monitor?**
A: Yes! Uncheck "Show live progress monitor" in the GUI. It will run in subprocess mode like before.

**Q: Does the progress monitor slow down analysis?**
A: No! Overhead is <0.5% (a few milliseconds per model). You won't notice any difference.

**Q: Can I run from command line and still see progress?**
A: Command line shows text progress: `[150/350] Testing...` but no GUI window. Use the GUI app for the visual progress monitor.

**Q: What if I close my laptop lid during analysis?**
A: Analysis continues running! Just open laptop and restore the window to see current progress.

**Q: Can I see progress for past analyses?**
A: No, progress is only shown during live analysis. But results are always saved to `outputs/results.csv`.

**Q: How do I test the progress monitor without real data?**
A: Run `python test_progress_monitor.py` for a quick demo!

---

## 🎉 Summary

**Phase 2 gives you complete visibility into your analysis!**

✅ No more staring at a blank console wondering if it's working
✅ Know exactly how long to wait
✅ See best results as they're discovered
✅ Cancel if needed
✅ Minimize and multitask

**Enjoy your new progress monitor!** 🚀

---

**Questions or Issues?**
- Read `IMPROVEMENTS_PHASE2.md` for technical details
- Read `IMPROVEMENTS_PHASE1.md` for model ranking improvements
- Check `outputs/results.csv` for detailed results

**Ready to analyze?**
```bash
python spectral_predict_gui.py
```

Happy analyzing! 📊✨
