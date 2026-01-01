# Recovery Agent Prompt

Copy everything below this line and paste to a new Claude Code session:

---

## CONTEXT

You are helping recover functionality for Spectral Predict V1, a Tkinter-based spectral modeling application. The codebase was reset to an earlier commit and we need to restore/implement features.

**Key Files:**
- Main GUI: `spectral_predict_gui_optimized.py`
- Backend: `src/spectral_predict/`
- Backup of Dec 16-20 work: `backup_2025-12-20/`

**Read first:**
- `CLAUDE.md` - Project overview and recovery process
- `RECOVERY_TODO.md` - Full task list with status

---

## TASK LIST (Implement in order)

### USER PRIORITY FEATURES (1-17)

**UI/Layout:**
1. Move Data Management to bottom of Advanced section (currently in Data section)
2. Remove extra space between Data Management icon and text
3. Data Management should always open on Import & Preview tab

**Imbalance Handling:**
4. Add SMOTE for regression (not just classification)
5. Add SMOTE-Tomek for regression
6. Add artificial sampling for regression imbalances

**Variable Selection:**
7. Add CARS (Competitive Adaptive Reweighted Sampling)
8. Add VCPA-IRIV (Variable Combination Population Analysis with IRIV)

**Genetic Algorithm:**
9. Add GA Variable Selection - CODE EXISTS: `backup_2025-12-20/src/spectral_predict/ga_pls.py` and `ga_lightgbm.py`
10. Add GA Preprocessing

**Ensemble/Buttons:**
11. Redo ALL buttons after Ensemble Models - FROM SCRATCH (text overlapping graphics, half don't work)
12. Check if Ensembles work with classification - if not, disable; if should work, implement

**Optimization:**
13. Add NSGA-II as addition to Bayesian - CODE EXISTS: `backup_2025-12-20/src/spectral_predict/nsga2_search.py`
    - Must work beyond just PLS/Ridge

**Model Config Cleanup:**
14. Remove 'Modern' from Gradient Boosting header, make it blue like others
15. Remove ALL "new" logos (prerelease = everything is new)
16. Rename 'Advanced Models' to 'Other Models'

**Preprocessing:**
17. Verify SG3 and SG4 are properly integrated (were broken at one point)

### DECEMBER 16-20 FEATURES (18-21)

18. Integrate Spectral Library Search
    - CODE: `backup_2025-12-20/src/spectral_predict/library_search.py`
    - Persistent local library, duplicate detection, wavelength alignment

19. Integrate Similarity Metrics
    - CODE: `backup_2025-12-20/src/spectral_predict/similarity_metrics.py`
    - HQI, SAM, Euclidean, derivative correlation

20. Integrate Search Controller
    - CODE: `backup_2025-12-20/src/spectral_predict/search_controller.py`
    - Thread-safe pause/resume/end for long searches

21. Integrate Test Suite
    - CODE: `backup_2025-12-20/tests/`
    - Pytest fixtures, synthetic data generators, gold standards

---

## WORKFLOW

1. **Work on ONE feature at a time** - complete and verify before moving on
2. **Check backup folder first** - if code exists in `backup_2025-12-20/`, use it
3. **Check `recovery_blobs/`** - IGNORE, all ancient Julia-era code (obsolete)
4. **Test after each change** - run the GUI to verify
5. **Commit working code** - before moving to next feature
6. **Update RECOVERY_TODO.md** - mark items complete as you go

---

## IMPORTANT NOTES

- V1 uses Tkinter, NOT Julia (Julia was removed 2 months ago)
- Ignore `recovery_blobs/` folder - all obsolete
- The `archive/` folder contains old docs - do NOT read
- V2 is abandoned - only work on V1
- Entry point: `python spectral_predict_gui_optimized.py`

---

## START

Read `RECOVERY_TODO.md` and begin with feature #1: Move Data Management to Advanced section.
