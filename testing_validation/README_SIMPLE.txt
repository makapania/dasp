================================================================================
SIMPLE INSTRUCTIONS - DASP VALIDATION TESTING
================================================================================

YOUR R IS INSTALLED BUT NOT IN PATH - NO PROBLEM!

I've created FIXED batch files that automatically find R.

================================================================================
EASIEST WAY - ONE CLICK
================================================================================

Double-click: START_HERE.bat

This does EVERYTHING automatically:
  - Installs Python packages
  - Installs R packages
  - Runs R tests
  - Runs DASP tests
  - Compares results

Time: 35-65 minutes
You can walk away!

================================================================================
STEP-BY-STEP WAY
================================================================================

If you prefer to run steps individually:

1. STEP0_Install_Python_Packages.bat
   (Installs xgboost and scipy)

2. STEP1_Install_R_Packages_FIXED.bat
   (Installs R packages - auto-finds R!)

3. STEP2_Run_R_Tests_FIXED.bat
   (Runs R tests - auto-finds R!)

4. STEP3_Run_DASP_Tests.bat
   (Runs DASP tests)

5. STEP4_Compare_Results.bat
   (Compares and generates report)

================================================================================
WHICH FILES TO USE
================================================================================

OLD FILES (don't use - these need R in PATH):
  X STEP1_Install_R_Packages.bat
  X STEP2_Run_R_Tests.bat

NEW FILES (use these - auto-find R):
  ✓ STEP1_Install_R_Packages_FIXED.bat
  ✓ STEP2_Run_R_Tests_FIXED.bat
  ✓ START_HERE.bat (RECOMMENDED!)

================================================================================
WHAT YOU'LL GET
================================================================================

After running, check:
  results\comparisons\comparison_report.md

This tells you:
  - Does DASP match R? (yes/no)
  - R² correlation (should be >0.99)
  - Best models for each dataset
  - Complete statistical analysis

================================================================================
TROUBLESHOOTING
================================================================================

If START_HERE.bat fails:

1. Make sure you ran it from the testing_validation folder
2. Check that R is actually installed:
   - Look in C:\Program Files\R\
   - You should see folders like R-4.5.2 or R-4.4.3

3. If Python errors:
   - Run STEP0_Install_Python_Packages.bat first

4. If still fails:
   - Run steps individually (STEP0, STEP1_FIXED, STEP2_FIXED, etc.)
   - This lets you see which step is failing

================================================================================

READY TO GO!

Just double-click: START_HERE.bat

Then wait 35-65 minutes for complete validation!

================================================================================
