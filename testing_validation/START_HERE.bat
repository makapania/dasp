@echo off
echo ================================================================================
echo DASP VALIDATION TESTING - COMPLETE WORKFLOW
echo ================================================================================
echo.
echo This script will run the complete validation:
echo   1. Install Python packages (xgboost, scipy)
echo   2. Install R packages (pls, glmnet, randomForest, etc.)
echo   3. Run R regression tests
echo   4. Run DASP regression tests
echo   5. Compare results
echo.
echo Total time: 35-65 minutes (mostly automated)
echo.
echo You can walk away and come back!
echo.
pause

cd /d "%~dp0"

REM =============================================================================
REM Find Python
REM =============================================================================

set PYTHON=
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON=python
) else if exist "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" (
    set PYTHON="C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe"
) else (
    echo ERROR: Cannot find Python!
    pause
    exit /b 1
)

echo Found Python: %PYTHON%

REM =============================================================================
REM Find R
REM =============================================================================

set RSCRIPT=
where Rscript >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set RSCRIPT=Rscript
) else if exist "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
) else if exist "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
) else if exist "C:\Program Files\R\R-4.5.0\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.5.0\bin\Rscript.exe"
) else if exist "C:\Program Files\R\R-4.4.3\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.4.3\bin\Rscript.exe"
) else if exist "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
) else (
    echo ERROR: Cannot find R!
    echo Please install R from https://cran.r-project.org/
    pause
    exit /b 1
)

echo Found R: %RSCRIPT%
echo.

REM =============================================================================
REM STEP 0: Install Python packages
REM =============================================================================

echo.
echo ================================================================================
echo [1/5] Installing Python packages (xgboost, scipy)...
echo ================================================================================
echo.

%PYTHON% -m pip install --quiet xgboost scipy

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Python package installation had issues
    echo Continuing anyway...
)

echo Python packages ready.

REM =============================================================================
REM STEP 1: Install R packages
REM =============================================================================

echo.
echo ================================================================================
echo [2/5] Installing R packages (5-10 minutes)...
echo ================================================================================
echo.

cd r_scripts
%RSCRIPT% install_packages.R

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: R package installation failed!
    pause
    exit /b 1
)

cd ..
echo R packages ready.

REM =============================================================================
REM STEP 2: Run R tests
REM =============================================================================

echo.
echo ================================================================================
echo [3/5] Running R regression tests (15-30 minutes)...
echo ================================================================================
echo.
echo Testing:
echo   - Bone Collagen (36 train / 13 test)
echo   - Enamel d13C (105 train / 35 test)
echo.
echo You can grab coffee - this will take a while!
echo.

cd r_scripts
%RSCRIPT% regression_models_comprehensive.R

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: R tests failed!
    pause
    exit /b 1
)

cd ..
echo R tests complete.

REM =============================================================================
REM STEP 3: Run DASP tests
REM =============================================================================

echo.
echo ================================================================================
echo [4/5] Running DASP regression tests (10-20 minutes)...
echo ================================================================================
echo.

%PYTHON% dasp_regression.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: DASP tests failed!
    pause
    exit /b 1
)

echo DASP tests complete.

REM =============================================================================
REM STEP 4: Compare results
REM =============================================================================

echo.
echo ================================================================================
echo [5/5] Comparing DASP vs R results...
echo ================================================================================
echo.

%PYTHON% compare_regression_results.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Comparison failed!
    pause
    exit /b 1
)

REM =============================================================================
REM DONE!
REM =============================================================================

echo.
echo ================================================================================
echo SUCCESS! ALL TESTS COMPLETE!
echo ================================================================================
echo.
echo Results are in:
echo   results\comparisons\comparison_report.md
echo.
echo Open that file to see if DASP matches R!
echo.
echo Also generated:
echo   - results\comparisons\bone_collagen_comparison.csv
echo   - results\comparisons\d13c_comparison.csv
echo   - results\comparisons\comparison_summary.json
echo.
echo ================================================================================
echo.

pause
