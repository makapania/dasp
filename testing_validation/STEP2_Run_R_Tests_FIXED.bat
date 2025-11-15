@echo off
echo ================================================================================
echo STEP 2: Run R Regression Tests (FIXED - Auto-finds R)
echo ================================================================================
echo.
echo This will test all regression models in R.
echo.
echo Datasets tested:
echo   - Bone Collagen (36 train / 13 test)
echo   - Enamel d13C (105 train / 35 test)
echo.
echo Models tested:
echo   - PLS (12 configs)
echo   - Ridge (5 configs)
echo   - Lasso (4 configs)
echo   - Random Forest (6 configs)
echo   - XGBoost (8 configs)
echo.
echo Time required: 15-30 minutes
echo.
pause

cd /d "%~dp0\r_scripts"

echo.
echo Looking for R installation...
echo.

REM Try to find Rscript in common locations
set RSCRIPT=

REM Check if Rscript is in PATH
where Rscript >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set RSCRIPT=Rscript
    echo Found Rscript in PATH
    goto :found
)

REM Check for R in Program Files (use the latest version)
if exist "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
    echo Found R 4.5.2
    goto :found
)

if exist "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
    echo Found R 4.5.1
    goto :found
)

if exist "C:\Program Files\R\R-4.5.0\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.5.0\bin\Rscript.exe"
    echo Found R 4.5.0
    goto :found
)

if exist "C:\Program Files\R\R-4.4.3\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.4.3\bin\Rscript.exe"
    echo Found R 4.4.3
    goto :found
)

if exist "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" (
    set RSCRIPT="C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
    echo Found R 4.4.2
    goto :found
)

REM If not found, show error
echo ERROR: Could not find R installation!
echo.
pause
exit /b 1

:found
echo Using: %RSCRIPT%
echo.
echo Running R regression tests...
echo This will take 15-30 minutes...
echo.

%RSCRIPT% regression_models_comprehensive.R

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: R tests failed!
    echo.
    echo Check the output above for error messages.
    echo.
) else (
    echo.
    echo ================================================================================
    echo SUCCESS! R regression tests complete.
    echo ================================================================================
    echo.
    echo Results saved to: ..\results\r_regression\
    echo.
    echo Next step: Run STEP3_Run_DASP_Tests.bat
    echo.
)

pause
