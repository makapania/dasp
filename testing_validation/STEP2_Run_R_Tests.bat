@echo off
echo ================================================================================
echo STEP 2: Run R Regression Tests
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
echo Running R regression tests...
echo.

Rscript regression_models_comprehensive.R

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
