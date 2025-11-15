@echo off
echo ================================================================================
echo STEP 4: Compare DASP vs R Results
echo ================================================================================
echo.
echo This will compare the results and generate reports.
echo.
echo Time required: 1-2 minutes
echo.
pause

cd /d "%~dp0"

echo.
echo Comparing results...
echo.

REM Try to find Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python compare_regression_results.py
) else (
    "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" compare_regression_results.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Comparison failed!
    echo.
    echo Make sure both R and DASP tests have been run first.
    echo.
) else (
    echo.
    echo ================================================================================
    echo SUCCESS! Comparison complete.
    echo ================================================================================
    echo.
    echo Reports generated:
    echo   - results\comparisons\comparison_report.md
    echo   - results\comparisons\bone_collagen_comparison.csv
    echo   - results\comparisons\d13c_comparison.csv
    echo.
    echo Open comparison_report.md to see the results!
    echo.
)

pause
