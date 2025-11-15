@echo off
echo ================================================================================
echo STEP 3: Run DASP Regression Tests
echo ================================================================================
echo.
echo This will test the same models in DASP (Python).
echo.
echo Time required: 10-20 minutes
echo.
pause

cd /d "%~dp0"

echo.
echo Running DASP regression tests...
echo.

REM Try to find Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python dasp_regression.py
) else (
    "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" dasp_regression.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: DASP tests failed!
    echo.
    echo Check the output above for error messages.
    echo.
) else (
    echo.
    echo ================================================================================
    echo SUCCESS! DASP regression tests complete.
    echo ================================================================================
    echo.
    echo Results saved to: results\dasp_regression\
    echo.
    echo Next step: Run STEP4_Compare_Results.bat
    echo.
)

pause
