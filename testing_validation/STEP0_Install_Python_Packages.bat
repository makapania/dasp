@echo off
echo ================================================================================
echo STEP 0: Install Python Packages
echo ================================================================================
echo.
echo This will install required Python packages for DASP testing.
echo.
echo Packages to install:
echo   - xgboost
echo   - scipy (for statistical tests)
echo.
echo Time required: 2-5 minutes
echo.
pause

cd /d "%~dp0"

echo.
echo Installing Python packages...
echo.

REM Try to find Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using python from PATH
    python -m pip install xgboost scipy
) else (
    echo Using specific Python path
    "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" -m pip install xgboost scipy
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Package installation failed!
    echo.
    echo Try running manually:
    echo   python -m pip install xgboost scipy
    echo.
) else (
    echo.
    echo ================================================================================
    echo SUCCESS! Python packages installed.
    echo ================================================================================
    echo.
    echo Installed packages:
    echo   - xgboost (for gradient boosting models)
    echo   - scipy (for statistical tests)
    echo.
    echo Next step: Run STEP1_Install_R_Packages.bat
    echo.
)

pause
