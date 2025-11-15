@echo off
echo ================================================================================
echo DASP Validation Testing - Automated Runner
echo ================================================================================
echo.

REM Change to the testing_validation directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Found Python:
    python --version
    echo.
    echo Running tests with python...
    python run_all_tests.py
    goto :end
)

REM Try the specific Python path we know works
if exist "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" (
    echo Found Python at: C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe
    echo.
    echo Running tests...
    "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" run_all_tests.py
    goto :end
)

REM If we get here, Python wasn't found
echo ERROR: Could not find Python!
echo.
echo Please run manually:
echo   "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" run_all_tests.py
echo.

:end
echo.
echo ================================================================================
echo Press any key to close this window...
pause >nul
