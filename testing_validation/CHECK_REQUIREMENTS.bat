@echo off
echo ================================================================================
echo DASP Testing - Requirements Check
echo ================================================================================
echo.

cd /d "%~dp0"

echo Checking Python...
echo.

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python found in PATH
    python --version
) else (
    echo [WARNING] Python not in PATH
)

if exist "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" (
    echo [OK] Python found at specific path
    "C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" --version
) else (
    echo [ERROR] Python not found at expected path
)

echo.
echo Checking Python packages...
echo.

"C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" -c "import pandas; print('[OK] pandas version:', pandas.__version__)"
"C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" -c "import numpy; print('[OK] numpy version:', numpy.__version__)"
"C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" -c "import sklearn; print('[OK] scikit-learn version:', sklearn.__version__)"
"C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" -c "import xgboost; print('[OK] xgboost version:', xgboost.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [MISSING] xgboost - Run STEP0_Install_Python_Packages.bat
)
"C:\Users\sponheim\AppData\Local\Programs\Python\Python314\python.exe" -c "import scipy; print('[OK] scipy version:', scipy.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [MISSING] scipy - Run STEP0_Install_Python_Packages.bat
)

echo.
echo Checking R...
echo.

where Rscript >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Rscript found in PATH
    Rscript --version 2>&1 | findstr /C:"R scripting"
) else (
    echo [ERROR] Rscript not found in PATH
    echo.
    echo R is not installed or not in your PATH.
    echo.
    echo To fix:
    echo   1. Download R from: https://cran.r-project.org/bin/windows/base/
    echo   2. Install R (make sure to check "Add to PATH" during installation)
    echo   3. Restart this script
    echo.
)

echo.
echo ================================================================================
echo Summary
echo ================================================================================
echo.
echo If you see [MISSING] or [ERROR] above, you need to:
echo   - [MISSING] xgboost or scipy: Run STEP0_Install_Python_Packages.bat
echo   - [ERROR] Rscript: Install R from https://cran.r-project.org/
echo.
echo If everything shows [OK], you're ready to run the tests!
echo.

pause
