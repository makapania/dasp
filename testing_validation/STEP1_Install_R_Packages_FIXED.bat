@echo off
echo ================================================================================
echo STEP 1: Install R Packages (FIXED - Auto-finds R)
echo ================================================================================
echo.
echo This will install required R packages for testing.
echo This only needs to be run once.
echo.
echo Time required: 5-10 minutes
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
echo Checked:
echo   - PATH
echo   - C:\Program Files\R\R-4.5.2\
echo   - C:\Program Files\R\R-4.5.1\
echo   - C:\Program Files\R\R-4.5.0\
echo   - C:\Program Files\R\R-4.4.3\
echo   - C:\Program Files\R\R-4.4.2\
echo.
echo Please check where R is installed.
echo.
pause
exit /b 1

:found
echo Using: %RSCRIPT%
echo.
echo Installing R packages...
echo.

%RSCRIPT% install_packages.R

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: R package installation failed!
    echo.
    echo Check the output above for error messages.
    echo.
) else (
    echo.
    echo ================================================================================
    echo SUCCESS! R packages installed.
    echo ================================================================================
    echo.
    echo Next step: Run STEP2_Run_R_Tests_FIXED.bat
    echo.
)

pause
