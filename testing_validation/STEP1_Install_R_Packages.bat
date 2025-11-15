@echo off
echo ================================================================================
echo STEP 1: Install R Packages (One-time setup)
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
echo Installing R packages...
echo.

Rscript install_packages.R

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: R package installation failed!
    echo.
    echo Make sure R is installed and Rscript is in your PATH.
    echo You can download R from: https://cran.r-project.org/
    echo.
) else (
    echo.
    echo ================================================================================
    echo SUCCESS! R packages installed.
    echo ================================================================================
    echo.
    echo Next step: Run STEP2_Run_R_Tests.bat
    echo.
)

pause
