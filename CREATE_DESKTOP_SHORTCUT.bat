@echo off
REM ============================================================================
REM Create Desktop Shortcut for Spectral Predict
REM ============================================================================
REM Run this script ONCE to create a shortcut on your desktop
REM ============================================================================

echo.
echo ================================================================================
echo    Creating Desktop Shortcut for Spectral Predict
echo ================================================================================
echo.

REM Get the script's directory
set "SCRIPT_DIR=%~dp0"

REM Create VBScript to make shortcut
set "VBSCRIPT=%TEMP%\create_shortcut.vbs"

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%VBSCRIPT%"
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\Spectral Predict.lnk" >> "%VBSCRIPT%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%VBSCRIPT%"
echo oLink.TargetPath = "%SCRIPT_DIR%RUN_SPECTRAL_PREDICT.bat" >> "%VBSCRIPT%"
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> "%VBSCRIPT%"
echo oLink.Description = "Spectral Predict - Julia-Powered Spectral Analysis" >> "%VBSCRIPT%"
echo oLink.IconLocation = "%%SystemRoot%%\System32\SHELL32.dll,165" >> "%VBSCRIPT%"
echo oLink.Save >> "%VBSCRIPT%"

REM Run the VBScript
cscript //nologo "%VBSCRIPT%"

REM Clean up
del "%VBSCRIPT%"

echo.
echo ================================================================================
echo    SUCCESS! Shortcut created on your desktop.
echo.
echo    Look for: "Spectral Predict" icon on your desktop
echo    Double-click it to launch the application
echo ================================================================================
echo.
pause
