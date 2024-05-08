@echo off
setlocal

set LIB_NAME=PFRTool

pip list | findstr /R /C:"^%LIB_NAME% " > nul
if errorlevel 1 (
    echo %LIB_NAME% is not installed. Installing...
    pip install .
) else (
    echo %LIB_NAME% is already installed. Upgrading...
    pip install . --upgrade
)

endlocal
