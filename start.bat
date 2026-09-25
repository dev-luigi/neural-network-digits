@echo off
rem ==========================================================================
rem  QUICK START
rem    double-click                      opens the graphical interface
rem    start.bat train --epochs 100      (from the terminal) runs one step
rem  The first time it installs by itself the libraries it needs.
rem ==========================================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install it from https://www.python.org
    echo and during the installation tick "Add python.exe to PATH".
    pause
    exit /b 1
)

rem Quick check (without loading them) that the libraries are there; if they are missing I install them
python -c "import importlib.util as u, sys; sys.exit(not all(u.find_spec(m) for m in ('numpy', 'PIL', 'matplotlib')))"
if errorlevel 1 (
    echo First time: installing the required libraries...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

if "%~1"=="" (
    rem No command: I open the interface without the black terminal window
    start "" pythonw start.py
) else (
    python start.py %*
)
