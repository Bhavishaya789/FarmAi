@echo off
title FarmAI - Crop Recommendation System
color 0A

echo.
echo  ========================================
echo    FarmAI - Crop Recommendation System
echo  ========================================
echo.

REM Change to the directory where this bat file is located
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python from https://python.org
    pause
    exit /b 1
)

REM Fix bcrypt compatibility issue and install all deps
echo  [1/3] Checking dependencies...
python -m pip install "bcrypt==4.0.1" --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo  [2/3] Starting FarmAI backend server...
echo.

REM Start the server in a new visible window
start "FarmAI Server" cmd /k "cd /d "%~dp0" && python -m uvicorn crop:app --host 127.0.0.1 --port 8000"

REM Wait for server to boot up
timeout /t 3 /nobreak >nul

REM Open browser
echo  [3/3] Opening browser...
start "" "http://127.0.0.1:8000"

echo.
echo  ========================================
echo    FarmAI is running!
echo    URL: http://127.0.0.1:8000
echo  ========================================
echo.
echo  Keep the "FarmAI Server" window open.
echo  Close it to stop the server.
echo.
pause
