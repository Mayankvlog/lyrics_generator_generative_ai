@echo off
REM Windows Batch Script to Start Streamlit App with VPS Configuration

setlocal enabledelayedexpansion

echo.
echo ============================================
echo   Lyrics Generator - Streamlit App Starter
echo ============================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "myenv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Creating virtual environment...
    python -m venv myenv
)

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Check if requirements are installed
echo.
echo Checking dependencies...
python -m pip install --upgrade pip -q
pip install -q -r requirements.txt 2>nul

REM Set Streamlit server configuration
set STREAMLIT_SERVER_PORT=8501
set STREAMLIT_SERVER_ADDRESS=0.0.0.0
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_BROWSER_GATHERUSAGESTATS=false

REM Display access information
echo.
echo ============================================
echo   🚀 Starting Streamlit App
echo ============================================
echo.
echo VPS IP: 167.71.235.91
echo Port: 8501
echo.
echo Access from:
echo - Local:   http://localhost:8501
echo - VPS:     http://167.71.235.91:8501
echo.
echo Press Ctrl+C to stop the app
echo ============================================
echo.

REM Run Streamlit
python -m streamlit run main.py --server.port=8501 --server.address=0.0.0.0

pause
