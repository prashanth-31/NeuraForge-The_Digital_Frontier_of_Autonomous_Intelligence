@echo off
echo NeuraForge Server Launcher
echo =======================
echo.

REM Set the project root
set PROJECT_ROOT=%~dp0

REM Activate the virtual environment
echo Activating virtual environment...
call "%PROJECT_ROOT%myenv\Scripts\activate.bat"

if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. Please make sure it exists.
    exit /b 1
)

REM Install required packages if missing
echo Checking required packages...
pip install -q "langchain>=0.1.0" "langchain-core>=0.1.23" "langchain-community>=0.0.10" "fastapi" "uvicorn" "python-dotenv" "pydantic" 2>nul
echo.

REM Set up PYTHONPATH
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%
echo Set PYTHONPATH to include: %PROJECT_ROOT%
echo.

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama does not appear to be running on http://localhost:11434.
    echo Please start Ollama before continuing.
    echo.
    echo Press any key to continue anyway or Ctrl+C to cancel...
    pause >nul
)

REM Launch the server
echo.
echo Launching NeuraForge server...
echo.
uvicorn server:app --reload --host 0.0.0.0 --port 8000

REM Deactivate the virtual environment when done
deactivate
