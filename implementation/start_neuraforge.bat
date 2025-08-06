@echo off
echo Starting NeuraForge Development Environment...
echo.
echo ========================================================
echo Make sure Ollama is running with LLaMA 3.1 model loaded!
echo ========================================================
echo.

REM Activate the Python virtual environment and start the backend server
start cmd /k "cd %~dp0backend && %~dp0backend\myenv\Scripts\activate && python server.py"

REM Wait a moment to ensure the backend starts first
timeout /t 3 > NUL

REM Start the frontend server in a new window
start cmd /k "cd %~dp0frontend && npm run dev"

echo.
echo NeuraForge servers started!
echo.
echo Backend API: http://127.0.0.1:8000
echo Frontend UI: http://localhost:3000
echo.
echo Press any key to close this window...
pause > NUL
