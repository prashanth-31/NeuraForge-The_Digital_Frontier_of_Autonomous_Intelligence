@echo off
echo Starting NeuraForge Development Environment...

REM Start the backend server in a new window
start cmd /k "cd %~dp0implementation\backend && python server.py"

REM Wait a moment to ensure the backend starts first
timeout /t 2 > NUL

REM Start the frontend server in a new window
start cmd /k "cd %~dp0implementation\frontend && npm run dev"

echo NeuraForge servers started!
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to close this window...
pause > NUL
