#!/bin/bash
# NeuraForge Server Launcher for Linux/macOS

echo "NeuraForge Server Launcher"
echo "=========================="
echo

# Set the project root
PROJECT_ROOT="$(dirname "$0")"

# Activate the virtual environment
echo "Activating virtual environment..."
if [ -d "$PROJECT_ROOT/myenv/bin" ]; then
    source "$PROJECT_ROOT/myenv/bin/activate"
elif [ -d "$PROJECT_ROOT/myenv/Scripts" ]; then
    source "$PROJECT_ROOT/myenv/Scripts/activate"
else
    echo "Failed to activate virtual environment. Please make sure it exists."
    exit 1
fi

# Install required packages if missing
echo "Checking required packages..."
pip install -q "langchain>=0.1.0" "langchain-core>=0.1.23" "langchain-community>=0.0.10" "fastapi" "uvicorn" "python-dotenv" "pydantic" 2>/dev/null
echo

# Set up PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "Set PYTHONPATH to include: $PROJECT_ROOT"
echo

# Check if Ollama is running
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "WARNING: Ollama does not appear to be running on http://localhost:11434."
    echo "Please start Ollama before continuing."
    echo
    read -p "Press Enter to continue anyway or Ctrl+C to cancel..." </dev/tty
fi

# Launch the server
echo
echo "Launching NeuraForge server..."
echo
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Deactivate the virtual environment when done
deactivate
