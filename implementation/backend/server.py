"""NeuraForge backend server entry point."""

import os

import uvicorn
from dotenv import load_dotenv

from neuraforge.api import app

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment variables
    # Use 127.0.0.1 for local development (direct browser access)
    # Use 0.0.0.0 only if you need to access the API from other devices on your network
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Start the server
    print(f"Starting NeuraForge API server on {host}:{port}")
    print(f"Access the API in your browser at: http://127.0.0.1:{port} or http://localhost:{port}")
    # Using the app as an import string to enable reload and workers
    uvicorn.run("neuraforge.api:app", host=host, port=port, reload=debug)
