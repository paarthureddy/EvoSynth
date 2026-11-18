"""
Root convenience wrapper for scripts/api_server.py
"""
import uvicorn
from scripts.api_server import app

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
