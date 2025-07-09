#!/usr/bin/env python3
"""
Script to run the FastAPI backend for the Dynamic Agent System
"""
import subprocess
import sys
import os

def run_backend():
    """Run the FastAPI backend server"""
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run FastAPI with uvicorn
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running FastAPI backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_backend() 