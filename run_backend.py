#!/usr/bin/env python3
"""
Script to run the FastAPI backend for the Dynamic Agent System
"""
import subprocess
import sys
import os

def run_backend():
    """Run the FastAPI backend server"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root  # <-- Set PYTHONPATH here

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running FastAPI backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_backend()
