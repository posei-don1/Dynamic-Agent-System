#!/usr/bin/env python3
"""
Script to run the Streamlit UI for the Dynamic Agent System
"""
import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit app"""
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.headless=false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit() 