#!/usr/bin/env python3
"""
YOLO Use Cases App Launcher
Run this script to start the Streamlit YOLO vision models platform.
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found. Please run this script from the yolo_usecases directory.")
        sys.exit(1)

    # Check if requirements are installed
    try:
        import streamlit
        import ultralytics
        import cv2
        import numpy
        import requests
        from PIL import Image
        import pandas
    except ImportError as e:
        print("❌ Missing dependencies. Please run 'pip install -r requirements.txt' first.")
        print(f"Missing module: {e}")
        sys.exit(1)

    print("🚀 Starting YOLO Vision Models Platform...")
    print("📱 App will open in your default browser at http://localhost:8501")
    print("❌ Press Ctrl+C to stop the server")
    print("-" * 60)

    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"],
                      cwd=os.getcwd(), check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
