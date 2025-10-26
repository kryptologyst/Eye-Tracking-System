#!/usr/bin/env python3
"""
Setup script for Eye Tracking System

This script helps users set up the eye tracking system quickly
by checking dependencies, creating necessary directories, and
providing setup instructions.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import urllib.request
import zipfile


def print_header():
    """Print setup header."""
    print("👁️ Eye Tracking System Setup")
    print("=" * 40)


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "opencv-python",
        "dlib", 
        "numpy",
        "yaml",
        "loguru",
        "streamlit",
        "plotly",
        "pandas",
        "matplotlib",
        "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "opencv-python":
                import cv2
            elif package == "dlib":
                import dlib
            elif package == "numpy":
                import numpy
            elif package == "yaml":
                import yaml
            elif package == "loguru":
                import loguru
            elif package == "streamlit":
                import streamlit
            elif package == "plotly":
                import plotly
            elif package == "pandas":
                import pandas
            elif package == "matplotlib":
                import matplotlib
            elif package == "seaborn":
                import seaborn
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data/models",
        "data/synthetic", 
        "config",
        "tests",
        "exports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")


def download_dlib_predictor():
    """Download dlib facial landmark predictor if not present."""
    print("\n🔽 Checking for dlib predictor...")
    
    predictor_path = Path("data/models/shape_predictor_68_face_landmarks.dat")
    
    if predictor_path.exists():
        print("✅ dlib predictor already exists")
        return True
    
    print("⚠️  dlib predictor not found!")
    print("   Please download shape_predictor_68_face_landmarks.dat from:")
    print("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("   Extract and place it in data/models/")
    
    # Try to download automatically
    try:
        print("   Attempting automatic download...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        
        # Download compressed file
        compressed_path = predictor_path.with_suffix('.dat.bz2')
        urllib.request.urlretrieve(url, compressed_path)
        
        # Extract
        import bz2
        with bz2.BZ2File(compressed_path, 'rb') as source:
            with open(predictor_path, 'wb') as target:
                target.write(source.read())
        
        # Clean up
        compressed_path.unlink()
        
        print("✅ Successfully downloaded and extracted dlib predictor!")
        return True
        
    except Exception as e:
        print(f"❌ Automatic download failed: {e}")
        print("   Please download manually and place in data/models/")
        return False


def test_camera():
    """Test camera availability."""
    print("\n📷 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not available!")
            print("   Check camera permissions and connections")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✅ Camera is working!")
            return True
        else:
            print("❌ Camera not responding!")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def run_basic_test():
    """Run basic functionality test."""
    print("\n🧪 Running basic test...")
    
    try:
        # Test imports
        sys.path.append(str(Path(__file__).parent / "src"))
        from eye_tracking import EyeTracker, EyeTrackingConfig
        
        # Test config loading
        config = EyeTracker.__new__(EyeTracker)._get_default_config()
        print("✅ Configuration system working")
        
        # Test synthetic data generation
        sys.path.append(str(Path(__file__).parent / "data/synthetic"))
        from generator import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator(duration=1.0, fps=10.0)
        data = generator.generate_synthetic_data(pattern_type="random")
        print(f"✅ Synthetic data generation working ({len(data)} points)")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n🚀 Next Steps:")
    print("=" * 40)
    print("1. Run real-time eye tracking:")
    print("   python cli.py --mode realtime")
    print()
    print("2. Start web interface:")
    print("   python cli.py --mode web")
    print()
    print("3. Generate synthetic data:")
    print("   python cli.py --mode synthetic --pattern circular")
    print()
    print("4. Run tests:")
    print("   python cli.py --mode test")
    print()
    print("5. Read the README.md for detailed usage instructions")


def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create directories
    create_directories()
    
    # Download dlib predictor
    predictor_ok = download_dlib_predictor()
    
    # Test camera
    camera_ok = test_camera()
    
    # Run basic test
    test_ok = run_basic_test()
    
    # Summary
    print("\n📋 Setup Summary:")
    print("=" * 40)
    print(f"Python version: {'✅' if sys.version_info >= (3, 8) else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"dlib predictor: {'✅' if predictor_ok else '❌'}")
    print(f"Camera: {'✅' if camera_ok else '❌'}")
    print(f"Basic test: {'✅' if test_ok else '❌'}")
    
    if all([deps_ok, predictor_ok, test_ok]):
        print("\n🎉 Setup completed successfully!")
        print_next_steps()
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("   Please address the issues above before running the system.")
        print_next_steps()


if __name__ == "__main__":
    main()
