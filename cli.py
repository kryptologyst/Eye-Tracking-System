"""
Command Line Interface for Eye Tracking System

This module provides a command-line interface for the eye tracking system,
making it easy to run from the terminal with various options.
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from eye_tracking import EyeTracker
from data.synthetic.generator import SyntheticDatasetGenerator


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Modern Eye Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run real-time eye tracking
  python cli.py --mode realtime
  
  # Generate synthetic data
  python cli.py --mode synthetic --pattern circular --duration 30
  
  # Run with custom config
  python cli.py --mode realtime --config custom_config.yaml
  
  # Run web interface
  python cli.py --mode web
        """
    )
    
    # Main mode selection
    parser.add_argument(
        "--mode",
        choices=["realtime", "synthetic", "web", "test"],
        default="realtime",
        help="Mode to run the eye tracking system"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    # Synthetic data generation options
    parser.add_argument(
        "--pattern",
        choices=["random", "circular", "horizontal", "vertical"],
        default="random",
        help="Eye movement pattern for synthetic data"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of synthetic data in seconds"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for synthetic data"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic",
        help="Output directory for synthetic data"
    )
    
    # Camera options
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height"
    )
    
    # Testing options
    parser.add_argument(
        "--test-pattern",
        type=str,
        help="Run specific test pattern"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def run_realtime_mode(args) -> None:
    """Run real-time eye tracking mode."""
    print("🎬 Starting real-time eye tracking...")
    print(f"📷 Camera: {args.camera}")
    print(f"📐 Resolution: {args.width}x{args.height}")
    print("Press ESC to exit")
    
    try:
        tracker = EyeTracker(args.config)
        tracker.run_realtime()
    except KeyboardInterrupt:
        print("\n⏹️ Eye tracking stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_synthetic_mode(args) -> None:
    """Run synthetic data generation mode."""
    print("🎭 Generating synthetic eye tracking data...")
    print(f"📊 Pattern: {args.pattern}")
    print(f"⏱️ Duration: {args.duration}s")
    print(f"🎞️ FPS: {args.fps}")
    print(f"📁 Output: {args.output}")
    
    try:
        generator = SyntheticDatasetGenerator(
            frame_width=args.width,
            frame_height=args.height,
            duration=args.duration,
            fps=args.fps
        )
        
        # Generate data
        data = generator.generate_synthetic_data(pattern_type=args.pattern)
        
        # Save dataset
        generator.save_dataset(data, args.output)
        
        # Create video
        video_path = f"{args.output}/synthetic_eye_tracking.mp4"
        generator.create_synthetic_video(data, video_path)
        
        print("✅ Synthetic data generation complete!")
        print(f"📊 Generated {len(data)} data points")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_web_mode(args) -> None:
    """Run web interface mode."""
    print("🌐 Starting web interface...")
    print("Open your browser and go to: http://localhost:8501")
    
    try:
        import subprocess
        import os
        
        # Change to web_app directory
        web_app_dir = Path(__file__).parent / "web_app"
        os.chdir(web_app_dir)
        
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_test_mode(args) -> None:
    """Run test mode."""
    print("🧪 Running tests...")
    
    try:
        import subprocess
        
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v",
            "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    required_packages = [
        "cv2", "dlib", "numpy", "yaml", "loguru"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "dlib":
                import dlib
            elif package == "numpy":
                import numpy
            elif package == "yaml":
                import yaml
            elif package == "loguru":
                import loguru
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Print header
    print("👁️ Modern Eye Tracking System")
    print("=" * 40)
    
    if args.verbose:
        print(f"🔧 Configuration: {args.config}")
        print(f"🎯 Mode: {args.mode}")
    
    # Run appropriate mode
    if args.mode == "realtime":
        run_realtime_mode(args)
    elif args.mode == "synthetic":
        run_synthetic_mode(args)
    elif args.mode == "web":
        run_web_mode(args)
    elif args.mode == "test":
        run_test_mode(args)
    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
