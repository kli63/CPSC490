#!/usr/bin/env python3
"""
Simple script to run the renderer with a model.npz file.
Automatically handles both GUI and headless modes.
"""
import os
import subprocess
import sys
import argparse
from pathlib import Path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the SDF renderer with a model file')
    parser.add_argument('model_path', nargs='?', default='improved_model.npz', 
                        help='Path to the model .npz file (default: improved_model.npz)')
    parser.add_argument('--headless', '-H', action='store_true',
                        help='Run in headless mode without GUI')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Build directory
    build_dir = script_dir / "build"
    
    # Check if build directory exists
    if not build_dir.exists():
        print(f"Build directory not found at {build_dir}")
        print("Run build.sh first to compile the renderer.")
        return 1
    
    # Check if sdfRenderer exists
    renderer_path = build_dir / "sdfRenderer"
    if not renderer_path.exists():
        print(f"Renderer not found at {renderer_path}")
        print("Run build.sh first to compile the renderer.")
        return 1
    
    # Determine model path
    if args.model_path:
        # If it's a relative path and doesn't exist directly, try to resolve against script dir
        model_path = Path(args.model_path)
        if not model_path.is_absolute() and not model_path.exists():
            potential_path = script_dir / args.model_path
            if potential_path.exists():
                model_path = potential_path
    else:
        # Default model path
        model_path = script_dir / "improved_model.npz"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please provide a valid model path.")
        return 1
    
    # Prepare command
    cmd = [str(renderer_path)]
    if args.headless:
        cmd.append("--headless")
    cmd.append(str(model_path))
    
    # Run the renderer
    print(f"Running renderer with model: {model_path}")
    print(f"Mode: {'Headless' if args.headless else 'GUI'}")
    
    try:
        result = subprocess.run(cmd, cwd=build_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\nRenderer stopped by user.")
        return 0
    except Exception as e:
        print(f"Error running renderer: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())