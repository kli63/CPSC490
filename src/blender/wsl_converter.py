#!/usr/bin/env python3
"""
WSL Converter for Windows Blender Integration

This script provides a simplified interface for the NGLOD converter
that can be easily called from Windows through the WSL command.

Example usage from Windows:
wsl /path/to/python /path/to/wsl_converter.py --model /path/to/model.pth --output /mnt/c/Temp/output.obj
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

def main():
    """Main entry point for the WSL converter."""
    parser = argparse.ArgumentParser(description="WSL NGLOD model to mesh converter")
    
    # Core arguments
    parser.add_argument("--model", required=True, help="Path to the .pth model file")
    parser.add_argument("--output", required=True, help="Output mesh file path (.obj or .ply)")
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution")
    parser.add_argument("--feature-dim", type=int, default=32, help="Feature dimension")
    parser.add_argument("--num-lods", type=int, default=5, help="Number of LOD levels")
    parser.add_argument("--active-lod", type=int, default=4, help="Which LOD level to use (0=lowest, higher=more detail)")
    
    args = parser.parse_args()
    
    try:
        # Import the converter
        from src.blender.nglod_converter import NGLODConverter
        
        # Print configuration for debugging
        print(f"=== WSL NGLOD Converter ===")
        print(f"Model path: {args.model}")
        print(f"Output path: {args.output}")
        print(f"Resolution: {args.resolution}")
        print(f"Feature dimension: {args.feature_dim}")
        print(f"LOD levels: {args.num_lods}")
        print(f"Active LOD: {args.active_lod}")
        print("=========================")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # Initialize converter
        converter = NGLODConverter()
        
        # Convert model to mesh
        model = converter.load_model(
            args.model, 
            args.feature_dim,
            args.num_lods
        )
        
        # Set the active LOD level
        model.lod = args.active_lod
        print(f"Set active LOD level to: {args.active_lod}")
        
        # Sample SDF grid
        sdf_grid = converter.sample_sdf_grid(model, args.resolution)
        
        # Extract mesh with marching cubes
        mesh = converter.extract_mesh(sdf_grid)
        
        # Save the mesh
        converter.save_mesh(mesh, args.output)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"Mesh saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in WSL converter: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())