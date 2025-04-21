#!/usr/bin/env python3
"""
Convert NGLOD Model to Mesh

Command-line utility for converting trained NGLOD models to triangle meshes
that can be imported into Blender or other 3D software.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path to enable imports
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

def main():
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(description="Convert NGLOD model to mesh")
    
    # Input/output options
    parser.add_argument("--model", required=True, help="Path to the .pth model file")
    parser.add_argument("--output", required=True, help="Output mesh file path (.obj or .ply)")
    
    # Mesh generation options
    parser.add_argument("--resolution", type=int, default=256, 
                       help="Grid resolution for marching cubes (higher = more detailed)")
    
    # Model parameters
    parser.add_argument("--feature-dim", type=int, default=32, 
                       help="Feature dimension used when training the model")
    parser.add_argument("--num-lods", type=int, default=5, 
                       help="Number of LOD levels used when training the model")
    
    # Additional options
    parser.add_argument("--gpu", type=int, default=0, 
                       help="GPU device ID to use (if multiple GPUs are available)")
    parser.add_argument("--batch-size", type=int, default=32768, 
                       help="Batch size for SDF evaluation")
    
    args = parser.parse_args()
    
    try:
        # Import here to avoid errors when paths aren't set up
        from nglod_converter import NGLODConverter
        import torch
        
        # Set GPU device if specified
        if torch.cuda.is_available():
            if args.gpu >= 0:
                device = torch.device(f'cuda:{args.gpu}')
            else:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("Warning: CUDA not available, using CPU (this will be very slow)")
        
        # Create converter
        converter = NGLODConverter(device=device)
        
        # Convert model to mesh
        print(f"=== Converting NGLOD Model to Mesh ===")
        print(f"Model: {args.model}")
        print(f"Output: {args.output}")
        print(f"Resolution: {args.resolution}")
        print(f"Feature dimension: {args.feature_dim}")
        print(f"LOD levels: {args.num_lods}")
        print(f"Device: {device}")
        print(f"Batch size: {args.batch_size}")
        print("====================================")
        
        # Load the model
        model = converter.load_model(
            args.model, 
            args.feature_dim,
            args.num_lods
        )
        
        # Sample SDF grid
        sdf_grid = converter.sample_sdf_grid(
            model,
            args.resolution,
            args.batch_size
        )
        
        # Extract mesh with marching cubes
        mesh = converter.extract_mesh(sdf_grid)
        
        # Save the mesh
        converter.save_mesh(mesh, args.output)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"Mesh saved to: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error converting model: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())