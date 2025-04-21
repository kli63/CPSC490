#!/usr/bin/env python3
"""
NGLOD to Mesh Converter

This module handles converting NGLOD neural SDF models to triangle meshes
using the marching cubes algorithm. It provides functionality to:
1. Load trained NGLOD models (.pth files)
2. Sample SDF values on a uniform grid
3. Run marching cubes to extract triangle meshes
4. Export meshes to OBJ/PLY formats for Blender import
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Tuple

# Add project root to path to enable imports
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "nglod/sdf-net"))
sys.path.insert(0, str(script_dir / "shap-e"))

# Import NGLOD and Shap-E components
from lib.models import OctreeSDF
from shap_e.rendering.mc import marching_cubes


class NGLODConverter:
    """Converter for NGLOD neural SDF models to triangle meshes."""
    
    def __init__(self, 
                device: Optional[torch.device] = None):
        """Initialize the converter.
        
        Args:
            device: The device to use for computation (defaults to CUDA if available)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_model(self, 
                  model_path: str, 
                  feature_dim: int = 32,
                  num_lods: int = 5,
                  net_type: str = "OctreeSDF") -> torch.nn.Module:
        """Load a trained NGLOD model.
        
        Args:
            model_path: Path to the .pth model file
            feature_dim: Feature dimension used in training
            num_lods: Number of LOD levels used in training
            net_type: Network architecture name
            
        Returns:
            The loaded model
        """
        print(f"Loading NGLOD model from {model_path}")
        
        # Import NGLOD modules
        from lib.models import OctreeSDF, BaseSDF, BaseLOD
        
        # Create model args
        args = type('Args', (), {})()
        args.feature_dim = feature_dim
        args.num_lods = num_lods
        args.net = net_type
        
        # Additional params for OctreeSDF initialization
        args.feature_size = 4
        args.hidden_dim = 128
        args.pos_invariant = False
        args.interpolate = None
        args.base_lod = 2
        
        # Parameters for BaseSDF
        args.input_dim = 3
        args.ff_dim = -1
        args.ff_width = 16.0
        args.pos_enc = False
        args.skip = None
        args.num_layers = 1
        args.joint_feature = False
        args.joint_decoder = False
        args.periodic = False
        args.freeze = -1
        args.feat_sum = False
        args.grow_every = -1
        args.growth_strategy = "increase"
        
        # Create model
        if net_type == "OctreeSDF":
            net = OctreeSDF(args)
        else:
            raise ValueError(f"Unsupported model type: {net_type}")
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        net.load_state_dict(state_dict)
        
        # Set model for inference
        net.eval()
        net = net.to(self.device)
        
        print(f"Model loaded successfully with {sum(p.numel() for p in net.parameters())} parameters")
        return net
    
    def sample_sdf_grid(self, 
                      model: torch.nn.Module, 
                      resolution: int = 256, 
                      batch_size: int = 32768) -> torch.Tensor:
        """Sample SDF values on a uniform grid.
        
        Args:
            model: The loaded NGLOD model
            resolution: Grid resolution (higher = more detailed mesh)
            batch_size: Batch size for SDF evaluation
            
        Returns:
            A 3D grid of SDF values with shape (resolution, resolution, resolution)
        """
        print(f"Sampling SDF grid at resolution {resolution}³")
        
        # Create a uniform 3D grid of query points
        grid = torch.linspace(-1, 1, resolution, device=self.device)
        try:
            # For PyTorch 1.10+ with indexing parameter
            X, Y, Z = torch.meshgrid(grid, grid, grid, indexing='ij')
        except TypeError:
            # For older PyTorch versions (default is ij indexing)
            X, Y, Z = torch.meshgrid(grid, grid, grid)
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        
        # Evaluate SDF values in batches to avoid OOM
        sdf_values = []
        
        with torch.no_grad():
            for i in range(0, points.shape[0], batch_size):
                batch_points = points[i:i+batch_size]
                batch_sdf = model(batch_points)  # Evaluate SDF at these points
                sdf_values.append(batch_sdf)
        
        # Reshape to 3D grid
        sdf_grid = torch.cat(sdf_values, dim=0).reshape(resolution, resolution, resolution)
        
        print(f"SDF grid sampled successfully")
        return sdf_grid
    
    def extract_mesh(self, 
                   sdf_grid: torch.Tensor,
                   min_point: torch.Tensor = None,
                   size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a triangle mesh from the SDF grid using marching cubes.
        
        Args:
            sdf_grid: 3D grid of SDF values
            min_point: Minimum point of the bounding box
            size: Size of the bounding box
            
        Returns:
            A tuple of (vertices, faces, normals)
        """
        print("Extracting mesh using marching cubes")
        
        # Set default bounding box if not provided
        if min_point is None:
            min_point = torch.tensor([-1, -1, -1], device=self.device)
        if size is None:
            size = torch.tensor([2, 2, 2], device=self.device)
        
        # Run marching cubes
        torch_mesh = marching_cubes(
            sdf_grid,
            min_point, 
            size
        )
        
        # Get mesh components
        verts = torch_mesh.verts
        faces = torch_mesh.faces
        
        # Compute vertex normals
        normals = torch.nn.functional.normalize(verts)
        
        print(f"Mesh extracted with {verts.shape[0]} vertices and {faces.shape[0]} faces")
        return torch_mesh
    
    def save_mesh(self, 
                torch_mesh, 
                output_path: str):
        """Save the mesh to a file.
        
        Args:
            torch_mesh: The mesh from marching cubes
            output_path: Path to save the mesh
        """
        print(f"Saving mesh to {output_path}")
        
        # Convert to CPU triangle mesh
        tri_mesh = torch_mesh.tri_mesh()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine file type and save
        if output_path.lower().endswith('.obj'):
            with open(output_path, 'w') as f:
                tri_mesh.write_obj(f)
        elif output_path.lower().endswith('.ply'):
            with open(output_path, 'wb') as f:
                tri_mesh.write_ply(f)
        else:
            raise ValueError(f"Unsupported file format: {output_path}")
        
        print(f"Mesh saved successfully to {output_path}")
    
    def convert_model_to_mesh(self, 
                            model_path: str, 
                            output_path: str,
                            resolution: int = 256,
                            feature_dim: int = 32,
                            num_lods: int = 5):
        """Convert a NGLOD model to a triangle mesh.
        
        Args:
            model_path: Path to the .pth model file
            output_path: Path to save the mesh
            resolution: Grid resolution for marching cubes
            feature_dim: Feature dimension used in training
            num_lods: Number of LOD levels used in training
        """
        # Load the model
        model = self.load_model(model_path, feature_dim, num_lods)
        
        # Sample SDF grid
        sdf_grid = self.sample_sdf_grid(model, resolution)
        
        # Extract mesh
        mesh = self.extract_mesh(sdf_grid)
        
        # Save mesh
        self.save_mesh(mesh, output_path)
        
        return True


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert NGLOD model to mesh")
    parser.add_argument("--model", required=True, help="Path to the .pth model file")
    parser.add_argument("--output", required=True, help="Output mesh file path (.obj or .ply)")
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution")
    parser.add_argument("--feature-dim", type=int, default=32, help="Feature dimension")
    parser.add_argument("--num-lods", type=int, default=5, help="Number of LOD levels")
    
    args = parser.parse_args()
    
    converter = NGLODConverter()
    converter.convert_model_to_mesh(
        args.model,
        args.output,
        args.resolution,
        args.feature_dim,
        args.num_lods
    )