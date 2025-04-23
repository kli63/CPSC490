#!/usr/bin/env python3
"""
Module for generating videos from rendered 3D models.
Contains utilities for rotation videos and spherical view videos.
"""
import os
import argparse
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm

def ensure_dir(dir_path):
    """Make sure directory exists."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def generate_rotation_video(model_path, 
                          frames=60, 
                          fps=30, 
                          name=None, 
                          output_dir="_results/videos",
                          temp_dir="_temp_render",
                          rotation_axis="y",
                          cam_radius=4.0,
                          net="OctreeSDF",
                          feature_dim=None,
                          num_lods=None,
                          use_legacy_output=False,
                          camera_view="angle",
                          custom_camera_origin=None):
    """Generate a video showing a 360° rotation of the model."""
    
    # Get NGLOD path
    script_dir = Path(__file__).parent.parent.parent
    nglod_path = os.path.join(script_dir, 'nglod/sdf-net')
    
    # Determine model name for output naming
    if name is None:
        name = Path(model_path).stem
    
    # Create output directories
    temp_dir = Path(temp_dir)
    ensure_dir(temp_dir)
    ensure_dir(output_dir)
    
    # Path to the sdf_renderer.py script in nglod/sdf-net
    renderer_path = Path(os.path.join(nglod_path, "app/sdf_renderer.py"))
    
    # Check if renderer exists
    if not renderer_path.exists():
        raise FileNotFoundError(f"Error: Renderer not found at {renderer_path}")
    
    # Auto-detect model parameters if they're not provided
    if feature_dim is None or num_lods is None:
        try:
            import torch
            model_state = torch.load(model_path)
            
            # Check if this is a state dict
            if isinstance(model_state, dict):
                # Auto-detect feature_dim from model
                if feature_dim is None:
                    for key in model_state.keys():
                        if 'features.0.fm' in key:
                            # Get feature dimension from tensor shape
                            feature_dim = model_state[key].shape[1]
                            print(f"Auto-detected feature_dim={feature_dim} from model")
                            break
                    
                    # Use default if not found
                    if feature_dim is None:
                        feature_dim = 32
                        print(f"Using default feature_dim={feature_dim}")
                
                # Auto-detect num_lods from model
                if num_lods is None:
                    # Count the number of feature modules
                    feature_count = 0
                    for key in model_state.keys():
                        if 'features.' in key and '.fm' in key:
                            feature_idx = int(key.split('.')[1])
                            feature_count = max(feature_count, feature_idx + 1)
                    
                    if feature_count > 0:
                        num_lods = feature_count
                        print(f"Auto-detected num_lods={num_lods} from model")
                    else:
                        num_lods = 5
                        print(f"Using default num_lods={num_lods}")
        except Exception as e:
            print(f"Could not auto-detect model parameters: {e}")
            # Set defaults if auto-detection fails
            if feature_dim is None:
                feature_dim = 32
                print(f"Using default feature_dim={feature_dim}")
            if num_lods is None:
                num_lods = 5
                print(f"Using default num_lods={num_lods}")
    
    # Generate rotation frames
    print(f"Generating {frames} rotation frames around the {rotation_axis}-axis...")
    
    # Define camera origin based on selected view
    camera_origins = {
        "front": [0, 0, cam_radius],          # Front view (z-axis)
        "top": [0, cam_radius, 0],            # Top-down view (y-axis)
        "side": [cam_radius, 0, 0],           # Side view (x-axis)
        "diagonal": [-cam_radius/1.732, cam_radius/1.732, -cam_radius/1.732],  # Diagonal view (3/4 view)
        "custom": custom_camera_origin if custom_camera_origin else [-cam_radius/1.732, cam_radius/1.732, -cam_radius/1.732]
    }
    
    # Choose the camera origin based on the view option
    selected_origin = camera_origins.get(camera_view, camera_origins["diagonal"])
    
    print(f"Using camera view: {camera_view} at position {selected_origin}")
    
    # Command parameters
    cmd = [
        "python", str(renderer_path),
        "--pretrained", model_path,
        "--net", net, 
        "--tracer", "SphereTracer",
        "--img-dir", str(temp_dir),
        "--r360",  # Enable rotation rendering
        "--nb-poses", str(frames),  # Number of poses to render
        "--feature-dim", str(feature_dim),  # Always include feature_dim
        "--num-lods", str(num_lods),   # Always include num_lods
        "--camera-origin", str(selected_origin[0]), str(selected_origin[1]), str(selected_origin[2])
    ]
    
    # Run the renderer to generate all frames
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Capture output to look for specific error patterns
        output = []
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            output.append(line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            # Check for model loading error
            error_message = "\n".join(output)
            if "Error(s) in loading state_dict" in error_message and "Unexpected key(s) in state_dict" in error_message:
                print("\n⚠️ ERROR: Model parameter mismatch detected!")
                print("Try using different --feature-dim and --num-lods values to match the trained model.")
                print(f"Current values: feature_dim={feature_dim}, num_lods={num_lods}")
                
                # Extract keys from error message to suggest parameters
                import re
                feature_keys = re.findall(r'features\.(\d+)\.fm', error_message)
                if feature_keys:
                    max_lod = max([int(k) for k in feature_keys]) + 1
                    print(f"Suggested parameters based on error: --num-lods={max_lod}")
                
                print("\nExample command:")
                print(f"./src/scripts/render_video.py {model_path} --name {name} --frames {frames} --feature-dim 32 --num-lods 5")
            
            return False
        
        print("Rendering completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running renderer: {e}")
        return False
    
    # Build the video from frames
    frames_dir = temp_dir / name / "rgb"
    output_path = Path(output_dir) / f"{name}_rotation.mp4"
    
    if not frames_dir.exists():
        print(f"Error: Rendered frames not found at {frames_dir}")
        return False
    
    # Use ffmpeg to create video
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob", 
        "-i", f"{frames_dir}/*.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video created successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False
    
    # Clean up temporary files (if desired)
    if temp_dir.exists():
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    
    return True

def generate_sphere_view_video(model_path, 
                             poses=64, 
                             fps=30, 
                             name=None, 
                             output_dir="_results/videos",
                             temp_dir="_temp_render",
                             cam_radius=4.0,
                             net="OctreeSDF",
                             feature_dim=None,
                             num_lods=None,
                             use_legacy_output=False,
                             camera_view="angle",
                             custom_camera_origin=None):
    """Generate a video showing views from around a sphere."""
    
    # Get NGLOD path
    script_dir = Path(__file__).parent.parent.parent
    nglod_path = os.path.join(script_dir, 'nglod/sdf-net')
    
    # Determine model name for output naming
    if name is None:
        name = Path(model_path).stem
    
    # Create output directories
    temp_dir = Path(temp_dir)
    ensure_dir(temp_dir)
    ensure_dir(output_dir)
    
    # Path to the sdf_renderer.py script in nglod/sdf-net
    renderer_path = Path(os.path.join(nglod_path, "app/sdf_renderer.py"))
    
    # Check if renderer exists
    if not renderer_path.exists():
        raise FileNotFoundError(f"Error: Renderer not found at {renderer_path}")
    
    # Auto-detect model parameters if they're not provided
    if feature_dim is None or num_lods is None:
        try:
            import torch
            model_state = torch.load(model_path)
            
            # Check if this is a state dict
            if isinstance(model_state, dict):
                # Auto-detect feature_dim from model
                if feature_dim is None:
                    for key in model_state.keys():
                        if 'features.0.fm' in key:
                            # Get feature dimension from tensor shape
                            feature_dim = model_state[key].shape[1]
                            print(f"Auto-detected feature_dim={feature_dim} from model")
                            break
                    
                    # Use default if not found
                    if feature_dim is None:
                        feature_dim = 32
                        print(f"Using default feature_dim={feature_dim}")
                
                # Auto-detect num_lods from model
                if num_lods is None:
                    # Count the number of feature modules
                    feature_count = 0
                    for key in model_state.keys():
                        if 'features.' in key and '.fm' in key:
                            feature_idx = int(key.split('.')[1])
                            feature_count = max(feature_count, feature_idx + 1)
                    
                    if feature_count > 0:
                        num_lods = feature_count
                        print(f"Auto-detected num_lods={num_lods} from model")
                    else:
                        num_lods = 5
                        print(f"Using default num_lods={num_lods}")
        except Exception as e:
            print(f"Could not auto-detect model parameters: {e}")
            # Set defaults if auto-detection fails
            if feature_dim is None:
                feature_dim = 32
                print(f"Using default feature_dim={feature_dim}")
            if num_lods is None:
                num_lods = 5
                print(f"Using default num_lods={num_lods}")
    
    # Generate sphere viewpoint frames
    print(f"Generating {poses} viewpoints distributed on a sphere...")
    
    # Define camera origin based on selected view - this affects starting point for sphere view
    camera_origins = {
        "front": [0, 0, cam_radius],          # Front view (z-axis)
        "top": [0, cam_radius, 0],            # Top-down view (y-axis)
        "side": [cam_radius, 0, 0],           # Side view (x-axis)
        "diagonal": [-cam_radius/1.732, cam_radius/1.732, -cam_radius/1.732],  # Diagonal view (3/4 view)
        "custom": custom_camera_origin if custom_camera_origin else [-cam_radius/1.732, cam_radius/1.732, -cam_radius/1.732]
    }
    
    # Choose the camera origin based on the view option
    selected_origin = camera_origins.get(camera_view, camera_origins["diagonal"])
    
    print(f"Using camera view: {camera_view} at initial position {selected_origin}")
    
    # Command parameters
    cmd = [
        "python", str(renderer_path),
        "--pretrained", model_path,
        "--net", net, 
        "--tracer", "SphereTracer",
        "--img-dir", str(temp_dir),
        "--rsphere",  # Enable sphere rendering
        "--nb-poses", str(poses),  # Number of poses to render
        "--cam-radius", str(cam_radius),  # Camera distance
        "--feature-dim", str(feature_dim),  # Always include feature_dim
        "--num-lods", str(num_lods),   # Always include num_lods
        "--camera-origin", str(selected_origin[0]), str(selected_origin[1]), str(selected_origin[2])
    ]
    
    # Run the renderer to generate all frames
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Capture output to look for specific error patterns
        output = []
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            output.append(line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            # Check for model loading error
            error_message = "\n".join(output)
            if "Error(s) in loading state_dict" in error_message and "Unexpected key(s) in state_dict" in error_message:
                print("\n⚠️ ERROR: Model parameter mismatch detected!")
                print("Try using different --feature-dim and --num-lods values to match the trained model.")
                print(f"Current values: feature_dim={feature_dim}, num_lods={num_lods}")
                
                # Extract keys from error message to suggest parameters
                import re
                feature_keys = re.findall(r'features\.(\d+)\.fm', error_message)
                if feature_keys:
                    max_lod = max([int(k) for k in feature_keys]) + 1
                    print(f"Suggested parameters based on error: --num-lods={max_lod}")
                
                print("\nExample command:")
                print(f"./src/scripts/render_sphere_views.py {model_path} --name {name} --poses {poses} --feature-dim 32 --num-lods 5")
            
            return False
        
        print("Rendering completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running renderer: {e}")
        return False
    
    # Build the video from frames
    frames_dir = temp_dir / name / "rgb"
    output_path = Path(output_dir) / f"{name}_sphere.mp4"
    
    if not frames_dir.exists():
        print(f"Error: Rendered frames not found at {frames_dir}")
        return False
    
    # Use ffmpeg to create video
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob", 
        "-i", f"{frames_dir}/*.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video created successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False
    
    # Clean up temporary files (if desired)
    if temp_dir.exists():
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    
    return True