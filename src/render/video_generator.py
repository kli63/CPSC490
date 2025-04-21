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
                          use_legacy_output=False):
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
    
    # Generate rotation frames
    print(f"Generating {frames} rotation frames around the {rotation_axis}-axis...")
    
    # Command parameters
    cmd = [
        "python", str(renderer_path),
        "--pretrained", model_path,
        "--net", net, 
        "--tracer", "SphereTracer",
        "--img-dir", str(temp_dir),
        "--r360",  # Enable rotation rendering
        "--nb-poses", str(frames)  # Number of poses to render
    ]
    
    # Add optional parameters if specified
    if feature_dim:
        cmd.extend(["--feature-dim", str(feature_dim)])
    if num_lods:
        cmd.extend(["--num-lods", str(num_lods)])
    
    # Run the renderer to generate all frames
    try:
        subprocess.run(cmd, check=True)
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
                             use_legacy_output=False):
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
    
    # Generate sphere viewpoint frames
    print(f"Generating {poses} viewpoints distributed on a sphere...")
    
    # Command parameters
    cmd = [
        "python", str(renderer_path),
        "--pretrained", model_path,
        "--net", net, 
        "--tracer", "SphereTracer",
        "--img-dir", str(temp_dir),
        "--rsphere",  # Enable sphere rendering
        "--nb-poses", str(poses),  # Number of poses to render
        "--cam-radius", str(cam_radius)  # Camera distance
    ]
    
    # Add optional parameters if specified
    if feature_dim:
        cmd.extend(["--feature-dim", str(feature_dim)])
    if num_lods:
        cmd.extend(["--num-lods", str(num_lods)])
    
    # Run the renderer to generate all frames
    try:
        subprocess.run(cmd, check=True)
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