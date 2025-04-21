#!/usr/bin/env python3
"""
Script to render a model from viewpoints distributed on a sphere and stitch into a video.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, script_dir)

from src.render.video_generator import generate_sphere_view_video

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a video from multiple renders around a sphere')
    parser.add_argument('model_path', help='Path to the model file (.pth or .npz)')
    parser.add_argument('--name', default=None, help='Name for the output (default: model filename)')
    parser.add_argument('--output-dir', default='_results/videos', help='Directory to save the video')
    parser.add_argument('--use-legacy-output', action='store_true', help='Use legacy directory structure instead of unified _results')
    parser.add_argument('--poses', type=int, default=64, help='Number of viewpoints to render')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second in output video')
    parser.add_argument('--cam-radius', type=float, default=4.0, help='Camera distance from object')
    parser.add_argument('--temp-dir', default='_temp_render', help='Temporary directory for frames')
    parser.add_argument('--net', default='OctreeSDF', help='Network architecture (OctreeSDF, etc.)')
    parser.add_argument('--feature-dim', type=int, default=None, help='Feature dimension used in training')
    parser.add_argument('--num-lods', type=int, default=None, help='Number of LODs used in training')
    args = parser.parse_args()
    
    # Generate video
    success = generate_sphere_view_video(
        model_path=args.model_path,
        poses=args.poses,
        fps=args.fps,
        name=args.name,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        cam_radius=args.cam_radius,
        net=args.net,
        feature_dim=args.feature_dim,
        num_lods=args.num_lods,
        use_legacy_output=args.use_legacy_output
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())