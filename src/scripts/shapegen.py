#!/usr/bin/env python3
"""
Unified 3D Generation and LOD Pipeline

This script provides a complete workflow for:
1. Generating 3D objects from text descriptions using Shap-E
2. Creating LOD representations using NGLOD
3. Rendering the optimized models

Compatible with RTX 4050 GPUs.
"""

import os
import sys
import argparse

# Add the project root to the path
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, script_dir)

from src.pipeline import Pipeline
from src.utils.dependencies import check_dependencies

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified 3D Generation and LOD Pipeline')
    
    # General options
    parser.add_argument('--skip-generate', action='store_true', help='Skip 3D generation step')
    parser.add_argument('--skip-train', action='store_true', help='Skip LOD training step')
    parser.add_argument('--skip-render', action='store_true', help='Skip rendering step')
    parser.add_argument('--existing-model', type=str, help='Path to existing trained model (.pth)')
    parser.add_argument('--skip-dependency-check', action='store_true', help='Skip dependency checking')
    parser.add_argument('--use-legacy-output', action='store_true', help='Use the legacy directory structure instead of unified _results')
    
    # Test mode options
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode with multiple predefined prompts')
    parser.add_argument('--test-prompts', type=str,
                       help='Comma-separated list of test prompts (no spaces after commas)')
    
    # Shap-E options
    shap_e_group = parser.add_argument_group('Shap-E')
    shap_e_group.add_argument('--text', type=str, default='a chair', help='Text prompt for 3D generation')
    shap_e_group.add_argument('--guidance-scale', type=float, default=15.0, help='Guidance scale for generation')
    shap_e_group.add_argument('--karras-steps', type=int, default=64, help='Number of Karras steps')
    shap_e_group.add_argument('--output', type=str, default=None, help='Output OBJ file path (if not specified, will be named based on the text prompt)')
    
    # NGLOD training options
    nglod_train_group = parser.add_argument_group('NGLOD Training')
    nglod_train_group.add_argument('--net', type=str, default='OctreeSDF', help='Network architecture')
    nglod_train_group.add_argument('--num-lods', type=int, default=5, help='Number of LOD levels')
    nglod_train_group.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    nglod_train_group.add_argument('--feature-dim', type=int, default=None, help='Feature dimension')
    nglod_train_group.add_argument('--exp-name', type=str, default='generated_model', help='Experiment name')
    nglod_train_group.add_argument('--input-obj', type=str, default=None, help='Input OBJ file path (overrides --output)')
    
    # NGLOD rendering options
    nglod_render_group = parser.add_argument_group('NGLOD Rendering')
    nglod_render_group.add_argument('--render-width', type=int, default=1280, help='Render width')
    nglod_render_group.add_argument('--render-height', type=int, default=720, help='Render height')
    nglod_render_group.add_argument('--shading-mode', type=str, default='matcap', help='Shading mode')
    nglod_render_group.add_argument('--lod', type=int, default=4, help='Specific LOD level to render')
    nglod_render_group.add_argument('--export-model', action='store_true', help='Export model to NPZ format')
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies(args.skip_dependency_check)
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Run the appropriate pipeline
    if args.test_mode:
        # Parse test prompts
        test_prompts = None
        if args.test_prompts:
            test_prompts = args.test_prompts.split(',')
            
        success = pipeline.run_test_mode(
            test_prompts=test_prompts,
            epochs=args.epochs,
            skip_generate=args.skip_generate,
            skip_train=args.skip_train,
            skip_render=args.skip_render,
            net=args.net,
            num_lods=args.num_lods,
            feature_dim=args.feature_dim,
            use_legacy_output=args.use_legacy_output
        )
    else:
        success = pipeline.run_pipeline(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())