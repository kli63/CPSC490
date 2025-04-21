#!/usr/bin/env python3
"""
RTX 4050 Compatibility Test for NGLOD Rendering
This script tests the rendering capabilities of NGLOD on RTX 4050 GPUs.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Add nglod/sdf-net directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'nglod/sdf-net'))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def test_model_export(model_path, output_path):
    """Test exporting a trained model to NPZ format for the C++ renderer."""
    print_header("Testing Model Export")
    
    try:
        # Import required modules
        from lib.models.SOL_NGLOD import SOL_NGLOD
        from lib.models.OctreeSDF import OctreeSDF
        from lib.options import parse_options
        
        # Parse options with default arguments
        args, _ = parse_options()
        args.net = "OctreeSDF"
        args.num_lods = 5
        args.pretrained = model_path
        
        # Load the trained model
        print(f"Loading model from {model_path}")
        model = OctreeSDF(args)
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        
        # Convert to SOL_NGLOD for export
        print("Converting to SOL_NGLOD format...")
        sol_model = SOL_NGLOD(model)
        
        # Export the model
        print(f"Exporting to {output_path}...")
        sol_model.save(output_path)
        print(f"✅ Model exported successfully to {output_path}")
        
        # Verify the exported model
        print("Verifying exported model...")
        data = np.load(output_path, allow_pickle=True)
        print(f"Exported model contains {len(data.keys())} arrays")
        print(f"Model keys: {sorted(data.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Model export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rendering(model_path, output_dir):
    """Test rendering a trained model."""
    print_header("Testing Rendering")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import required modules
        from lib.models.OctreeSDF import OctreeSDF
        from lib.options import parse_options
        from lib.tracer.SphereTracer import SphereTracer
        from lib.renderer import Renderer
        
        # Parse options with default arguments
        args, _ = parse_options()
        args.net = "OctreeSDF"
        args.num_lods = 5
        args.pretrained = model_path
        args.render_res = [800, 600]
        args.camera_origin = [0, 1, 4]
        args.camera_lookat = [0, 0, 0]
        args.camera_fov = 30
        args.tracer = "SphereTracer"
        args.shading_mode = "matcap"
        
        # Load model
        print(f"Loading model from {model_path}")
        model = OctreeSDF(args)
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        
        # Create tracer and renderer
        tracer = SphereTracer(args)
        renderer = Renderer(tracer, args=args, device='cuda')
        
        # Render image
        print("Rendering image...")
        start_time = time.time()
        out = renderer.shade_images(
            net=model,
            f=args.camera_origin,
            t=args.camera_lookat,
            fov=args.camera_fov,
            aa=True
        )
        render_time = time.time() - start_time
        print(f"Rendering completed in {render_time:.2f} seconds")
        
        # Save images
        img_out = out.image().byte().numpy()
        model_name = os.path.basename(model_path).split('.')[0]
        
        rgb_path = os.path.join(output_dir, f"{model_name}_rgb.png")
        depth_path = os.path.join(output_dir, f"{model_name}_depth.png")
        normal_path = os.path.join(output_dir, f"{model_name}_normal.png")
        hit_path = os.path.join(output_dir, f"{model_name}_hit.png")
        
        Image.fromarray(img_out.rgb).save(rgb_path, mode='RGB')
        Image.fromarray(img_out.depth).save(depth_path, mode='RGB')
        Image.fromarray(img_out.normal).save(normal_path, mode='RGB')
        Image.fromarray(img_out.hit).save(hit_path, mode='L')
        
        print(f"✅ Images saved to {output_dir}")
        print(f"  RGB: {rgb_path}")
        print(f"  Depth: {depth_path}")
        print(f"  Normal: {normal_path}")
        print(f"  Hit: {hit_path}")
        
        return True
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """Test rendering for all available models."""
    print_header("RTX 4050 NGLOD Rendering Test")
    
    # System info
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU architecture: {torch.cuda.get_device_capability(0)}")
    
    # Check CUDA extensions
    try:
        import mesh2sdf
        import sol_nglod
        print("✅ CUDA extensions loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to load CUDA extensions: {e}")
        return 1
    
    # Find all models
    models_dir = os.path.join(script_dir, 'nglod/sdf-net', '_results', 'models')
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return 1
    
    model_files = list(Path(models_dir).glob('*.pth'))
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        return 1
    
    # Output directory
    output_dir = os.path.join(script_dir, 'nglod/sdf-net', '_results', 'rtx_render_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each model
    success_count = 0
    for model_path in model_files:
        model_name = model_path.stem
        print(f"\nTesting model: {model_name}")
        
        # Test export
        npz_path = os.path.join(output_dir, f"{model_name}.npz")
        export_success = test_model_export(str(model_path), npz_path)
        
        # Test rendering
        render_success = test_rendering(str(model_path), output_dir)
        
        if export_success and render_success:
            success_count += 1
    
    # Summary
    print_header("Test Summary")
    print(f"Total models tested: {len(model_files)}")
    print(f"Successfully rendered: {success_count}")
    
    if success_count == len(model_files):
        print("\n✅ All models rendered successfully on RTX 4050 GPU!")
        return 0
    else:
        print(f"\n⚠️ {len(model_files) - success_count} models failed to render correctly.")
        return 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test NGLOD rendering on RTX 4050')
    parser.add_argument('--model', type=str, help='Specific model to test (path to .pth file)')
    parser.add_argument('--output-dir', type=str, default='nglod/sdf-net/_results/rtx_render_test',
                       help='Output directory for rendered images')
    args = parser.parse_args()
    
    if args.model:
        # Test specific model
        output_dir = os.path.join(script_dir, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = os.path.basename(args.model).split('.')[0]
        npz_path = os.path.join(output_dir, f"{model_name}.npz")
        
        export_success = test_model_export(args.model, npz_path)
        render_success = test_rendering(args.model, output_dir)
        
        if export_success and render_success:
            print("\n✅ Model rendered successfully on RTX 4050 GPU!")
            return 0
        else:
            print("\n❌ Failed to render model on RTX 4050 GPU.")
            return 1
    else:
        # Test all models
        return test_all_models()

if __name__ == "__main__":
    sys.exit(main())