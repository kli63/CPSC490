#!/usr/bin/env python3
"""
Core 3D generation and training pipeline.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import subprocess
import multiprocessing
from typing import Dict, Optional, List, Union, Any, Tuple

# Add module paths
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NGLOD_PATH = os.path.join(SCRIPT_DIR, 'nglod/sdf-net')
SHAPE_PATH = os.path.join(SCRIPT_DIR, 'shap-e')

sys.path.insert(0, NGLOD_PATH)
sys.path.insert(0, SHAPE_PATH)

from .utils.dependencies import print_header, check_dependencies

# Set multiprocessing start method at module load
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    # Context already set, ignore
    pass


class Pipeline:
    """3D generation and LOD pipeline."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shape_e_models = None
        
    def load_shape_e_models(self):
        """Load Shap-E models once to avoid reloading."""
        if self.shape_e_models is not None:
            return self.shape_e_models
            
        print(f"Using device: {self.device}")
        print("Loading Shap-E models...")
        
        # Import Shap-E modules
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config
        
        # Load models
        xm = load_model('transmitter', device=self.device)
        model = load_model('text300M', device=self.device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        
        self.shape_e_models = {
            'xm': xm, 
            'model': model, 
            'diffusion': diffusion, 
            'device': self.device
        }
        
        return self.shape_e_models
    
    def generate_3d_from_text(self, 
                           text: str,
                           output_path: str = None,
                           guidance_scale: float = 15.0,
                           karras_steps: int = 64,
                           use_legacy_output: bool = False) -> Optional[str]:
        """Generate a 3D object from a text prompt using Shap-E."""
        print_header(f"Generating 3D Model from Text: '{text}'")
        
        # Generate filename from text if not provided
        if output_path is None:
            sanitized_name = text.replace(" ", "_").replace("'", "").replace('"', "").replace(",", "")
            if use_legacy_output:
                output_path = f"output/{sanitized_name}.obj"
            else:
                output_path = f"_results/meshes/{sanitized_name}.obj"
        
        try:
            # Ensure models are loaded
            models = self.load_shape_e_models()
            
            # Import models
            from shap_e.diffusion.sample import sample_latents
            from shap_e.util.notebooks import decode_latent_mesh
            
            print(f"Generating 3D model from text prompt: '{text}'")
            
            # Sample latents
            latents = sample_latents(
                batch_size=1,
                model=models['model'],
                diffusion=models['diffusion'],
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[text]),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=karras_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export to OBJ
            latent = latents[0]
            print(f"Converting latent to mesh...")
            t = decode_latent_mesh(models['xm'], latent).tri_mesh()
            
            print(f"Saving mesh to {output_path}...")
            with open(output_path, 'w') as f:
                t.write_obj(f)
            
            print(f"✅ 3D model successfully generated and saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to generate 3D model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_lod(self, 
               input_obj_path: str,
               exp_name: str = "generated_model",
               net: str = "OctreeSDF",
               num_lods: int = 5,
               epochs: int = 5,
               feature_dim: Optional[int] = None,
               use_legacy_output: bool = False) -> Optional[str]:
        """Train an LOD model on the generated 3D object using NGLOD."""
        print_header(f"Training LOD Model: {exp_name}")
        
        try:
            # Import NGLOD modules
            from lib.trainer import Trainer
            from lib.options import parse_options
            
            # Monkey patch multiprocessing to avoid "context already set" error
            import multiprocessing
            original_set_start_method = multiprocessing.set_start_method
            
            def patched_set_start_method(*args, **kwargs):
                try:
                    original_set_start_method(*args, **kwargs)
                except RuntimeError:
                    # Already set, ignore the error
                    pass
                    
            # Apply the patch
            multiprocessing.set_start_method = patched_set_start_method
            
            # Ensure the model output directory exists
            if use_legacy_output:
                model_dir = os.path.join(NGLOD_PATH, "_results", "models")
                os.makedirs(model_dir, exist_ok=True)
            else:
                # Creating absolute paths for models directory
                model_dir = os.path.join(os.path.abspath("_results"), "models")
                os.makedirs(model_dir, exist_ok=True)
            
            # Direct training with subprocess for better log visibility and reliability
            print("Running NGLOD training...")
            # Convert input path to absolute path to avoid CWD issues
            input_obj_path = os.path.abspath(input_obj_path)
            
            cmd = [
                sys.executable,
                os.path.join(NGLOD_PATH, "app", "main.py"),
                "--net", net,
                "--num-lods", str(num_lods),
                "--epochs", str(epochs),
                "--dataset-path", input_obj_path,
                "--exp-name", exp_name,
                "--matcap-path", os.path.join(NGLOD_PATH, "data/matcap/green.png")
            ]
            
            if feature_dim is not None:
                cmd.extend(["--feature-dim", str(feature_dim)])
            
            # Run the training process with output displayed
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=NGLOD_PATH  # Run from NGLOD directory
            )
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                raise Exception(f"Training process exited with code {return_code}")
                
            # Output model path and verify it exists
            if use_legacy_output:
                model_path = os.path.join(NGLOD_PATH, "_results", "models", f"{exp_name}.pth")
            else:
                # Check both possible locations since training happens in the NGLOD directory
                model_path = os.path.join(SCRIPT_DIR, "_results", "models", f"{exp_name}.pth")
                legacy_model_path = os.path.join(NGLOD_PATH, "_results", "models", f"{exp_name}.pth")
                
                # If the unified path doesn't exist but the legacy path does, copy the file
                if not os.path.isfile(model_path) and os.path.isfile(legacy_model_path):
                    import shutil
                    print(f"Model found in legacy location, copying to unified location")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    shutil.copy2(legacy_model_path, model_path)
            
            # Verify the file exists
            if not os.path.isfile(model_path):
                # Final check for legacy path as fallback
                legacy_model_path = os.path.join(NGLOD_PATH, "_results", "models", f"{exp_name}.pth")
                if os.path.isfile(legacy_model_path):
                    print(f"Warning: Model found at {legacy_model_path} but not at expected location {model_path}")
                    model_path = legacy_model_path
                else:
                    raise FileNotFoundError(f"Expected model file not found at {model_path}")
            
            # Success - model was created
            print(f"✅ LOD model trained successfully: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Failed to train LOD model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def render_model(self, 
                  model_path: str,
                  exp_name: str = "generated_model",
                  net: str = "OctreeSDF",
                  num_lods: int = 5,
                  render_width: int = 1280,
                  render_height: int = 720,
                  shading_mode: str = "matcap",
                  lod: Optional[int] = 4,
                  feature_dim: Optional[int] = None,
                  export_model: bool = False,
                  use_legacy_output: bool = False) -> bool:
        """Render the trained LOD model."""
        print_header(f"Rendering LOD Model: {os.path.basename(model_path)}")
        
        try:
            # First, verify the model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            # Run the renderer as a subprocess for better visibility and reliability
            print(f"Running NGLOD renderer for {model_path}...")
            
            # Determine output directory based on whether legacy output is used
            if use_legacy_output:
                output_dir = os.path.join(NGLOD_PATH, "_results", "render_app", "imgs", exp_name)
            else:
                output_dir = os.path.join(SCRIPT_DIR, "_results", "renders", exp_name)
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Make sure the model path is absolute
            model_path_abs = os.path.abspath(model_path)
            
            cmd = [
                sys.executable,
                os.path.join(NGLOD_PATH, "app", "sdf_renderer.py"),
                "--net", net,
                "--num-lods", str(num_lods),
                "--pretrained", model_path_abs,
                "--render-res", str(render_width), str(render_height),
                "--shading-mode", shading_mode,
                "--matcap-path", os.path.join(NGLOD_PATH, "data/matcap/green.png")
            ]
            
            if lod is not None:
                cmd.extend(["--lod", str(lod)])
            
            if feature_dim is not None:
                cmd.extend(["--feature-dim", str(feature_dim)])
                
            if export_model:
                if use_legacy_output:
                    output_npz = os.path.join(os.path.dirname(model_path), f"{exp_name}.npz")
                else:
                    output_npz = os.path.join(SCRIPT_DIR, "_results", "models", f"{exp_name}.npz")
                cmd.extend(["--export", output_npz])
            
            # Run the rendering process with output displayed
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=NGLOD_PATH  # Run from NGLOD directory
            )
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                raise Exception(f"Rendering process exited with code {return_code}")
            
            # Check if the render was successful
            rgb_path = os.path.join(output_dir, f"{exp_name}_rgb.png")
            
            # Also check the legacy location if using unified output
            if not use_legacy_output:
                legacy_rgb_path = os.path.join(NGLOD_PATH, "_results", "render_app", "imgs", exp_name, f"{exp_name}_rgb.png")
                
                # If the render appears in the legacy location but not the unified one, copy it
                if not os.path.exists(rgb_path) and os.path.exists(legacy_rgb_path):
                    import shutil
                    print(f"Render found in legacy location, copying to unified location")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Copy all render files (rgb, normal, hit, depth)
                    legacy_dir = os.path.join(NGLOD_PATH, "_results", "render_app", "imgs", exp_name)
                    for filename in os.listdir(legacy_dir):
                        if filename.startswith(exp_name):
                            shutil.copy2(os.path.join(legacy_dir, filename), os.path.join(output_dir, filename))
                            
                    print(f"Copied render files from {legacy_dir} to {output_dir}")
            
            if export_model:
                # Check if NPZ file was created
                if os.path.exists(output_npz):
                    print(f"✅ Model exported to {output_npz}")
                    return True
                else:
                    # Check legacy location for NPZ file
                    legacy_npz = os.path.join(os.path.dirname(legacy_model_path), f"{exp_name}.npz")
                    if os.path.exists(legacy_npz):
                        import shutil
                        shutil.copy2(legacy_npz, output_npz)
                        print(f"✅ Model exported to {output_npz} (copied from legacy location)")
                        return True
                    else:
                        raise FileNotFoundError(f"Exported NPZ file not found at {output_npz}")
            else:
                # Check if render outputs were created
                if os.path.exists(rgb_path):
                    print(f"✅ Images saved to {output_dir}")
                    print(f"  RGB: {rgb_path}")
                    return True
                else:
                    # Final check for legacy location
                    if not use_legacy_output:
                        legacy_rgb_path = os.path.join(NGLOD_PATH, "_results", "render_app", "imgs", exp_name, f"{exp_name}_rgb.png")
                        if os.path.exists(legacy_rgb_path):
                            print(f"✅ Images found in legacy location: {os.path.dirname(legacy_rgb_path)}")
                            return True
                    
                    raise FileNotFoundError(f"Render output not found at {rgb_path}")
            
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_test_mode(self, 
                   test_prompts=None, 
                   epochs=5, 
                   skip_generate=False, 
                   skip_train=False, 
                   skip_render=False,
                   use_legacy_output=False,
                   **kwargs):
        """Run the pipeline on multiple test prompts."""
        print_header("Running Test Mode")
        
        # Default test prompts
        if test_prompts is None:
            test_prompts = [
                "a shark", 
                "a penguin", 
                "an ice cream cone", 
                "an airplane that looks like a banana"
            ]
        
        print(f"Running test mode with {len(test_prompts)} prompts:")
        for i, prompt in enumerate(test_prompts):
            print(f"  {i+1}. {prompt}")
        
        # Create output directories
        if use_legacy_output:
            os.makedirs("output/test", exist_ok=True)
        else:
            os.makedirs("_results/meshes/test", exist_ok=True)
        
        # Load Shap-E models once for all generations
        if not skip_generate:
            self.load_shape_e_models()
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            print_header(f"Processing Test Prompt {i+1}/{len(test_prompts)}: '{prompt}'")
            
            # Setup paths for this prompt
            sanitized_name = prompt.replace(" ", "_").replace("'", "").replace('"', "").replace(",", "")
            
            # Use legacy directory structure if specified, otherwise use unified structure
            if use_legacy_output:
                output_obj = f"output/test/a_{sanitized_name}.obj"
                model_path = os.path.join(NGLOD_PATH, "_results", "models", f"test_a_{sanitized_name}.pth")
                render_path = os.path.join(NGLOD_PATH, "_results", "render_app", "imgs", f"test_a_{sanitized_name}")
            else:
                output_obj = f"_results/meshes/test/a_{sanitized_name}.obj"
                model_path = f"_results/models/test_a_{sanitized_name}.pth"
                render_path = f"_results/renders/test_a_{sanitized_name}"
            
            exp_name = f"test_a_{sanitized_name}"
            
            # 1. Generate 3D model
            if not skip_generate:
                output_path = self.generate_3d_from_text(
                    text=prompt, 
                    output_path=output_obj,
                    use_legacy_output=use_legacy_output,
                    **kwargs
                )
                if output_path is None:
                    print(f"❌ Failed to generate 3D model for '{prompt}', skipping to next prompt.")
                    continue
            else:
                print("Skipping 3D generation step")
                
            # 2. Train LOD model
            if not skip_train:
                model_path = self.train_lod(
                    input_obj_path=output_obj,
                    exp_name=exp_name,
                    use_legacy_output=use_legacy_output,
                    **kwargs
                )
                if model_path is None:
                    print(f"❌ Failed to train LOD model for '{prompt}', skipping to next prompt.")
                    continue
            else:
                print("Skipping LOD training step")
                
            # 3. Render the model
            if not skip_render:
                success = self.render_model(
                    model_path=model_path,
                    exp_name=exp_name,
                    use_legacy_output=use_legacy_output,
                    **kwargs
                )
                if not success:
                    print(f"❌ Failed to render model for '{prompt}'.")
            else:
                print("Skipping rendering step")
            
            # Record result
            results.append({
                "prompt": prompt,
                "obj_path": output_obj,
                "model_path": model_path,
                "render_path": render_path
            })
        
        # Summary
        print_header("Test Results Summary")
        print(f"Processed {len(results)}/{len(test_prompts)} prompts successfully.")
        for i, result in enumerate(results):
            print(f"{i+1}. '{result['prompt']}'")
            print(f"   - OBJ: {result['obj_path']}")
            print(f"   - Model: {result['model_path']}")
            print(f"   - Renders: {result['render_path']}")
        
        return len(results) > 0

    def run_pipeline(self, args):
        """Run the complete pipeline for a single 3D model."""
        # If output path not provided, generate one from the text prompt
        if args.output is None and args.text:
            sanitized_name = args.text.replace(" ", "_").replace("'", "").replace('"', "").replace(",", "")
            if args.use_legacy_output:
                args.output = f"output/{sanitized_name}.obj"
            else:
                args.output = f"_results/meshes/{sanitized_name}.obj"
                
        # Create required directories
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        if args.use_legacy_output:
            os.makedirs(os.path.join(NGLOD_PATH, "_results", "models"), exist_ok=True)
            os.makedirs(os.path.join(NGLOD_PATH, "_results", "render_app", "imgs"), exist_ok=True)
        else:
            os.makedirs("_results/models", exist_ok=True)
            os.makedirs("_results/renders", exist_ok=True)
        
        # Resolve input file paths
        if args.input_obj is None:
            args.input_obj = args.output
        
        # 1. Generate 3D model from text
        if not args.skip_generate:
            success = self.generate_3d_from_text(
                text=args.text,
                output_path=args.output,
                guidance_scale=args.guidance_scale,
                karras_steps=args.karras_steps,
                use_legacy_output=args.use_legacy_output
            )
            if not success:
                print("❌ Failed to generate 3D model, aborting pipeline.")
                return False
        else:
            print_header("Skipping 3D Generation")
        
        # 2. Train LOD model
        if not args.skip_train:
            if not os.path.exists(args.input_obj):
                print(f"❌ Input OBJ file not found: {args.input_obj}")
                return False
                
            model_path = self.train_lod(
                input_obj_path=args.input_obj,
                exp_name=args.exp_name,
                net=args.net,
                num_lods=args.num_lods,
                epochs=args.epochs,
                feature_dim=args.feature_dim,
                use_legacy_output=args.use_legacy_output
            )
            if model_path is None:
                print("❌ Failed to train LOD model, aborting pipeline.")
                return False
        else:
            print_header("Skipping LOD Training")
            if args.existing_model:
                model_path = args.existing_model
            else:
                if args.use_legacy_output:
                    model_path = os.path.join(NGLOD_PATH, "_results", "models", f"{args.exp_name}.pth")
                else:
                    model_path = os.path.join("_results", "models", f"{args.exp_name}.pth")
                
                if not os.path.exists(model_path):
                    print(f"❌ No existing model found at {model_path}. Please provide --existing-model.")
                    return False
        
        # 3. Render trained model
        if not args.skip_render:
            success = self.render_model(
                model_path=model_path,
                exp_name=args.exp_name,
                net=args.net,
                num_lods=args.num_lods,
                render_width=args.render_width,
                render_height=args.render_height,
                shading_mode=args.shading_mode,
                lod=args.lod,
                feature_dim=args.feature_dim,
                export_model=args.export_model,
                use_legacy_output=args.use_legacy_output
            )
            if not success:
                print("❌ Failed to render model.")
                return False
        else:
            print_header("Skipping Rendering")
        
        print_header("Pipeline Complete")
        print("✅ All steps completed successfully!")
        return True