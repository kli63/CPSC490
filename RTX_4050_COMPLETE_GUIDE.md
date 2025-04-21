# Complete Guide: NGLOD on RTX 4050 GPUs

This guide provides comprehensive instructions for using the Neural Geometric Level of Detail (NGLOD) framework on NVIDIA RTX 4050 GPUs.

## Quick Start

```bash
# 1. Setup environment
conda create -n cpsc490_env python=3.10
conda activate cpsc490_env
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib tqdm scikit-image pybind11 trimesh Pillow scipy moviepy plyfile polyscope pyexr openexr tensorboard

# 2. Build CUDA extensions
cd nglod/sdf-net/lib/extensions
chmod +x build_ext.sh
./build_ext.sh

# 3. Train a model
cd ../../
python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/my.obj \
    --epoch 10 \
    --exp-name testobj

# 4. Render the model
python app/sdf_renderer.py \
    --net OctreeSDF \
    --num-lods 5 \
    --pretrained _results/models/testobj.pth \
    --render-res 1280 720 \
    --shading-mode matcap \
    --lod 4
```

## Detailed Setup Guide

### 1. Environment Setup

The RTX 4050 has compute capability 8.9, which requires a newer version of PyTorch than the original codebase used. We need to create a dedicated environment:

```bash
# Create and activate the environment
conda create -n cpsc490_env python=3.10
conda activate cpsc490_env

# Install PyTorch with CUDA 11.8 support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install matplotlib tqdm scikit-image pybind11 trimesh Pillow scipy moviepy plyfile polyscope pyexr openexr tensorboard
```

### 2. CUDA Extension Modifications

The CUDA extensions need to be modified to target the RTX 4050 architecture:

1. **mesh2sdf_cuda/setup.py**:
   ```python
   from setuptools import setup
   from torch.utils.cpp_extension import BuildExtension, CUDAExtension
   import os

   # Set CUDA architecture specifically for RTX 4050 (Compute Capability 8.9)
   os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
   os.environ['PYTHONNOUSERSITE'] = '1'  # Avoid using user site packages

   setup(
       name='mesh2sdf',
       ext_modules=[
           CUDAExtension('mesh2sdf', [
               'mesh2sdf_kernel.cu',
           ],
           extra_compile_args={
               'cxx': ['-O3', '-std=c++14'],
               'nvcc': ['-O3']
           })
       ],
       cmdclass={
           'build_ext': BuildExtension
       })
   ```

2. **sol_nglod/setup.py**:
   ```python
   from setuptools import setup
   from torch.utils.cpp_extension import BuildExtension, CUDAExtension
   import os

   # Set CUDA architecture specifically for RTX 4050 (Compute Capability 8.9)
   os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
   os.environ['PYTHONNOUSERSITE'] = '1'  # Avoid using user site packages

   setup(
       name='sol_nglod',
       ext_modules=[
           CUDAExtension('sol_nglod', [
               'sol_nglod_kernel.cu',
           ],
           extra_compile_args={
               'cxx': ['-O3', '-std=c++14'],
               'nvcc': ['-O3']
           })
       ],
       cmdclass={
           'build_ext': BuildExtension
       })
   ```

### 3. Dependency Replacements

Replace tinyobjloader with trimesh in `lib/torchgp/load_obj.py`:

```python
import torch
import numpy as np
import trimesh  # Replacement for tinyobjloader

def load_obj(fname: str, load_materials: bool = False):
    """Load .obj file using trimesh (replacement for tinyobjloader)"""
    
    # Load mesh using trimesh
    mesh = trimesh.load(fname, force='mesh', process=True)
    
    # Extract vertices and faces
    vertices = torch.FloatTensor(mesh.vertices)
    faces = torch.LongTensor(mesh.faces)
    
    # Create minimal material response to maintain compatibility
    materials = {}
    if load_materials and hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        # Extract material information when available
        material = mesh.visual.material
        material_name = getattr(material, 'name', 'default')
        
        materials[material_name] = {
            'diffuse': torch.FloatTensor([0.8, 0.8, 0.8]) if not hasattr(material, 'diffuse') else torch.FloatTensor(material.diffuse[:3]),
            'specular': torch.FloatTensor([0.2, 0.2, 0.2]) if not hasattr(material, 'specular') else torch.FloatTensor(material.specular[:3]),
            'ambient': torch.FloatTensor([0.1, 0.1, 0.1]) if not hasattr(material, 'ambient') else torch.FloatTensor(material.ambient[:3]),
            'texture_path': getattr(material, 'texture', None)
        }
    
    return {
        'vertices': vertices,
        'faces': faces,
        'materials': materials
    }
```

### 4. Building CUDA Extensions

```bash
cd nglod/sdf-net/lib/extensions
chmod +x build_ext.sh
./build_ext.sh
```

### 5. Verification

Use the provided verification script to ensure everything is set up correctly:

```bash
python rtx_render_test.py
```

## Complete NGLOD Workflow on RTX 4050

### 1. Training Models

NGLOD uses the octree-based SDF representation to learn 3D shapes:

```bash
python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/my.obj \
    --epoch 10 \
    --exp-name testobj
```

Key parameters:
- `--net`: Network architecture (OctreeSDF is the main architecture)
- `--num-lods`: Number of levels of detail (higher values = more detail but slower training)
- `--dataset-path`: Path to the 3D mesh file (.obj)
- `--epoch`: Number of training epochs
- `--exp-name`: Name for the experiment/model

### 2. Rendering Models

Render the trained model to visualize the learned 3D shape:

```bash
python app/sdf_renderer.py \
    --net OctreeSDF \
    --num-lods 5 \
    --pretrained _results/models/testobj.pth \
    --render-res 1280 720 \
    --shading-mode matcap \
    --lod 4
```

Key parameters:
- `--net`: Same network architecture used for training
- `--num-lods`: Same number of levels as training
- `--pretrained`: Path to the trained model (.pth file)
- `--render-res`: Resolution for rendering (width height)
- `--shading-mode`: Visualization method (matcap, normal, lambertian)
- `--lod`: Which level of detail to render (usually highest level)

### 3. Exporting Models for C++ Renderer

Export the trained model to NPZ format for use with the C++ renderer:

```bash
python app/sdf_renderer.py \
    --net OctreeSDF \
    --num-lods 5 \
    --pretrained _results/models/testobj.pth \
    --export testobj.npz
```

### 4. Using the C++ Renderer

The C++ renderer in the sol-renderer directory provides real-time rendering:

```bash
cd nglod/sol-renderer
./build.sh
./render_image.sh testobj.npz
```

## Performance Optimization for RTX 4050

To get the best performance with NGLOD on the RTX 4050:

1. **Batch Size**: Adjust the batch size based on available GPU memory
   ```
   --batch-size 4096  # Default, reduce if you get OOM errors
   ```

2. **Resolution**: Lower resolution for faster training/rendering
   ```
   --render-res 800 600  # For faster rendering
   ```

3. **LOD Selection**: Use lower LOD for faster rendering during development
   ```
   --lod 3  # Lower LOD for faster rendering (vs. default 4)
   ```

4. **Progressive Training**: Use for complex models
   ```
   --progressive  # Start with low resolution, progressively increase
   ```

## Troubleshooting

### 1. CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce batch size: `--batch-size 2048` (default is 4096)
- Use fewer samples: `--samples-per-voxel 16` (default is 32)
- Decrease the number of levels: `--num-lods 4` (default is 5)

### 2. Import Errors

If you encounter import errors:
- Ensure you're in the correct environment: `conda activate cpsc490_env`
- Check that extensions were built correctly:
  ```bash
  python -c "import mesh2sdf; import sol_nglod; print('Extensions loaded successfully!')"
  ```
  
### 3. Rendering Issues

If rendering produces incorrect images:
- Ensure you're using the same network architecture and parameters used during training
- Try different shading modes: `--shading-mode normal`
- Adjust camera parameters: `--camera-origin 0 1 4 --camera-lookat 0 0 0`

## Testing the Complete Workflow

Use the rtx_render_test.py script to test model export and rendering:

```bash
python rtx_render_test.py
```

This will:
1. Test the CUDA extensions and environment setup
2. Export each trained model to NPZ format
3. Render each model and save the output images
4. Provide a detailed report on compatibility with your RTX 4050

## Additional Resources

- Original NGLOD paper: "Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes"
- NGLOD repository: https://github.com/nv-tlabs/nglod
- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/