# CPSC490: Neural 3D Modeling and Rendering

This repository contains an implementation for neural 3D modeling and rendering techniques using Neural Geometric Level of Detail (NGLOD) and Shap-E.

## Overview

This project integrates two major neural 3D representations:

1. **NGLOD (Neural Geometric Level of Detail)**: Fast neural SDF (Signed Distance Function) representation with multi-scale octree acceleration structure
2. **Shap-E**: Text-to-3D model by OpenAI that generates 3D objects from text prompts

Both frameworks are configured to work with RTX 4050 GPUs and use a unified Python environment.

## Features

- **Text-to-3D Generation**: Create 3D models from text descriptions
- **Neural SDF Representation**: Efficient representation of 3D models using signed distance functions
- **Multi-view Rendering**: Generate videos showing 3D models from multiple viewpoints
- **360° Rotation**: Create videos showing smooth rotation around models
- **Spherical Views**: Generate videos from viewpoints distributed around a sphere

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CPSC490.git
cd CPSC490

# Set up the unified environment
chmod +x build_rtx.sh
./build_rtx.sh

# Activate the environment
conda activate cpsc490_env
```

## Results Directory

All outputs are stored in a single `_results` directory:

```
_results/
  ├── models/     # Trained models (.pth and .npz files)
  ├── meshes/     # Generated 3D meshes (.obj files)
  ├── renders/    # Still image renders
  ├── videos/     # Generated videos
  └── logs/       # Training logs and TensorBoard files
```

## Pipeline

The pipeline script provides a complete end-to-end workflow:

```bash
# Generate 3D model from text, train LOD, and render in one command
./src/scripts/shapegen.py --text "a chair that looks like an avocado" --exp-name avocado_chair
```

### Pipeline Options

```bash
# Skip specific steps if needed
./src/scripts/shapegen.py --skip-generate --input-obj existing_model.obj  # Use existing model
./src/scripts/shapegen.py --skip-train --existing-model path/to/model.pth  # Use pre-trained model
./src/scripts/shapegen.py --skip-render  # Only generate and train

# Custom parameters
./src/scripts/shapegen.py --text "a modern lamp" --guidance-scale 20 --epochs 500 --num-lods 6
```

### Individual Pipeline Steps

```bash
# Only generate 3D model from text
./src/scripts/shapegen.py --text "a chair that looks like an avocado" --skip-train --skip-render

# Only train LOD model
./src/scripts/shapegen.py --skip-generate --input-obj model.obj --exp-name my_model --skip-render

# Only render trained model
./src/scripts/shapegen.py --skip-generate --skip-train --existing-model _results/models/my_model.pth --exp-name model_name
```

## Component Usage

### NGLOD

Train a neural SDF model:

```bash
cd nglod/sdf-net
python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/my.obj \
    --epoch 10 \
    --exp-name mymodel
```

Render the trained model:

```bash
python app/sdf_renderer.py \
    --net OctreeSDF \
    --num-lods 5 \
    --pretrained _results/models/mymodel.pth \
    --render-res 1280 720 \
    --shading-mode matcap
```

For detailed instructions on using NGLOD with RTX 4050 GPUs, see [RTX_4050_COMPLETE_GUIDE.md](RTX_4050_COMPLETE_GUIDE.md).

### Shap-E

Generate 3D models from text:

```bash
cd shap-e
python -m shap_e.examples.sample_text_to_3d \
    --prompt "a teapot" \
    --output-dir outputs/teapot
```

View the Jupyter notebooks in `shap-e/shap_e/examples/` for more detailed usage examples.

## Video Generation

Create videos of 3D models from different viewpoints:

```bash
# Generate a 360° rotation video
./src/scripts/render_video.py _results/models/my_model.pth --name my_model --frames 120 --fps 30

# Generate a video with views from around a sphere
./src/scripts/render_sphere_views.py _results/models/my_model.pth --name my_model --poses 64 --cam-radius 3.5
```

### Advanced Video Options

```bash
# Specify model architecture parameters
./src/scripts/render_video.py _results/models/my_model.pth --net OctreeSDF --feature-dim 32 --num-lods 5

# Customize camera and video settings
./src/scripts/render_sphere_views.py _results/models/my_model.pth --net OctreeSDF --cam-radius 5.0 --poses 128

# Change output location
./src/scripts/render_video.py _results/models/my_model.pth --output-dir custom_videos

# Change camera view perspective
./src/scripts/render_video.py _results/models/my_model.pth --camera-view diagonal   # Diagonal view (default, 3/4 view)
./src/scripts/render_video.py _results/models/my_model.pth --camera-view front      # Front view (z-axis)
./src/scripts/render_video.py _results/models/my_model.pth --camera-view top        # Top-down view (y-axis)
./src/scripts/render_video.py _results/models/my_model.pth --camera-view side       # Side view (x-axis)

# Use custom camera position
./src/scripts/render_video.py _results/models/my_model.pth --camera-view custom --custom-camera-origin 3.0 2.0 1.0
```

## Testing

To test the NGLOD renderer:

```bash
python rtx_render_test.py
```

## Project Structure

```
├── _results/            # Unified directory for all outputs
│   ├── models/          # Trained models (.pth and .npz files)
│   ├── meshes/          # Generated 3D meshes (.obj files)
│   ├── renders/         # Still image renders
│   ├── videos/          # Generated videos
│   └── logs/            # Training logs and TensorBoard files
│
├── src/                 # Source code
│   ├── render/          # Rendering-related functionality
│   ├── utils/           # Utility functions
│   ├── scripts/         # Command-line interface scripts
│   │   ├── shapegen.py  # Main pipeline script for text-to-3D generation
│   │   ├── render_video.py  # Generate 360° rotation videos
│   │   └── render_sphere_views.py  # Generate videos with views from a sphere
│   └── pipeline.py      # Core implementation of the pipeline
│
├── nglod/               # Neural Geometric Level of Detail implementation
│   ├── sdf-net/         # Python implementation of NGLOD
│   └── sol-renderer/    # C++ renderer for real-time visualization
│
├── shap-e/              # OpenAI's Shap-E text-to-3D model
│
├── build_rtx.sh         # Script to set up the unified Python environment
├── requirements.txt     # Python dependencies
├── RTX_4050_COMPLETE_GUIDE.md  # Guide for using NGLOD with RTX 4050 GPUs
└── setup.py             # Package installation configuration
```

## Command Line Reference

### Pipeline: src/scripts/shapegen.py

```bash
./src/scripts/shapegen.py [options]
```

#### General Options:
- `--skip-generate`: Skip 3D generation step
- `--skip-train`: Skip LOD training step  
- `--skip-render`: Skip rendering step
- `--existing-model PATH`: Path to existing trained model (.pth)
- `--skip-dependency-check`: Skip dependency checking
- `--use-legacy-output`: Use legacy directory structure instead of unified _results
- `--test-mode`: Run in test mode with multiple predefined prompts
- `--test-prompts TEXT1,TEXT2,...`: Comma-separated list of test prompts (no spaces after commas)

#### 3D Generation Options (Shap-E):
- `--text TEXT`: Text prompt for 3D generation (default: "a chair")
- `--guidance-scale FLOAT`: Guidance scale for generation (default: 15.0)
- `--karras-steps INT`: Number of Karras steps (default: 64)
- `--output PATH`: Output OBJ file path (default: "_results/meshes/generated.obj")

#### Training Options (NGLOD):
- `--net STRING`: Network architecture (default: "OctreeSDF")
- `--num-lods INT`: Number of LOD levels (default: 5)
- `--epochs INT`: Number of training epochs (default: 5)
- `--feature-dim INT`: Feature dimension (default is 32)
- `--exp-name STRING`: Experiment name (default: "generated_model")
- `--input-obj PATH`: Input OBJ file path (overrides --output)

#### Rendering Options (NGLOD):
- `--render-width INT`: Render width (default: 1280)
- `--render-height INT`: Render height (default: 720)
- `--shading-mode STRING`: Shading mode (default: "matcap")
- `--lod INT`: Specific LOD level to render (default: 4)
- `--camera-view {diagonal,front,top,side,custom}`: Camera view perspective (default: "diagonal")
- `--custom-camera-origin X Y Z`: Custom camera origin as 3 float values
- `--export-model`: Export model to NPZ format

### Video Generation: src/scripts/render_video.py

```bash
./src/scripts/render_video.py [MODEL_PATH] [options]
```

#### Main Options:
- `MODEL_PATH`: Path to the model file (.pth or .npz)
- `--name STRING`: Name for the output (default: model filename)
- `--output-dir PATH`: Directory to save the video (default: "_results/videos")
- `--use-legacy-output`: Use legacy directory structure instead of unified _results
- `--frames INT`: Number of frames to render (default: 60)
- `--fps INT`: Frames per second in output video (default: 30)
- `--rotation {y,x,z}`: Axis to rotate around (default: "y")
- `--cam-radius FLOAT`: Camera distance from object (default: 4.0)
- `--temp-dir PATH`: Temporary directory for frames (default: "_temp_render")
- `--camera-view {diagonal,front,top,side,custom}`: Camera view perspective (default: "diagonal")
- `--custom-camera-origin X Y Z`: Custom camera origin as 3 float values

#### Network Options:
- `--net STRING`: Network architecture (default: "OctreeSDF")
- `--feature-dim INT`: Feature dimension used in training
- `--num-lods INT`: Number of LODs used in training

### Spherical View Generation: src/scripts/render_sphere_views.py

```bash
./src/scripts/render_sphere_views.py [MODEL_PATH] [options]
```

#### Main Options:
- `MODEL_PATH`: Path to the model file (.pth or .npz)
- `--name STRING`: Name for the output (default: model filename)
- `--output-dir PATH`: Directory to save the video (default: "_results/videos")
- `--use-legacy-output`: Use legacy directory structure instead of unified _results
- `--poses INT`: Number of viewpoints to render (default: 64)
- `--fps INT`: Frames per second in output video (default: 30)
- `--cam-radius FLOAT`: Camera distance from object (default: 4.0)
- `--temp-dir PATH`: Temporary directory for frames (default: "_temp_render")
- `--camera-view {diagonal,front,top,side,custom}`: Starting camera view (default: "diagonal")
- `--custom-camera-origin X Y Z`: Custom camera origin as 3 float values

#### Network Options:
- `--net STRING`: Network architecture (default: "OctreeSDF")
- `--feature-dim INT`: Feature dimension used in training
- `--num-lods INT`: Number of LODs used in training

## Requirements

- NVIDIA GPU with CUDA support (configured for RTX 4050)
- CUDA 11.8 and compatible drivers
- Python 3.10
- Conda package manager

## License

See individual repositories for licensing information.
- NGLOD: [LICENSE](nglod/LICENSE)
- Shap-E: [LICENSE](shap-e/LICENSE)