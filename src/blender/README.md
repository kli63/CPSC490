# NGLOD Blender Integration

This module provides Blender integration for NGLOD neural SDF models, allowing you to convert them to triangle meshes for use in Blender.

## Core Files:

- **nglod_blender_direct.py**: Blender add-on that integrates with WSL to import NGLOD models
- **nglod_converter.py**: Core conversion library for NGLOD models to meshes
- **wsl_converter.py**: WSL specific script for Windows-WSL integration
- **convert_model.py**: Command-line utility for batch conversion

## Quick Start:

### For Windows with WSL:

See [WINDOWS_INSTALLATION.md](WINDOWS_INSTALLATION.md) for detailed setup instructions.

1. Install `nglod_blender_direct.py` as a Blender add-on
2. Set your WSL Python path and converter script path in the add-on preferences
3. Import models through File > Import > NGLOD Model (Direct WSL)

### For Linux/Mac or Standalone Usage:

```bash
# Convert a model to OBJ format
python convert_model.py --model /path/to/model.pth --output /path/to/output.obj --resolution 256
```

## Model Resolution and Quality:

The `resolution` parameter controls the quality of the generated mesh:

- **64-128**: Low resolution, fast conversion
- **256**: Medium resolution, good balance 
- **512**: High resolution, more detailed but slower
- **1024+**: Very high resolution, best quality but very slow

## Requirements:

- PyTorch
- Blender 2.8+ for the add-on
- WSL with Python + PyTorch for Windows users