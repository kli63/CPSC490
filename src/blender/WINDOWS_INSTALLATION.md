# Installing the NGLOD Blender Add-on on Windows with WSL

This guide explains how to install and use the NGLOD Blender add-on on Windows when your model files and NGLOD code are in WSL.

## Prerequisites

1. Windows 10/11 with WSL2 installed
2. Blender installed on Windows
3. Your CPSC490 project with trained NGLOD models in WSL
4. Python with PyTorch installed in your WSL environment

## Installation Options

There are two add-on versions available:

1. **Direct WSL Version (Recommended)** - `nglod_blender_direct.py`
   - More reliable path handling
   - Simpler configuration
   - Better for Blender 4.0+

2. **Standard WSL Version** - `nglod_blender_addon_windows.py`
   - Original implementation
   - More complex path handling
   - May have issues with some path configurations

The instructions below focus on the recommended Direct WSL version.

## Installation Steps

### 1. Create Blender's Add-ons Directory (if it doesn't exist)

1. Navigate to Blender's user scripts directory:
   - For Blender 4.x: `%APPDATA%\Blender Foundation\Blender\4.x\scripts\`
   - For Blender 3.x: `%APPDATA%\Blender Foundation\Blender\3.x\scripts\`

2. If the `addons` folder doesn't exist, create it:
   - Create a new folder named `addons` inside the `scripts` folder

### 2. Copy the Add-on File

1. Copy the `nglod_blender_direct.py` file to the Blender add-ons directory you created or located.

### 3. Enable the Add-on in Blender

1. Open Blender
2. Go to Edit > Preferences > Add-ons
3. Click "Install..." and navigate to the `nglod_blender_direct.py` file (alternatively, if you already copied it to the addons directory, just search for "NGLOD")
4. Check the box next to "Import-Export: NGLOD Model Importer (Direct WSL)"

### 4. Configure the Add-on

In the add-on preferences, set:

1. **WSL Python Path**: The full path to your Python executable in WSL
   - Example: `/home/kli63/miniconda3/envs/cpsc490_env/bin/python`
   - This must be exactly your username and path in WSL, not "username"

2. **WSL Converter Path**: The full path to the wsl_converter.py script in your project
   - Example: `/home/kli63/dev/CPSC490/src/blender/wsl_converter.py`
   - This must be exactly your username and path in WSL, not "username"

3. **Temporary Folder**: A Windows folder to use for temporary files
   - Example: `C:\Temp\nglod`
   - Create this folder if it doesn't exist
   - Ensure it's a simple path without spaces

## Using the Add-on

1. In Blender, go to File > Import > NGLOD Model (Direct WSL)

2. Navigate to your .pth model file:
   - You can access WSL files from Windows using `\\wsl$\Ubuntu\path\to\files`
   - Or enter a WSL path directly, e.g., `/home/username/dev/CPSC490/_results/models/model.pth`

3. Configure the import settings:
   - **Resolution**: Grid resolution (128-256 recommended for balance of quality and speed)
   - **Feature Dimension**: Usually 32
   - **LOD Levels**: Usually 5
   - **Import Format**: OBJ or PLY
   - **Center Model**: Whether to center the model
   - **Smooth Shading**: Whether to use smooth shading
   - **Show Debug Info**: Show detailed debug information in the console

4. Click "Import NGLOD Model (Direct WSL)" to start the conversion and import

## Troubleshooting

### General Troubleshooting Steps

1. **Enable "Show Debug Info"** in the import dialog to get detailed information
2. **Open Blender's System Console** (Window > Toggle System Console) to see error messages
3. **Test the converter manually** from Windows Command Prompt to verify it works

### Common Issues and Solutions

#### WSL Path Issues

If you get errors about file not found:
- Make sure your paths use ***your actual WSL username***, not "username"
- Double-check all paths using `wsl ls -la <path>` in Command Prompt to verify they exist
- Ensure you're accessing your model through the Windows file browser using the path `\\wsl$\Ubuntu\home\your-username\dev\CPSC490\_results\models\`

#### Correct Path Format Examples

For a user with WSL username "kli63":
- WSL Python path: `/home/kli63/miniconda3/envs/cpsc490_env/bin/python`
- WSL Converter path: `/home/kli63/dev/CPSC490/src/blender/wsl_converter.py`
- Windows temp folder: `C:\Temp\nglod`

#### Browser Paths vs. WSL Internal Paths

There are two ways to access your model files:

1. **Using Windows File Explorer (Recommended):**  
   Navigate to `\\wsl$\Ubuntu\home\your-username\dev\CPSC490\_results\models\`  
   This allows you to see the files in the standard file browser dialog.

2. **Entering path directly:**  
   Type `/home/your-username/dev/CPSC490/_results/models/your-model.pth`  
   This requires you to know the exact path and filename.

#### Cannot Find Libraries in WSL

If conversion fails due to missing Python libraries:
- Ensure PyTorch is installed in your WSL environment
- Check that you're using the correct Python path (e.g., to the conda environment)
- Activate your conda environment before running manual tests

### Testing the Converter Manually

You can test the WSL converter directly from Windows Command Prompt:

```cmd
wsl /home/username/miniconda3/envs/cpsc490_env/bin/python /home/username/dev/CPSC490/src/blender/wsl_converter.py --model /home/username/dev/CPSC490/_results/models/dino_nugget.pth --output /mnt/c/Temp/test_model.obj --resolution 128
```

This should generate a mesh file at `C:\Temp\test_model.obj` that you can import into Blender.