#!/usr/bin/env python3
"""
NGLOD Blender Add-on (Direct WSL Access)

A simplified Blender add-on that directly calls the wsl_converter.py script
without trying to construct paths through os.path.join, which can cause issues
with mixed Windows/WSL paths.

This version uses raw string concatenation which is more reliable for
WSL path handling from Windows.
"""

import os
import sys
import bpy
import tempfile
import subprocess
from pathlib import Path
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
from bpy.types import Operator, Panel, AddonPreferences
from bpy_extras.io_utils import ImportHelper

# Add-on information
bl_info = {
    "name": "NGLOD Model Importer (Direct WSL)",
    "author": "CPSC490",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > NGLOD Model (Direct WSL)",
    "description": "Import NGLOD neural SDF models from WSL (direct version)",
    "category": "Import-Export",
}

class NGLODAddonPreferencesDirect(AddonPreferences):
    """Add-on preferences for storing paths."""
    
    bl_idname = __name__
    
    wsl_python_path: StringProperty(
        name="WSL Python Path",
        description="Full path to Python in WSL (e.g., /home/user/miniconda3/envs/cpsc490_env/bin/python)",
        default=""
    )
    
    wsl_converter_path: StringProperty(
        name="WSL Converter Path",
        description="Full path to wsl_converter.py in WSL (e.g., /home/user/dev/CPSC490/src/blender/wsl_converter.py)",
        default=""
    )
    
    temp_folder: StringProperty(
        name="Temporary Folder",
        description="Windows folder to use for temporary files (must be accessible from both WSL and Windows)",
        subtype='DIR_PATH',
        default="C:\\Temp\\nglod"
    )
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="NGLOD Model Importer (Direct WSL) Settings")
        
        box = layout.box()
        box.label(text="WSL Configuration:")
        box.prop(self, "wsl_python_path")
        box.prop(self, "wsl_converter_path")
        box.prop(self, "temp_folder")
        
        # Help text
        layout.label(text="WSL Python Path should be the full path to Python with PyTorch")
        layout.label(text="WSL Converter Path should be the full path to wsl_converter.py in your project")
        layout.label(text="Temporary Folder must be accessible from both WSL and Windows")
        
        if not self.wsl_python_path or not self.wsl_converter_path:
            layout.label(text="Please fill in all the required paths to enable the importer", icon='ERROR')
        
        # Check for import support in this Blender version
        obj_import_available = hasattr(bpy.ops.wm, "obj_import") or hasattr(bpy.ops.import_scene, "obj")
        ply_import_available = hasattr(bpy.ops.wm, "ply_import") or hasattr(bpy.ops.import_mesh, "ply")
        
        box = layout.box()
        box.label(text="Blender Import Support:")
        
        row = box.row()
        if obj_import_available:
            row.label(text="✓ OBJ Import supported", icon='CHECKMARK')
        else:
            row.label(text="❌ OBJ Import not available in this Blender version", icon='ERROR')
            
        row = box.row()
        if ply_import_available:
            row.label(text="✓ PLY Import supported", icon='CHECKMARK')
        else:
            row.label(text="❌ PLY Import not available in this Blender version", icon='ERROR')

def windows_to_wsl_path(windows_path):
    """Convert a Windows path to a WSL path."""
    # Check if this looks like a Windows path (has drive letter with colon)
    if len(windows_path) > 2 and windows_path[1] == ':':
        # Extract the drive letter and convert to lowercase
        drive = windows_path[0].lower()
        
        # Get the rest of the path, replacing backslashes with forward slashes
        rest_path = windows_path[2:].replace('\\', '/')
        
        # If the rest doesn't start with a slash, add one
        if not rest_path.startswith('/'):
            rest_path = '/' + rest_path
            
        # Return the WSL path format
        return f"/mnt/{drive}{rest_path}"
    
    # If it's already a WSL path or doesn't look like a Windows path, return as is
    return windows_path

class NGLOD_OT_ImportModelDirectWSL(Operator, ImportHelper):
    """Import a NGLOD model from WSL as a mesh (direct version)"""
    
    bl_idname = "import_mesh.nglod_direct_wsl"
    bl_label = "Import NGLOD Model (Direct WSL)"
    bl_description = "Import a NGLOD neural SDF model (.pth) from WSL as a mesh (direct version)"
    bl_options = {'REGISTER', 'UNDO'}
    
    # File selection filter
    filename_ext = ".pth"
    filter_glob: StringProperty(
        default="*.pth",
        options={'HIDDEN'},
    )
    
    # Import properties
    resolution: IntProperty(
        name="Resolution",
        description="Grid resolution for marching cubes (higher = more detailed)",
        default=128,
        min=32, 
        max=512
    )
    
    feature_dim: IntProperty(
        name="Feature Dimension",
        description="Feature dimension used when training the model",
        default=32,
        min=1, 
        max=128
    )
    
    num_lods: IntProperty(
        name="Number of LOD Levels",
        description="Total number of LOD levels in the model",
        default=5,
        min=1, 
        max=10
    )
    
    active_lod: IntProperty(
        name="Active LOD Level",
        description="Which LOD level to render (0 = lowest detail, higher = more detail)",
        default=4,
        min=0, 
        max=9
    )
    
    render_quality: EnumProperty(
        items=[
            ('PREVIEW', 'Preview', 'Fast, low-quality rendering for interactive use'),
            ('MEDIUM', 'Medium', 'Balanced quality and speed'),
            ('HIGH', 'High', 'High quality rendering, slower'),
        ],
        name="Render Quality",
        description="Quality preset for rendering",
        default='MEDIUM'
    )
    
    import_mode: EnumProperty(
        items=[
            ('OBJ', 'OBJ', 'Import as OBJ format'),
            ('PLY', 'PLY', 'Import as PLY format'),
        ],
        name="Import Format",
        description="File format for temporary mesh file",
        default='OBJ'
    )
    
    centered: BoolProperty(
        name="Center Model",
        description="Center the model at the origin after import",
        default=True
    )
    
    use_smooth: BoolProperty(
        name="Smooth Shading",
        description="Use smooth shading for the imported mesh",
        default=True
    )
    
    show_debug: BoolProperty(
        name="Show Debug Info",
        description="Show detailed debugging information in the console",
        default=True
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Quality preset with presets for quick resolution selection
        box = layout.box()
        box.label(text="Quality Settings:")
        box.prop(self, "render_quality")
        
        # When "Preview" is selected, use lower resolution automatically
        if self.render_quality == 'PREVIEW':
            self.resolution = 64
        elif self.render_quality == 'MEDIUM':
            self.resolution = 128
        elif self.render_quality == 'HIGH':
            self.resolution = 256
            
        # Resolution can still be manually adjusted
        box.prop(self, "resolution")
        
        # LOD Level selection
        box = layout.box()
        box.label(text="LOD Settings:")
        
        # Number of LODs in the model
        box.prop(self, "num_lods")
        
        # Which LOD level to render
        row = box.row()
        row.prop(self, "active_lod")
        
        # Ensure active_lod doesn't exceed num_lods-1
        if self.active_lod >= self.num_lods:
            self.active_lod = self.num_lods - 1
            
        # Visual LOD slider
        row = box.row()
        row.scale_y = 0.4
        for i in range(self.num_lods):
            if i == self.active_lod:
                row.label(text="▲", icon='CHECKMARK')
            else:
                row.label(text="·")
        
        # Advanced parameters
        box = layout.box()
        box.label(text="Advanced Parameters:")
        box.prop(self, "feature_dim")
        
        # Import options
        box = layout.box()
        box.label(text="Import Settings:")
        box.prop(self, "import_mode")
        box.prop(self, "centered")
        box.prop(self, "use_smooth")
        box.prop(self, "show_debug")
    
    def execute(self, context):
        addon_prefs = context.preferences.addons[__name__].preferences
        
        # Check if configuration is complete
        if not addon_prefs.wsl_python_path or not addon_prefs.wsl_converter_path:
            self.report({'ERROR'}, "Please set WSL Python Path and Converter Path in the add-on preferences")
            return {'CANCELLED'}
        
        # Make sure temp folder exists
        temp_folder = addon_prefs.temp_folder
        os.makedirs(temp_folder, exist_ok=True)
        
        # Create paths for temporary files
        win_temp_mesh = os.path.join(temp_folder, f"temp_mesh.{self.import_mode.lower()}")
        wsl_temp_mesh = windows_to_wsl_path(win_temp_mesh)
        
        # Get the model path
        model_path = self.filepath
        
        # Handle Windows network path to WSL
        # Pattern like: \\wsl$\Ubuntu\path\to\file.pth
        if model_path.startswith('\\\\wsl$\\'):
            # Split path into components and extract relevant parts
            parts = model_path.split('\\')
            
            # Make sure we have enough parts for a valid path
            if len(parts) >= 4:
                # Try to find 'home' in the path
                try:
                    home_index = parts.index('home', 3)  # Start searching from index 3
                    if home_index > 0:
                        # Join all parts from 'home' onwards with forward slashes
                        wsl_path = '/' + '/'.join(parts[home_index:])
                        wsl_model_path = wsl_path
                        self.report({'INFO'}, f"Converted WSL network path to: {wsl_model_path}")
                    else:
                        raise ValueError("'home' not found in expected position")
                except ValueError:
                    # If 'home' not found, try a more generic approach - just remove the \\wsl$\Ubuntu part
                    if len(parts) > 3:
                        wsl_path = '/' + '/'.join(parts[3:])
                        wsl_model_path = wsl_path
                        self.report({'INFO'}, f"Converted WSL network path (without home): {wsl_model_path}")
                    else:
                        self.report({'ERROR'}, f"Could not parse WSL path: {model_path}")
                        return {'CANCELLED'}
            else:
                self.report({'ERROR'}, f"WSL path too short: {model_path}")
                return {'CANCELLED'}
        # Regular Windows path
        elif len(model_path) > 2 and model_path[1] == ':':
            wsl_model_path = windows_to_wsl_path(model_path)
        # Assume it's already a WSL path
        else:
            wsl_model_path = model_path
        
        # Construct direct WSL command - avoid os.path.join for mixed path types
        wsl_cmd = [
            "wsl",
            addon_prefs.wsl_python_path,
            addon_prefs.wsl_converter_path,
            "--model", wsl_model_path,
            "--output", wsl_temp_mesh,
            "--resolution", str(self.resolution),
            "--feature-dim", str(self.feature_dim),
            "--num-lods", str(self.num_lods),
            "--active-lod", str(self.active_lod)  # Pass the active LOD level
        ]
        
        if self.show_debug:
            debug_cmd = " ".join(wsl_cmd)
            self.report({'INFO'}, f"Converting model using direct WSL command:")
            self.report({'INFO'}, debug_cmd)
        else:
            self.report({'INFO'}, f"Converting model. This may take a few minutes for high resolutions...")
        
        # Check if model path ends with .pth (basic validation)
        if not wsl_model_path.lower().endswith('.pth'):
            self.report({'ERROR'}, f"Model path must end with .pth: {wsl_model_path}")
            self.report({'ERROR'}, f"You might be trying to select a directory rather than a file.")
            self.report({'ERROR'}, f"Please select a specific .pth model file, not a directory.")
            return {'CANCELLED'}
        
        try:
            # Execute the conversion process
            result = subprocess.run(
                wsl_cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Print output for debugging
            if self.show_debug:
                for line in result.stdout.splitlines():
                    self.report({'INFO'}, line)
            
            # Check if the converted mesh file exists
            if not os.path.exists(win_temp_mesh):
                self.report({'ERROR'}, f"Conversion failed, mesh file not created: {win_temp_mesh}")
                return {'CANCELLED'}
            
            # Import the mesh into Blender
            self.report({'INFO'}, f"Importing mesh: {win_temp_mesh}")
            
            try:
                # Import using Blender 4.x core importers
                if self.import_mode == 'OBJ':
                    # Use the core OBJ importer in recent Blender versions
                    self.report({'INFO'}, f"Importing using Blender's core OBJ importer...")
                    
                    # Find the most appropriate import function
                    if hasattr(bpy.ops.wm, "obj_import"):  # Blender 4.x
                        bpy.ops.wm.obj_import(filepath=win_temp_mesh)
                    elif hasattr(bpy.ops.import_scene, "obj"):  # Older versions
                        bpy.ops.import_scene.obj(filepath=win_temp_mesh)
                    else:
                        # If neither method exists, read and create mesh manually as last resort
                        self.report({'ERROR'}, "No OBJ importer found in this Blender version.")
                        self.report({'INFO'}, "Manual mesh import will be attempted.")
                        
                        # At this point, we could implement manual OBJ parsing
                        # For simplicity we'll just use a cube placeholder
                        bpy.ops.mesh.primitive_cube_add()
                        bpy.context.active_object.name = os.path.splitext(os.path.basename(self.filepath))[0]
                        
                        # Inform user where to find the actual mesh
                        self.report({'INFO'}, f"The converted mesh is saved at: {win_temp_mesh}")
                        self.report({'INFO'}, "You can import it manually using File > Import > Wavefront (.obj)")
                        return {'FINISHED'}
                else:  # PLY
                    # Use the core PLY importer
                    self.report({'INFO'}, f"Importing using Blender's core PLY importer...")
                    
                    # Find the most appropriate import function
                    if hasattr(bpy.ops.wm, "ply_import"):  # Blender 4.x
                        bpy.ops.wm.ply_import(filepath=win_temp_mesh) 
                    elif hasattr(bpy.ops.import_mesh, "ply"):  # Older versions
                        bpy.ops.import_mesh.ply(filepath=win_temp_mesh)
                    else:
                        # If neither method exists, create placeholder
                        self.report({'ERROR'}, "No PLY importer found in this Blender version.")
                        bpy.ops.mesh.primitive_cube_add()
                        bpy.context.active_object.name = os.path.splitext(os.path.basename(self.filepath))[0]
                        
                        # Inform user where to find the actual mesh
                        self.report({'INFO'}, f"The converted mesh is saved at: {win_temp_mesh}")
                        self.report({'INFO'}, "You can import it manually using File > Import > Stanford PLY (.ply)")
                        return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"Error importing mesh: {str(e)}")
                self.report({'INFO'}, "This may be a temporary issue or bug in Blender's importer")
                self.report({'INFO'}, f"The model was successfully converted and saved to: {win_temp_mesh}")
                self.report({'INFO'}, "You can manually import it via File > Import > Wavefront (.obj)")
                
                # Create a primitive as fallback
                bpy.ops.mesh.primitive_cube_add()
                bpy.context.active_object.name = os.path.splitext(os.path.basename(self.filepath))[0]
                return {'FINISHED'}
            
            # Get the imported object
            obj = context.active_object
            if obj:
                # Set smooth shading if requested
                if self.use_smooth:
                    for polygon in obj.data.polygons:
                        polygon.use_smooth = True
                
                # Center the object if requested
                if self.centered:
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                    obj.location = (0, 0, 0)
                
                # Rename the object based on the input file
                obj.name = os.path.splitext(os.path.basename(self.filepath))[0]
                
                self.report({'INFO'}, f"NGLOD model imported successfully as {obj.name}")
            
            return {'FINISHED'}
        
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"WSL conversion failed with code {e.returncode}")
            
            # Show detailed error message from stderr
            if e.stderr:
                for line in e.stderr.splitlines():
                    self.report({'ERROR'}, f"Error: {line}")
            
            # Show any stdout output that might help diagnose the issue
            if e.stdout:
                for line in e.stdout.splitlines():
                    self.report({'INFO'}, f"Output: {line}")
                    
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error importing NGLOD model: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

class NGLOD_PT_SidePanel(Panel):
    """NGLOD Import Panel in the sidebar"""
    bl_label = "NGLOD Models"
    bl_idname = "NGLOD_PT_side_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NGLOD"
    
    def draw(self, context):
        layout = self.layout
        
        # Section for importing models
        box = layout.box()
        box.label(text="Import NGLOD Model")
        op = box.operator(NGLOD_OT_ImportModelDirectWSL.bl_idname, text="Import from WSL", icon='IMPORT')
        
        # Display currently imported models
        box = layout.box()
        box.label(text="Imported Models")
        
        # Check if any NGLOD models are in the scene
        found_models = False
        for obj in bpy.data.objects:
            if obj.name.endswith('.pth'):
                found_models = True
                row = box.row()
                row.label(text=obj.name, icon='MESH_DATA')
                
                # Add buttons for operations
                sub = row.row(align=True)
                sub.scale_x = 0.6
                sub.operator("object.select_all", text="", icon='CHECKBOX_HLT').action = 'DESELECT'
                # select just this object
                op = sub.operator("object.select_all", text="", icon='RESTRICT_SELECT_OFF')
                op.action = 'SELECT'
                # op.selected_objects = [obj]  # Not valid in Blender 4.0+
                
        if not found_models:
            box.label(text="No NGLOD models imported", icon='INFO')
        
        # Add help and info
        box = layout.box()
        box.label(text="Import Settings")
        box.label(text="LOD: Controls detail level", icon='INFO')
        box.label(text="Resolution: Mesh quality", icon='INFO')

def menu_import(self, context):
    """Add to the import menu."""
    self.layout.operator(NGLOD_OT_ImportModelDirectWSL.bl_idname, text="NGLOD Model (Direct WSL)")

# Registration
classes = (
    NGLODAddonPreferencesDirect,
    NGLOD_OT_ImportModelDirectWSL,
    NGLOD_PT_SidePanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()