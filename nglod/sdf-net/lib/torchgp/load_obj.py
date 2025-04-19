# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import numpy as np
import torch
from PIL import Image

# Replacement for tinyobjloader dependency - using trimesh
import trimesh

# Keep the original texopts for compatibility
texopts = [
    'ambient_texname',
    'diffuse_texname',
    'specular_texname',
    'specular_highlight_texname',
    'bump_texname',
    'displacement_texname',
    'alpha_texname',
    'reflection_texname',
    'roughness_texname',
    'metallic_texname',
    'sheen_texname',
    'emissive_texname',
    'normal_texname'
]

def load_mat(fname : str):
    img = torch.FloatTensor(np.array(Image.open(fname)))
    img = img / 255.0
    return img

def load_obj(
    fname : str, 
    load_materials : bool = False):
    """Load .obj file using trimesh (replacement for tinyobjloader)
    
    Args:
        fname (str): path to Wavefront .obj file
        load_materials (bool): whether to load materials
    """
    assert fname is not None and os.path.exists(fname), \
        'Invalid file path and/or format, must be an existing Wavefront .obj'

    print(f"Loading OBJ file: {fname} with trimesh")
    
    # Load mesh using trimesh with triangulation
    mesh = trimesh.load(fname, force='mesh', process=True)
    
    # Get vertices as tensor
    vertices = torch.FloatTensor(mesh.vertices)
    
    # Get faces as tensor
    faces = torch.LongTensor(mesh.faces)
    
    print(f"Loaded OBJ with {vertices.shape[0]} vertices and {faces.shape[0]} faces")
    
    # Handle materials if requested
    if load_materials:
        # Create empty textures since we can't easily extract them with trimesh
        texv = torch.zeros((1, 2), dtype=torch.float32)
        texf = torch.zeros((faces.shape[0], 4), dtype=torch.long)
        mats = {0: {'diffuse': torch.FloatTensor([0.8, 0.8, 0.8])}}
        
        print("Warning: Material loading with trimesh is limited. Using default materials.")
        return vertices, faces, texv, texf, mats
    
    return vertices, faces