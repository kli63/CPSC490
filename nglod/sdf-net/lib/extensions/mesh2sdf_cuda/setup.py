from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

PACKAGE_NAME = 'mesh2sdf'
VERSION = '0.1.0'
DESCRIPTION = 'Fast CUDA kernel for computing SDF of triangle mesh'
AUTHOR = 'Zekun Hao et al.'
URL = 'https://github.com/zekunhao1995/DualSDF'
LICENSE = 'MIT'

# Set CUDA architecture specifically for RTX 4050 (Compute Capability 8.9)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
os.environ['PYTHONNOUSERSITE'] = '1'  # Avoid using user site packages

# Update setup with C++17 for newer PyTorch
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    url=URL,
    license=LICENSE,
    ext_modules=[
        CUDAExtension(
            name='mesh2sdf',
            sources=['mesh2sdf_kernel.cu'],
            extra_compile_args={
                'cxx': ['-std=c++17', '-ffast-math'], 
                'nvcc': ['-std=c++17']
            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)