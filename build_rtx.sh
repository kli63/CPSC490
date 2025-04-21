#!/bin/bash
# Build script for NGLOD on RTX 4050
# This script creates the conda environment and builds CUDA extensions
# for compatibility with the RTX 4050 GPU (compute capability 8.9)

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  NGLOD RTX 4050 Compatibility Setup  ${NC}"
echo -e "${GREEN}=======================================${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed or not in PATH.${NC}"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Define environment name and project root
ENV_NAME="cpsc490_env"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Creating new conda environment: $ENV_NAME${NC}"
    conda create -y -n $ENV_NAME python=3.10
else
    echo -e "${YELLOW}Using existing conda environment: $ENV_NAME${NC}"
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA 11.8 if not already installed
if ! python -c "import torch; print(torch.__version__)" &> /dev/null; then
    echo -e "${YELLOW}Installing PyTorch with CUDA 11.8...${NC}"
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
fi

# Install NGLOD dependencies
echo -e "${YELLOW}Installing NGLOD dependencies...${NC}"
pip install matplotlib tqdm scikit-image trimesh Pillow scipy moviepy pyexr tensorboard einops pybind11 plyfile polyscope

# Install Shap-E dependencies
echo -e "${YELLOW}Installing Shap-E dependencies...${NC}"
pip install filelock humanize requests blobfile fire pyyaml typing-extensions attrs "clip @ git+https://github.com/openai/CLIP.git" ninja

# Install Shap-E package
echo -e "${YELLOW}Installing Shap-E package...${NC}"
cd "$PROJECT_ROOT/shap-e"
pip install -e .
cd "$PROJECT_ROOT"

# Build the CUDA extensions
echo -e "${YELLOW}Building CUDA extensions...${NC}"
cd "$PROJECT_ROOT/nglod/sdf-net/lib/extensions"
chmod +x build_ext.sh
./build_ext.sh

# Return to the original directory
cd "$PROJECT_ROOT"

# Create required output directories
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p "$PROJECT_ROOT/output"
mkdir -p "$PROJECT_ROOT/nglod/sdf-net/_results/models"
mkdir -p "$PROJECT_ROOT/nglod/sdf-net/_results/render_app/imgs"

# Verify the installation
echo -e "${YELLOW}Verifying installation...${NC}"
if python -c "import torch; import mesh2sdf; import sol_nglod; print('PyTorch:', torch.__version__); print('CUDA Version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'); print('Extensions loaded successfully!')"; then
    echo -e "${GREEN}✅ NGLOD extensions loaded successfully!${NC}"
else
    echo -e "${RED}❌ NGLOD extensions failed to load. Check the error messages above.${NC}"
    exit 1
fi

if python -c "import shap_e; print('Shap-E version:', shap_e.__version__ if hasattr(shap_e, '__version__') else 'Unknown'); print('✅ Shap-E loaded successfully!')"; then
    echo -e "${GREEN}✅ Shap-E loaded successfully!${NC}"
else
    echo -e "${RED}❌ Shap-E failed to load. Check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}   Unified Environment Setup Complete   ${NC}"
echo -e "${GREEN}=======================================${NC}"
echo -e "To use the unified environment:"
echo -e "1. ${YELLOW}conda activate $ENV_NAME${NC}"
echo -e "2. Run the unified pipeline: ${YELLOW}python shapegen.py --text \"a chair that looks like an avocado\" --exp-name avocado_chair${NC}"
echo -e ""
echo -e "Or use the individual components:"
echo -e "3. NGLOD: Train a model: ${YELLOW}cd nglod/sdf-net && python app/main.py --net OctreeSDF --num-lods 5 --dataset-path data/my.obj --epoch 10 --exp-name mymodel${NC}"
echo -e "4. NGLOD: Render a model: ${YELLOW}cd nglod/sdf-net && python app/sdf_renderer.py --net OctreeSDF --num-lods 5 --pretrained _results/models/mymodel.pth --render-res 1280 720 --shading-mode matcap${NC}"
echo -e "5. Test rendering: ${YELLOW}python rtx_render_test.py${NC}"
echo -e ""
echo -e "For more detailed instructions, see: ${YELLOW}RTX_4050_COMPLETE_GUIDE.md${NC}"