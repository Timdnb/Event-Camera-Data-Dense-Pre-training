#!/bin/bash

# Exit on error
set -e

# Step 1: Create and activate conda environment
echo "Creating conda environment..."
conda create -n ecddp python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecddp

# Step 2: Upgrade pip
echo "Upgrading pip..."
python -m pip install pip==23.3.2

# Step 3: Install PyTorch, Torchvision, and CUDA
echo "Installing PyTorch, Torchvision, and CUDA..."
conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Step 4: Install additional Python packages
echo "Installing additional Python packages..."
pip install pytorch-lightning==1.6.4
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
pip install numpy==1.26.4

# Step 5: Set CUDA_HOME
echo "Setting CUDA_HOME..."
export CUDA_HOME=$CONDA_PREFIX/

# Step 6: Install CUDA NVCC
echo "Installing CUDA NVCC..."
conda install nvidia/label/cuda-11.8.0::cuda-nvcc -y

# Step 7: Install and update MMEngine and MMCV
echo "Installing MMEngine and MMCV..."
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0

# Step 8: Install MMSegmentation
echo "Installing MMSegmentation..."
pip install mmsegmentation==1.2.2

# Step 9: Update conda environment with additional dependencies
if [ -f "environment.yml" ]; then
  echo "Updating conda environment from environment.yml..."
  conda env update --name ecddp_test --file environment.yml
else
  echo "environment.yml not found. Skipping..."
fi

# Step 10: Modify mmseg __init__.py
echo "Modifying mmseg __init__.py..."
MMSEG_INIT_FILE="$CONDA_PREFIX/lib/python3.10/site-packages/mmseg/__init__.py"
if [ -f "$MMSEG_INIT_FILE" ]; then
  sed -i 's/assert (mmcv_min_version <= mmcv_version < mmcv_max_version)/assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)/' "$MMSEG_INIT_FILE"
  echo "Modification complete."
else
  echo "mmseg __init__.py not found. Skipping modification..."
fi

echo "Setup completed successfully!"