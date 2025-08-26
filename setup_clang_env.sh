#!/bin/bash
# setup_clang_env.sh - Create separate conda environment for Clang build

set -e

ENV_NAME="pytorch_clang"

echo "Creating conda environment: $ENV_NAME"

# Create separate conda environment for Clang build
conda create -n $ENV_NAME python=3.11 -y

echo "Activating environment and installing dependencies..."

# Activate and install dependencies
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install dependencies  
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing_extensions

echo "Environment $ENV_NAME created and dependencies installed."
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo "  ./build_clang.sh"
echo ""
echo "Current environment info:"
echo "  Python: $(python --version)"
echo "  Clang: $(clang++ --version | head -1)"