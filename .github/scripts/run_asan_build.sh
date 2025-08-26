#!/bin/bash
# .github/scripts/run_asan_build.sh
#
# This script is now run INSIDE the Docker container.
# It assumes system dependencies like clang and python dependencies are already installed.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Setting up ASAN environment variables ---"
# Tell the build system to use Clang and enable ASAN
export CC=clang
export CXX=clang++
export USE_ASAN=1
# Explicitly disable CUDA
export USE_CUDA=0
# Use all available CPU cores to speed up the build
export MAX_JOBS=$(nproc)

echo "--- Building PyTorch with ASAN enabled (CUDA disabled) ---"
python setup.py build

echo "--- Running targeted tests ---"
# Run a small, high-value set of tests.
pytest test/test_torch.py test/test_nn.py

