#!/bin/bash
# activate_clang.sh - Source this to use Clang build environment

# Source this file with: source activate_clang.sh

export PYTORCH_BUILD_DIR="build_clang"
export PYTORCH_INSTALL_DIR="$(pwd)/install_clang"
export CC=clang
export CXX=clang++
export CMAKE_C_COMPILER=clang
export CMAKE_CXX_COMPILER=clang++
export CMAKE_BUILD_DIR="build_clang"

# Add build-specific flags
export CXXFLAGS="-O2 -g -march=native -stdlib=libc++"

# Set PYTHONPATH to use Clang-compiled PyTorch
export PYTHONPATH="$PYTORCH_INSTALL_DIR/lib/python3.12/site-packages:$PYTHONPATH"

# Remove any existing PyTorch from path to ensure clean import
export PYTHONDONTWRITEBYTECODE=1

echo "🔧 Switched to Clang build environment"
echo "   Build dir: $PYTORCH_BUILD_DIR"
echo "   Install dir: $PYTORCH_INSTALL_DIR"
echo "   CC: $CC ($(clang --version 2>/dev/null | head -1))"
echo "   CXX: $CXX"
echo "   PYTHONPATH: $PYTHONPATH"

# Check if build exists
if [ -d "$PYTORCH_BUILD_DIR" ]; then
    echo "   ✅ Clang build directory exists"
else
    echo "   ⚠️  Clang build directory not found - run ./build_clang.sh first"
fi

# Check if installation exists
if [ -d "$PYTORCH_INSTALL_DIR/lib/python3.12/site-packages/torch" ]; then
    echo "   ✅ Clang PyTorch installation found"
else
    echo "   ⚠️  Clang PyTorch installation not found - run ./build_clang.sh first"
fi