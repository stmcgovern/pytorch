#!/bin/bash
# build_clang.sh - Build PyTorch with Clang in separate directory

set -e

# Compiler settings
export CC=clang
export CXX=clang++
export CMAKE_C_COMPILER=clang
export CMAKE_CXX_COMPILER=clang++

# Build directory for Clang
BUILD_DIR="build_clang"
CMAKE_BUILD_TYPE=RelWithDebInfo

# Set install directory for Clang build  
INSTALL_DIR="$(pwd)/install_clang"
# Note: PyTorch warns against setting CMAKE_INSTALL_PREFIX in environment

# Optional: Clang-specific optimizations with libc++
export CXXFLAGS="-O2 -g -march=native -stdlib=libc++"
export MAX_JOBS=20

# Enable ccache if available
if command -v ccache >/dev/null 2>&1; then
    export CC="ccache clang"
    export CXX="ccache clang++"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    echo "Using ccache for Clang build"
fi

# Clean and create build and install directories
rm -rf $BUILD_DIR $INSTALL_DIR
mkdir -p $BUILD_DIR $INSTALL_DIR

echo "Building PyTorch with Clang..."
echo "Compiler: $(clang++ --version | head -1)"
echo "Build dir: $BUILD_DIR"
echo "Install dir: $INSTALL_DIR"

# Use the new PYTORCH_BUILD_DIR environment variable
export PYTORCH_BUILD_DIR="$BUILD_DIR"

echo "Using PYTORCH_BUILD_DIR: $PYTORCH_BUILD_DIR"

# Now use PyTorch's standard build process which will use our BUILD_DIR
python setup.py build
python setup.py install --prefix=$INSTALL_DIR

echo "Clang build completed!"
echo "Build artifacts: $BUILD_DIR"
echo "Installation: $INSTALL_DIR"

# Verify the build worked by checking for libraries
TORCH_LIB="$INSTALL_DIR/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"
if [ -f "$TORCH_LIB" ]; then
    echo "✅ Clang-compiled PyTorch libraries found in install directory"
    
    # Check compiler signature
    echo "Checking compiler signature:"
    strings "$TORCH_LIB" | grep -E "clang|GCC" | head -3
else
    echo "❌ Build libraries not found in install directory"
    echo "Contents of install directory:"
    find $INSTALL_DIR -name "*.so" | head -5
    echo ""
    echo "Contents of build directory:"
    find $BUILD_DIR -name "*.so" | head -5
fi