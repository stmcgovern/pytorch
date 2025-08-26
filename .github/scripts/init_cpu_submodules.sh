#!/bin/bash
# .github/scripts/init_cpu_submodules.sh
#
# This script selectively initializes and clones only the Git submodules
# that are essential for a CPU-only build of PyTorch.

set -e

echo "--- Initializing minimal submodules for CPU build ---"

# A list of essential third-party submodules for a CPU build.
# This list excludes all CUDA, ROCm, and other hardware-specific libraries.
CPU_SUBMODULES=(
    "third_party/pybind11"
    "third_party/protobuf"
    "third_party/onnx"
    "third_party/fbgemm"
    "third_party/gloo"
    "third_party/cpuinfo"
    "third_party/xnnpack"
    "third_party/sleef"
    "third_party/ideep"
    "third_party/ittapi"
    "third_party/fmt"
)

# Use git submodule update --init on each required path.
# The --jobs flag parallelizes the download to speed it up.
git submodule update --init --jobs=$(nproc) "${CPU_SUBMODULES[@]}"

echo "--- Minimal submodule checkout complete ---"

