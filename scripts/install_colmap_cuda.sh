#!/usr/bin/env bash
# Build COLMAP with CUDA so photogrammetry can do DENSE MVS (much denser point
# clouds than the CPU/sparse fallback). The Ubuntu apt `colmap` is CPU-only and
# cannot run patch_match_stereo. Run on a machine with an NVIDIA GPU + CUDA
# toolkit (nvcc). Requires sudo for build deps.
#
# After building, point the app at it:
#   export COLMAP_BIN=/opt/colmap-cuda/bin/colmap
#   export PHOTOGRAMMETRY_DEVICE=auto   # or gpu / gpu_strict
set -euo pipefail

PREFIX="${1:-/opt/colmap-cuda}"
CUDA_ARCH="${CUDA_ARCH:-native}"   # e.g. 120 for RTX 5060 (sm_120); native lets CMake detect

echo "== Building CUDA COLMAP into ${PREFIX} (arch=${CUDA_ARCH}) =="

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Install the CUDA toolkit and put it on PATH." >&2
  echo "       (e.g. export PATH=/usr/local/cuda/bin:\$PATH)" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
  libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev \
  qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

WORK="$(mktemp -d)"
git clone https://github.com/colmap/colmap.git "${WORK}/colmap"
cd "${WORK}/colmap"
mkdir build && cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"
ninja
sudo ninja install

echo
echo "Done. Verify CUDA support (should NOT say 'requires CUDA'):"
echo "  ${PREFIX}/bin/colmap patch_match_stereo -h | head"
echo "Then set: export COLMAP_BIN=${PREFIX}/bin/colmap"
