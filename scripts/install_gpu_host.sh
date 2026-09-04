#!/usr/bin/env bash
# Install the NVIDIA Container Toolkit on the HOST so Docker can pass the GPU
# into containers (needed for `make up-gpu`). Run on the host, not in a
# container. Requires sudo and an NVIDIA driver already installed.
#
# Verify afterwards:
#   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
set -euo pipefail

echo "== NVIDIA Container Toolkit installer (host) =="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. Install the NVIDIA driver first." >&2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker first." >&2
  exit 1
fi

# Already present?
if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
  echo "nvidia runtime already registered with Docker — nothing to do."
  exit 0
fi

# Official repo (Debian/Ubuntu).
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Register the runtime with Docker and restart it.
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker || echo "restart docker manually if not using systemd"

echo
echo "Done. Verify with:"
echo "  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
