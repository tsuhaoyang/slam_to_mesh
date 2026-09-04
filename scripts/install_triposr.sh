#!/usr/bin/env bash
# Set up TripoSR (single-image → 3D) in its OWN isolated venv. Kept out of
# Docker because its deps are heavy/version-pinned and GPU-arch-specific. The
# app calls it out of process; point the app at the checkout with TRIPOSR_DIR.
#
# Encodes the gotchas found during validation (see docs/triposr_2d_to_3d.md):
#   - RTX 5060 (sm_120 / Blackwell) needs PyTorch **nightly cu128**; stable
#     cu124 reports CUDA available but fails with "no kernel image".
#   - torchmcubes won't build against torch 2.12 nightly + CUDA 13 → we patch
#     TripoSR's single marching-cubes call to use scikit-image (CPU) instead.
#   - torchvision isn't used by TripoSR; remove it to avoid an ABI clash.
#
# Requires: git, uv (or python3-venv+pip), an NVIDIA GPU. Run on the host.
set -euo pipefail

DEST="${1:-$(pwd)/TripoSR}"
REPO="https://github.com/VAST-AI-Research/TripoSR.git"

echo "== TripoSR setup into ${DEST} =="

command -v git >/dev/null || { echo "ERROR: git required" >&2; exit 1; }
PIP_INSTALL="pip install"
if command -v uv >/dev/null 2>&1; then USE_UV=1; else USE_UV=0; fi

[ -d "${DEST}/.git" ] || git clone --depth 1 "${REPO}" "${DEST}"
cd "${DEST}"

# venv
if [ "${USE_UV}" = "1" ]; then uv venv .venv --python 3.10; else python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="/usr/local/cuda/bin:${PATH}"
export PIP_DEFAULT_TIMEOUT=1200

# 1) PyTorch nightly cu128 (sm_120). No torchvision (unused; avoids ABI clash).
python -m pip install --upgrade pip
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall --no-cache-dir
python -m pip uninstall -y torchvision 2>/dev/null || true
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
x = torch.randn(512, 512, device="cuda"); float((x @ x).sum())
print("torch", torch.__version__, "GPU OK:", torch.cuda.get_device_name(0))
PY

# 2) TripoSR runtime deps (no torchvision, no torchmcubes) + scikit-image.
$PIP_INSTALL omegaconf==2.3.0 "Pillow>=10.1.0" einops==0.7.0 transformers==4.35.0 \
  "trimesh>=4.0.5" rembg huggingface-hub "imageio[ffmpeg]" xatlas==0.0.9 \
  moderngl==5.10.0 onnxruntime scikit-image

# 3) Patch tsr/models/isosurface.py to use scikit-image marching cubes if the
#    repo copy still imports torchmcubes. (Idempotent: only patches once.)
ISO="tsr/models/isosurface.py"
if grep -q "from torchmcubes import marching_cubes" "${ISO}"; then
  echo "Patching ${ISO} to use scikit-image marching cubes…"
  python - "$ISO" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
s = s.replace("from torchmcubes import marching_cubes", "from skimage import measure")
# Replace the marching_cubes call site with a skimage wrapper.
s = s.replace(
    "self.mc_func: Callable = marching_cubes",
    "self.mc_func: Callable = _skimage_mc",
)
wrapper = '''

def _skimage_mc(volume, threshold):
    import numpy as np, torch
    from skimage import measure
    vol = volume.detach().cpu().numpy().astype("float32")
    try:
        v, f, _n, _val = measure.marching_cubes(vol, level=threshold)
    except (ValueError, RuntimeError):
        return (torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long))
    import numpy as _np
    return (torch.from_numpy(_np.ascontiguousarray(v)).float(),
            torch.from_numpy(_np.ascontiguousarray(f)).long())
'''
s = s.replace("from skimage import measure\n", "from skimage import measure\n" + wrapper, 1)
open(p, "w").write(s)
print("patched", p)
PY
fi

echo
echo "Done. Try a run:"
echo "  cd ${DEST} && source .venv/bin/activate && python run.py examples/chair.png --output-dir /tmp/triposr_out --mc-resolution 256"
echo "Then point the app at it: export TRIPOSR_DIR=${DEST}"
