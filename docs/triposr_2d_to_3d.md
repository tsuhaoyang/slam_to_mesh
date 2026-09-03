# TripoSR — verified install & feasibility notes

Single image → 3D mesh, as a **front-end for the slam_to_mesh retopo pipeline**.
This documents the path that was **actually verified working** on this machine
(RTX 5060 Laptop, 8 GB, sm_120 / Blackwell; CUDA 13.3), not a guess.

## Verdict: WORKS (feasibility validated)

- TripoSR (VAST-AI-Research, MIT license, Tripo AI + Stability AI). Weights are
  MIT too. Legitimate, commercial-friendly.
- End-to-end confirmed: `examples/chair.png` → TripoSR → 84,156-face watertight
  mesh → `slam2mesh` (clean → QuadriFlow → project) → **3,008-face, 100% quad,
  watertight, mean surface error 0.069% of bbox**.
- Timing: model inference 2.1 s (GPU), mesh extraction 2.3 s (CPU marching
  cubes), export 0.1 s. First run downloads ~1.68 GB weights + 176 MB rembg.
- See `slam_to_mesh/reports/triposr_pipeline.png` (2D → TripoSR tris → retopo quad).

## Environment isolation

TripoSR has heavy, version-pinned deps that conflict with the main project, so
it lives in **its own venv**: `/home/howard/workspace/TripoSR/.venv`.
Never mix it with `slam_to_mesh/.venv`.

## Two real obstacles and how they were solved

### 1. sm_120 (RTX 5060 is Blackwell) — stable PyTorch doesn't support it
`torch==2.6.0+cu124` reports `cuda available: True` but any GPU op fails with
`no kernel image is available for execution on the device` (its arch list stops
at sm_90). **Fix: PyTorch nightly cu128**, which includes sm_120.

### 2. torchmcubes won't build against torch 2.12 nightly + CUDA 13
`torchmcubes` (TripoSR's GPU marching-cubes dep) fails to compile: its use of
new ATen headers under nvcc 13.3 throws `need 'typename' ... dependent scope`
errors in `ATen/core/List_inl.h`. This is a dead end with this toolchain combo.
**Fix: drop torchmcubes entirely and use scikit-image CPU marching cubes.**
TripoSR only calls marching cubes at one spot (`tsr/models/isosurface.py`); the
heavy neural inference stays on the GPU. The CPU iso-surface extraction adds ~2 s
— acceptable. (Patch already applied to the local checkout; see that file.)

## Working install (what was actually run)

```bash
cd /home/howard/workspace/TripoSR
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export UV_HTTP_TIMEOUT=1200

# 1. PyTorch nightly cu128 (sm_120 support). torch first, then a matching-date
#    torchvision IF you need it — TripoSR does NOT use torchvision, so we skip it.
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall --no-cache-dir
python -m pip uninstall -y torchvision   # remove any cu124 torchvision to avoid ABI clash

# verify GPU actually computes:
python -c "import torch; x=torch.randn(2000,2000,device='cuda'); print('GPU OK', float((x@x).sum()))"

# 2. TripoSR runtime deps (no torchvision, no torchmcubes)
uv pip install omegaconf==2.3.0 "Pillow>=10.1.0" einops==0.7.0 transformers==4.35.0 \
  "trimesh>=4.0.5" rembg huggingface-hub "imageio[ffmpeg]" xatlas==0.0.9 \
  moderngl==5.10.0 onnxruntime scikit-image

# 3. Patch tsr/models/isosurface.py to use scikit-image marching cubes
#    (already done in this checkout; the file imports skimage.measure and defines
#     _skimage_marching_cubes as a drop-in for torchmcubes.marching_cubes).

# 4. Run
python run.py examples/chair.png --output-dir /tmp/triposr_out --mc-resolution 256
# → /tmp/triposr_out/0/mesh.obj
```

## Feed into slam_to_mesh

```bash
# from the slam_to_mesh venv:
slam2mesh run /tmp/triposr_out/0/mesh.obj --out out_chair \
  --target-faces 3000 --backend quadriflow --formats glb,obj
```

## Future integration (NOT done yet — deferred)

To make this a service input type alongside mesh/point-cloud:
- Add an image branch at ingest that shells out to the TripoSR venv
  (`subprocess` calling its `python run.py`) to produce a mesh, then continues
  the normal pipeline. Cross-venv via subprocess keeps the conflicting deps
  isolated.
- Gate it behind availability detection (like the QuadriFlow binary): if the
  TripoSR venv/weights are absent, the image input is simply unavailable.
- Accept `.png/.jpg` uploads in `POST /jobs`; record `input_kind = "image"`.
- UI: an upload path + a note that generation takes a few seconds on GPU.

Deferred by decision — this pass validated feasibility only.
