# Deployment

How to run slam_to_mesh, and what lives in Docker vs. what must be installed on
the host. Designed so a new (production) machine needs the fewest manual steps.

## What runs where

| Piece | Location | Why |
|---|---|---|
| Pipeline core, CLI, FastAPI service + UI | **Docker image** | one command to run |
| pymeshlab (+ OpenGL libs), Open3D, xatlas, usd-core | **Docker image** | awkward native deps, frozen |
| QuadriFlow (field-aligned quad remesh) | **Docker image** (built from source) | no packaged binary |
| COLMAP **CPU** (photogrammetry, sparse) | **Docker image** (apt) | works headless, no GPU |
| ffmpeg (video → frames) | **Docker image** | — |
| **NVIDIA Container Toolkit** | **Host** (`scripts/install_gpu_host.sh`) | lets Docker see the GPU |
| **COLMAP CUDA** (dense photogrammetry) | **Host** (`scripts/install_colmap_cuda.sh`) | dense MVS needs CUDA; mount into the GPU container |
| **TripoSR** (single-image → 3D) | **Host venv** (`scripts/install_triposr.sh`) | heavy, GPU-arch-specific deps; called out of process |

Rule of thumb: **CPU + reproducible → in the image; GPU-specific or huge → host
script + mounted/env-pointed into the container.**

## Prerequisites

- **Always**: Docker + Docker Compose (v2). Verify: `docker compose version`.
- **For GPU features** (dense photogrammetry / faster remesh / TripoSR):
  - NVIDIA driver on the host (`nvidia-smi` works).
  - NVIDIA Container Toolkit: run `scripts/install_gpu_host.sh` (host, sudo).
  - Verify: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.

## Quick start (CPU — the common case)

```bash
make build      # build the CPU image (first time; ~minutes)
make up         # start the service
# → http://localhost:8000/ui/   (health: /healthz)
make logs       # tail logs
make down       # stop
```

This gives you: mesh / point-cloud / **image-set (.zip)** / **video** inputs,
the full retopo pipeline, QuadriFlow quads, and the multi-view UI. Photogrammetry
runs **sparse** (CPU COLMAP) — enough to work, fewer points than dense.

Run the tests in-container: `make test`.

## GPU deployment (optional, for dense photogrammetry / TripoSR)

1. Host setup:
   ```bash
   scripts/install_gpu_host.sh        # NVIDIA Container Toolkit
   scripts/install_colmap_cuda.sh     # CUDA COLMAP → /opt/colmap-cuda (optional, for dense)
   scripts/install_triposr.sh /opt/TripoSR   # image→3D (optional)
   ```
2. In `docker-compose.yml`, uncomment the GPU service's mounts/env for the pieces
   you installed:
   ```yaml
   environment:
     COLMAP_BIN: /opt/colmap-cuda/bin/colmap
     TRIPOSR_DIR: /opt/TripoSR
     PHOTOGRAMMETRY_DEVICE: auto
   volumes:
     - /opt/colmap-cuda:/opt/colmap-cuda:ro
     - /opt/TripoSR:/opt/TripoSR:ro
   ```
3. Start the GPU service:
   ```bash
   make up-gpu
   ```

## Environment variables

| Var | Purpose | Default |
|---|---|---|
| `JOBS_ROOT` | where the service stores jobs | `/data/service_jobs` (image) |
| `QUADRIFLOW_BIN` | path to the QuadriFlow binary | `/opt/quadriflow/quadriflow` (image) |
| `COLMAP_BIN` | path to a COLMAP binary (e.g. CUDA build) | `colmap` on PATH |
| `TRIPOSR_DIR` | path to the TripoSR checkout (enables image input) | sibling `TripoSR/` if present |
| `PHOTOGRAMMETRY_DEVICE` | `auto` / `cpu` / `gpu` / `gpu_strict` | `cpu` (CPU image) |

Availability is always **detected at runtime**: choosing GPU doesn't assume a
GPU — the app checks the binary's CUDA support and the hardware, and either uses
it, falls back (auto/gpu), or errors clearly (gpu_strict). Missing pieces just
disable the corresponding input (e.g. no TripoSR → image upload returns 503;
`GET /capabilities` reports what's available).

## Notes / limitations

- CPU COLMAP cannot do dense MVS (`patch_match_stereo` requires CUDA); the
  backend falls back to exporting sparse SfM points. Use `install_colmap_cuda.sh`
  for dense.
- Photogrammetry capture: 10–50 **overlapping, sharp, textured** views; too few
  or non-overlapping images make SfM fail to build a model.
- TripoSR image size/deps make it unsuitable to bake into the main image; the
  host-venv + subprocess design keeps it isolated and swappable (e.g. TRELLIS
  later).
- First image/photogrammetry runs download model weights / take minutes; the UI
  sets expectations.

## Photogrammetry quality & the CUDA COLMAP fusion workaround

- **CPU COLMAP** does sparse only (a few hundred points) → **not usable** for a
  real surface; results look like a blob. Use CUDA COLMAP for dense.
- **CUDA COLMAP dense** on very new GPUs (observed: RTX 5060, sm_120, CUDA 13.3):
  the PatchMatch **geometric consistency filter zeroes all depth**, so COLMAP's
  own `stereo_fusion` yields 0 points (bug across COLMAP 3.9 and 3.11). The
  photometric depth maps are still valid, so `ColmapBackend` **falls back to a
  custom depth-map fusion** (`core/depth_fusion.py`) that back-projects the valid
  depth via the recovered poses. `reconstruction` reports `dense_custom_fusion`.
- **Quality still depends heavily on capture.** The custom fusion does not yet do
  multi-view consistency filtering, so noisy/sparse captures produce noisy clouds
  → blobby meshes. For a usable object: **orbit densely, 40–60 photos, >70%
  overlap, textured object, even lighting, plain background**; ideally two height
  rings.
- Robustness fixes so the pipeline always completes on messy reconstructions:
  Poisson uses KNN normals (+ consistent orientation) and caps the mesh to a
  safe face budget; the QuadriFlow backend retries **without `-sharp`** if the
  sharp run hangs/fails (common on holey photogrammetry meshes).
