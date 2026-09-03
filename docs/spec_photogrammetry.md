# Spec: Multi-view photogrammetry input (images / video → point cloud)

Status: **draft** · Depends on: existing point-cloud pipeline
(`core/pointcloud.py`, ingest point-cloud branch), backend-registry pattern.

## 1. Summary

Add **measurement-based** 3D from many overlapping views, for objects that
single-image generation gets wrong (rigid, straight-edged things like a
workstation desk). Two new inputs:

- **image set** — a `.zip` of 10–50 photos taken around the object.
- **video** — a `.mp4/.mov`; frames are extracted, then treated as an image set.

Reconstruction is **photogrammetry** (Structure-from-Motion + Multi-View Stereo)
producing a **dense point cloud**, which then flows into the *existing*
point-cloud path (Poisson → clean → QuadriFlow → …). So this feature is mostly a
new front-end that yields a point cloud; everything downstream is reused.

Unlike single-image generation (a guess), photogrammetry **measures** geometry,
so straight edges stay straight — at the cost of needing many views and minutes
of compute.

## 2. Inputs & detection

- `.zip` → `input_kind = "images"`. Unzip, collect image files (png/jpg/…),
  require a sane minimum (≥ ~8; warn if < 10 or > ~200).
- `.mp4/.mov/.mkv/.webm` → `input_kind = "video"`. Extract frames (ffmpeg) at a
  target count (default ~40, evenly spaced), then proceed as an image set.

## 3. Backend registry (pluggable, like remesh / image3d)

`backends/photogrammetry.py`:
- `PhotogrammetryBackend` protocol: `id`, `is_available()`,
  `reconstruct(images_dir: Path, out_points: Path) -> dict`.
- Registry: `register_backend` / `get_backend(id|None)` / `available_backends()`.
- First impl: **COLMAP** (`backends/photogrammetry_colmap.py`), invoked as a
  subprocess CLI. Availability = `colmap` binary on PATH (or `COLMAP_BIN`).
- Later swappable: Meshroom/AliceVision, or learning-based Dust3R/VGGT.

COLMAP flow (automatic reconstruction, then export dense points):
```
colmap feature_extractor  → colmap exhaustive_matcher →
colmap mapper (SfM)       → colmap image_undistorter  →
colmap patch_match_stereo → colmap stereo_fusion       → fused.ply (dense points)
```
(We may use `colmap automatic_reconstructor` for simplicity first, then refine.)
GPU is used when COLMAP is built with CUDA; else CPU (slower). Availability
gating means no COLMAP → these inputs are simply unavailable.

## 4. Frame extraction (video)

`core/frames.py::extract_frames(video, out_dir, n=40) -> count`:
- Prefer system **ffmpeg** (fast, robust); fall back to imageio/OpenCV.
- Evenly sample `n` frames across the duration.
- Note in docs: video frames are lower quality than deliberate photos (motion
  blur, compression) — advise slow, well-lit capture.

## 5. Ingest wiring

New branch order in `core/stages/ingest.py`:
1. image (single) → image3d backend (existing).
2. **zip / video → photogrammetry**:
   - video: extract frames → images dir.
   - zip: unzip → images dir.
   - `reconstruct(images_dir, 00_points.ply)` via photogrammetry backend.
   - retain the dense point cloud as `00_points.ply`, set `has_pointcloud=True`,
     `input_kind = "images"` (or "video").
   - then reconstruct a mesh from those points (reuse `reconstruct_poisson`) as
     the ingest mesh — identical to the point-cloud input path.
3. point cloud → Poisson (existing).
4. mesh → direct (existing).

So after ingest, an images/video job behaves exactly like a point-cloud job
(point cloud retained + mesh for the surface pipeline).

## 6. Service

- `POST /jobs` accepts `.zip` and video extensions; image-set/video gated on
  photogrammetry availability (503 if COLMAP absent), mirroring image gating.
- `GET /capabilities` adds `photogrammetry` (bool) + `photogrammetry_backends`.
- Optional form field `frames` (video frame count, default 40).

## 7. Config

`PipelineConfig`:
- `photogrammetry_backend: str | None = "colmap"` (None = first available).
- `video_frames: int = 40`.

## 8. Testing

- `core/frames.py`: extract N frames from a tiny synthetic video (generate with
  imageio) → N images. (ffmpeg optional; test the fallback path.)
- Registry: availability/selection/None-when-absent (COLMAP likely absent in CI
  → tests assert graceful unavailability + clear errors).
- Ingest: zip/video detection routes to photogrammetry; clear error when backend
  unavailable (the common CI case).
- Heavy e2e (real COLMAP on a real image set): `skipif` COLMAP missing.

## 9. Rollout / task order

1. `core/frames.py` (video → frames) + tests.
2. `backends/photogrammetry.py` registry + `photogrammetry_colmap.py` (subprocess
   + availability) + tests (graceful-absent).
3. Ingest zip/video branch → dense points → reuse point-cloud path; config +
   `input_kind`.
4. Service: accept zip/video, gating, `/capabilities`, `frames` field.
5. Frontend: allow `.zip`/video in the upload input; note processing time.
6. COLMAP install (user, when ready) + real end-to-end validation.
7. Docs (README + ROADMAP) + commits.

## 10. Non-goals / notes
- Not real-time; photogrammetry takes minutes. UI must set expectations.
- Quality depends on capture (overlap, lighting, sharpness). Document guidance.
- Camera intrinsics: let COLMAP self-calibrate (single-camera assumption); expose
  nothing extra for now.
