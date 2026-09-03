# Roadmap

Status of features described in the README vs. what is implemented. The 9-stage
pipeline, CLI, FastAPI service, backend registry, resumable manifest, and a
48-test suite are all implemented and verified end-to-end on real scan data
(Stanford Armadillo: 99,976 → 8,948 faces, 100% quad, mean surface error 0.09%
of bbox). See `reports/armadillo_compare.png`.

## Environment

- GPU: **NVIDIA GeForce RTX 5060 Laptop, 8 GB VRAM** (driver 596.13, CUDA 13.2 runtime).
- `nvcc` / CUDA toolkit **not installed** (runtime only) — prefer prebuilt wheels
  or bundled binaries over source builds for any GPU dependency.
- No `quadriflow` / `InstantMeshes` binary on PATH.

## Priority order (as agreed)

### 1. Normal-map baking — **DONE**
`core/stages/bake.py` now bakes a tangent-space normal map (`07_bake_normal.png`)
alongside color: per-texel it samples the reference surface normal, transforms it
into the texel's tangent frame (tangents from the UV gradient, Gram-Schmidt
against the low-poly normal), and encodes 0.5-centered RGB. Attached on export as
a glTF `normalTexture` / USD `normal` input. Verified: mean RGB ≈ (127,128,253),
the signature of a valid tangent-space map.

### 2. Baker performance / vectorization — **DONE**
Rewrote the rasterizer to compute barycentrics for each triangle's pixel bounding
box in one numpy pass (no per-pixel Python loop), color + normal share a single
rasterization, and the closest-point query uses Open3D `RaycastingScene`
(vectorized, GPU-friendly) with a trimesh fallback. **~14× faster** (28.6 s → 2.0 s
at 1024² on the test mesh).

### 3. QuadriFlow remesh backend — **DONE (field-aligned)**
`backends/quadriflow.py` drives the QuadriFlow CLI as a subprocess
(`quadriflow -i in.obj -o out.obj -f <faces> [-sharp]`), registered as
`quadriflow` and `quadriflow_gpu`. Binary discovery: `QUADRIFLOW_BIN` env →
`PATH` → a sibling `QuadriFlow/build/quadriflow`. When no binary is present
`is_available()` is False and the registry falls back to the CPU backend, so the
pipeline always completes. Built from source (CPU + OpenMP, CMake) against
system Eigen 3.4 + Boost 1.74.

Verified on the Stanford Armadillo (target 8000): QuadriFlow produces genuinely
field-aligned, **100% quad, watertight** topology whose edge flow follows the
surface curvature — vs the CPU backend's regular-but-irregular-flow, non-
watertight output. Mean surface error 0.071% of bbox (vs 0.090% CPU). See
`reports/backend_compare.png` for the edge-flow difference.

Acceleration note: acceleration is a property of how the binary was built. The
current build is CPU+OpenMP (already field-aligned). A CUDA-enabled build (the
box has CUDA 13.3 / sm_120) would use the GPU without any code change — same
`quadriflow_gpu` id, same subprocess interface. Wiring the CUDA solver build is a
future optimization, not required for correct field-aligned output.

### 4. `quad_ratio` metric consistency — **DONE**
`backends/remesh_pymeshlab.py` now parses its own saved OBJ (`_count_quads`) and
reports the true `quad_ratio` / `quads` / `polygon_faces`. The remesh stage
prefers the backend-reported metrics (single source of truth) and only falls back
to re-parsing when a backend omits them.

### 5. `feature_lines` guidance — **DONE (explicit, backend-dependent)**
The CPU pymeshlab backend cannot consume an arbitrary feature-line file, so
instead of silently ignoring `RemeshConfig.feature_lines` it now reports
`feature_lines_provided` / `feature_lines_used` / `feature_lines_note` in the
remesh metrics and surfaces the note in the stage message (e.g. "provided but not
consumed by pymeshlab_cpu; using dihedral-angle detection instead"). A
field-aligned GPU backend can honor the file when wired in (#3).

## Quality / robustness backlog (works, could be better)

### 6. Hole filling — **DONE (iterative, measured)**
`core/stages/clean.py` now fills holes iteratively: it measures boundary edges
via pymeshlab topological measures before/after, retries `close_holes` across up
to `CleanConfig.max_hole_fill_iterations` passes, and inserts a non-manifold
repair to unblock closing when a pass stalls (the common failure on fragmented
input). It reports `hole_boundary_edges_closed` and `hole_fill_passes`, and stops
early once no boundary edges remain or a pass makes no progress. Verified: a
sphere with 80 faces removed closes from 36 boundary edges to 0 (watertight) in 3
passes; `fill_holes=False` leaves holes open.

### 7. USD material/texture export — **DONE**
`core/stages/export.py::_export_usd` now builds a `UsdPreviewSurface` material
with `UsdUVTexture` readers (shared `st` primvar reader), binding the baked color
map to `diffuseColor` and the normal map to `normal` (raw color space), via
`MaterialBindingAPI`. Verified against `usd-core`.

### 8. QC watertight signal — **DONE**
QC now measures geometric watertightness on the pre-unwrap PROJECT mesh (so UV
seams don't create false holes) and reports it as `is_watertight` — the number to
trust. It also reports `final_mesh_watertight` (the raw UV mesh, usually False)
and a `seam_vertex_split` flag that trips when the pre-seam mesh is closed but the
final isn't. Note: the current CPU remesher's output is not perfectly watertight
even from closed input, so `is_watertight` honestly reflects that (a field-aligned
GPU backend, #3, would improve it).

## Done / verified
9 stages (ingest, clean, decimate, remesh, project, unwrap, bake-color, export,
qc), CLI (`run`, `resume-from`, `inspect`, `list-backends`), FastAPI service
(create/status/download), resumable job manifest, backend registry + CPU
fallback, 48 passing tests.

## Feature: Interactive quad decimation (LOD preview + export) — **DONE**

Spec: `docs/spec_interactive_decimation.md`.

Web UI (`/ui/`) where the user drags a slider (% of original faces) to reduce the
mesh, sees a rotatable Three.js preview with an optional bake toggle
(color/normal), reads fidelity metrics, and exports the chosen LOD. Decimation =
**re-running QuadriFlow at a different resolution** (keeps regular quad topology;
QEM would not). `core/lod.py::build_lod` produces + caches each LOD; LOD API
endpoints (`POST /lod`, `GET /lod/{n}/model.glb`, `GET /lods`, `POST /export-lod`)
extend the service; the Three.js frontend is served as static assets. Verified
end-to-end (build + cache + glb download + export zip); 64 tests pass.

## Feature: Flexible inputs + multi-view + point clouds — **DONE**

Spec: `docs/spec_flexible_io.md`.

- **Inputs**: triangle mesh **or** point cloud. Point clouds are Poisson-
  reconstructed to a mesh at ingest (`core/pointcloud.py`), original points
  retained (`00_points.ply`); `input_kind`/`has_pointcloud` on the manifest.
- **Multi-view UI**: six checkbox representations — triangle, triangle-decimated
  (QEM), quad, quad-decimated (QuadriFlow), point cloud, point-cloud-downsample
  (voxel). Each opens its own synced Three.js pane; layout is dynamic.
- **Independent decimation**: triangle (QEM, exact faces) and quad (QuadriFlow,
  re-remesh) have separate sliders; point-cloud voxel downsample is a separate
  slider. They do not drive each other (no point↔mesh correspondence).
- **Gating**: point-cloud downsample is selectable only when the point cloud is
  shown alone; any mesh representation disables it and vice versa.
- **Point-cloud ops**: generate from a mesh surface (opt-in) and download
  original/downsampled points.
- API: `POST /tri-lod`, `GET /tri-lod/{n}/model.glb`, `GET /pointcloud.json`,
  `POST /pointcloud/downsample`, `POST /pointcloud/generate`,
  `GET /pointcloud/download`, plus `input_kind`/`has_pointcloud` on `GET /jobs/{id}`.
- Verified via Playwright (mesh → 3 synced panes; point cloud → downsample compare
  with correct gating). 80 tests pass.

### 2D image → 3D (TripoSR) — **INTEGRATED (pluggable backend)**
Image input is a first-class input type. Single-image 3D is a **swappable backend
registry** (`backends/image3d.py`, mirroring the remesh registry): **TripoSR** is
the built-in impl (`backends/image3d_triposr.py`, isolated venv via subprocess).
Ingest detects `.png/.jpg/.jpeg/.webp`, picks the configured
`PipelineConfig.image_backend` (default `triposr`; `None` = first available),
generates a mesh, then the normal pipeline runs; `input_kind = "image"`. Gated by
availability — image uploads return 503 when no backend is available, and
`GET /capabilities` reports `image_backends`. A stronger model such as **TRELLIS**
(Microsoft, MIT) can be added as another backend on a bigger-VRAM server and
selected via `image_backend` (form field / config) without touching the pipeline.

TripoSR is MIT-licensed (VAST-AI-Research; Tripo AI + Stability AI), incl.
weights, and runs in its own venv. Two obstacles solved: (1) needs PyTorch
**nightly cu128** for sm_120; (2) `torchmcubes` won't build against torch 2.12
nightly + CUDA 13, so its single marching-cubes call was replaced with
**scikit-image CPU** marching cubes (inference stays on GPU). Install notes:
`docs/triposr_2d_to_3d.md`; the isosurface patch: `docs/triposr_isosurface_cpu.patch`.

### Known limitation
- Re-remeshing at high face targets is slow (QuadriFlow ~1–2 s small, tens of
  seconds for very dense targets); the UI is snappy at low percentages. A
  precomputed LOD ladder would remove the wait (future optimization).
