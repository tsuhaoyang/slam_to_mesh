# slam_to_mesh

Convert messy 3D SLAM meshes into **lightweight, quad-dominant, UV-unwrapped**
meshes that are cheap to display and easy to texture in **NVIDIA Omniverse**
(USD / glTF).

SLAM output (Poisson / TSDF / marching-cubes) is a byproduct of surface
sampling: hundreds of thousands to millions of irregular triangles, with holes,
non-manifold edges and floating islands. This project retopologizes it into a
clean, regular, lightweight mesh suitable for material authoring and real-time
display.

> **Reality check:** you cannot *perfectly* recover the deliberate quad topology
> a human artist would author — that design intent never existed in the scan.
> The goal here is a **regular, quad-dominant, low-poly mesh that is
> geometrically faithful and production-usable**.

## Pipeline

```
input SLAM mesh (.ply/.obj)
  1. Ingest & Analyze   load, stats, detect problems
  2. Clean              remove islands, fill holes, fix non-manifold, unify normals
  3. Decimate           QEM simplify to target face count
  4. Remesh (quad)      field-aligned quad-dominant remesh
  5. Project            snap remeshed verts back onto original surface
  6. UV unwrap          xatlas atlas
  7. Bake (optional)    original color / normal detail -> low-poly textures
  8. Export             USD / glTF + textures
  9. QC report          quad ratio, Hausdorff distance, manifoldness, face count
```

## Design principles

- **CPU-first.** Every stage runs without a GPU. GPU-capable stages
  (remesh, bake) go through a **backend abstraction** so they can be swapped for
  accelerated implementations on a GPU server later.
- **Resumable.** Each stage writes a named intermediate artifact and records its
  parameters in a job manifest. Users can inspect results, tweak a stage's
  parameters, and re-run only from that stage onward.
- **Shared core.** The same pipeline core powers both the CLI and (later) the
  FastAPI service.

## Layout

```
src/slam_to_mesh/
  core/       pipeline stages, orchestration, data model
  backends/   swappable implementations (CPU now, GPU later)
  cli/        typer CLI
  service/    FastAPI service (later)
```

## Install (CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional: USD export
pip install -e ".[usd]"
```

## Usage

```bash
slam2mesh run input.ply --out out/ --target-faces 20000
slam2mesh resume-from remesh --job out/job.json
slam2mesh inspect out/job.json
```

## Service (FastAPI)

```bash
pip install -e ".[service]"
uvicorn slam_to_mesh.service.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /jobs` — upload a mesh (multipart `file`) plus optional form fields
  (`target_faces`, `decimate_faces`, `backend`, `bake`, `formats`, `project`).
  Returns a `job_id`; processing runs in the background.
- `GET /jobs/{job_id}` — job status and per-stage results.
- `GET /jobs/{job_id}/download` — zip of final artifacts, or `?fmt=glb` for a
  single file.
- `GET /backends` — available remesh backends.

Interactive decimation (LOD) endpoints:

- `POST /jobs/{job_id}/lod` — build (or return cached) a quad LOD at a chosen
  resolution. Body: `{"target_faces": N}` or `{"ratio": 0.0–1.0}` (ratio is of
  the original input face count), plus optional `"bake": true` to bake
  color/normal textures into the preview. Returns the LOD's metrics and a
  `glb_url`.
- `GET /jobs/{job_id}/lod/{target_faces}/model.glb?baked=` — the LOD glb.
- `GET /jobs/{job_id}/lods` — all LODs built so far.
- `POST /jobs/{job_id}/export-lod` — build a LOD and download a zip of the
  chosen formats (`glb`/`usd`/`obj`) with baked textures.

Because every LOD is produced by **re-running the quad remesher** at a new
resolution (not QEM edge-collapse), each level stays regular quad-dominant and
export-ready. High-curvature/feature regions keep denser quads automatically
(QuadriFlow's field follows curvature), so reducing faces preserves detail where
it matters. Results are cached per (face bucket, baked) so revisiting a level is
instant.

## Interactive UI

With the service running, open `http://localhost:8000/ui/` for a Three.js
multi-view page that lets you:

- paste a `job_id` and load it,
- choose which **representations** to display via checkboxes — each opens its own
  synced, rotatable viewer pane:
  - **Triangle mesh** (the normalized/ingest surface),
  - **Triangle — decimated** (QEM edge-collapse, exact target face count),
  - **Quad mesh** and **Quad — decimated** (field-aligned QuadriFlow, shown as a
    real quad wireframe),
  - **Point cloud** and **Point cloud — downsample** (Open3D voxel),
- drive independent sliders per representation (triangle faces, quad faces,
  point count),
- read fidelity metrics (actual faces, quad ratio, mean surface error),
- generate a point cloud from a mesh input and download point clouds.

Gating: point-cloud downsampling is a dedicated original-vs-reduced compare mode,
selectable only when the point cloud is shown alone; selecting any mesh
representation disables it (and vice versa).

### Inputs

The service accepts **triangle meshes** (`.ply/.obj/.stl/.glb/.off`), **point
clouds** (`.ply` points / `.pcd` / `.xyz`), and **2D images**
(`.png/.jpg/.webp`). Point clouds are reconstructed to a triangle mesh (Open3D
Poisson) at ingest so the surface pipeline can run; the original points are
retained for the point-cloud viewer and downsampling. Images are turned into a
3D mesh at ingest via a **pluggable single-image backend**
(`slam_to_mesh.backends.image3d`) — **TripoSR** is the built-in implementation
(isolated venv, called out of process); a stronger model such as **TRELLIS**
(MIT) can be registered as another backend on a bigger-VRAM server and selected
via `image_backend` without touching the pipeline. If no image backend is
installed the image input is simply unavailable (see `GET /capabilities` and
`docs/triposr_2d_to_3d.md`).

For **rigid, straight-edged objects** (where single-image generation distorts),
the service also accepts **multi-view input**: a **`.zip` of photos** (10–50,
taken around the object) or a **video** (`.mp4/.mov/.mkv/.webm`; frames are
sampled — set the count with the `frames` field). These go through
**photogrammetry** (Structure-from-Motion + MVS via a pluggable
`slam_to_mesh.backends.photogrammetry` backend, **COLMAP** built-in) to produce a
dense point cloud, which then flows into the normal point-cloud path. This
*measures* geometry (straight edges stay straight) rather than guessing, at the
cost of minutes of compute. Requires COLMAP; unavailable otherwise
(see `docs/spec_photogrammetry.md`).

The service and CLI share the same pipeline core (`slam_to_mesh.core.pipeline`)
and on-disk job manifest, so jobs are inspectable and resumable either way.

## GPU note

The remesh stage goes through a backend registry
(`slam_to_mesh.backends.remesh`). A field-aligned **QuadriFlow** backend is wired
in (built from source); it is selected via `--backend quadriflow` /
`RemeshConfig.backend`, and the registry falls back to the CPU remesher when the
binary is absent so the pipeline always completes.

