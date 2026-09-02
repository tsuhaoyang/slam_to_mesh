# Spec: Interactive Quad Decimation (LOD preview + export)

Status: **draft** · Owner: slam_to_mesh · Depends on: existing pipeline core,
QuadriFlow backend, FastAPI service.

## 1. Summary

Let a user interactively reduce the face count of an already-processed mesh from
a web page: drag a slider (target faces / percentage), see the resulting
quad-dominant mesh rendered in 3D (rotatable), read fidelity metrics, and when
satisfied, export the chosen level of detail (LOD) with textures.

The output at every level **must remain regular, field-aligned quad-dominant**
geometry suitable for Omniverse export. Therefore decimation is implemented by
**re-running QuadriFlow at a different resolution**, not by QEM edge-collapse
(which would destroy quad regularity). See §3 for the rationale.

## 2. Goals / Non-goals

### Goals
- Interactive slider (target faces or % of a reference count) → new LOD.
- Every LOD is regular quad-dominant, projected back onto the original surface.
- 3D rotatable preview in the browser (Three.js), plus per-LOD metrics
  (actual face count, quad ratio, mean surface error vs original).
- Response within ~2 s per slider change (user-accepted latency).
- Result caching so revisiting a face count is instant.
- "Confirm & export" produces the final glTF/USD (+ optional bake) at the
  chosen LOD, reusing the existing export/bake stages.

### Non-goals (this phase)
- User-painted protection masks (deferred to a future phase; see §9).
- Continuous progressive-mesh (single collapse hierarchy) — not compatible with
  regular quads.
- ML-based semantic importance.

## 3. Why re-run QuadriFlow, not QEM

"Feature importance" is handled by QuadriFlow's curvature-following field: it
naturally places smaller quads in high-curvature/feature regions and larger
quads on flat areas, at any target resolution. QEM edge-collapse is a triangle
algorithm and produces irregular topology, so it cannot preserve the regular
quad layout the export path requires.

| Aspect | Re-run QuadriFlow (`-f`) | QEM decimation |
|---|---|---|
| Output topology | Regular quad (field-aligned) | Irregular tris/mixed |
| Meets export need | Yes | No (must re-quad, quality lost) |
| Feature preservation | Curvature field auto-densifies features | Via QEM error, but breaks quads |
| Exact face count | Approximate (`-f` is a target) | Exact |
| Single-call speed | ~1–2 s | ms |
| Complexity | Low (reuse backend) | Medium |

Decision: **re-run QuadriFlow**. The only downsides (≈2 s latency, approximate
face count) are acceptable given the requirements.

## 4. Importance model (how "important" is decided)

Phase 1 relies entirely on QuadriFlow's built-in curvature alignment — no manual
marking required. The importance signal is implicit:

- **Curvature** — high curvature ⇒ feature ⇒ QuadriFlow keeps denser quads.
- **Boundaries/seams** — preserved by the projection step onto the original
  surface (positions snapped, connectivity kept).

Phase 2 (future) adds an explicit per-region **sizing field** so a user-painted
mask can force extra density in chosen areas (see §9).

## 5. Backend design

### 5.1 New pipeline capability: re-remesh at a target

Add a reusable core function that, given an existing job, produces a new LOD:

```
core/lod.py
  build_lod(manifest, target_faces: int, bake: bool = False) -> LodResult
```

Steps (reusing existing stages/backends):
1. Source = the cleaned/decimated tri mesh already in the job
   (`03_decimate.ply`, else `02_clean.ply`, else ingest). This is the QuadriFlow
   input — do NOT feed a previous quad LOD (avoids compounding error).
2. Run the configured remesh backend with `target_faces = N`
   (`backends.remesh.get_backend(...)`, i.e. QuadriFlow when available).
3. Project the result back onto the reference surface (reuse project stage
   logic) to restore fidelity.
4. If `bake`: run unwrap + color/normal bake on the LOD mesh (reuse those
   stages) so the exported glb carries textures.
5. Compute metrics: actual polygon faces, quad ratio, mean/Hausdorff surface
   distance vs the ingest mesh (reuse qc helpers).
6. Write artifacts under a per-LOD subdir:
   `lod/<N>[/baked]/model.obj`, `model.glb`, `preview.json`.

```python
class LodResult(BaseModel):
    target_faces: int
    actual_faces: int
    quad_ratio: float
    mean_dist_pct_bbox: float
    hausdorff_pct_bbox: float
    baked: bool = False   # whether this LOD's glb carries color/normal textures
    glb: str          # path relative to job dir
    obj: str
```

### 5.2 Caching

- Key: `(target_faces bucket, baked)` — `target_faces` rounded to a bucket
  (nearest 250 to bound cache size) plus whether textures were baked. Store
  built LODs in `lod/<N>[/baked]/` and an index in the manifest
  (`manifest.lods: dict[str, LodResult]`, keyed e.g. `"8000"` / `"8000+baked"`).
- On request, return the cached LOD if that key exists; else build and cache.

### 5.3 New API endpoints (extend `service/app.py`)

```
POST /jobs/{job_id}/lod
  body: { "target_faces": int,   # or "ratio": float (vs original input faces)
          "bake": bool }         # optional, default false
  200:  LodResult (+ url fields for glb/preview)
  409:  if the job hasn't reached at least the remesh-input stage
  Behavior: build-or-return-cached; runs synchronously in the CPU pool.
            bake=false: ~1–2 s, untextured quad glb.
            bake=true:  slower (adds unwrap + color/normal bake), textured glb.

GET  /jobs/{job_id}/lod/{target_faces}/model.glb
  200: the glb for that LOD (FileResponse)

GET  /jobs/{job_id}/lods
  200: { "lods": [LodResult, ...] }   # everything built so far

POST /jobs/{job_id}/export-lod
  body: { "target_faces": int, "formats": ["glb","usd"], "bake": bool }
  200: triggers the existing export (and optional bake) at the chosen LOD;
       returns download info. Reuses unwrap→bake→export stages on the LOD mesh.
```

Notes:
- LOD build is CPU-bound → run in the existing `ThreadPoolExecutor` (serialized).
- `ratio` is resolved against the **original input face count**
  (`manifest.input_stats.faces`), so a slider at 50% targets half the raw scan's
  faces regardless of intermediate stages. The resolved `target_faces` is echoed
  in the response.

### 5.4 Face-count bounds

- Clamp `target_faces` to `[min_faces, max_faces]` where `min_faces` ≈ 200 and
  `max_faces` ≈ the decimate budget (no point remeshing denser than the input to
  QuadriFlow). Return the clamped value in the response so the UI can reflect it.

## 6. Frontend design (Three.js)

Single-page UI served by the API (static assets) or a small dev server.

Components:
- **Viewer**: Three.js `WebGLRenderer` + `OrbitControls` (rotate/zoom/pan),
  loads the LOD glb via `GLTFLoader`. Lighting: hemisphere + directional.
- **Slider**: percentage of the **original input face count** (e.g. 5%–100%),
  with a live numeric readout of the resolved target face count.
- **Bake toggle (button)**: switches the preview between untextured (fast,
  shaded geometry) and textured (color + normal baked). Toggling re-requests
  the LOD with `bake=true/false`; results are cached per variant.
- **Metrics panel**: on each applied LOD show actual faces, quad ratio, mean
  surface error (% bbox) — so the user sees the fidelity cost of reducing.
- **Debounce**: fire the API call on slider *release* (`change`), not on every
  input event, to respect the ~2 s cost. Show a spinner while building.
- **Confirm & Export**: format checkboxes (glb/usd), bake toggle → calls
  `export-lod`, then offers the download.

Interaction loop:
```
drag slider → release → POST /lod {target_faces, bake}
  → receive glb url + metrics → GLTFLoader.load(url) → swap mesh in scene
  → update metrics panel
toggle bake → re-POST /lod {same target_faces, bake=!bake} → reload glb
```

Caching UX: because the backend caches by bucket, dragging back to a previous
value returns instantly.

Tech: plain Three.js (CDN or bundled). No framework required for phase 1; keep
it a single `index.html` + `app.js` + minimal CSS under
`src/slam_to_mesh/service/static/`.

## 7. Data & artifact layout

```
<job_dir>/
  job.json                 # manifest, now also holds `lods` index
  01_ingest.ply ...        # existing stage artifacts
  lod/
    2000/  model.obj  model.glb  preview.json
    4000/  ...
    8000/  ...
```

## 8. Testing

- **Core**: `build_lod` produces a quad-dominant mesh at ≈ target; metrics
  present; caching returns the same artifact (skipif no QuadriFlow binary → use
  CPU backend which is also quad-dominant, so the test still validates shape).
- **API**: `POST /lod` builds and returns metrics; second call is served from
  cache; `GET /lod/{n}/model.glb` returns a file; `export-lod` produces outputs.
  Use FastAPI `TestClient` with a synchronous executor (as existing service
  tests do).
- **Bounds**: out-of-range `target_faces` is clamped and reported.
- **Frontend**: manual smoke test (load, drag, rotate, export). Optional: a
  headless check that `index.html`/`app.js` are served.

## 9. Future phases

- **Protection mask**: user paints regions in the viewer; frontend sends a
  vertex/face selection; backend converts it to a QuadriFlow **sizing field**
  (denser where protected). Requires a backend that accepts a per-vertex target
  edge length (QuadriFlow supports an adaptive scale input; wire it in the
  QuadriFlow backend).
- **Precomputed LOD ladder**: build 5–7 resolutions up front so slider dragging
  swaps instantly; release triggers exact rebuild.
- **Progressive streaming**: serve lower LOD first, refine.

## 10. Rollout / task breakdown

1. `core/lod.py::build_lod` + `LodResult` + manifest `lods` index + caching.
2. Refactor project/qc helpers into reusable functions callable from `build_lod`.
3. API: `POST /lod`, `GET /lod/{n}/model.glb`, `GET /lods`, `POST /export-lod`.
4. Frontend: Three.js viewer + slider + metrics + export, served as static.
5. Tests: core + API + bounds.
6. Docs: README section + update ROADMAP.
