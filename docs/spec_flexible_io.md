# Spec: Flexible inputs + multi-view display + point-cloud support

Status: **draft** · Depends on: existing pipeline core, QuadriFlow backend, LOD
API + Three.js UI (`docs/spec_interactive_decimation.md`).

## 1. Summary

Make the service accept multiple input types and let the user view several
representations side by side, each in its own synced viewer, chosen via
checkboxes. Add point-cloud input (reconstructed to a mesh) and an independent
point-cloud **downsampling** mode. Two independent decimation controls:
triangle decimation (QEM) and quad decimation (QuadriFlow re-remesh).

Frontend stays a **web page** (Three.js). Heavy work stays on the FastAPI
backend. (Browser-extension packaging was considered and rejected: extensions
can't run the CPU/GPU reconstruction/remeshing and would still just call the
backend.)

## 2. Inputs (pluggable, detected at ingest)

1. **Point cloud** — `.ply` (points), `.pcd`, `.xyz`.
2. **Triangle mesh** — `.ply`, `.obj`, `.stl`, `.glb`, `.off`.
3. **2D image** — **PENDING** (TripoSR track paused; see
   `TripoSR/INSTALL_STEPS.md`). Not implemented in this spec.

Ingest normalizes every input into a common **triangle mesh** (plus, optionally,
a retained/generated point cloud):

- Point cloud → estimate normals if missing → **Open3D Poisson surface
  reconstruction** → triangle mesh. Original points retained for the point-cloud
  viewer and for downsampling.
- Triangle mesh → used directly. A point cloud can be **generated on request**
  by surface-sampling (user opt-in, Q3).
- Image → (pending).

## 3. Representations the user can display

Checkbox items (each opens its own viewer pane when checked):

| Item | Source | Decimation control |
|---|---|---|
| Triangle | normalized/ingest triangle mesh | — (static) |
| Triangle-decimated | QEM decimation of the triangle mesh (pymeshlab) | **faces slider (tri)** |
| Quad | QuadriFlow at default resolution | — (static) |
| Quad-decimated | QuadriFlow re-remesh at target | **faces slider (quad)** |
| Point cloud | original (or generated) points | — (static) |
| Point-cloud-downsampled | Open3D voxel downsample | **points slider** |

### Decimation methods (confirmed)
- **Triangle-decimated** → QEM edge-collapse (pymeshlab
  `meshing_decimation_quadric_edge_collapse`). Fast, irregular tris, any exact
  target face count. Reuses/extends the existing decimate stage logic.
- **Quad-decimated** → re-run QuadriFlow at target `-f` (existing `core/lod.py`).
  Regular, export-ready, ~1–2 s.
- **Point-cloud-downsampled** → Open3D `voxel_down_sample`. Independent slider.

The three sliders are **independent**; one never drives another (point cloud and
mesh have no vertex correspondence — a downsampled cloud does not change any
mesh, and mesh decimation does not change the cloud).

## 4. Checkbox gating rules (Q1)

- **Point-cloud-downsample** is selectable **only when "Point cloud" is the sole
  checked item** (no mesh representation checked).
- When any mesh item (triangle / triangle-decimated / quad / quad-decimated) is
  checked, "Point-cloud-downsample" is **disabled** (greyed out).
- When "Point-cloud-downsample" is checked, mesh items are **disabled**.
- Rationale: point-cloud downsampling is a dedicated original-vs-reduced compare
  mode (left = original points, right = downsampled). It owns the layout.

Allowed multi-mesh combos (any subset, free arrangement), e.g.:
- Triangle + Triangle-decimated + Quad
- Quad + Quad-decimated + Triangle-decimated
- Triangle + Quad

## 5. Viewer layout

- N checked items → N panes laid out side by side (responsive; 1–3 typical,
  cap sensible max e.g. 4).
- All mesh panes share a **synced camera** (drag one → all rotate).
- Point-cloud panes render via Three.js `Points`; mesh panes via glb (shaded) or
  quad wireframe (as today).
- Each pane shows a small header with its label + face/point count.

## 6. Sliders / controls

- **Tri faces**: shown when Triangle-decimated is checked → drives QEM target.
- **Quad faces**: shown when Quad-decimated is checked → drives QuadriFlow target.
- **Points**: shown only in point-cloud-downsample mode → drives voxel size /
  target point count.
- Debounced on release (~build cost). Cache results per (type, target).

## 7. Point-cloud options (Q3)

- **Generate point cloud** (checkbox/setting): for mesh inputs, sample the
  surface to produce a point cloud for viewing/downsampling. Off by default.
- **Export point cloud** (checkbox at export): include the (possibly
  downsampled) point cloud in the export bundle.

## 8. Backend design

### 8.1 Ingest changes
- Detect input type by extension + content (points vs faces).
- Point cloud path: `core/stages/ingest.py` (or a new `reconstruct` helper)
  runs Open3D normal estimation + Poisson; stores `00_points.ply` (original) and
  the reconstructed triangle mesh as the ingest artifact.
- Record `input_kind` ("pointcloud" | "mesh") in the manifest.

### 8.2 New/extended core functions
- `core/pointcloud.py`:
  - `reconstruct_poisson(points_path, out_mesh, depth=9) -> stats`
  - `voxel_downsample(points_path, out_points, voxel_size) -> {points_before, points_after}`
  - `sample_points_from_mesh(mesh_path, out_points, n) -> count`
- `core/lod.py`: already does quad LOD. Add `build_tri_lod(manifest,
  target_faces)` for QEM triangle decimation (reuse decimate stage), cached like
  quad LODs.

### 8.3 API endpoints
```
POST /jobs/{id}/tri-lod          { target_faces }        -> TriLodResult (+ glb/wire)
GET  /jobs/{id}/tri-lod/{n}/model.glb
POST /jobs/{id}/pointcloud/downsample { voxel_size | target_points } -> {count, url}
GET  /jobs/{id}/pointcloud.(ply|json)                    # original or generated
GET  /jobs/{id}/pointcloud/downsampled.(ply|json)
```
Plus existing quad LOD endpoints. Point data for the browser is returned as a
compact JSON `{positions:[...]}` (or a binary .ply the frontend parses) for
Three.js `Points`.

### 8.4 Manifest
- `input_kind`, `has_pointcloud`, LOD caches for quad + tri, downsample cache.

## 9. Frontend design

- Checkbox panel with the 6 items + gating logic (§4).
- Dynamic pane container: render one `<canvas>` viewer per checked item.
- A `Viewer` abstraction (already exists) reused; add a `Points` loader and a
  quad-wireframe loader (exists).
- Sliders appear contextually (§6). Debounced fetch → update the matching pane.
- Export panel: formats (glb/usd/obj) for meshes + optional point-cloud export.

## 10. Testing

- Core: Poisson reconstruction on a sphere point cloud → watertight-ish mesh;
  voxel_downsample reduces point count; QEM tri-lod reduces faces to ~target;
  sample_points_from_mesh returns N points.
- API: pointcloud upload → ingest reconstructs; downsample endpoint reduces
  count; tri-lod builds + caches; gating not enforced server-side (client rule)
  but endpoints independently valid.
- Frontend: manual + a served-assets smoke test.

## 11. Rollout / task order

1. `core/pointcloud.py` (Poisson, voxel downsample, sample-from-mesh) + tests.
2. Ingest: detect point-cloud input, reconstruct, retain original points;
   `input_kind` in manifest.
3. `build_tri_lod` (QEM) in core + cache + API (`/tri-lod`).
4. Point-cloud API (`/pointcloud*`, downsample).
5. Frontend: multi-pane dynamic layout + checkbox gating + contextual sliders +
   `Points` rendering.
6. Export: point-cloud export option.
7. Tests + docs (README + ROADMAP).

## 12. Non-goals / pending
- 2D image → 3D (TripoSR): **pending**, tracked separately.
- No cross-linking between point-cloud downsampling and mesh decimation (they are
  independent by geometric necessity).
