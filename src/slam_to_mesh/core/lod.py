"""Interactive level-of-detail (LOD) building.

Produces a lower- (or higher-) resolution version of an already-processed mesh
by **re-running the quad remesh backend at a different target face count**, then
projecting back onto the original surface. Re-remeshing keeps the output regular
and quad-dominant (QEM decimation would not), which the export path requires.

This is the backend behind the interactive decimation UI: a slider chooses a
target face count, :func:`build_lod` produces (and caches) the corresponding
quad LOD plus fidelity metrics, and the frontend previews it.

See ``docs/spec_interactive_decimation.md``.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from ..backends.remesh import RemeshRequest, get_backend
from .meshio import bbox_diagonal, load_mesh
from .model import JobManifest, LodResult, Stage

#: Face-count clamp bounds for interactive LODs.
MIN_LOD_FACES = 200
#: Bucket size for caching (target faces rounded to this granularity).
FACE_BUCKET = 250


def _remesh_source(manifest: JobManifest) -> Path:
    """The triangle mesh fed to the remesher for LOD building.

    Prefer the decimated mesh, then cleaned, then ingest, then the raw input.
    We deliberately re-remesh from a triangle source (never a previous quad LOD)
    so error does not compound across LOD changes.
    """
    for stage in (Stage.DECIMATE, Stage.CLEAN, Stage.INGEST):
        p = manifest.artifact_path(stage)
        if p is not None and p.exists():
            return p
    return Path(manifest.input_path)


def _reference_surface(manifest: JobManifest) -> trimesh.Trimesh:
    """High-res surface to project onto: clean, else ingest, else input."""
    for stage in (Stage.CLEAN, Stage.INGEST):
        p = manifest.artifact_path(stage)
        if p is not None and p.exists():
            return load_mesh(p)
    return load_mesh(manifest.input_path)


def resolve_target_faces(
    manifest: JobManifest,
    target_faces: int | None = None,
    ratio: float | None = None,
) -> int:
    """Resolve and clamp a target face count.

    ``ratio`` is interpreted against the **original input face count**. Exactly
    one of ``target_faces`` / ``ratio`` should be given; ``target_faces`` wins if
    both are present.
    """
    if target_faces is None:
        if ratio is None:
            raise ValueError("provide target_faces or ratio")
        base = manifest.input_stats.faces if manifest.input_stats else 0
        target_faces = round(base * float(ratio))

    upper = _upper_bound(manifest)
    return max(MIN_LOD_FACES, min(int(target_faces), upper))


def _upper_bound(manifest: JobManifest) -> int:
    """Densest sensible LOD: the remesh source's triangle count."""
    src = _remesh_source(manifest)
    try:
        n = len(load_mesh(src).faces)
    except Exception:  # noqa: BLE001
        n = 100000
    return max(MIN_LOD_FACES, n)


def _cache_key(target_faces: int, baked: bool) -> str:
    bucket = int(round(target_faces / FACE_BUCKET) * FACE_BUCKET)
    bucket = max(MIN_LOD_FACES, bucket)
    return f"{bucket}+baked" if baked else str(bucket)


def build_lod(
    manifest: JobManifest,
    target_faces: int | None = None,
    ratio: float | None = None,
    bake: bool = False,
    force: bool = False,
) -> LodResult:
    """Build (or return a cached) quad LOD at *target_faces*.

    Steps: re-remesh the triangle source at the target → project onto the
    reference surface → (optionally) unwrap + bake color/normal → export glb +
    obj → compute fidelity metrics. Results are cached in ``manifest.lods`` and
    on disk under ``lod/<bucket>[/baked]/``.
    """
    resolved = resolve_target_faces(manifest, target_faces, ratio)
    key = _cache_key(resolved, bake)

    if not force and key in manifest.lods:
        cached = manifest.lods[key]
        glb = Path(manifest.job_dir) / cached.glb if cached.glb else None
        if glb is not None and glb.exists():
            return cached

    job_dir = Path(manifest.job_dir)
    sub = job_dir / "lod" / (f"{_cache_key(resolved, False)}" + ("/baked" if bake else ""))
    sub.mkdir(parents=True, exist_ok=True)

    src = _remesh_source(manifest)
    ref = _reference_surface(manifest)
    diag = bbox_diagonal(ref)

    # 1. Re-remesh at the target resolution (quad-dominant, field-aligned).
    remesh_obj = sub / "remesh.obj"
    backend = get_backend(manifest.config.remesh.backend)
    backend.remesh(
        RemeshRequest(
            input_path=src,
            output_path=remesh_obj,
            target_faces=int(resolved),
            quads=True,
            preserve_sharp=bool(manifest.config.remesh.preserve_sharp),
            bbox_diagonal=diag,
        )
    )

    # 2. Project back onto the reference surface (keeps quad connectivity).
    from .stages.project import project_obj_onto_surface

    project_obj = sub / "project.obj"
    project_obj_onto_surface(
        remesh_obj, project_obj, ref,
        max_snap_ratio=manifest.config.project.max_snap_ratio,
    )

    mesh_for_export = project_obj
    color_tex = normal_tex = None

    # 3. Optional unwrap + bake so the preview/exported glb carries textures.
    if bake:
        mesh_for_export, color_tex, normal_tex = _unwrap_and_bake(
            manifest, project_obj, sub
        )

    # 4. Export obj + glb (with textures if baked).
    obj_out = sub / "model.obj"
    glb_out = sub / "model.glb"
    _export_lod_mesh(mesh_for_export, obj_out, glb_out, color_tex, normal_tex)

    # 5. Fidelity metrics vs the ingest/reference surface.
    from .stages.qc import surface_distance_metrics

    quad_mesh = load_mesh(project_obj)
    _, polygon_faces, quad_ratio = _quad_stats(project_obj)
    dist = surface_distance_metrics(quad_mesh, ref)

    result = LodResult(
        target_faces=int(resolved),
        actual_faces=int(polygon_faces),
        quad_ratio=round(quad_ratio, 4),
        mean_dist_pct_bbox=dist["mean_dist_pct_bbox"],
        hausdorff_pct_bbox=dist["hausdorff_pct_bbox"],
        baked=bool(bake),
        glb=str(glb_out.relative_to(job_dir)),
        obj=str(obj_out.relative_to(job_dir)),
    )
    manifest.lods[key] = result
    manifest.save()
    return result


def _quad_stats(obj_path: Path) -> tuple[int, int, float]:
    """(quads, total_polygon_faces, quad_ratio) from an OBJ."""
    quads = total = 0
    try:
        with open(obj_path) as fh:
            for line in fh:
                if line.startswith("f "):
                    total += 1
                    if len(line.split()) - 1 == 4:
                        quads += 1
    except OSError:
        return 0, 0, 0.0
    return quads, total, (quads / total if total else 0.0)


def _unwrap_and_bake(manifest: JobManifest, project_obj: Path, sub: Path):
    """Unwrap the LOD mesh and bake color/normal; return (uv_obj, color, normal)."""
    import numpy as np
    import xatlas

    from .stages.bake import (
        _bake_color,
        _bake_normal,
        _rasterize_texels,
        _reference_mesh_for,
        _save_png,
    )

    # --- unwrap (triangulated UV obj) ---
    mesh = load_mesh(project_obj)
    positions = np.asarray(mesh.vertices, dtype=np.float32)
    indices = np.asarray(mesh.faces, dtype=np.uint32)
    vmapping, out_indices, uvs = xatlas.parametrize(positions, indices)
    out_positions = positions[vmapping]
    out_faces = np.asarray(out_indices, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(uvs, dtype=np.float64)

    uv_obj = sub / "unwrap.obj"
    _write_uv_obj(uv_obj, out_positions, out_faces, uvs)

    # --- bake color + normal ---
    uv_mesh = trimesh.load(str(uv_obj), process=False)
    low_normals = np.asarray(uv_mesh.vertex_normals, dtype=np.float64)
    size = int(manifest.config.bake.texture_size)
    tex = _rasterize_texels(
        np.asarray(uv_mesh.vertices, dtype=np.float64),
        np.asarray(uv_mesh.faces, dtype=np.int64),
        np.asarray(uv_mesh.visual.uv, dtype=np.float64),
        low_normals,
        size,
    )
    ref = _reference_mesh_for(manifest)
    color_path = normal_path = None
    from .stages.bake import _has_vertex_colors

    if _has_vertex_colors(ref):
        img, _ = _bake_color(tex, ref, size)
        color_path = sub / "bake_color.png"
        _save_png(img, color_path)
    img, _ = _bake_normal(tex, ref, size)
    normal_path = sub / "bake_normal.png"
    _save_png(img, normal_path)
    return uv_obj, color_path, normal_path


def _write_uv_obj(path, positions, faces, uvs) -> None:
    with open(path, "w") as fh:
        fh.write("# slam_to_mesh LOD UV mesh\n")
        fh.writelines(f"v {p[0]:.8g} {p[1]:.8g} {p[2]:.8g}\n" for p in positions)
        fh.writelines(f"vt {uv[0]:.8g} {uv[1]:.8g}\n" for uv in uvs)
        for f in faces:
            a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
            fh.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def _export_lod_mesh(mesh_path, obj_out, glb_out, color_tex, normal_tex) -> None:
    """Export the LOD mesh to obj + glb, attaching textures via PBR if present."""
    from .stages.export import _load_uv_mesh_with_texture

    mesh = _load_uv_mesh_with_texture(mesh_path, color_tex, normal_tex)
    mesh.export(str(obj_out))
    mesh.export(str(glb_out))
