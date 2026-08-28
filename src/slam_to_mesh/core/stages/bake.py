"""Stage 7: Texture bake (optional).

Default OFF (``BakeConfig.enabled = False``). When enabled, bakes the original
high-resolution surface's vertex colors into a texture aligned to the low-poly
mesh's UV atlas, so the lightweight mesh looks like the detailed scan in
Omniverse.

CPU algorithm (no GPU):

1. Rasterize each UV triangle into the texture image. For every covered texel,
   compute its barycentric coordinates to recover the corresponding 3D point on
   the low-poly surface.
2. Find the closest point on the reference (high-res) surface and sample its
   interpolated vertex color there.
3. Write the color into the texel.

This is a straightforward forward-rasterization baker; a GPU backend can replace
it later. Normal-map baking is stubbed (records intent) pending a tangent-space
implementation.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..context import StageContext
from ..meshio import load_mesh
from ..model import Stage, StageResult


def _reference_with_color(ctx: StageContext) -> trimesh.Trimesh:
    """Load the highest-detail available mesh that may carry vertex colors."""
    for stage in (Stage.INGEST, Stage.CLEAN):
        p = ctx.manifest.artifact_path(stage)
        if p is not None and p.exists():
            m = load_mesh(p)
            if _has_vertex_colors(m):
                return m
    return load_mesh(ctx.manifest.input_path)


def _has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    try:
        vc = mesh.visual.vertex_colors
        return vc is not None and len(vc) == len(mesh.vertices)
    except Exception:  # noqa: BLE001
        return False


def _load_uv_mesh(path):
    """Load a UV OBJ, returning (positions, faces, uvs) with UVs per vertex."""
    mesh = trimesh.load(str(path), process=False)
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    return positions, faces, uv


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.BAKE)
    result.mark_running()
    cfg = ctx.manifest.config.bake
    try:
        src = ctx.input_for(Stage.BAKE)  # the UV-unwrapped mesh

        if not cfg.enabled:
            from ..model import StageStatus

            result.params = cfg.model_dump()
            result.message = "bake disabled (skipped)"
            result.status = StageStatus.SKIPPED
            result.finished_at = result.started_at
            return result

        positions, faces, uv = _load_uv_mesh(src)
        if uv is None:
            raise ValueError("input mesh has no UVs; run unwrap stage first")

        ref = _reference_with_color(ctx)
        size = int(cfg.texture_size)

        texels_written = 0
        color_texture = None
        if cfg.bake_color and _has_vertex_colors(ref):
            color_texture, texels_written = _bake_color(
                positions, faces, uv, ref, size
            )
            out_png = ctx.job_dir / "07_bake_color.png"
            _save_png(color_texture, out_png)
            result.extra_artifacts.append(ctx.rel(out_png))

        # The bake stage does not change geometry; its artifact is the UV mesh
        # so downstream export picks up the same mesh plus textures.
        result.artifact = ctx.rel(src) if src.parent == ctx.job_dir else None
        result.params = cfg.model_dump()
        result.metrics = {
            "texture_size": size,
            "texels_written": int(texels_written),
            "color_baked": bool(color_texture is not None),
            "normal_baked": False,
            "reference_has_color": _has_vertex_colors(ref),
        }
        result.message = (
            f"baked color to {size}x{size} ({texels_written} texels)"
            if color_texture is not None
            else "no vertex colors on reference; nothing to bake"
        )
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result


def _bake_color(positions, faces, uv, ref, size):
    """Rasterize UV triangles and sample reference vertex colors per texel."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    written = np.zeros((size, size), dtype=bool)

    ref_colors = np.asarray(ref.visual.vertex_colors)[:, :3].astype(np.float64)

    # Accumulate all texel 3D positions first, then do one batched proximity
    # query for speed.
    texel_xy: list[tuple[int, int]] = []
    texel_p3d: list[np.ndarray] = []

    for tri in faces:
        uv_tri = uv[tri]  # (3,2) in [0,1]
        p_tri = positions[tri]  # (3,3)
        # Pixel-space triangle (v flipped: image row 0 = top).
        px = uv_tri.copy()
        px[:, 0] = uv_tri[:, 0] * (size - 1)
        px[:, 1] = (1.0 - uv_tri[:, 1]) * (size - 1)

        min_x = max(int(np.floor(px[:, 0].min())), 0)
        max_x = min(int(np.ceil(px[:, 0].max())), size - 1)
        min_y = max(int(np.floor(px[:, 1].min())), 0)
        max_y = min(int(np.ceil(px[:, 1].max())), size - 1)
        if max_x < min_x or max_y < min_y:
            continue

        # Barycentric setup.
        x0, y0 = px[0]
        x1, y1 = px[1]
        x2, y2 = px[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        for yy in range(min_y, max_y + 1):
            for xx in range(min_x, max_x + 1):
                a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
                b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
                c = 1.0 - a - b
                if a < -1e-4 or b < -1e-4 or c < -1e-4:
                    continue
                p3d = a * p_tri[0] + b * p_tri[1] + c * p_tri[2]
                texel_xy.append((yy, xx))
                texel_p3d.append(p3d)

    if not texel_p3d:
        return image, 0

    pts = np.asarray(texel_p3d)
    closest, _, tri_ids = trimesh.proximity.closest_point(ref, pts)

    # Interpolate reference color at each closest point via its triangle.
    ref_faces = ref.faces
    for (yy, xx), fid, cp in zip(texel_xy, tri_ids, closest):
        vids = ref_faces[fid]
        tri_pts = ref.vertices[vids]
        bary = trimesh.triangles.points_to_barycentric(
            tri_pts[None, :, :], cp[None, :]
        )[0]
        col = (bary[:, None] * ref_colors[vids]).sum(axis=0)
        image[yy, xx] = np.clip(col, 0, 255).astype(np.uint8)
        written[yy, xx] = True

    return image, int(written.sum())


def _save_png(image, path):
    from PIL import Image

    Image.fromarray(image, mode="RGB").save(str(path))
