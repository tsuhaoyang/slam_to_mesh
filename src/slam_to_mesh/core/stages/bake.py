"""Stage 7: Texture bake (optional).

Default OFF (``BakeConfig.enabled = False``). When enabled, transfers the
original high-resolution scan's detail onto textures aligned to the low-poly
mesh's UV atlas, so the lightweight mesh looks like the detailed scan in
Omniverse. Two maps are supported:

* **color** — the reference surface's interpolated vertex color per texel.
* **normal** — the reference surface's interpolated normal per texel, expressed
  in the low-poly texel's **tangent space** and encoded as RGB (the standard
  tangent-space normal map consumed by glTF/USD).

CPU algorithm (no GPU), fully vectorized:

1. Rasterize every UV triangle into the texture grid. For all covered texels we
   compute barycentric coordinates in one numpy pass per triangle (no per-pixel
   Python loop), recovering for each texel: its 3D position on the low-poly
   surface, its interpolated low-poly normal, and its tangent/bitangent from the
   UV gradient.
2. Do a single batched closest-point query of all texel positions against the
   reference (high-res) surface.
3. Sample the reference's interpolated color / normal at each hit and write it
   (normals rotated into the per-texel tangent frame) into the texel.

A GPU backend can replace this later; the interface (``run(ctx)``) is stable.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..context import StageContext
from ..meshio import load_mesh
from ..model import Stage, StageResult, StageStatus


def _reference_mesh(ctx: StageContext, need_color: bool) -> trimesh.Trimesh:
    """Load the highest-detail available reference mesh.

    For color baking we prefer a mesh that actually carries vertex colors; for
    normals any high-res surface works.
    """
    for stage in (Stage.INGEST, Stage.CLEAN):
        p = ctx.manifest.artifact_path(stage)
        if p is not None and p.exists():
            m = load_mesh(p)
            if not need_color or _has_vertex_colors(m):
                return m
    return load_mesh(ctx.manifest.input_path)


def _reference_mesh_for(manifest, prefer_color: bool = True) -> trimesh.Trimesh:
    """Manifest-based reference loader (reused by LOD baking).

    Returns the highest-detail available mesh, preferring one with vertex colors
    when ``prefer_color`` is set.
    """
    for stage in (Stage.INGEST, Stage.CLEAN):
        p = manifest.artifact_path(stage)
        if p is not None and p.exists():
            m = load_mesh(p)
            if not prefer_color or _has_vertex_colors(m):
                return m
    return load_mesh(manifest.input_path)


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
    return positions, faces, uv, mesh


# --------------------------------------------------------------------------- #
# Rasterization
# --------------------------------------------------------------------------- #


def _rasterize_texels(positions, faces, uv, normals, size):
    """Rasterize all UV triangles; return per-texel attributes.

    Returns a dict of parallel arrays (one entry per covered texel):
      * ``yx``      (N,2) int   texel row/col
      * ``p3d``     (N,3) float 3D position on the low-poly surface
      * ``normal``  (N,3) float interpolated low-poly normal (unit)
      * ``tangent`` (N,3) float per-texel tangent from the UV gradient (unit)

    Vectorized per triangle: for each triangle we build the pixel bounding box
    as a grid, compute barycentric coords for the whole grid at once, and keep
    the inside-triangle texels. This removes the per-pixel Python loop.
    """
    yx_list = []
    p3d_list = []
    n_list = []
    t_list = []

    for tri in faces:
        uv_tri = uv[tri]  # (3,2)
        p_tri = positions[tri]  # (3,3)
        n_tri = normals[tri]  # (3,3)

        # Pixel-space triangle (row 0 = top, so flip v).
        px = np.empty((3, 2))
        px[:, 0] = uv_tri[:, 0] * (size - 1)
        px[:, 1] = (1.0 - uv_tri[:, 1]) * (size - 1)

        min_x = max(int(np.floor(px[:, 0].min())), 0)
        max_x = min(int(np.ceil(px[:, 0].max())), size - 1)
        min_y = max(int(np.floor(px[:, 1].min())), 0)
        max_y = min(int(np.ceil(px[:, 1].max())), size - 1)
        if max_x < min_x or max_y < min_y:
            continue

        x0, y0 = px[0]
        x1, y1 = px[1]
        x2, y2 = px[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        gx, gy = np.meshgrid(xs, ys)  # (H,W)
        gx = gx.ravel()
        gy = gy.ravel()

        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        c = 1.0 - a - b

        inside = (a >= -1e-4) & (b >= -1e-4) & (c >= -1e-4)
        if not np.any(inside):
            continue

        a = a[inside]
        b = b[inside]
        c = c[inside]
        gx = gx[inside]
        gy = gy[inside]
        bary = np.stack([a, b, c], axis=1)  # (M,3)

        p3d = bary @ p_tri  # (M,3)
        nrm = bary @ n_tri  # (M,3)

        # Per-triangle tangent from the UV gradient (constant over the triangle).
        # dP/du direction: solve [duv1; duv2] [T; B] = [dp1; dp2].
        duv1 = uv_tri[1] - uv_tri[0]
        duv2 = uv_tri[2] - uv_tri[0]
        dp1 = p_tri[1] - p_tri[0]
        dp2 = p_tri[2] - p_tri[0]
        r_det = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        if abs(r_det) < 1e-12:
            tangent = dp1  # degenerate UV; fall back to an edge direction
        else:
            r = 1.0 / r_det
            tangent = (dp1 * duv2[1] - dp2 * duv1[1]) * r
        tangent = np.broadcast_to(tangent, p3d.shape)

        yx_list.append(np.stack([gy, gx], axis=1))
        p3d_list.append(p3d)
        n_list.append(nrm)
        t_list.append(tangent)

    if not yx_list:
        empty_i = np.zeros((0, 2), dtype=np.int64)
        empty_f = np.zeros((0, 3), dtype=np.float64)
        return {"yx": empty_i, "p3d": empty_f, "normal": empty_f, "tangent": empty_f}

    return {
        "yx": np.concatenate(yx_list).astype(np.int64),
        "p3d": np.concatenate(p3d_list),
        "normal": np.concatenate(n_list),
        "tangent": np.concatenate(t_list),
    }


def _normalize(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def _sample_reference(ref, p3d):
    """Batched closest-point on the reference; return (bary, tri_vertex_ids).

    bary: (N,3) barycentric weights of the hit within its triangle.
    vids: (N,3) reference vertex indices of that triangle.

    Uses Open3D's ``RaycastingScene`` (vectorized, GPU-friendly) when available,
    which is far faster than trimesh's per-point proximity for large texel
    counts. Falls back to trimesh if Open3D is missing or errors.
    """
    ref_faces = np.asarray(ref.faces)
    closest, tri_ids = _closest_point_open3d(ref, p3d)
    if closest is None:
        closest, _, tri_ids = trimesh.proximity.closest_point(ref, p3d)
    vids = ref_faces[tri_ids]  # (N,3)
    tri_pts = ref.vertices[vids]  # (N,3,3)
    bary = trimesh.triangles.points_to_barycentric(tri_pts, closest)  # (N,3)
    return bary, vids


def _closest_point_open3d(ref, p3d):
    """Return (closest_points, triangle_ids) via Open3D, or (None, None)."""
    try:
        import open3d as o3d
    except Exception:  # noqa: BLE001
        return None, None
    try:
        scene = o3d.t.geometry.RaycastingScene()
        verts = np.asarray(ref.vertices, dtype=np.float32)
        tris = np.asarray(ref.faces, dtype=np.uint32)
        scene.add_triangles(
            o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(tris, dtype=o3d.core.Dtype.UInt32),
        )
        query = o3d.core.Tensor(
            np.asarray(p3d, dtype=np.float32), dtype=o3d.core.Dtype.Float32
        )
        res = scene.compute_closest_points(query)
        closest = res["points"].numpy().astype(np.float64)
        tri_ids = res["primitive_ids"].numpy().astype(np.int64)
        return closest, tri_ids
    except Exception:  # noqa: BLE001
        return None, None


def _bake_color(tex, ref, size, sample=None):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    if len(tex["p3d"]) == 0:
        return image, 0
    ref_colors = np.asarray(ref.visual.vertex_colors)[:, :3].astype(np.float64)
    bary, vids = sample if sample is not None else _sample_reference(ref, tex["p3d"])
    cols = np.einsum("nk,nkc->nc", bary, ref_colors[vids])  # (N,3)
    yy = tex["yx"][:, 0]
    xx = tex["yx"][:, 1]
    image[yy, xx] = np.clip(cols, 0, 255).astype(np.uint8)
    return image, len(yy)


def _bake_normal(tex, ref, size, sample=None):
    """Bake a tangent-space normal map.

    For each texel we sample the reference world-space normal, then express it
    in the texel's tangent frame (T, B, N derived from the low-poly surface and
    UV gradient) and encode to RGB with the usual 0.5-centered convention.
    """
    # Flat, unwritten normal-map value is (0.5, 0.5, 1.0) -> +Z in tangent space.
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = (128, 128, 255)
    if len(tex["p3d"]) == 0:
        return image, 0

    if ref.vertex_normals is None or len(ref.vertex_normals) != len(ref.vertices):
        ref_vn = ref.vertex_normals  # trimesh computes lazily
    else:
        ref_vn = ref.vertex_normals
    ref_vn = np.asarray(ref_vn, dtype=np.float64)

    bary, vids = sample if sample is not None else _sample_reference(ref, tex["p3d"])
    world_n = _normalize(np.einsum("nk,nkc->nc", bary, ref_vn[vids]))  # (N,3)

    # Build the per-texel tangent frame from the low-poly surface.
    N = _normalize(tex["normal"])
    T = tex["tangent"]
    # Gram-Schmidt: remove the normal component from the tangent.
    T = _normalize(T - N * np.sum(T * N, axis=1, keepdims=True))
    B = np.cross(N, T)

    # Express the world-space reference normal in (T, B, N).
    nx = np.sum(world_n * T, axis=1)
    ny = np.sum(world_n * B, axis=1)
    nz = np.sum(world_n * N, axis=1)
    ts = np.stack([nx, ny, nz], axis=1)
    ts = _normalize(ts)

    rgb = np.clip((ts * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    yy = tex["yx"][:, 0]
    xx = tex["yx"][:, 1]
    image[yy, xx] = rgb
    return image, len(yy)


def _save_png(image, path):
    from PIL import Image

    Image.fromarray(image, mode="RGB").save(str(path))


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.BAKE)
    result.mark_running()
    cfg = ctx.manifest.config.bake
    try:
        src = ctx.input_for(Stage.BAKE)  # the UV-unwrapped mesh

        if not cfg.enabled:
            result.params = cfg.model_dump()
            result.message = "bake disabled (skipped)"
            result.status = StageStatus.SKIPPED
            result.finished_at = result.started_at
            return result

        positions, faces, uv, uv_mesh = _load_uv_mesh(src)
        if uv is None:
            raise ValueError("input mesh has no UVs; run unwrap stage first")

        size = int(cfg.texture_size)
        low_normals = np.asarray(uv_mesh.vertex_normals, dtype=np.float64)

        want_color = bool(cfg.bake_color)
        want_normal = bool(cfg.bake_normal)

        # Load the reference once and reuse it for both bakes where possible. The
        # color-capable reference (ingest/clean mesh) also carries normals, so a
        # single object serves both — letting us share the expensive closest-
        # point query below.
        ref = None
        if want_color or want_normal:
            ref = _reference_mesh(ctx, need_color=want_color)
        ref_has_color = ref is not None and _has_vertex_colors(ref)
        # If color was requested but this reference has no colors, still use it
        # for normals.
        ref_color = ref if ref_has_color else None
        ref_normal = ref if want_normal else None

        # One rasterization pass shared by both bakes.
        tex = _rasterize_texels(positions, faces, uv, low_normals, size)

        color_written = 0
        normal_written = 0
        color_baked = False
        normal_baked = False

        # The closest-point query is the expensive step. When color and normal
        # bake against the same reference mesh (the common case: the ingest mesh
        # carries both color and normals), run it once and share the result.
        same_ref = (
            want_color
            and want_normal
            and ref_has_color
            and ref_normal is not None
            and ref_color is ref_normal
        )
        shared = None
        if same_ref and len(tex["p3d"]):
            shared = _sample_reference(ref_color, tex["p3d"])

        if want_color and ref_has_color:
            img, color_written = _bake_color(tex, ref_color, size, sample=shared)
            _save_png(img, ctx.job_dir / "07_bake_color.png")
            result.extra_artifacts.append(ctx.rel(ctx.job_dir / "07_bake_color.png"))
            color_baked = True

        if want_normal and ref_normal is not None:
            img, normal_written = _bake_normal(tex, ref_normal, size, sample=shared)
            _save_png(img, ctx.job_dir / "07_bake_normal.png")
            result.extra_artifacts.append(ctx.rel(ctx.job_dir / "07_bake_normal.png"))
            normal_baked = True

        result.artifact = ctx.rel(src) if src.parent == ctx.job_dir else None
        result.params = cfg.model_dump()
        result.metrics = {
            "texture_size": size,
            "texels_rasterized": len(tex["p3d"]),
            "color_texels_written": int(color_written),
            "normal_texels_written": int(normal_written),
            "color_baked": color_baked,
            "normal_baked": normal_baked,
            "reference_has_color": ref_has_color,
        }
        parts = []
        if color_baked:
            parts.append("color")
        if normal_baked:
            parts.append("normal")
        result.message = (
            f"baked {', '.join(parts)} to {size}x{size}"
            if parts
            else "nothing baked (no color reference / maps disabled)"
        )
        result.mark_done()
    except Exception as e:
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
