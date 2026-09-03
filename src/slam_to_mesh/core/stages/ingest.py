"""Stage 1: Ingest & Analyze.

Load the input, normalize it into a triangle mesh, compute statistics, detect
common SLAM defects, and persist a normalized copy as the ingest artifact so
later stages have a consistent starting point.

Inputs may be a **triangle mesh** or a **point cloud**. Point clouds are
reconstructed into a triangle mesh (Poisson) so the surface-based pipeline can
run; the original points are retained as ``00_points.ply`` for the point-cloud
viewer and downsampling.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ..context import StageContext
from ..frames import extract_frames, is_video_file
from ..image_to_mesh import generate_mesh, is_available, is_image_file
from ..meshio import compute_stats, load_mesh, save_mesh
from ..model import Stage, StageResult
from ..pointcloud import is_point_cloud_file, reconstruct_poisson


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.INGEST)
    result.mark_running()
    try:
        src = ctx.input_for(Stage.INGEST)

        recon_stats = None
        image_src = None
        if is_image_file(src):
            # Image input: generate a 3D mesh with the configured image3d
            # backend (isolated venv, subprocess), then treat it as the source.
            backend_id = ctx.manifest.config.image_backend
            if not is_available(backend_id):
                raise RuntimeError(
                    "image input requires an image-to-3D backend "
                    f"(requested={backend_id!r}, none available); see "
                    "docs/triposr_2d_to_3d.md"
                )
            ctx.manifest.input_kind = "image"
            image_src = src
            gen_mesh = ctx.job_dir / "00_generated.obj"
            gen_mesh.parent.mkdir(parents=True, exist_ok=True)
            gen_stats = generate_mesh(src, gen_mesh, backend_id=backend_id)
            mesh = load_mesh(gen_mesh)
        elif is_video_file(src) or Path(src).suffix.lower() == ".zip":
            # Multi-view input: images (.zip) or video (frames) → photogrammetry
            # → dense point cloud, then reuse the point-cloud path (Poisson).
            kind = "video" if is_video_file(src) else "images"
            ctx.manifest.input_kind = kind
            points_out = ctx.job_dir / "00_points.ply"
            recon_stats = _photogrammetry_points(ctx, src, kind, points_out)
            ctx.manifest.has_pointcloud = True

            recon_mesh = ctx.job_dir / "00_reconstructed.ply"
            poisson_stats = reconstruct_poisson(points_out, recon_mesh)
            recon_stats.update(poisson_stats)
            mesh = load_mesh(recon_mesh)
        elif is_point_cloud_file(src):
            # Point-cloud input: retain the original points, then reconstruct a
            # triangle mesh to feed the rest of the pipeline.
            ctx.manifest.input_kind = "pointcloud"
            ctx.manifest.has_pointcloud = True
            points_out = ctx.job_dir / "00_points.ply"
            points_out.parent.mkdir(parents=True, exist_ok=True)
            _copy_points(src, points_out)

            recon_mesh = ctx.job_dir / "00_reconstructed.ply"
            recon_stats = reconstruct_poisson(src, recon_mesh)
            mesh = load_mesh(recon_mesh)
        else:
            ctx.manifest.input_kind = "mesh"
            mesh = load_mesh(src)

        stats = compute_stats(mesh)
        ctx.manifest.input_stats = stats

        # Persist a normalized copy (PLY preserves color/normals well).
        out = ctx.out_path(Stage.INGEST, "ply")
        save_mesh(mesh, out)

        result.artifact = ctx.rel(out)
        result.params = {"source": str(src), "input_kind": ctx.manifest.input_kind}
        result.metrics = {
            "vertices": stats.vertices,
            "faces": stats.faces,
            "components": stats.components,
            "boundary_edges": stats.boundary_edges,
            "non_manifold_edges": stats.non_manifold_edges,
            "is_watertight": stats.is_watertight,
        }
        if recon_stats is not None:
            result.metrics["reconstructed_from_points"] = recon_stats["input_points"]
        if image_src is not None:
            result.metrics["generated_from_image"] = str(image_src)
            result.metrics["image_backend"] = gen_stats.get("backend")

        # Human-readable problem summary.
        problems = []
        if ctx.manifest.input_kind == "image":
            problems.append("generated from image (TripoSR)")
        if ctx.manifest.input_kind in ("images", "video"):
            n_imgs = recon_stats.get("images") if recon_stats else None
            problems.append(
                f"photogrammetry from {n_imgs} views" if n_imgs
                else "photogrammetry reconstruction"
            )
        if ctx.manifest.input_kind == "pointcloud":
            problems.append(
                f"reconstructed from {recon_stats['input_points']} points"
            )
        if stats.non_manifold_edges > 0:
            problems.append(f"{stats.non_manifold_edges} non-manifold edges")
        if not stats.is_watertight:
            problems.append("not watertight (holes present)")
        if stats.components > 1:
            problems.append(f"{stats.components} disconnected components")
        result.message = "; ".join(problems) if problems else "no obvious defects"

        result.mark_done()
    except Exception as e:
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result


def _copy_points(src: Path, dst: Path) -> None:
    """Retain the original point cloud as a PLY next to the job artifacts."""
    src = Path(src)
    if src.suffix.lower() == ".ply":
        shutil.copyfile(src, dst)
    else:
        # Re-encode other point formats (.pcd/.xyz) as PLY for the viewer.
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(src))
        o3d.io.write_point_cloud(str(dst), pcd)


def _photogrammetry_points(ctx, src: Path, kind: str, points_out: Path) -> dict:
    """Prepare an images dir (from zip/video) and run photogrammetry → points.

    Returns the backend's stats dict. Raises RuntimeError with a clear message
    when no photogrammetry backend is available.
    """
    from ...backends import photogrammetry as pg

    src = Path(src)
    images_dir = ctx.job_dir / "00_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if kind == "video":
        n = int(ctx.manifest.config.video_frames)
        count = extract_frames(src, images_dir, n=n)
        if count == 0:
            raise RuntimeError(f"no frames extracted from video {src.name}")
    else:  # images zip
        with zipfile.ZipFile(src) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name or member.endswith("/"):
                    continue
                if Path(name).suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
                }:
                    # Flatten into images_dir (ignore archive subfolders).
                    with zf.open(member) as fh, open(images_dir / name, "wb") as out:
                        shutil.copyfileobj(fh, out)

    backend_id = ctx.manifest.config.photogrammetry_backend
    backend = pg.get_backend(backend_id)
    if backend is None:
        raise RuntimeError(
            "image-set / video input requires a photogrammetry backend "
            f"(requested={backend_id!r}, none available); install COLMAP "
            "(see docs/spec_photogrammetry.md)"
        )
    stats = backend.reconstruct(images_dir, points_out)
    return dict(stats)
