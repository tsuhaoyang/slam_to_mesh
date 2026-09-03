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
from pathlib import Path

from ..context import StageContext
from ..meshio import compute_stats, load_mesh, save_mesh
from ..model import Stage, StageResult
from ..pointcloud import is_point_cloud_file, reconstruct_poisson


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.INGEST)
    result.mark_running()
    try:
        src = ctx.input_for(Stage.INGEST)

        recon_stats = None
        if is_point_cloud_file(src):
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

        # Human-readable problem summary.
        problems = []
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
