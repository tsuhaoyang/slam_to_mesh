"""Stage 1: Ingest & Analyze.

Load the input mesh, compute statistics, detect common SLAM defects
(non-manifold edges, holes, floating components), and persist a normalized copy
as the ingest artifact so later stages have a consistent starting point.
"""

from __future__ import annotations

from ..context import StageContext
from ..meshio import compute_stats, load_mesh, save_mesh
from ..model import Stage, StageResult


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.INGEST)
    result.mark_running()
    try:
        src = ctx.input_for(Stage.INGEST)
        mesh = load_mesh(src)

        stats = compute_stats(mesh)
        ctx.manifest.input_stats = stats

        # Persist a normalized copy (PLY preserves color/normals well).
        out = ctx.out_path(Stage.INGEST, "ply")
        save_mesh(mesh, out)

        result.artifact = ctx.rel(out)
        result.params = {"source": str(src)}
        result.metrics = {
            "vertices": stats.vertices,
            "faces": stats.faces,
            "components": stats.components,
            "boundary_edges": stats.boundary_edges,
            "non_manifold_edges": stats.non_manifold_edges,
            "is_watertight": stats.is_watertight,
        }
        # Human-readable problem summary.
        problems = []
        if stats.non_manifold_edges > 0:
            problems.append(f"{stats.non_manifold_edges} non-manifold edges")
        if not stats.is_watertight:
            problems.append("not watertight (holes present)")
        if stats.components > 1:
            problems.append(f"{stats.components} disconnected components")
        result.message = "; ".join(problems) if problems else "no obvious defects"

        result.mark_done()
    except Exception as e:  # noqa: BLE001 - surface any failure into the manifest
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
