"""Stage 3: Decimate (QEM).

Reduce the cleaned high-resolution mesh to a target face budget using quadric
edge-collapse decimation. This shrinks the working set before the (more
expensive) quad remesh while preserving overall shape, boundaries, and normals.

We do not decimate below the remesh target; the goal is a faithful reduced tri
mesh that the remesher can then convert into a regular quad-dominant layout.
"""

from __future__ import annotations

from .. import _libfix

_libfix.ensure_opengl_loaded()

import pymeshlab as ml  # noqa: E402 - must import after the OpenGL preload

from ..context import StageContext
from ..meshio import compute_stats, load_mesh
from ..model import Stage, StageResult


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.DECIMATE)
    result.mark_running()
    cfg = ctx.manifest.config.decimate
    try:
        src = ctx.input_for(Stage.DECIMATE)

        ms = ml.MeshSet()
        ms.load_new_mesh(str(src))

        before = ms.current_mesh().face_number()
        target = int(cfg.target_faces)

        skipped = False
        if before <= target:
            # Already at or below budget; nothing to do but pass through.
            skipped = True
        else:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target,
                qualitythr=float(cfg.quality_threshold),
                preserveboundary=bool(cfg.preserve_boundary),
                preservenormal=bool(cfg.preserve_normals),
                preservetopology=True,
                optimalplacement=True,
                autoclean=True,
            )

        out = ctx.out_path(Stage.DECIMATE, "ply")
        out.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(out))

        after = ms.current_mesh().face_number()
        stats = compute_stats(load_mesh(out))

        result.artifact = ctx.rel(out)
        result.params = cfg.model_dump()
        result.metrics = {
            "faces_before": int(before),
            "faces_after": int(after),
            "target_faces": target,
            "reduction_ratio": round(1.0 - (after / before), 4) if before else 0.0,
            "vertices": stats.vertices,
            "components": stats.components,
        }
        result.message = (
            "skipped (already within budget)"
            if skipped
            else f"decimated {before} -> {after} faces"
        )
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
