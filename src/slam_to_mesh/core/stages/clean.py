"""Stage 2: Clean.

Repair common SLAM mesh defects with pymeshlab:

* remove floating islands (small disconnected components)
* remove duplicate / unreferenced / degenerate geometry
* repair non-manifold edges and vertices
* close holes up to a bounded size
* recompute consistent vertex normals

The :mod:`.._libfix` preload ensures pymeshlab's ``meshing`` plugin is available
on headless CPU environments (WSL without a display).
"""

from __future__ import annotations

from .. import _libfix

_libfix.ensure_opengl_loaded()

import pymeshlab as ml  # noqa: E402 - must import after the OpenGL preload

from ..context import StageContext
from ..meshio import compute_stats, load_mesh
from ..model import Stage, StageResult


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.CLEAN)
    result.mark_running()
    cfg = ctx.manifest.config.clean
    try:
        src = ctx.input_for(Stage.CLEAN)

        ms = ml.MeshSet()
        ms.load_new_mesh(str(src))

        applied: list[str] = []

        # --- basic topological hygiene -----------------------------------
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_unreferenced_vertices()
        applied += [
            "remove_duplicate_vertices",
            "remove_duplicate_faces",
            "remove_null_faces",
            "remove_unreferenced_vertices",
        ]

        # --- floating islands --------------------------------------------
        if cfg.remove_islands:
            face_count = ms.current_mesh().face_number()
            min_faces = max(1, int(face_count * cfg.min_component_face_ratio))
            ms.meshing_remove_connected_component_by_face_number(
                mincomponentsize=min_faces, removeunref=True
            )
            applied.append(f"remove_islands(min_faces={min_faces})")

        # --- non-manifold repair -----------------------------------------
        if cfg.fix_non_manifold:
            try:
                ms.meshing_repair_non_manifold_edges()
                applied.append("repair_non_manifold_edges")
            except Exception:  # noqa: BLE001
                pass
            try:
                ms.meshing_repair_non_manifold_vertices()
                applied.append("repair_non_manifold_vertices")
            except Exception:  # noqa: BLE001
                pass

        # --- hole filling -------------------------------------------------
        if cfg.fill_holes:
            try:
                ms.meshing_close_holes(
                    maxholesize=int(cfg.max_hole_edges),
                    selfintersection=True,
                    refinehole=False,
                )
                applied.append(f"close_holes(max={cfg.max_hole_edges})")
            except Exception:  # noqa: BLE001
                # Non-manifold geometry can block hole closing; leave as-is.
                pass

        # --- normals ------------------------------------------------------
        if cfg.unify_normals:
            ms.compute_normal_per_vertex()
            applied.append("unify_normals")

        out = ctx.out_path(Stage.CLEAN, "ply")
        out.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(out))

        # Recompute stats on the cleaned mesh for reporting.
        cleaned = load_mesh(out)
        stats = compute_stats(cleaned)

        result.artifact = ctx.rel(out)
        result.params = cfg.model_dump()
        result.params["filters"] = applied
        result.metrics = {
            "vertices": stats.vertices,
            "faces": stats.faces,
            "components": stats.components,
            "boundary_edges": stats.boundary_edges,
            "non_manifold_edges": stats.non_manifold_edges,
            "is_watertight": stats.is_watertight,
        }
        result.message = f"applied {len(applied)} operations"
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
