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

import pymeshlab as ml

from ..context import StageContext
from ..meshio import compute_stats, load_mesh
from ..model import Stage, StageResult


def _boundary_edge_count(ms: ml.MeshSet) -> int:
    """Number of boundary (hole) edges in the current mesh.

    Uses pymeshlab's topological measures. Returns 0 if unavailable so the
    caller can proceed without crashing on odd meshes.
    """
    try:
        measures = ms.get_topological_measures()
    except Exception:  # noqa: BLE001
        return 0
    for key in ("boundary_edges", "boundary_edge_num", "number_holes"):
        if key in measures:
            try:
                return int(measures[key])
            except (TypeError, ValueError):
                continue
    return 0


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
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                ms.meshing_repair_non_manifold_vertices()
                applied.append("repair_non_manifold_vertices")
            except Exception:  # noqa: BLE001, S110
                pass

        # --- hole filling (iterative, robust for fragmented input) --------
        holes_closed_edges = 0
        fill_passes = 0
        if cfg.fill_holes:
            before_boundary = _boundary_edge_count(ms)
            max_iter = max(1, int(cfg.max_hole_fill_iterations))
            prev_boundary = before_boundary
            for _ in range(max_iter):
                closed_any = False
                try:
                    ms.meshing_close_holes(
                        maxholesize=int(cfg.max_hole_edges),
                        selfintersection=True,
                        refinehole=False,
                    )
                    closed_any = True
                except Exception:  # noqa: BLE001, S110
                    # Non-manifold geometry commonly blocks closing. Try to
                    # unblock it with a repair pass, then loop to retry.
                    pass

                cur_boundary = _boundary_edge_count(ms)
                if closed_any:
                    fill_passes += 1
                # Stop when no boundary edges remain, or a pass made no progress.
                if cur_boundary == 0 or cur_boundary >= prev_boundary:
                    if cur_boundary >= prev_boundary and not closed_any:
                        # Blocked with no progress: attempt one repair to unblock.
                        try:
                            ms.meshing_repair_non_manifold_edges()
                        except Exception:  # noqa: BLE001
                            break
                        # If repair didn't reduce boundaries either, give up.
                        if _boundary_edge_count(ms) >= prev_boundary:
                            break
                    elif cur_boundary >= prev_boundary:
                        break
                prev_boundary = cur_boundary

            after_boundary = _boundary_edge_count(ms)
            holes_closed_edges = max(0, before_boundary - after_boundary)
            applied.append(
                f"close_holes(max={cfg.max_hole_edges}, passes={fill_passes}, "
                f"boundary {before_boundary}->{after_boundary})"
            )

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
            "hole_boundary_edges_closed": int(holes_closed_edges),
            "hole_fill_passes": int(fill_passes),
        }
        result.message = f"applied {len(applied)} operations"
        result.mark_done()
    except Exception as e:
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
