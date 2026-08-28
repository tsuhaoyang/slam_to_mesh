"""Stage 5: Project.

Remeshing moves vertices off the original surface (isotropic smoothing pulls
them toward the average). To restore geometric fidelity we snap each remeshed
vertex to the closest point on a reference surface — the cleaned, pre-remesh
mesh — provided the move is within a bounded distance (fraction of the bbox
diagonal). Bounding the snap distance prevents thin features from collapsing.

Quad connectivity is preserved: we only move vertex positions, not topology.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..context import StageContext
from ..meshio import bbox_diagonal, load_mesh
from ..model import STAGE_ORDER, Stage, StageResult, stage_index


def _reference_surface(ctx: StageContext) -> trimesh.Trimesh:
    """The high-res surface to project onto: prefer clean, else ingest, else input."""
    for stage in (Stage.CLEAN, Stage.INGEST):
        p = ctx.manifest.artifact_path(stage)
        if p is not None and p.exists():
            return load_mesh(p)
    return load_mesh(ctx.manifest.input_path)


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.PROJECT)
    result.mark_running()
    cfg = ctx.manifest.config.project
    try:
        src = ctx.input_for(Stage.PROJECT)  # remeshed (quad) mesh

        if not cfg.enabled:
            # Pass-through: copy input as the project artifact.
            out = ctx.out_path(Stage.PROJECT, "obj")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(src.read_bytes())
            result.artifact = ctx.rel(out)
            result.params = cfg.model_dump()
            result.message = "projection disabled (pass-through)"
            result.mark_done()
            return result

        # Load the remeshed mesh WITHOUT triangulation to keep quad faces.
        # trimesh triangulates, so we preserve original face lines by editing
        # only the vertex block of the OBJ and rewriting it.
        verts, faces_lines, other_lines = _read_obj_verts_faces(src)
        ref = _reference_surface(ctx)

        diag = bbox_diagonal(ref)
        max_snap = cfg.max_snap_ratio * diag

        # Closest point on reference surface for each remeshed vertex.
        closest, distances, _ = trimesh.proximity.closest_point(ref, verts)

        moved = distances <= max_snap
        new_verts = verts.copy()
        new_verts[moved] = closest[moved]

        out = ctx.out_path(Stage.PROJECT, "obj")
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_obj(out, new_verts, faces_lines, other_lines)

        result.artifact = ctx.rel(out)
        result.params = cfg.model_dump()
        result.metrics = {
            "vertices": int(len(verts)),
            "vertices_snapped": int(np.count_nonzero(moved)),
            "max_snap_distance": round(float(max_snap), 6),
            "mean_snap_distance": round(float(distances[moved].mean()), 6)
            if np.any(moved)
            else 0.0,
            "max_observed_distance": round(float(distances.max()), 6),
        }
        result.message = (
            f"snapped {int(np.count_nonzero(moved))}/{len(verts)} verts "
            f"(<= {max_snap:.4g})"
        )
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result


def _read_obj_verts_faces(path):
    """Parse an OBJ into (vertices array, face lines, other lines).

    Preserves polygon faces verbatim so quad connectivity survives projection.
    """
    verts = []
    face_lines = []
    other_lines = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                face_lines.append(line.rstrip("\n"))
            elif line.startswith(("vn ", "vt ")):
                # Drop stale normals/uvs; normals get recomputed downstream.
                continue
            else:
                other_lines.append(line.rstrip("\n"))
    return np.asarray(verts, dtype=float), face_lines, other_lines


def _write_obj(path, verts, face_lines, other_lines):
    """Write an OBJ preserving the original (polygon) face lines."""
    with open(path, "w") as fh:
        for ln in other_lines:
            if ln.strip():
                fh.write(ln + "\n")
        for v in verts:
            fh.write(f"v {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n")
        for fl in face_lines:
            fh.write(fl + "\n")
