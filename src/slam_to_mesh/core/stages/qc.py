"""Stage 9: QC report.

Produce a quality-control summary the user can trust before importing to
Omniverse:

* **face count / reduction** — how light the result is vs the input.
* **quad ratio** — how quad-dominant the topology is (from the Stage 4 remesh).
* **Hausdorff / mean surface distance** — geometric fidelity vs the original,
  sampled both ways and normalized by the bounding-box diagonal.
* **manifoldness / watertightness** — topological health of the result.

The report is written as ``09_qc.json`` and mirrored into the manifest metrics.
"""

from __future__ import annotations

import json

import numpy as np
import trimesh

from ..context import StageContext
from ..meshio import bbox_diagonal, load_mesh
from ..model import Stage, StageResult


def _sample_surface(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


def _distances(a_pts: np.ndarray, b_mesh: trimesh.Trimesh) -> np.ndarray:
    _, dist, _ = trimesh.proximity.closest_point(b_mesh, a_pts)
    return dist


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.QC)
    result.mark_running()
    try:
        # Final mesh: prefer the projected/unwrapped result.
        final_path = ctx.input_for(Stage.QC)
        final = load_mesh(final_path)

        # Original reference for fidelity: the ingest artifact (pre-clean).
        ref_path = ctx.manifest.artifact_path(Stage.INGEST) or ctx.manifest.input_path
        ref = load_mesh(ref_path)

        diag = bbox_diagonal(ref) or 1.0

        # Bidirectional surface distance (sampled).
        n = min(20000, max(2000, len(final.vertices) * 5))
        fp = _sample_surface(final, n)
        rp = _sample_surface(ref, n)
        d_final_to_ref = _distances(fp, ref)
        d_ref_to_final = _distances(rp, final)

        hausdorff = float(max(d_final_to_ref.max(), d_ref_to_final.max()))
        mean_dist = float((d_final_to_ref.mean() + d_ref_to_final.mean()) / 2.0)

        # Quad ratio from the remesh stage metrics, if present.
        remesh_res = ctx.manifest.results.get(Stage.REMESH)
        quad_ratio = (
            float(remesh_res.metrics.get("quad_ratio", 0.0)) if remesh_res else 0.0
        )

        input_faces = (
            ctx.manifest.input_stats.faces if ctx.manifest.input_stats else None
        )

        report = {
            "final_vertices": int(len(final.vertices)),
            "final_faces": int(len(final.faces)),
            "input_faces": input_faces,
            "face_reduction_ratio": (
                round(1.0 - len(final.faces) / input_faces, 4)
                if input_faces
                else None
            ),
            "quad_ratio": round(quad_ratio, 4),
            "hausdorff_distance": round(hausdorff, 6),
            "hausdorff_pct_bbox": round(hausdorff / diag * 100, 4),
            "mean_surface_distance": round(mean_dist, 6),
            "mean_dist_pct_bbox": round(mean_dist / diag * 100, 4),
            "is_watertight": bool(final.is_watertight),
            "is_winding_consistent": bool(final.is_winding_consistent),
            "bbox_diagonal": round(diag, 6),
        }

        out = ctx.job_dir / "09_qc.json"
        out.write_text(json.dumps(report, indent=2))

        result.artifact = ctx.rel(out)
        result.metrics = report
        result.message = (
            f"faces={report['final_faces']} "
            f"quad={report['quad_ratio']:.0%} "
            f"meanDist={report['mean_dist_pct_bbox']:.3f}%bbox "
            f"watertight={report['is_watertight']}"
        )
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
