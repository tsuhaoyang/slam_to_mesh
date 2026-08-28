"""Stage 4: Quad remesh.

Delegates to a swappable remesh backend (see :mod:`...backends.remesh`) to turn
the regularized triangle mesh into a quad-dominant mesh. The backend id comes
from the job manifest, so a GPU backend can be selected later without changing
this stage.

We also measure the resulting quad ratio directly from the saved mesh (parsing
polygon faces) so the QC report reflects reality rather than the backend's claim.
"""

from __future__ import annotations

from pathlib import Path

from ...backends.remesh import RemeshRequest, get_backend
from ..context import StageContext
from ..meshio import bbox_diagonal, load_mesh
from ..model import Stage, StageResult


def _quad_ratio_from_obj(path: Path) -> tuple[int, int, float]:
    """Count quad vs total polygon faces in an OBJ file.

    Returns (quads, total_faces, quad_ratio). pymeshlab writes native polygon
    faces to OBJ, so we can detect 4-vertex faces directly from ``f`` lines.
    """
    quads = 0
    total = 0
    try:
        with open(path, "r") as fh:
            for line in fh:
                if line.startswith("f "):
                    total += 1
                    n = len(line.split()) - 1
                    if n == 4:
                        quads += 1
    except OSError:
        return 0, 0, 0.0
    ratio = (quads / total) if total else 0.0
    return quads, total, ratio


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.REMESH)
    result.mark_running()
    cfg = ctx.manifest.config.remesh
    try:
        src = ctx.input_for(Stage.REMESH)
        diag = bbox_diagonal(load_mesh(src))

        # OBJ preserves polygon (quad) faces; PLY from trimesh would triangulate.
        out = ctx.out_path(Stage.REMESH, "obj")
        out.parent.mkdir(parents=True, exist_ok=True)

        backend = get_backend(cfg.backend)
        req = RemeshRequest(
            input_path=src,
            output_path=out,
            target_faces=int(cfg.target_faces),
            quads=bool(cfg.quads),
            preserve_sharp=bool(cfg.preserve_sharp),
            feature_lines=cfg.feature_lines,
            bbox_diagonal=diag,
        )
        res = backend.remesh(req)

        quads, total, ratio = _quad_ratio_from_obj(out)

        result.artifact = ctx.rel(out)
        result.params = cfg.model_dump()
        result.params["backend_used"] = backend.id
        result.metrics = dict(res.metrics)
        result.metrics.update(
            {"quads": quads, "polygon_faces": total, "quad_ratio": round(ratio, 4)}
        )
        result.message = (
            f"backend={backend.id}, quad_ratio={ratio:.1%}, faces={total}"
        )
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
