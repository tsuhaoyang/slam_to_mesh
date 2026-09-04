"""QuadriFlow remesh backend (field-aligned quad remeshing via subprocess).

QuadriFlow (Huang et al., SGP 2018) produces a genuinely field-aligned,
quad-dominant mesh whose edge flow follows surface curvature — a real step up
from the CPU isotropic-remesh + tri-to-quad pairing fallback. It ships as a
small C++ CLI (`quadriflow -i in.obj -o out.obj -f <faces> [-sharp]`), so we
drive it as a subprocess rather than binding it in-process.

Binary discovery order:
1. ``QUADRIFLOW_BIN`` environment variable (explicit path).
2. ``quadriflow`` on ``PATH``.
3. A locally built copy under a sibling ``QuadriFlow/build/`` directory (the
   conventional spot when built from source next to this repo).

If no binary is found, :meth:`is_available` returns ``False`` and the registry
transparently falls back to the CPU backend, so the pipeline always completes.

Note: this backend uses whatever acceleration the binary was built with. A
plain CPU+OpenMP build is already field-aligned; a CUDA-enabled build uses the
GPU. We expose the same backend under both ``quadriflow`` and ``quadriflow_gpu``
ids — the id is cosmetic, the acceleration is a property of the built binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .remesh import RemeshRequest, RemeshResult


def _count_quads(path) -> tuple[int, int]:
    """Count (quad_faces, total_polygon_faces) in an OBJ file."""
    quads = 0
    total = 0
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("f "):
                    total += 1
                    if len(line.split()) - 1 == 4:
                        quads += 1
    except OSError:
        return 0, 0
    return quads, total


def find_quadriflow_binary() -> str | None:
    """Locate the quadriflow binary, or return None."""
    env = os.environ.get("QUADRIFLOW_BIN")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env

    on_path = shutil.which("quadriflow")
    if on_path:
        return on_path

    # Conventional local build location: a sibling QuadriFlow checkout.
    here = Path(__file__).resolve()
    candidates = []
    for parent in here.parents:
        candidates.append(parent / "QuadriFlow" / "build" / "quadriflow")
        candidates.append(parent.parent / "QuadriFlow" / "build" / "quadriflow")
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


class QuadriFlowRemeshBackend:
    """Field-aligned quad remesh via the QuadriFlow CLI (subprocess)."""

    id = "quadriflow"

    def is_available(self) -> bool:
        return find_quadriflow_binary() is not None

    def remesh(self, req: RemeshRequest) -> RemeshResult:
        from .remesh import RemeshResult

        binary = find_quadriflow_binary()
        if binary is None:
            raise RuntimeError("quadriflow binary not found")

        # QuadriFlow needs a triangle OBJ input. The upstream artifact may be a
        # PLY (from decimate); convert to OBJ if needed.
        in_path = Path(req.input_path)
        tmp_obj = None
        if in_path.suffix.lower() != ".obj":
            import trimesh

            tmp_obj = req.output_path.with_name("_quadriflow_input.obj")
            trimesh.load(str(in_path), process=False, force="mesh").export(str(tmp_obj))
            in_path = tmp_obj

        # ``-f`` is the desired quad face resolution.
        resolution = max(int(req.target_faces), 100)

        def _run(use_sharp: bool, timeout: int):
            cmd = [binary, "-i", str(in_path), "-o", str(req.output_path),
                   "-f", str(resolution)]
            if use_sharp:
                cmd.append("-sharp")
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )

        # QuadriFlow's -sharp SAT/index-map step can hang or fail on messy
        # (photogrammetry) meshes with many non-manifold edges/holes. Try it
        # with a bounded timeout first; on failure/timeout, retry without it so
        # the pipeline still produces a clean quad mesh.
        want_sharp = bool(req.preserve_sharp)
        sharp_used = want_sharp
        ok = False
        if want_sharp:
            try:
                proc = _run(True, timeout=300)
                ok = proc.returncode == 0 and Path(req.output_path).exists()
            except subprocess.TimeoutExpired:
                ok = False
            if not ok:
                sharp_used = False  # fall back to a robust non-sharp run
        if not ok:
            proc = _run(False, timeout=1800)
            ok = proc.returncode == 0 and Path(req.output_path).exists()

        if not ok:
            raise RuntimeError(
                f"quadriflow failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        if tmp_obj is not None:
            tmp_obj.unlink(missing_ok=True)

        quads, polygon_faces = _count_quads(req.output_path)
        quad_ratio = (quads / polygon_faces) if polygon_faces else 0.0

        # QuadriFlow consumes sharp features via -sharp (angle-based), not an
        # external feature-line file; acknowledge the request honestly.
        feature_lines_provided = req.feature_lines is not None
        feature_note = None
        if feature_lines_provided:
            feature_note = (
                "quadriflow uses -sharp angle-based feature detection; "
                "an external feature-line file is not consumed"
            )

        return RemeshResult(
            output_path=req.output_path,
            metrics={
                "polygon_faces": int(polygon_faces),
                "quads": int(quads),
                "quad_ratio": round(quad_ratio, 6),
                "resolution": int(resolution),
                "sharp": sharp_used,
                "backend_binary": binary,
                "feature_lines_provided": bool(feature_lines_provided),
                "feature_lines_used": False,
                "feature_lines_note": feature_note,
            },
        )
