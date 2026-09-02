"""CPU quad-dominant remesh backend (pymeshlab).

Strategy (fully CPU, no external binaries):

1. **Isotropic explicit remeshing** — rebuild the surface with near-uniform edge
   lengths and regular vertex valence. The target edge length is derived from
   the desired face budget and the mesh's bounding-box diagonal. This is what
   turns an irregular SLAM triangulation into a *regular* triangulation.
2. **Smart tri-to-quad pairing** — merge adjacent triangles into quads, yielding
   a quad-dominant mesh.

This does not match a GPU field-aligned remesher (Instant Meshes / QuadriFlow)
for edge-flow quality, but it is deterministic, dependency-light, and produces a
regular, low-poly, quad-dominant mesh suitable for Omniverse display and
material authoring. A GPU backend can replace it later via the registry.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..core import _libfix

_libfix.ensure_opengl_loaded()

import pymeshlab as ml

from .remesh import RemeshRequest, RemeshResult


def _count_quads(path) -> tuple[int, int]:
    """Count (quad_faces, total_polygon_faces) in an OBJ file.

    pymeshlab writes native polygon faces to OBJ, so a face line with four
    vertex references is a quad. Returns (0, 0) if the file cannot be read.
    """
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


def _target_edge_length(target_faces: int, bbox_diag: float) -> float:
    """Estimate a uniform edge length for a triangle budget.

    Surface area is unknown a priori here, so we approximate using the
    bounding-box diagonal as a proxy for scale. The resulting length is refined
    by the remesher's split/collapse iterations; being approximate is fine.
    """
    target_faces = max(target_faces, 4)
    # Heuristic: edge length scales with diagonal / sqrt(faces).
    return max(bbox_diag / math.sqrt(target_faces) * 1.5, 1e-6)


class PyMeshLabRemeshBackend:
    """CPU quad-dominant remesh via pymeshlab."""

    id = "pymeshlab_cpu"

    def is_available(self) -> bool:
        # Requires the meshing plugin, which needs the OpenGL preload we did.
        # Probe with a tiny mesh loaded, since some filters report defaults only
        # when a mesh is present.
        try:
            ms = ml.MeshSet()
            verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            faces = [[0, 1, 2]]
            ms.add_mesh(ml.Mesh(verts, faces))
            ms.filter_parameter_values("meshing_isotropic_explicit_remeshing")
            return True
        except Exception:  # noqa: BLE001
            return False

    def remesh(self, req: RemeshRequest) -> RemeshResult:
        ms = ml.MeshSet()
        ms.load_new_mesh(str(req.input_path))

        tris_before = ms.current_mesh().face_number()
        target_len = _target_edge_length(req.target_faces, req.bbox_diagonal)

        # Feature-line handling. A user-supplied feature-line file is meant to
        # guide edge flow. The pymeshlab CPU backend detects features internally
        # by dihedral angle (``featuredeg``) and cannot consume an arbitrary
        # external feature-line file, so we surface exactly what happened rather
        # than silently ignoring the parameter. A field-aligned GPU backend
        # (Instant Meshes / QuadriFlow) can honor such a file when wired in.
        feature_lines_provided = req.feature_lines is not None
        feature_lines_used = False
        feature_note = None
        if feature_lines_provided:
            fl = Path(req.feature_lines)
            if not fl.exists():
                feature_note = f"feature_lines file not found: {fl}"
            else:
                feature_note = (
                    "feature_lines provided but not consumed by pymeshlab_cpu; "
                    "using dihedral-angle feature detection (featuredeg) instead"
                )

        # 1. Regularize the triangulation.
        feature_deg = 30.0 if req.preserve_sharp else 180.0
        ms.meshing_isotropic_explicit_remeshing(
            iterations=10,
            adaptive=False,
            targetlen=ml.PureValue(target_len),
            featuredeg=feature_deg,
            checksurfdist=True,
            reprojectflag=True,
        )
        tris_remeshed = ms.current_mesh().face_number()

        # 2. Convert to quad-dominant.
        if req.quads:
            try:
                ms.meshing_tri_to_quad_by_smart_triangle_pairing()
            except Exception:  # noqa: BLE001
                # Fallback pairing method.
                ms.meshing_tri_to_quad_dominant(level=0)

        ms.save_current_mesh(str(req.output_path), save_face_color=False)

        faces_after = ms.current_mesh().face_number()

        # Measure the true quad ratio from the saved polygon mesh. pymeshlab
        # writes native polygon faces to OBJ, so 4-vertex ``f`` lines are quads.
        quads, polygon_faces = _count_quads(req.output_path)
        quad_ratio = (quads / polygon_faces) if polygon_faces else 0.0

        return RemeshResult(
            output_path=req.output_path,
            metrics={
                "tris_before": int(tris_before),
                "tris_remeshed": int(tris_remeshed),
                "faces_after": int(faces_after),
                "polygon_faces": int(polygon_faces),
                "quads": int(quads),
                "target_edge_length": round(target_len, 6),
                "quad_pairing": bool(req.quads),
                "quad_ratio": round(quad_ratio, 6),
                "feature_lines_provided": bool(feature_lines_provided),
                "feature_lines_used": bool(feature_lines_used),
                "feature_lines_note": feature_note,
            },
        )
