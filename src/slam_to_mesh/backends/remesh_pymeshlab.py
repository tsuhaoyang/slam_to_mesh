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

from ..core import _libfix

_libfix.ensure_opengl_loaded()

import pymeshlab as ml  # noqa: E402 - after OpenGL preload

from .remesh import RemeshRequest, RemeshResult


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

        quad_ratio = 0.0
        # 2. Convert to quad-dominant.
        if req.quads:
            try:
                ms.meshing_tri_to_quad_by_smart_triangle_pairing()
            except Exception:  # noqa: BLE001
                # Fallback pairing method.
                ms.meshing_tri_to_quad_dominant(level=0)

        ms.save_current_mesh(str(req.output_path), save_face_color=False)

        faces_after = ms.current_mesh().face_number()

        return RemeshResult(
            output_path=req.output_path,
            metrics={
                "tris_before": int(tris_before),
                "tris_remeshed": int(tris_remeshed),
                "faces_after": int(faces_after),
                "target_edge_length": round(target_len, 6),
                "quad_pairing": bool(req.quads),
                "quad_ratio": quad_ratio,
            },
        )
