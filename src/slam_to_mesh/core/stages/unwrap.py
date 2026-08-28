"""Stage 6: UV unwrap (xatlas).

xatlas operates on triangle meshes, so we triangulate the projected
quad-dominant mesh, generate a UV atlas, and write a triangulated OBJ carrying
per-vertex UVs. Omniverse (USD/glTF) triangulates on import anyway, so a
triangulated + UV-mapped mesh is the right hand-off for texturing.

The quad topology from Stage 5 is preserved as its own artifact; this stage adds
the UV-ready mesh alongside it.

``xatlas.parametrize`` returns:
* ``vmapping`` — for each output vertex, the index of the source vertex it came
  from (charts split vertices along seams).
* ``indices``  — triangle indices into the output vertex list.
* ``uvs``      — per output-vertex UV coordinates in [0, 1].
"""

from __future__ import annotations

import numpy as np
import trimesh
import xatlas

from ..context import StageContext
from ..meshio import load_mesh
from ..model import Stage, StageResult


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.UNWRAP)
    result.mark_running()
    cfg = ctx.manifest.config.unwrap
    try:
        src = ctx.input_for(Stage.UNWRAP)

        # trimesh triangulates polygon faces on load -> tris for xatlas.
        mesh = load_mesh(src)
        positions = np.asarray(mesh.vertices, dtype=np.float32)
        indices = np.asarray(mesh.faces, dtype=np.uint32)

        # Generate the atlas.
        vmapping, out_indices, uvs = xatlas.parametrize(positions, indices)

        # Build the UV-mapped mesh: vertices remapped through vmapping.
        out_positions = positions[vmapping]
        out_faces = np.asarray(out_indices, dtype=np.int64).reshape(-1, 3)
        uvs = np.asarray(uvs, dtype=np.float64)

        out = ctx.out_path(Stage.UNWRAP, "obj")
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_obj_with_uv(out, out_positions, out_faces, uvs)

        result.artifact = ctx.rel(out)
        result.params = cfg.model_dump()
        result.metrics = {
            "input_vertices": int(len(positions)),
            "atlas_vertices": int(len(out_positions)),
            "faces": int(len(out_faces)),
            "seam_vertex_increase": int(len(out_positions) - len(positions)),
            "uv_min": [float(uvs[:, 0].min()), float(uvs[:, 1].min())],
            "uv_max": [float(uvs[:, 0].max()), float(uvs[:, 1].max())],
        }
        result.message = (
            f"unwrapped: {len(positions)} -> {len(out_positions)} verts "
            f"(seams), {len(out_faces)} tris"
        )
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result


def _write_obj_with_uv(path, positions, faces, uvs):
    """Write a triangulated OBJ with per-vertex texture coordinates.

    Vertex and UV indices are aligned 1:1 (xatlas gives one UV per output
    vertex), so face lines use ``v/vt`` with identical indices.
    """
    with open(path, "w") as fh:
        fh.write("# slam_to_mesh UV-unwrapped mesh\n")
        for p in positions:
            fh.write(f"v {p[0]:.8g} {p[1]:.8g} {p[2]:.8g}\n")
        for uv in uvs:
            fh.write(f"vt {uv[0]:.8g} {uv[1]:.8g}\n")
        for f in faces:
            a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
            fh.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
