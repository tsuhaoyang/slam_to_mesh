"""Stage 8: Export.

Write the final lightweight mesh to Omniverse-friendly formats. glTF/glb/obj are
always available via trimesh. USD (``.usd``/``.usdc``) is written when
``usd-core`` (the ``pxr`` module) is installed; otherwise it is skipped with a
clear message so the pipeline still completes on a minimal install.

The UV-unwrapped mesh from Stage 6 is the geometry source. If Stage 7 produced a
baked color texture, it is attached to the exported material.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..context import StageContext
from ..model import Stage, StageResult


def _load_uv_mesh_with_texture(mesh_path: Path, texture_path: Path | None):
    """Load the UV mesh and attach a texture image if provided."""
    mesh = trimesh.load(str(mesh_path), process=False)
    if texture_path is not None and texture_path.exists():
        try:
            from PIL import Image

            img = Image.open(str(texture_path)).convert("RGB")
            uv = getattr(mesh.visual, "uv", None)
            if uv is not None:
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv, image=img
                )
        except Exception:  # noqa: BLE001
            pass
    return mesh


def _find_bake_texture(ctx: StageContext) -> Path | None:
    res = ctx.manifest.results.get(Stage.BAKE)
    if res:
        for rel in res.extra_artifacts:
            p = ctx.job_dir / rel
            if p.suffix.lower() == ".png" and p.exists():
                return p
    return None


def _export_usd(mesh: trimesh.Trimesh, path: Path) -> bool:
    """Export to USD via pxr if available. Returns True on success."""
    try:
        from pxr import Sdf, Usd, UsdGeom
    except Exception:  # noqa: BLE001
        return False

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    mesh_prim = UsdGeom.Mesh.Define(stage, "/Mesh")

    points = mesh.vertices.astype(float)
    faces = mesh.faces.astype(int)
    mesh_prim.CreatePointsAttr([tuple(p) for p in points])
    mesh_prim.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh_prim.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())

    uv = getattr(mesh.visual, "uv", None)
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st = np.asarray(uv)[faces.reshape(-1)]
        pv.Set([tuple(map(float, c)) for c in st])

    stage.GetRootLayer().Save()
    return True


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.EXPORT)
    result.mark_running()
    cfg = ctx.manifest.config.export
    try:
        src = ctx.input_for(Stage.EXPORT)
        texture = _find_bake_texture(ctx)
        mesh = _load_uv_mesh_with_texture(src, texture)

        if cfg.scale != 1.0:
            mesh.apply_scale(cfg.scale)

        written: list[str] = []
        skipped: list[str] = []
        base = ctx.job_dir / "model"

        for fmt in cfg.formats:
            fmt = fmt.lower()
            if fmt in {"usd", "usdc", "usda"}:
                out = base.with_suffix(f".{fmt}")
                if _export_usd(mesh, out):
                    written.append(ctx.rel(out))
                else:
                    skipped.append(fmt)
            elif fmt in {"glb", "gltf", "obj", "ply", "stl", "off"}:
                out = base.with_suffix(f".{fmt}")
                mesh.export(str(out))
                written.append(ctx.rel(out))
            else:
                skipped.append(fmt)

        # Record the primary artifact (prefer glb, else first written).
        primary = next(
            (w for w in written if w.endswith(".glb")),
            written[0] if written else None,
        )
        result.artifact = primary
        result.extra_artifacts = [w for w in written if w != primary]
        if texture is not None:
            result.extra_artifacts.append(ctx.rel(texture))
        result.params = cfg.model_dump()
        result.metrics = {
            "formats_written": written,
            "formats_skipped": skipped,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "has_texture": texture is not None,
        }
        msg = f"exported {len(written)} format(s): {', '.join(written)}"
        if skipped:
            msg += f"; skipped: {', '.join(skipped)} (usd-core not installed?)"
        result.message = msg
        result.mark_done()
    except Exception as e:  # noqa: BLE001
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
