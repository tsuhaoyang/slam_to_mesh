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


def _load_uv_mesh_with_texture(
    mesh_path: Path,
    color_path: Path | None,
    normal_path: Path | None,
):
    """Load the UV mesh and attach baked color/normal textures via a PBR material."""
    mesh = trimesh.load(str(mesh_path), process=False)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return mesh

    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return mesh

    color_img = None
    normal_img = None
    if color_path is not None and color_path.exists():
        color_img = Image.open(str(color_path)).convert("RGB")
    if normal_path is not None and normal_path.exists():
        normal_img = Image.open(str(normal_path)).convert("RGB")

    if color_img is None and normal_img is None:
        return mesh

    # A PBR material can carry both a base-color and a tangent-space normal map;
    # glTF/glb exporters honor both. Fall back to a simple textured visual if
    # only a color map is present and PBR construction fails.
    try:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=color_img,
            normalTexture=normal_img,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    except Exception:  # noqa: BLE001
        if color_img is not None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=color_img)
    return mesh


def _find_bake_textures(ctx: StageContext) -> tuple[Path | None, Path | None]:
    """Return (color_png, normal_png) produced by the bake stage, if any."""
    color = None
    normal = None
    res = ctx.manifest.results.get(Stage.BAKE)
    if res:
        for rel in res.extra_artifacts:
            p = ctx.job_dir / rel
            if p.suffix.lower() != ".png" or not p.exists():
                continue
            if "normal" in p.name:
                normal = p
            elif "color" in p.name:
                color = p
    return color, normal


def _export_usd(
    mesh: trimesh.Trimesh,
    path: Path,
    color_tex: Path | None = None,
    normal_tex: Path | None = None,
) -> bool:
    """Export to USD via pxr if available. Returns True on success.

    Geometry + UVs are always written. If baked textures are present, a
    ``UsdPreviewSurface`` material is created and bound, with the color map on
    ``diffuseColor`` and the normal map on ``normal`` (via UsdUVTexture readers
    fed by a shared ``st`` primvar reader) — the standard Omniverse-friendly
    material graph.
    """
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdShade
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

    # Material graph (only when there is at least one texture and UVs).
    if uv is not None and (color_tex is not None or normal_tex is not None):
        material = UsdShade.Material.Define(stage, "/Mesh/Material")
        pbr = UsdShade.Shader.Define(stage, "/Mesh/Material/PBRShader")
        pbr.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(
            pbr.ConnectableAPI(), "surface"
        )

        # Shared UV reader.
        st_reader = UsdShade.Shader.Define(stage, "/Mesh/Material/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        if color_tex is not None:
            tex = UsdShade.Shader.Define(stage, "/Mesh/Material/diffuseTex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(color_tex.name)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(rgb)

        if normal_tex is not None:
            tex = UsdShade.Shader.Define(stage, "/Mesh/Material/normalTex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_tex.name)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            # Tangent-space normal maps use raw (non-color) sampling.
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            nrgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                nrgb
            )

        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    stage.GetRootLayer().Save()
    return True


def run(ctx: StageContext) -> StageResult:
    result = ctx.manifest.result(Stage.EXPORT)
    result.mark_running()
    cfg = ctx.manifest.config.export
    try:
        src = ctx.input_for(Stage.EXPORT)
        color_tex, normal_tex = _find_bake_textures(ctx)
        mesh = _load_uv_mesh_with_texture(src, color_tex, normal_tex)

        if cfg.scale != 1.0:
            mesh.apply_scale(cfg.scale)

        written: list[str] = []
        skipped: list[str] = []
        base = ctx.job_dir / "model"

        for fmt in cfg.formats:
            fmt = fmt.lower()
            if fmt in {"usd", "usdc", "usda"}:
                out = base.with_suffix(f".{fmt}")
                if _export_usd(mesh, out, color_tex, normal_tex):
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
        for tex in (color_tex, normal_tex):
            if tex is not None:
                result.extra_artifacts.append(ctx.rel(tex))
        result.params = cfg.model_dump()
        result.metrics = {
            "formats_written": written,
            "formats_skipped": skipped,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_color_texture": color_tex is not None,
            "has_normal_texture": normal_tex is not None,
        }
        msg = f"exported {len(written)} format(s): {', '.join(written)}"
        if skipped:
            msg += f"; skipped: {', '.join(skipped)} (usd-core not installed?)"
        result.message = msg
        result.mark_done()
    except Exception as e:
        result.mark_failed(f"{type(e).__name__}: {e}")
        raise
    return result
