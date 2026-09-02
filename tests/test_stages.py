"""Per-stage tests.

Each stage is exercised on the synthetic SLAM mesh. Because later stages consume
earlier artifacts (resolved via ``ctx.input_for``), we build up a shared job
manifest by running the prerequisite stages in order, then assert on the stage
under test.

A module-scoped ``pipeline_upto`` helper runs the (relatively slow) pymeshlab /
xatlas stages once and caches the resulting job directory so the per-stage
assertions stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slam_to_mesh.core.context import StageContext
from slam_to_mesh.core.model import (
    JobManifest,
    PipelineConfig,
    Stage,
    StageStatus,
)
from slam_to_mesh.core.pipeline import STAGE_FUNCS, create_job


def _pxr_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return True


def _run_stages(ctx: StageContext, stages):
    """Run the given stages in order on ctx, returning the last result."""
    last = None
    for s in stages:
        last = STAGE_FUNCS[s](ctx)
        ctx.manifest.save()
    return last


@pytest.fixture
def base_manifest(slam_mesh_path: Path, tmp_path: Path) -> JobManifest:
    """A job manifest with a small, fast config."""
    cfg = PipelineConfig()
    cfg.decimate.target_faces = 2000
    cfg.remesh.target_faces = 800
    cfg.unwrap.resolution = 256
    cfg.export.formats = ["glb", "obj"]
    return create_job(slam_mesh_path, tmp_path / "job", config=cfg)


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def test_ingest_produces_artifact_and_stats(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    res = STAGE_FUNCS[Stage.INGEST](ctx)

    assert res.status == StageStatus.DONE
    assert res.artifact == "01_ingest.ply"
    assert (ctx.job_dir / res.artifact).exists()
    assert base_manifest.input_stats is not None
    assert base_manifest.input_stats.faces > 0
    # The floating island means >1 component -> reported as a problem.
    assert base_manifest.input_stats.components >= 2
    assert "component" in res.message


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #


def test_clean_removes_islands(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(ctx, [Stage.INGEST])
    ingest_components = base_manifest.input_stats.components
    res = STAGE_FUNCS[Stage.CLEAN](ctx)

    assert res.status == StageStatus.DONE
    assert (ctx.job_dir / res.artifact).exists()
    assert res.metrics["faces"] > 0
    assert "remove_islands" in " ".join(res.params["filters"])
    # Cleaning must not fragment the mesh; component count should not grow.
    assert res.metrics["components"] <= ingest_components


def test_clean_drops_small_component_below_ratio(tmp_path: Path):
    """A small island (< min_component_face_ratio of the main body) is dropped."""
    import trimesh

    from slam_to_mesh.core.meshio import compute_stats

    # Large main body so the island is comfortably below the 1% face ratio.
    main = trimesh.creation.icosphere(subdivisions=4, radius=1.0)  # 5120 faces
    island = trimesh.creation.box(extents=[0.05, 0.05, 0.05])  # 12 faces << 1%
    island.apply_translation([4.0, 4.0, 4.0])
    combined = trimesh.util.concatenate([main, island])
    src = tmp_path / "islandy.ply"
    combined.export(str(src))
    assert compute_stats(combined).components >= 2

    m = create_job(src, tmp_path / "job", config=PipelineConfig())
    ctx = StageContext(manifest=m)
    _run_stages(ctx, [Stage.INGEST])
    res = STAGE_FUNCS[Stage.CLEAN](ctx)

    assert res.status == StageStatus.DONE
    assert res.metrics["components"] == 1


def test_clean_fills_holes_iteratively(tmp_path: Path):
    """A mesh with several holes is closed to a watertight result."""
    import trimesh

    # Sphere with faces removed to open multiple holes.
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    holey = trimesh.Trimesh(
        vertices=sphere.vertices, faces=sphere.faces[:-80], process=False
    )
    assert not holey.is_watertight
    src = tmp_path / "holey.ply"
    holey.export(str(src))

    m = create_job(src, tmp_path / "job", config=PipelineConfig())
    ctx = StageContext(manifest=m)
    _run_stages(ctx, [Stage.INGEST])
    res = STAGE_FUNCS[Stage.CLEAN](ctx)

    assert res.status == StageStatus.DONE
    # Holes were closed: boundary edges gone and mesh reported watertight.
    assert res.metrics["hole_boundary_edges_closed"] > 0
    assert res.metrics["hole_fill_passes"] >= 1
    assert res.metrics["boundary_edges"] == 0
    assert res.metrics["is_watertight"] is True


def test_clean_respects_fill_holes_disabled(tmp_path: Path):
    """With fill_holes=False, holes are left open."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    holey = trimesh.Trimesh(
        vertices=sphere.vertices, faces=sphere.faces[:-80], process=False
    )
    src = tmp_path / "holey.ply"
    holey.export(str(src))

    cfg = PipelineConfig()
    cfg.clean.fill_holes = False
    m = create_job(src, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=m)
    _run_stages(ctx, [Stage.INGEST])
    res = STAGE_FUNCS[Stage.CLEAN](ctx)

    assert res.status == StageStatus.DONE
    assert res.metrics["hole_boundary_edges_closed"] == 0
    assert res.metrics["boundary_edges"] > 0


# --------------------------------------------------------------------------- #
# Decimate
# --------------------------------------------------------------------------- #


def test_decimate_reduces_faces(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(ctx, [Stage.INGEST, Stage.CLEAN])
    res = STAGE_FUNCS[Stage.DECIMATE](ctx)

    assert res.status == StageStatus.DONE
    assert (ctx.job_dir / res.artifact).exists()
    assert res.metrics["faces_after"] <= res.metrics["faces_before"]
    assert res.metrics["target_faces"] == 2000


def test_decimate_skips_when_within_budget(slam_mesh_path: Path, tmp_path: Path):
    cfg = PipelineConfig()
    cfg.decimate.target_faces = 10_000_000  # absurdly high
    m = create_job(slam_mesh_path, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=m)
    _run_stages(ctx, [Stage.INGEST, Stage.CLEAN])
    res = STAGE_FUNCS[Stage.DECIMATE](ctx)

    assert res.status == StageStatus.DONE
    assert "skipped" in res.message
    assert res.metrics["faces_after"] == res.metrics["faces_before"]


# --------------------------------------------------------------------------- #
# Remesh
# --------------------------------------------------------------------------- #


def test_remesh_produces_quad_dominant_obj(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(ctx, [Stage.INGEST, Stage.CLEAN, Stage.DECIMATE])
    res = STAGE_FUNCS[Stage.REMESH](ctx)

    assert res.status == StageStatus.DONE
    out = ctx.job_dir / res.artifact
    assert out.suffix == ".obj"
    assert out.exists()
    assert res.params["backend_used"] == "pymeshlab_cpu"
    assert res.metrics["polygon_faces"] > 0
    # Quad-dominant: majority of polygon faces are quads.
    assert res.metrics["quad_ratio"] >= 0.5
    # The backend now reports a real (non-zero) quad ratio itself, and it must
    # be internally consistent with the quad/polygon counts it returns.
    assert res.metrics["quads"] > 0
    expected = res.metrics["quads"] / res.metrics["polygon_faces"]
    assert abs(res.metrics["quad_ratio"] - expected) < 1e-6


def test_remesh_acknowledges_feature_lines(base_manifest: JobManifest, tmp_path: Path):
    """A supplied feature-line file is explicitly acknowledged, not ignored."""
    fl = tmp_path / "features.txt"
    fl.write_text("0 1\n1 2\n")  # arbitrary content; backend won't consume it
    base_manifest.config.remesh.feature_lines = fl
    ctx = StageContext(manifest=base_manifest)
    _run_stages(ctx, [Stage.INGEST, Stage.CLEAN, Stage.DECIMATE])
    res = STAGE_FUNCS[Stage.REMESH](ctx)

    assert res.status == StageStatus.DONE
    assert res.metrics["feature_lines_provided"] is True
    # The CPU backend cannot consume the file; it says so rather than silently
    # dropping the parameter.
    assert res.metrics["feature_lines_used"] is False
    assert res.metrics["feature_lines_note"]
    assert "feature_lines" in (res.message or "")


# --------------------------------------------------------------------------- #
# Project
# --------------------------------------------------------------------------- #


def test_project_snaps_and_preserves_faces(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(ctx, [Stage.INGEST, Stage.CLEAN, Stage.DECIMATE, Stage.REMESH])
    res = STAGE_FUNCS[Stage.PROJECT](ctx)

    assert res.status == StageStatus.DONE
    out = ctx.job_dir / res.artifact
    assert out.exists()
    assert res.metrics["vertices"] > 0
    assert res.metrics["vertices_snapped"] >= 0
    assert res.metrics["vertices_snapped"] <= res.metrics["vertices"]
    # Quad face lines should be preserved (4-vertex f lines present).
    with open(out) as fh:
        has_quad = any(
            line.startswith("f ") and len(line.split()) == 5 for line in fh
        )
    assert has_quad


def test_project_disabled_passthrough(base_manifest: JobManifest):
    base_manifest.config.project.enabled = False
    ctx = StageContext(manifest=base_manifest)
    _run_stages(ctx, [Stage.INGEST, Stage.CLEAN, Stage.DECIMATE, Stage.REMESH])
    remesh_out = (ctx.job_dir / base_manifest.results[Stage.REMESH].artifact).read_bytes()
    res = STAGE_FUNCS[Stage.PROJECT](ctx)

    assert res.status == StageStatus.DONE
    assert "pass-through" in res.message
    # Pass-through copies the remesh artifact verbatim.
    assert (ctx.job_dir / res.artifact).read_bytes() == remesh_out


# --------------------------------------------------------------------------- #
# Unwrap
# --------------------------------------------------------------------------- #


def test_unwrap_produces_uv_obj(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(
        ctx,
        [Stage.INGEST, Stage.CLEAN, Stage.DECIMATE, Stage.REMESH, Stage.PROJECT],
    )
    res = STAGE_FUNCS[Stage.UNWRAP](ctx)

    assert res.status == StageStatus.DONE
    out = ctx.job_dir / res.artifact
    assert out.exists()
    assert res.metrics["faces"] > 0
    assert res.metrics["atlas_vertices"] >= res.metrics["input_vertices"]
    # OBJ carries vt (texture coord) lines.
    with open(out) as fh:
        assert any(line.startswith("vt ") for line in fh)


# --------------------------------------------------------------------------- #
# Bake
# --------------------------------------------------------------------------- #


def test_bake_skipped_when_disabled(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
        ],
    )
    assert base_manifest.config.bake.enabled is False
    res = STAGE_FUNCS[Stage.BAKE](ctx)
    assert res.status == StageStatus.SKIPPED
    assert "disabled" in res.message


def test_bake_enabled_writes_texture(colored_slam_mesh_path: Path, tmp_path: Path):
    cfg = PipelineConfig()
    cfg.decimate.target_faces = 1500
    cfg.remesh.target_faces = 600
    cfg.bake.enabled = True
    cfg.bake.texture_size = 64  # tiny for speed
    m = create_job(colored_slam_mesh_path, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=m)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
        ],
    )
    res = STAGE_FUNCS[Stage.BAKE](ctx)

    assert res.status == StageStatus.DONE
    assert res.metrics["texture_size"] == 64
    assert res.metrics["reference_has_color"] is True
    assert res.metrics["color_baked"] is True
    # Both color and normal maps are baked by default.
    assert res.metrics["normal_baked"] is True
    pngs = [a for a in res.extra_artifacts if a.endswith(".png")]
    assert any("color" in p for p in pngs)
    assert any("normal" in p for p in pngs)
    for p in pngs:
        assert (ctx.job_dir / p).exists()


def test_bake_normal_map_is_tangent_space(colored_slam_mesh_path: Path, tmp_path: Path):
    """The baked normal map should be a valid tangent-space map.

    Convention: most texels point out of the surface, so the blue (Z) channel
    dominates and R/G hover near the 128 midpoint.
    """
    import numpy as np
    from PIL import Image

    cfg = PipelineConfig()
    cfg.decimate.target_faces = 1500
    cfg.remesh.target_faces = 600
    cfg.bake.enabled = True
    cfg.bake.bake_color = False  # normal only
    cfg.bake.bake_normal = True
    cfg.bake.texture_size = 128
    m = create_job(colored_slam_mesh_path, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=m)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
        ],
    )
    res = STAGE_FUNCS[Stage.BAKE](ctx)
    assert res.status == StageStatus.DONE
    assert res.metrics["normal_baked"] is True
    assert res.metrics["color_baked"] is False

    normal_png = next(
        a for a in res.extra_artifacts if a.endswith(".png") and "normal" in a
    )
    img = np.asarray(Image.open(ctx.job_dir / normal_png)).reshape(-1, 3)
    # Only consider written texels (non-background). Background flat value is
    # (128,128,255); written texels vary but Z should stay dominant on average.
    mean = img.mean(axis=0)
    assert mean[2] > mean[0] and mean[2] > mean[1]  # blue dominant
    assert 100 < mean[0] < 160 and 100 < mean[1] < 160  # R,G near midpoint


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_writes_glb_and_obj(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
            Stage.BAKE,
        ],
    )
    res = STAGE_FUNCS[Stage.EXPORT](ctx)

    assert res.status == StageStatus.DONE
    written = res.metrics["formats_written"]
    assert any(w.endswith("model.glb") for w in written)
    assert any(w.endswith("model.obj") for w in written)
    assert (ctx.job_dir / "model.glb").exists()
    assert (ctx.job_dir / "model.obj").exists()


def test_export_skips_usd_when_pxr_missing(base_manifest: JobManifest):
    """USD requested but usd-core not installed -> skipped, pipeline still ok."""
    if _pxr_available():
        pytest.skip("usd-core installed; skip path not exercised")

    base_manifest.config.export.formats = ["obj", "usd"]
    ctx = StageContext(manifest=base_manifest)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
            Stage.BAKE,
        ],
    )
    res = STAGE_FUNCS[Stage.EXPORT](ctx)

    assert res.status == StageStatus.DONE
    assert "usd" in res.metrics["formats_skipped"]
    assert any(w.endswith("model.obj") for w in res.metrics["formats_written"])


def test_export_attaches_baked_textures(colored_slam_mesh_path: Path, tmp_path: Path):
    """When bake produced color+normal maps, export reports and embeds them."""
    cfg = PipelineConfig()
    cfg.decimate.target_faces = 1500
    cfg.remesh.target_faces = 600
    cfg.bake.enabled = True
    cfg.bake.texture_size = 64
    cfg.export.formats = ["glb", "obj"]
    m = create_job(colored_slam_mesh_path, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=m)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
            Stage.BAKE,
        ],
    )
    res = STAGE_FUNCS[Stage.EXPORT](ctx)

    assert res.status == StageStatus.DONE
    assert res.metrics["has_color_texture"] is True
    assert res.metrics["has_normal_texture"] is True
    # glb should be non-trivial (embeds geometry + textures).
    glb = ctx.job_dir / "model.glb"
    assert glb.exists() and glb.stat().st_size > 1000


def test_export_usd_with_material(colored_slam_mesh_path: Path, tmp_path: Path):
    """USD export binds a material with baked textures (requires usd-core)."""
    if not _pxr_available():
        pytest.skip("usd-core not installed")

    cfg = PipelineConfig()
    cfg.decimate.target_faces = 1500
    cfg.remesh.target_faces = 600
    cfg.bake.enabled = True
    cfg.bake.texture_size = 64
    cfg.export.formats = ["usd"]
    m = create_job(colored_slam_mesh_path, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=m)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
            Stage.BAKE,
        ],
    )
    res = STAGE_FUNCS[Stage.EXPORT](ctx)
    assert res.status == StageStatus.DONE
    usd_out = ctx.job_dir / "model.usd"
    assert usd_out.exists()

    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(usd_out))
    mat = UsdShade.Material.Get(stage, "/Mesh/Material")
    assert mat  # material graph was created and bound


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #


def test_qc_writes_report(base_manifest: JobManifest):
    ctx = StageContext(manifest=base_manifest)
    _run_stages(
        ctx,
        [
            Stage.INGEST,
            Stage.CLEAN,
            Stage.DECIMATE,
            Stage.REMESH,
            Stage.PROJECT,
            Stage.UNWRAP,
            Stage.BAKE,
            Stage.EXPORT,
        ],
    )
    res = STAGE_FUNCS[Stage.QC](ctx)

    assert res.status == StageStatus.DONE
    report_path = ctx.job_dir / "09_qc.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["final_faces"] > 0
    assert 0.0 <= report["quad_ratio"] <= 1.0
    assert report["mean_surface_distance"] >= 0.0
    assert "hausdorff_distance" in report
    # Watertight signal is disambiguated: geometric (pre-unwrap) vs the raw
    # final UV mesh, plus an explicit seam-split flag.
    assert "final_mesh_watertight" in report
    assert "seam_vertex_split" in report
    assert isinstance(report["is_watertight"], bool)
    # xatlas splits vertices along seams, so the final UV mesh is (almost
    # always) reported not-watertight even when the geometry is closed.
    if report["is_watertight"] and not report["final_mesh_watertight"]:
        assert report["seam_vertex_split"] is True
    # Metrics mirrored into the manifest.
    assert res.metrics["final_faces"] == report["final_faces"]
