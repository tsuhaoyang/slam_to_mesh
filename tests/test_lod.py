"""Tests for core.lod builders (quad re-remesh + triangle QEM)."""

from __future__ import annotations

from pathlib import Path

import trimesh

from slam_to_mesh.core import lod
from slam_to_mesh.core.context import StageContext
from slam_to_mesh.core.model import PipelineConfig
from slam_to_mesh.core.pipeline import create_job
from slam_to_mesh.core.stages import clean, decimate, ingest


def _prepared_job(tmp_path: Path):
    m = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    src = tmp_path / "in.ply"
    m.export(str(src))
    cfg = PipelineConfig()
    cfg.decimate.target_faces = 4000
    manifest = create_job(src, tmp_path / "job", config=cfg)
    ctx = StageContext(manifest=manifest)
    for fn in (ingest, clean, decimate):
        fn.run(ctx)
        manifest.save()
    return manifest


def test_build_tri_lod_hits_exact_target(tmp_path: Path):
    manifest = _prepared_job(tmp_path)
    res = lod.build_tri_lod(manifest, target_faces=1000)
    # QEM hits the exact target face count.
    assert res.actual_faces == 1000
    assert res.mean_dist_pct_bbox >= 0.0
    assert (Path(manifest.job_dir) / res.glb).exists()
    assert (Path(manifest.job_dir) / res.obj).exists()
    assert "1000" in manifest.tri_lods


def test_build_tri_lod_is_cached(tmp_path: Path):
    manifest = _prepared_job(tmp_path)
    a = lod.build_tri_lod(manifest, target_faces=1000)
    b = lod.build_tri_lod(manifest, target_faces=1000)
    assert a.glb == b.glb
    assert len(manifest.tri_lods) == 1  # same bucket, not rebuilt


def test_build_tri_lod_by_ratio(tmp_path: Path):
    manifest = _prepared_job(tmp_path)
    # 10% of the original (icosphere sub4 = 5120 faces) ≈ 512.
    res = lod.build_tri_lod(manifest, ratio=0.1)
    assert res.actual_faces > 0
    assert res.actual_faces <= 5120


def test_build_quad_lod_still_works(tmp_path: Path):
    """Regression: quad LOD builder unaffected by tri additions."""
    manifest = _prepared_job(tmp_path)
    res = lod.build_lod(manifest, target_faces=800, bake=False)
    assert res.actual_faces > 0
    assert 0.0 <= res.quad_ratio <= 1.0
    assert (Path(manifest.job_dir) / res.glb).exists()
