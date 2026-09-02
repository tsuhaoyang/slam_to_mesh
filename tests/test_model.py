"""Unit tests for the core data model (model.py)."""

from __future__ import annotations

from pathlib import Path

from slam_to_mesh.core.model import (
    STAGE_ORDER,
    JobManifest,
    PipelineConfig,
    Stage,
    StageResult,
    StageStatus,
    stage_index,
    stages_from,
)


def test_stage_order_is_canonical_pipeline():
    assert STAGE_ORDER == [
        Stage.INGEST,
        Stage.CLEAN,
        Stage.DECIMATE,
        Stage.REMESH,
        Stage.PROJECT,
        Stage.UNWRAP,
        Stage.BAKE,
        Stage.EXPORT,
        Stage.QC,
    ]
    # Values double as artifact prefixes and are unique.
    assert len({s.value for s in Stage}) == len(list(Stage))


def test_stage_index_and_stages_from():
    assert stage_index(Stage.INGEST) == 0
    assert stage_index(Stage.QC) == len(STAGE_ORDER) - 1
    assert stages_from(Stage.REMESH) == [
        Stage.REMESH,
        Stage.PROJECT,
        Stage.UNWRAP,
        Stage.BAKE,
        Stage.EXPORT,
        Stage.QC,
    ]
    assert stages_from(Stage.QC) == [Stage.QC]
    assert stages_from(Stage.INGEST) == STAGE_ORDER


def test_stage_result_lifecycle():
    r = StageResult(stage=Stage.CLEAN)
    assert r.status == StageStatus.PENDING
    r.mark_running()
    assert r.status == StageStatus.RUNNING
    assert r.started_at is not None
    r.mark_done()
    assert r.status == StageStatus.DONE
    assert r.finished_at is not None

    r2 = StageResult(stage=Stage.REMESH)
    r2.mark_running()
    r2.mark_failed("boom")
    assert r2.status == StageStatus.FAILED
    assert r2.message == "boom"
    assert r2.finished_at is not None


def test_manifest_result_creates_lazily():
    m = JobManifest(job_id="j1", input_path="/in.ply", job_dir="/tmp/j1")
    assert Stage.INGEST not in m.results
    res = m.result(Stage.INGEST)
    assert isinstance(res, StageResult)
    assert m.results[Stage.INGEST] is res
    # Second call returns the same object.
    assert m.result(Stage.INGEST) is res


def test_manifest_save_and_load_roundtrip(tmp_path: Path):
    m = JobManifest(
        job_id="abc123",
        input_path="/data/in.ply",
        job_dir=str(tmp_path),
        config=PipelineConfig(),
    )
    r = m.result(Stage.INGEST)
    r.artifact = "01_ingest.ply"
    r.mark_done()

    saved = m.save()
    assert saved == tmp_path / "job.json"
    assert saved.exists()

    loaded = JobManifest.load(saved)
    assert loaded.job_id == "abc123"
    assert loaded.input_path == "/data/in.ply"
    assert loaded.results[Stage.INGEST].artifact == "01_ingest.ply"
    assert loaded.results[Stage.INGEST].status == StageStatus.DONE


def test_artifact_path_resolution(tmp_path: Path):
    m = JobManifest(job_id="j", input_path="/in.ply", job_dir=str(tmp_path))
    assert m.artifact_path(Stage.INGEST) is None
    r = m.result(Stage.INGEST)
    r.artifact = "01_ingest.ply"
    assert m.artifact_path(Stage.INGEST) == tmp_path / "01_ingest.ply"


def test_last_completed_stage(tmp_path: Path):
    m = JobManifest(job_id="j", input_path="/in.ply", job_dir=str(tmp_path))
    assert m.last_completed_stage() is None

    m.result(Stage.INGEST).mark_done()
    assert m.last_completed_stage() == Stage.INGEST

    m.result(Stage.CLEAN).mark_done()
    m.result(Stage.DECIMATE).mark_done()
    assert m.last_completed_stage() == Stage.DECIMATE

    # A failed later stage does not count.
    m.result(Stage.REMESH).mark_failed("x")
    assert m.last_completed_stage() == Stage.DECIMATE


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.decimate.target_faces == 50000
    assert cfg.remesh.target_faces == 20000
    assert cfg.remesh.backend == "quadriflow_cpu"
    assert cfg.project.enabled is True
    assert cfg.bake.enabled is False
    assert "glb" in cfg.export.formats
