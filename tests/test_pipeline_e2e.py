"""End-to-end pipeline and resume-from tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_to_mesh.core.model import (
    STAGE_ORDER,
    JobManifest,
    PipelineConfig,
    Stage,
    StageStatus,
)
from slam_to_mesh.core.pipeline import create_job, run_pipeline


def _fast_config() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.decimate.target_faces = 2000
    cfg.remesh.target_faces = 800
    cfg.unwrap.resolution = 256
    cfg.export.formats = ["glb", "obj"]
    return cfg


def test_full_pipeline_runs_all_stages(slam_mesh_path: Path, tmp_path: Path):
    manifest = create_job(slam_mesh_path, tmp_path / "job", config=_fast_config())

    seen: list[str] = []
    run_pipeline(
        manifest,
        start=Stage.INGEST,
        on_stage=lambda r: seen.append(r.stage.value),
    )

    # Every stage was visited in canonical order.
    assert seen == [s.value for s in STAGE_ORDER]

    # Terminal states: bake skipped (disabled), everything else done.
    for stage in STAGE_ORDER:
        res = manifest.results[stage]
        if stage == Stage.BAKE:
            assert res.status == StageStatus.SKIPPED
        else:
            assert res.status == StageStatus.DONE, f"{stage} -> {res.status}"

    # Final exported artifacts exist.
    assert (Path(manifest.job_dir) / "model.glb").exists()
    assert (Path(manifest.job_dir) / "09_qc.json").exists()

    # Manifest persisted to disk and reloadable.
    reloaded = JobManifest.load(Path(manifest.job_dir) / "job.json")
    assert reloaded.last_completed_stage() == Stage.QC


def test_manifest_saved_after_every_stage(slam_mesh_path: Path, tmp_path: Path):
    manifest = create_job(slam_mesh_path, tmp_path / "job", config=_fast_config())
    job_json = Path(manifest.job_dir) / "job.json"

    # Run only through decimate via explicit stage list.
    run_pipeline(
        manifest,
        stages=[Stage.INGEST, Stage.CLEAN, Stage.DECIMATE],
    )

    reloaded = JobManifest.load(job_json)
    assert reloaded.results[Stage.DECIMATE].status == StageStatus.DONE
    # Later stages were not run.
    assert Stage.REMESH not in reloaded.results


def test_resume_from_reuses_prior_artifacts(slam_mesh_path: Path, tmp_path: Path):
    manifest = create_job(slam_mesh_path, tmp_path / "job", config=_fast_config())

    # First pass: run up to and including remesh.
    run_pipeline(
        manifest,
        stages=[Stage.INGEST, Stage.CLEAN, Stage.DECIMATE, Stage.REMESH],
    )
    assert manifest.results[Stage.REMESH].status == StageStatus.DONE
    remesh_artifact = Path(manifest.job_dir) / manifest.results[Stage.REMESH].artifact
    remesh_mtime = remesh_artifact.stat().st_mtime

    # Reload from disk (as the CLI `resume-from` does) and resume at project.
    reloaded = JobManifest.load(Path(manifest.job_dir) / "job.json")
    run_pipeline(reloaded, start=Stage.PROJECT)

    # Remesh artifact was NOT regenerated (resume reused it).
    assert remesh_artifact.stat().st_mtime == remesh_mtime
    # Downstream stages completed.
    assert reloaded.results[Stage.PROJECT].status == StageStatus.DONE
    assert reloaded.results[Stage.QC].status == StageStatus.DONE
    assert (Path(reloaded.job_dir) / "model.glb").exists()


def test_resume_with_changed_param_reruns_stage(slam_mesh_path: Path, tmp_path: Path):
    """Tweaking a stage param and resuming re-runs it with the new value."""
    manifest = create_job(slam_mesh_path, tmp_path / "job", config=_fast_config())
    run_pipeline(manifest, stages=[Stage.INGEST, Stage.CLEAN, Stage.DECIMATE, Stage.REMESH])

    first_faces = manifest.results[Stage.REMESH].metrics["polygon_faces"]

    # Change the remesh target and resume from remesh.
    manifest.config.remesh.target_faces = 300
    run_pipeline(manifest, start=Stage.REMESH)

    second_faces = manifest.results[Stage.REMESH].metrics["polygon_faces"]
    # A smaller target should not produce more faces than the larger target.
    assert second_faces <= first_faces
    assert manifest.results[Stage.QC].status == StageStatus.DONE


def test_pipeline_stops_on_failure(tmp_path: Path):
    """A stage failure halts the pipeline and is recorded in the manifest."""
    # Point at a non-existent input so ingest fails.
    bad = tmp_path / "missing.ply"
    (tmp_path / "job").mkdir(parents=True, exist_ok=True)
    m = JobManifest(
        job_id="failjob",
        input_path=str(bad),
        job_dir=str(tmp_path / "job"),
        config=PipelineConfig(),
    )

    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        run_pipeline(m, start=Stage.INGEST)

    # Ingest recorded as failed; later stages never ran.
    assert m.results[Stage.INGEST].status == StageStatus.FAILED
    assert Stage.CLEAN not in m.results
