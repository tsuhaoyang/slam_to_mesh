"""Unit tests for meshio helpers and the StageContext."""

from __future__ import annotations

from pathlib import Path

import trimesh

from slam_to_mesh.core.context import StageContext, artifact_name
from slam_to_mesh.core.meshio import (
    bbox_diagonal,
    compute_stats,
    load_mesh,
    save_mesh,
)
from slam_to_mesh.core.model import JobManifest, Stage


def test_load_mesh_returns_trimesh(slam_mesh_path: Path):
    mesh = load_mesh(slam_mesh_path)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_save_mesh_creates_parents(tmp_path: Path, slam_mesh: trimesh.Trimesh):
    out = tmp_path / "nested" / "dir" / "m.ply"
    returned = save_mesh(slam_mesh, out)
    assert returned == out
    assert out.exists()
    reloaded = load_mesh(out)
    assert len(reloaded.vertices) == len(slam_mesh.vertices)


def test_compute_stats_detects_components(slam_mesh: trimesh.Trimesh):
    stats = compute_stats(slam_mesh)
    # icosphere + island => at least 2 components.
    assert stats.components >= 2
    assert stats.faces == len(slam_mesh.faces)
    assert stats.vertices == len(slam_mesh.vertices)
    assert stats.bbox_min is not None and stats.bbox_max is not None
    # A pure icosphere is watertight; adding the box island keeps two closed
    # surfaces, so tris count matches faces.
    assert stats.tris == stats.faces
    assert stats.quads == 0


def test_bbox_diagonal_positive(slam_mesh: trimesh.Trimesh):
    diag = bbox_diagonal(slam_mesh)
    assert diag > 0


def test_artifact_name_prefixes():
    assert artifact_name(Stage.INGEST, "ply") == "01_ingest.ply"
    assert artifact_name(Stage.CLEAN, ".ply") == "02_clean.ply"
    assert artifact_name(Stage.QC, "json") == "09_qc.json"


def test_context_out_path_and_rel(tmp_path: Path):
    m = JobManifest(job_id="j", input_path="/in.ply", job_dir=str(tmp_path))
    ctx = StageContext(manifest=m)
    out = ctx.out_path(Stage.REMESH, "obj")
    assert out == tmp_path / "04_remesh.obj"
    assert ctx.rel(out) == "04_remesh.obj"
    assert ctx.job_dir == tmp_path


def test_context_input_for_falls_back_to_original(tmp_path: Path):
    m = JobManifest(job_id="j", input_path="/original/in.ply", job_dir=str(tmp_path))
    ctx = StageContext(manifest=m)
    # No prior artifacts => original input.
    assert ctx.input_for(Stage.INGEST) == Path("/original/in.ply")


def test_context_input_for_uses_latest_artifact(tmp_path: Path, slam_mesh: trimesh.Trimesh):
    m = JobManifest(job_id="j", input_path="/original/in.ply", job_dir=str(tmp_path))
    ctx = StageContext(manifest=m)

    # Produce ingest + clean artifacts on disk.
    ingest_out = ctx.out_path(Stage.INGEST, "ply")
    save_mesh(slam_mesh, ingest_out)
    m.result(Stage.INGEST).artifact = ctx.rel(ingest_out)

    clean_out = ctx.out_path(Stage.CLEAN, "ply")
    save_mesh(slam_mesh, clean_out)
    m.result(Stage.CLEAN).artifact = ctx.rel(clean_out)

    # Input for DECIMATE should be the most recent produced artifact (clean).
    assert ctx.input_for(Stage.DECIMATE) == clean_out
    # Input for CLEAN should be ingest.
    assert ctx.input_for(Stage.CLEAN) == ingest_out


def test_context_input_for_skips_missing_artifact(tmp_path: Path, slam_mesh: trimesh.Trimesh):
    m = JobManifest(job_id="j", input_path="/original/in.ply", job_dir=str(tmp_path))
    ctx = StageContext(manifest=m)

    # Record an ingest artifact path that does NOT exist on disk.
    m.result(Stage.INGEST).artifact = "01_ingest.ply"
    # Clean exists.
    clean_out = ctx.out_path(Stage.CLEAN, "ply")
    save_mesh(slam_mesh, clean_out)
    m.result(Stage.CLEAN).artifact = ctx.rel(clean_out)

    # Decimate walks back: clean exists -> use it.
    assert ctx.input_for(Stage.DECIMATE) == clean_out
    # Clean walks back: ingest missing -> fall back to original input.
    assert ctx.input_for(Stage.CLEAN) == Path("/original/in.ply")
