"""Tests for image → mesh (TripoSR) integration.

Availability / routing / gating tests always run. The heavy end-to-end test
(actually invoking TripoSR on the GPU) is skipped when TripoSR isn't available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_to_mesh.core import image_to_mesh as i2m


def _triposr_available() -> bool:
    return i2m.is_available()


def test_is_image_file():
    assert i2m.is_image_file("a.png") is True
    assert i2m.is_image_file("a.JPG") is True
    assert i2m.is_image_file("a.webp") is True
    assert i2m.is_image_file("a.obj") is False
    assert i2m.is_image_file("a.ply") is False


def test_find_triposr_dir_env(monkeypatch, tmp_path: Path):
    # With a fake TRIPOSR_DIR containing run.py, discovery finds it.
    (tmp_path / "run.py").write_text("# fake")
    monkeypatch.setenv("TRIPOSR_DIR", str(tmp_path))
    assert i2m.find_triposr_dir() == tmp_path


def test_generate_mesh_unavailable_raises(monkeypatch, tmp_path: Path):
    # No TripoSR dir → clear RuntimeError, not a crash.
    monkeypatch.setenv("TRIPOSR_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(i2m, "find_triposr_dir", lambda: None)
    with pytest.raises(RuntimeError, match="TripoSR not found"):
        i2m.generate_mesh(tmp_path / "x.png", tmp_path / "out.obj")


def test_ingest_image_requires_triposr(monkeypatch, tmp_path: Path):
    """Ingest on an image input errors clearly when TripoSR is unavailable."""
    from slam_to_mesh.core.context import StageContext
    from slam_to_mesh.core.model import PipelineConfig, Stage
    from slam_to_mesh.core.pipeline import create_job
    from slam_to_mesh.core.stages import ingest as ingest_stage

    monkeypatch.setattr(ingest_stage, "is_available", lambda: False)
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # not a real image; won't be reached
    manifest = create_job(img, tmp_path / "job", config=PipelineConfig())
    ctx = StageContext(manifest=manifest)
    with pytest.raises(RuntimeError, match="TripoSR"):
        ingest_stage.run(ctx)
    assert manifest.results[Stage.INGEST].status.value == "failed"


@pytest.mark.skipif(not _triposr_available(), reason="TripoSR not installed")
def test_generate_mesh_end_to_end(tmp_path: Path):
    """Heavy: actually run TripoSR on the bundled example image."""
    triposr = i2m.find_triposr_dir()
    example = triposr / "examples" / "chair.png"
    if not example.exists():
        pytest.skip("no example image")
    out = tmp_path / "mesh.obj"
    i2m.generate_mesh(example, out, mc_resolution=128)
    assert out.exists()
    import trimesh

    m = trimesh.load(str(out), force="mesh")
    assert len(m.faces) > 0
