"""Tests for image → mesh backend registry + TripoSR integration.

Availability / routing / gating tests always run. The heavy end-to-end test
(actually invoking TripoSR on the GPU) is skipped when TripoSR isn't available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_to_mesh.backends import image3d
from slam_to_mesh.backends.image3d_triposr import TripoSRBackend, find_triposr_dir
from slam_to_mesh.core import image_to_mesh as i2m


def _triposr_available() -> bool:
    return TripoSRBackend().is_available()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_triposr_registered():
    assert "triposr" in image3d._REGISTRY


def test_registry_get_and_available():
    has = _triposr_available()
    assert (image3d.get_backend("triposr") is not None) == has
    assert ("triposr" in image3d.available_backends()) == has
    assert image3d.any_available() == has


def test_registry_unknown_backend_falls_back_to_available():
    # An unknown id returns the first available backend (or None if none).
    b = image3d.get_backend("does_not_exist")
    if _triposr_available():
        assert b is not None and b.id == "triposr"
    else:
        assert b is None


# --------------------------------------------------------------------------- #
# Compat shim
# --------------------------------------------------------------------------- #


def test_is_image_file():
    assert i2m.is_image_file("a.png") is True
    assert i2m.is_image_file("a.JPG") is True
    assert i2m.is_image_file("a.obj") is False
    assert i2m.is_image_file("a.ply") is False


def test_generate_mesh_none_available_raises(monkeypatch, tmp_path: Path):
    # Force the registry to report nothing available → clear error.
    monkeypatch.setattr(image3d, "get_backend", lambda backend_id=None: None)
    with pytest.raises(RuntimeError, match="no image-to-3D backend"):
        i2m.generate_mesh(tmp_path / "x.png", tmp_path / "out.obj")


def test_find_triposr_dir_env(monkeypatch, tmp_path: Path):
    (tmp_path / "run.py").write_text("# fake")
    monkeypatch.setenv("TRIPOSR_DIR", str(tmp_path))
    assert find_triposr_dir() == tmp_path


# --------------------------------------------------------------------------- #
# Ingest routing
# --------------------------------------------------------------------------- #


def test_ingest_image_requires_backend(monkeypatch, tmp_path: Path):
    """Ingest on an image errors clearly when no image backend is available."""
    from slam_to_mesh.core.context import StageContext
    from slam_to_mesh.core.model import PipelineConfig, Stage
    from slam_to_mesh.core.pipeline import create_job
    from slam_to_mesh.core.stages import ingest as ingest_stage

    monkeypatch.setattr(ingest_stage, "is_available", lambda backend_id=None: False)
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest = create_job(img, tmp_path / "job", config=PipelineConfig())
    ctx = StageContext(manifest=manifest)
    with pytest.raises(RuntimeError, match="image-to-3D backend"):
        ingest_stage.run(ctx)
    assert manifest.results[Stage.INGEST].status.value == "failed"


# --------------------------------------------------------------------------- #
# Heavy end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _triposr_available(), reason="TripoSR not installed")
def test_generate_mesh_end_to_end(tmp_path: Path):
    triposr = find_triposr_dir()
    example = triposr / "examples" / "chair.png"
    if not example.exists():
        pytest.skip("no example image")
    out = tmp_path / "mesh.obj"
    i2m.generate_mesh(example, out, backend_id="triposr")
    assert out.exists()
    import trimesh

    m = trimesh.load(str(out), force="mesh")
    assert len(m.faces) > 0
