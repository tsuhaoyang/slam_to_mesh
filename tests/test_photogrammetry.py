"""Tests for the photogrammetry backend registry + COLMAP backend.

COLMAP is typically absent (CI, and here until installed), so tests focus on
graceful unavailability + registry behavior. The real reconstruction e2e is
skipif-gated on COLMAP being present."""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_to_mesh.backends import photogrammetry as pg
from slam_to_mesh.backends.photogrammetry_colmap import ColmapBackend


def _colmap_available() -> bool:
    return ColmapBackend().is_available()


def test_colmap_registered():
    assert "colmap" in pg._REGISTRY


def test_registry_availability_matches_binary():
    has = _colmap_available()
    assert pg.any_available() == has
    assert ("colmap" in pg.available_backends()) == has
    assert (pg.get_backend("colmap") is not None) == has
    if not has:
        assert pg.get_backend() is None


def test_colmap_reconstruct_unavailable_raises(monkeypatch, tmp_path: Path):
    """When COLMAP is absent, reconstruct raises a clear error."""
    monkeypatch.setattr(
        "slam_to_mesh.backends.photogrammetry_colmap.find_colmap_bin", lambda: None
    )
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.png").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="COLMAP not found"):
        ColmapBackend().reconstruct(tmp_path, tmp_path / "out.ply")


def test_colmap_reconstruct_too_few_images(monkeypatch, tmp_path: Path):
    """With a (pretended) available COLMAP but too few images → clear error."""
    monkeypatch.setattr(
        "slam_to_mesh.backends.photogrammetry_colmap.find_colmap_bin",
        lambda: "/usr/bin/true",
    )
    (tmp_path / "only.png").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="needs several images"):
        ColmapBackend().reconstruct(tmp_path, tmp_path / "out.ply")


@pytest.mark.skipif(not _colmap_available(), reason="COLMAP not installed")
def test_colmap_reconstruct_end_to_end(tmp_path: Path):  # pragma: no cover
    pytest.skip("real image set required; wire up when validating with COLMAP")


# --------------------------------------------------------------------------- #
# Ingest routing (zip / video → photogrammetry)
# --------------------------------------------------------------------------- #


def _make_image_zip(zip_path: Path, n: int = 3) -> None:
    import zipfile

    import numpy as np
    from PIL import Image

    tmp = zip_path.parent
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(n):
            p = tmp / f"img{i}.png"
            Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(p)
            zf.write(p, f"img{i}.png")


def test_ingest_zip_routes_to_photogrammetry(monkeypatch, tmp_path: Path):
    """A .zip is detected as an image set, unzipped, routed to photogrammetry.

    With COLMAP absent (the common case) it errors clearly — which confirms the
    routing + gating without needing COLMAP installed.
    """
    from slam_to_mesh.backends import photogrammetry as pg
    from slam_to_mesh.core.context import StageContext
    from slam_to_mesh.core.model import PipelineConfig
    from slam_to_mesh.core.pipeline import create_job
    from slam_to_mesh.core.stages import ingest as ingest_stage

    monkeypatch.setattr(pg, "get_backend", lambda backend_id=None: None)
    zp = tmp_path / "imgs.zip"
    _make_image_zip(zp, n=3)
    manifest = create_job(zp, tmp_path / "job", config=PipelineConfig())
    ctx = StageContext(manifest=manifest)
    with pytest.raises(RuntimeError, match="photogrammetry backend"):
        ingest_stage.run(ctx)
    assert manifest.input_kind == "images"
    assert (Path(manifest.job_dir) / "00_images").is_dir()
