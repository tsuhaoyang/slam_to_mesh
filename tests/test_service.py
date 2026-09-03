"""FastAPI service tests using Starlette's TestClient.

The service normally runs jobs in a background thread pool. For deterministic
tests we:

* redirect ``JOBS_ROOT`` to a temp dir (so no ``service_jobs/`` litters the repo),
* replace the executor's ``submit`` with a synchronous call, so the pipeline
  finishes before the ``POST /jobs`` response returns.

A small fast config is patched onto the pipeline via a lightweight monkeypatch of
``create_job`` defaults is unnecessary because we drive small inputs; instead we
keep the mesh tiny and pass small ``target_faces`` form fields.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slam_to_mesh.service import app as svc


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Redirect job storage to a temp dir.
    monkeypatch.setattr(svc, "JOBS_ROOT", tmp_path / "service_jobs")

    # Run submitted work synchronously; return a future-like so callers that do
    # `.result()` (the LOD endpoints) work, and callers that ignore the return
    # (the pipeline job) also work.
    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    class _SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            return _ImmediateFuture(fn(*args, **kwargs))

    monkeypatch.setattr(svc, "_executor", _SyncExecutor())
    return TestClient(svc.app)


def test_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_backends_endpoint(client: TestClient):
    r = client.get("/backends")
    assert r.status_code == 200
    backends = r.json()["backends"]
    assert "pymeshlab_cpu" in backends


def test_create_job_rejects_unsupported_format(client: TestClient):
    r = client.post(
        "/jobs",
        files={"file": ("bad.txt", b"not a mesh", "text/plain")},
    )
    assert r.status_code == 400


def test_get_job_404_for_unknown(client: TestClient):
    r = client.get("/jobs/deadbeef")
    assert r.status_code == 404


def test_full_job_lifecycle(client: TestClient, slam_mesh_path: Path):
    data = slam_mesh_path.read_bytes()

    # Create + (synchronously) run the job with a small budget.
    r = client.post(
        "/jobs",
        files={"file": ("slam_in.ply", data, "application/octet-stream")},
        data={
            "target_faces": "800",
            "decimate_faces": "2000",
            "formats": "glb,obj",
            "bake": "false",
            "project": "true",
        },
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert job_id

    # Status should be completed (ran synchronously).
    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["stages"]["qc"]["status"] == "done"
    assert body["stages"]["bake"]["status"] == "skipped"
    assert body["stages"]["export"]["status"] == "done"

    # Download a single format.
    r = client.get(f"/jobs/{job_id}/download", params={"fmt": "glb"})
    assert r.status_code == 200
    assert len(r.content) > 0

    # Download the full zip bundle.
    r = client.get(f"/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(__import__("io").BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert any(n.startswith("model.") for n in names)
    assert "09_qc.json" in names


def test_download_missing_format_404(client: TestClient, slam_mesh_path: Path):
    data = slam_mesh_path.read_bytes()
    r = client.post(
        "/jobs",
        files={"file": ("slam_in.ply", data, "application/octet-stream")},
        data={"target_faces": "800", "decimate_faces": "2000", "formats": "obj"},
    )
    job_id = r.json()["job_id"]
    # No STL was exported.
    r = client.get(f"/jobs/{job_id}/download", params={"fmt": "stl"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Interactive LOD endpoints
# --------------------------------------------------------------------------- #


def _create_completed_job(client: TestClient, slam_mesh_path: Path) -> str:
    """Create + run a job (synchronous executor) and return its id."""
    r = client.post(
        "/jobs",
        files={"file": ("slam_in.ply", slam_mesh_path.read_bytes(), "application/octet-stream")},
        data={"target_faces": "800", "decimate_faces": "2000", "formats": "glb,obj"},
    )
    assert r.status_code == 202
    return r.json()["job_id"]


def test_lod_build_and_cache(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)

    r = client.post(f"/jobs/{job_id}/lod", json={"target_faces": 600})
    assert r.status_code == 200
    body = r.json()
    lod = body["lod"]
    assert lod["actual_faces"] > 0
    assert 0.0 <= lod["quad_ratio"] <= 1.0
    assert lod["baked"] is False
    assert body["glb_url"].endswith("/model.glb")

    # The glb is downloadable.
    r = client.get(body["glb_url"])
    assert r.status_code == 200
    assert len(r.content) > 0

    # Listed among the job's LODs.
    r = client.get(f"/jobs/{job_id}/lods")
    assert r.status_code == 200
    assert len(r.json()["lods"]) >= 1


def test_lod_by_ratio(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.post(f"/jobs/{job_id}/lod", json={"ratio": 0.1})
    assert r.status_code == 200
    assert r.json()["lod"]["actual_faces"] > 0


def test_lod_requires_target_or_ratio(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.post(f"/jobs/{job_id}/lod", json={})
    assert r.status_code == 422


def test_lod_unknown_job_404(client: TestClient):
    r = client.post("/jobs/nope/lod", json={"target_faces": 500})
    assert r.status_code == 404


def test_lod_glb_404_before_build(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.get(f"/jobs/{job_id}/lod/9999/model.glb")
    assert r.status_code == 404


def test_export_lod_zip(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.post(
        f"/jobs/{job_id}/export-lod",
        json={"target_faces": 600, "bake": False, "formats": ["glb", "obj"]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert any(n.startswith("model.") for n in names)


def test_ui_served(client: TestClient):
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "multi-view" in r.text
    r = client.get("/ui/app.js")
    assert r.status_code == 200


def test_get_job_exposes_ingest_faces_for_ui(client: TestClient, slam_mesh_path: Path):
    """The frontend maps % → target faces using ingest metrics.faces."""
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    ingest = body["stages"]["ingest"]
    assert ingest["metrics"].get("faces", 0) > 0
    # Frontend gating fields.
    assert body["input_kind"] == "mesh"
    assert body["has_pointcloud"] is False
    assert body["input_faces"] > 0


def test_source_glb_served(client: TestClient, slam_mesh_path: Path):
    """The original mesh is available as glb for the surface viewer."""
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.get(f"/jobs/{job_id}/source.glb")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_lod_wire_quads(client: TestClient, slam_mesh_path: Path):
    """wire.json returns quad-polygon edges (positions + edge indices)."""
    job_id = _create_completed_job(client, slam_mesh_path)
    built = client.post(f"/jobs/{job_id}/lod", json={"target_faces": 600}).json()
    tf = built["lod"]["target_faces"]
    r = client.get(f"/jobs/{job_id}/lod/{tf}/wire.json")
    assert r.status_code == 200
    data = r.json()
    assert len(data["positions"]) % 3 == 0
    assert len(data["edges"]) % 2 == 0
    assert len(data["positions"]) > 0
    assert len(data["edges"]) > 0


def test_lod_wire_404_before_build(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.get(f"/jobs/{job_id}/lod/9999/wire.json")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Triangle LOD endpoints
# --------------------------------------------------------------------------- #


def test_tri_lod_build_and_glb(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.post(f"/jobs/{job_id}/tri-lod", json={"target_faces": 500})
    assert r.status_code == 200
    lod = r.json()["lod"]
    assert lod["actual_faces"] > 0
    # QEM hits the exact target.
    assert lod["actual_faces"] == 500
    r = client.get(r.json()["glb_url"])
    assert r.status_code == 200
    assert len(r.content) > 0


def test_tri_lod_requires_target_or_ratio(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.post(f"/jobs/{job_id}/tri-lod", json={})
    assert r.status_code == 422


def test_tri_lod_glb_404_before_build(client: TestClient, slam_mesh_path: Path):
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.get(f"/jobs/{job_id}/tri-lod/9999/model.glb")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Point-cloud endpoints
# --------------------------------------------------------------------------- #


def _point_cloud_ply(tmp_path: Path) -> Path:
    import numpy as np
    import open3d as o3d

    n = 6000
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    pcd.normals = o3d.utility.Vector3dVector(v)
    p = tmp_path / "cloud.ply"
    o3d.io.write_point_cloud(str(p), pcd)
    return p


def _create_pointcloud_job(client: TestClient, ply: Path) -> str:
    r = client.post(
        "/jobs",
        files={"file": ("cloud.ply", ply.read_bytes(), "application/octet-stream")},
        data={"target_faces": "800", "decimate_faces": "3000", "formats": "glb"},
    )
    assert r.status_code == 202
    return r.json()["job_id"]


def test_pointcloud_json_and_downsample(client: TestClient, tmp_path: Path):
    ply = _point_cloud_ply(tmp_path)
    job_id = _create_pointcloud_job(client, ply)

    # Original points are available.
    r = client.get(f"/jobs/{job_id}/pointcloud.json")
    assert r.status_code == 200
    pos = r.json()["positions"]
    assert len(pos) % 3 == 0
    original_count = len(pos) // 3
    assert original_count > 0

    # Downsample by target points.
    r = client.post(
        f"/jobs/{job_id}/pointcloud/downsample", json={"target_points": 800}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["points_after"] < body["stats"]["points_before"]
    assert len(body["positions"]) // 3 == body["stats"]["points_after"]


def test_pointcloud_downsample_requires_control(client: TestClient, tmp_path: Path):
    ply = _point_cloud_ply(tmp_path)
    job_id = _create_pointcloud_job(client, ply)
    r = client.post(f"/jobs/{job_id}/pointcloud/downsample", json={})
    assert r.status_code == 422


def test_pointcloud_json_404_for_mesh_job(client: TestClient, slam_mesh_path: Path):
    """A mesh-input job has no retained point cloud."""
    job_id = _create_completed_job(client, slam_mesh_path)
    r = client.get(f"/jobs/{job_id}/pointcloud.json")
    assert r.status_code == 404


def test_pointcloud_generate_for_mesh_job(client: TestClient, slam_mesh_path: Path):
    """A mesh job can generate a point cloud from its surface (opt-in)."""
    job_id = _create_completed_job(client, slam_mesh_path)
    # No point cloud initially.
    assert client.get(f"/jobs/{job_id}/pointcloud.json").status_code == 404

    r = client.post(f"/jobs/{job_id}/pointcloud/generate", json={"n": 2000})
    assert r.status_code == 200
    assert r.json()["stats"]["sampled_points"] > 0

    # Now available + job flips has_pointcloud.
    assert client.get(f"/jobs/{job_id}/pointcloud.json").status_code == 200
    assert client.get(f"/jobs/{job_id}").json()["has_pointcloud"] is True


def test_pointcloud_download(client: TestClient, tmp_path: Path):
    ply = _point_cloud_ply(tmp_path)
    job_id = _create_pointcloud_job(client, ply)
    r = client.get(f"/jobs/{job_id}/pointcloud/download")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_capabilities_endpoint(client: TestClient):
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert "image_input" in body
    assert "image_backends" in body
    assert ".png" in body["image_exts"]
    assert "photogrammetry" in body
    assert "photogrammetry_backends" in body
    assert ".mp4" in body["video_exts"]
    assert isinstance(body["backends"], list)


def test_zip_upload_rejected_when_photogrammetry_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """A .zip / video upload returns 503 when no photogrammetry backend exists."""
    import slam_to_mesh.backends.photogrammetry as pg

    monkeypatch.setattr(pg, "any_available", lambda: False)
    r = client.post(
        "/jobs",
        files={"file": ("imgs.zip", b"PK\x03\x04not-a-real-zip", "application/zip")},
    )
    assert r.status_code == 503


def test_image_upload_rejected_when_triposr_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Image upload returns 503 when no image backend is available."""
    import slam_to_mesh.core.image_to_mesh as i2m

    monkeypatch.setattr(i2m, "is_available", lambda backend_id=None: False)
    r = client.post(
        "/jobs",
        files={"file": ("pic.png", b"not-a-real-image", "image/png")},
    )
    assert r.status_code == 503
