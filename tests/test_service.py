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

    # Run submitted jobs synchronously so status is 'completed' right away.
    class _SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

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
