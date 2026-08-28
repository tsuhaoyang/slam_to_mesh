"""FastAPI service for slam_to_mesh.

Endpoints:

* ``POST /jobs``                 — upload a mesh (+ optional params); creates a
  job and runs the pipeline asynchronously in a background thread. Returns the
  job id immediately.
* ``GET  /jobs/{job_id}``        — job status: per-stage results from the
  manifest, plus overall state.
* ``GET  /jobs/{job_id}/download`` — download the result (single glb, or a zip of
  all final artifacts).
* ``GET  /backends``             — list available remesh backends.
* ``GET  /healthz``              — liveness probe.

The pipeline is synchronous and CPU-bound, so each job runs in a thread pool to
avoid blocking the event loop. Job state lives on disk (``job.json``), so status
survives process restarts and is shared with the CLI.
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..backends.remesh import available_backends
from ..core.model import (
    JobManifest,
    PipelineConfig,
    Stage,
    StageStatus,
)
from ..core.pipeline import create_job, run_pipeline

#: Root directory for all service jobs. Configurable via env in real deploys.
JOBS_ROOT = Path("service_jobs")

app = FastAPI(title="slam_to_mesh", version="0.1.0")

# Single-worker pool keeps CPU jobs serialized; raise for a bigger box/GPU.
_executor = ThreadPoolExecutor(max_workers=1)


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _manifest_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _run_job(job_id: str) -> None:
    """Background worker: load the manifest and run the full pipeline."""
    manifest = JobManifest.load(_manifest_path(job_id))
    run_pipeline(manifest, start=Stage.INGEST)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/backends")
def backends() -> dict:
    return {"backends": available_backends()}


@app.post("/jobs", status_code=202)
async def create_and_run_job(
    file: UploadFile = File(...),
    target_faces: int = Form(20000),
    decimate_faces: Optional[int] = Form(None),
    backend: str = Form("quadriflow_cpu"),
    bake: bool = Form(False),
    formats: str = Form("glb,obj"),
    project: bool = Form(True),
) -> dict:
    """Accept a mesh upload, create a job, and start processing in background."""
    suffix = Path(file.filename or "input.ply").suffix or ".ply"
    if suffix.lower() not in {".ply", ".obj", ".stl", ".off", ".glb"}:
        raise HTTPException(400, f"Unsupported input format: {suffix}")

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)

    # Stage the uploaded file, then create the job around it.
    import uuid

    job_id = uuid.uuid4().hex[:12]
    jd = _job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    input_path = jd / f"input{suffix}"
    input_path.write_bytes(await file.read())

    config = PipelineConfig()
    config.remesh.target_faces = target_faces
    config.remesh.backend = backend
    config.decimate.target_faces = decimate_faces or int(target_faces * 2.5)
    config.bake.enabled = bake
    config.project.enabled = project
    config.export.formats = [f.strip() for f in formats.split(",") if f.strip()]

    create_job(input_path, jd, config=config, job_id=job_id)

    _executor.submit(_run_job, job_id)

    return {"job_id": job_id, "status": "accepted"}


def _overall_status(manifest: JobManifest) -> str:
    results = [manifest.results.get(s) for s in Stage]
    statuses = [r.status for r in results if r is not None]
    if any(s == StageStatus.FAILED for s in statuses):
        return "failed"
    if manifest.results.get(Stage.QC) and \
            manifest.results[Stage.QC].status == StageStatus.DONE:
        return "completed"
    if any(s == StageStatus.RUNNING for s in statuses) or statuses:
        return "running"
    return "pending"


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Return job status and per-stage results."""
    mp = _manifest_path(job_id)
    if not mp.exists():
        raise HTTPException(404, "job not found")
    manifest = JobManifest.load(mp)

    stages = {}
    for stage in Stage:
        res = manifest.results.get(stage)
        stages[stage.value] = {
            "status": res.status.value if res else "pending",
            "message": res.message if res else None,
            "metrics": res.metrics if res else {},
        }

    return {
        "job_id": job_id,
        "status": _overall_status(manifest),
        "stages": stages,
    }


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str, fmt: Optional[str] = None):
    """Download the result.

    * ``?fmt=glb`` (or obj/gltf/...) returns that single file.
    * no fmt returns a zip of all final ``model.*`` files plus textures and QC.
    """
    jd = _job_dir(job_id)
    mp = _manifest_path(job_id)
    if not mp.exists():
        raise HTTPException(404, "job not found")
    manifest = JobManifest.load(mp)
    if _overall_status(manifest) != "completed":
        raise HTTPException(409, "job not completed yet")

    if fmt:
        target = jd / f"model.{fmt.lower()}"
        if not target.exists():
            raise HTTPException(404, f"no output in format '{fmt}'")
        return FileResponse(str(target), filename=target.name)

    # Zip all final artifacts.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in ("model.*", "07_bake_*.png", "09_qc.json"):
            for f in jd.glob(pattern):
                zf.write(f, arcname=f.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )
