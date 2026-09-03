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
from pydantic import BaseModel

from ..backends.remesh import available_backends
from ..core.lod import build_lod, build_tri_lod
from ..core.model import (
    JobManifest,
    PipelineConfig,
    Stage,
    StageStatus,
)
from ..core.pipeline import create_job, run_pipeline
from ..core.pointcloud import (
    points_to_positions,
    sample_points_from_mesh,
    voxel_downsample,
)

#: Root directory for all service jobs. Configurable via env in real deploys.
JOBS_ROOT = Path("service_jobs")

app = FastAPI(title="slam_to_mesh", version="0.1.0")

# Serve the interactive decimation UI (Three.js) as static files.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

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


@app.get("/capabilities")
def capabilities() -> dict:
    """Feature availability for the UI (e.g. whether image input is possible)."""
    from ..backends.image3d import available_backends as image_backends
    from ..backends.photogrammetry import available_backends as photo_backends
    from ..core.frames import VIDEO_EXTS
    from ..core.image_to_mesh import IMAGE_EXTS

    img_backends = image_backends()
    photogrammetry = photo_backends()
    return {
        "image_input": bool(img_backends),
        "image_backends": img_backends,
        "image_exts": sorted(IMAGE_EXTS),
        "photogrammetry": bool(photogrammetry),
        "photogrammetry_backends": photogrammetry,
        "video_exts": sorted(VIDEO_EXTS),
        "backends": available_backends(),
    }


@app.post("/jobs", status_code=202)
async def create_and_run_job(
    file: UploadFile = File(...),
    target_faces: int = Form(20000),
    decimate_faces: Optional[int] = Form(None),
    backend: str = Form("quadriflow_cpu"),
    bake: bool = Form(False),
    formats: str = Form("glb,obj"),
    project: bool = Form(True),
    image_backend: Optional[str] = Form(None),
    frames: int = Form(40),
) -> dict:
    """Accept a mesh / point-cloud / image / image-set / video upload."""
    suffix = (Path(file.filename or "input.ply").suffix or ".ply").lower()
    mesh_exts = {".ply", ".obj", ".stl", ".off", ".glb"}
    pointcloud_exts = {".pcd", ".xyz", ".xyzn", ".pts"}
    from ..backends.photogrammetry import any_available as photogrammetry_available
    from ..core.frames import VIDEO_EXTS
    from ..core.image_to_mesh import IMAGE_EXTS
    from ..core.image_to_mesh import is_available as image3d_available

    if suffix in IMAGE_EXTS:
        if not image3d_available(image_backend):
            raise HTTPException(
                503,
                "image input needs an image-to-3D backend "
                f"(requested={image_backend or 'any'}), which is not available",
            )
    elif suffix == ".zip" or suffix in VIDEO_EXTS:
        if not photogrammetry_available():
            raise HTTPException(
                503,
                "image-set / video input needs a photogrammetry backend "
                "(COLMAP), which is not available on this server",
            )
    elif suffix not in mesh_exts | pointcloud_exts:
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
    if image_backend:
        config.image_backend = image_backend
    config.video_frames = int(frames)

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


@app.get("/jobs/{job_id}/source.glb")
def get_source_glb(job_id: str):
    """Return the original (pre-remesh) mesh as glb for side-by-side preview.

    Prefers the ingest artifact (the normalized original), falling back to
    clean, then the raw input. Converted to glb via trimesh and cached on disk
    as ``source.glb``.
    """
    manifest = _require_manifest(job_id)
    jd = _job_dir(job_id)
    cached = jd / "source.glb"
    if not cached.exists():
        src = None
        for stage in (Stage.INGEST, Stage.CLEAN):
            p = manifest.artifact_path(stage)
            if p is not None and p.exists():
                src = p
                break
        if src is None:
            src = Path(manifest.input_path)
        if not src.exists():
            raise HTTPException(404, "no source mesh available")
        import trimesh

        trimesh.load(str(src), process=False, force="mesh").export(str(cached))
    return FileResponse(str(cached), filename="source.glb", media_type="model/gltf-binary")


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
        "input_kind": manifest.input_kind,
        "has_pointcloud": manifest.has_pointcloud,
        "input_faces": manifest.input_stats.faces if manifest.input_stats else None,
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


# --------------------------------------------------------------------------- #
# Interactive LOD (decimation) endpoints
# --------------------------------------------------------------------------- #


class LodRequest(BaseModel):
    """Body for POST /jobs/{id}/lod."""

    target_faces: int | None = None
    ratio: float | None = None
    bake: bool = False


class ExportLodRequest(BaseModel):
    """Body for POST /jobs/{id}/export-lod."""

    target_faces: int | None = None
    ratio: float | None = None
    bake: bool = True
    formats: list[str] = ["glb", "usd"]


def _require_manifest(job_id: str) -> JobManifest:
    mp = _manifest_path(job_id)
    if not mp.exists():
        raise HTTPException(404, "job not found")
    return JobManifest.load(mp)


def _lod_source_ready(manifest: JobManifest) -> bool:
    """A LOD needs at least one triangle source artifact (ingest onward)."""
    for stage in (Stage.DECIMATE, Stage.CLEAN, Stage.INGEST):
        p = manifest.artifact_path(stage)
        if p is not None and p.exists():
            return True
    return Path(manifest.input_path).exists()


def _build_lod_sync(job_id: str, target_faces, ratio, bake: bool):
    """Run build_lod in the CPU pool and persist to the on-disk manifest."""
    manifest = JobManifest.load(_manifest_path(job_id))
    result = build_lod(manifest, target_faces=target_faces, ratio=ratio, bake=bake)
    return result


def _export_lod_extra_formats(lod_dir: Path, formats: list[str]) -> None:
    """Write additional formats (currently USD) for a built LOD directory.

    glb/obj are already produced by build_lod. USD is generated from the LOD's
    UV mesh (unwrap.obj if present, else model.obj) with baked textures bound.
    """
    import trimesh

    from ..core.stages.export import _export_usd, _load_uv_mesh_with_texture

    wants_usd = any(f.lower() in {"usd", "usdc", "usda"} for f in formats)
    if not wants_usd:
        return

    mesh_src = lod_dir / "unwrap.obj"
    if not mesh_src.exists():
        mesh_src = lod_dir / "model.obj"
    if not mesh_src.exists():
        return

    color = lod_dir / "bake_color.png"
    normal = lod_dir / "bake_normal.png"
    color = color if color.exists() else None
    normal = normal if normal.exists() else None

    mesh = _load_uv_mesh_with_texture(mesh_src, color, normal)
    if not isinstance(mesh, trimesh.Trimesh):
        return
    _export_usd(mesh, lod_dir / "model.usd", color, normal)


@app.post("/jobs/{job_id}/lod")
def create_lod(job_id: str, req: LodRequest) -> dict:
    """Build (or return cached) a quad LOD at a target face count / ratio."""
    manifest = _require_manifest(job_id)
    if not _lod_source_ready(manifest):
        raise HTTPException(409, "job has no mesh to remesh yet")
    if req.target_faces is None and req.ratio is None:
        raise HTTPException(422, "provide target_faces or ratio")

    # Run synchronously in the CPU pool (spec: ~1-2 s per change).
    fut = _executor.submit(
        _build_lod_sync, job_id, req.target_faces, req.ratio, req.bake
    )
    result = fut.result()

    return {
        "job_id": job_id,
        "lod": result.model_dump(),
        "glb_url": f"/jobs/{job_id}/lod/{result.target_faces}/model.glb"
        + ("?baked=1" if result.baked else ""),
    }


@app.get("/jobs/{job_id}/lods")
def list_lods(job_id: str) -> dict:
    """All LODs built so far for this job."""
    manifest = _require_manifest(job_id)
    return {"job_id": job_id, "lods": {k: v.model_dump() for k, v in manifest.lods.items()}}


@app.get("/jobs/{job_id}/lod/{target_faces}/model.glb")
def get_lod_glb(job_id: str, target_faces: int, baked: bool = False):
    """Return the glb for a previously built LOD."""
    from ..core.lod import _cache_key

    manifest = _require_manifest(job_id)
    key = _cache_key(target_faces, baked)
    lod = manifest.lods.get(key)
    if lod is None or not lod.glb:
        raise HTTPException(404, "LOD not built; POST /jobs/{id}/lod first")
    path = _job_dir(job_id) / lod.glb
    if not path.exists():
        raise HTTPException(404, "LOD glb missing on disk")
    return FileResponse(str(path), filename="model.glb", media_type="model/gltf-binary")


@app.get("/jobs/{job_id}/lod/{target_faces}/wire.json")
def get_lod_wire(job_id: str, target_faces: int, baked: bool = False) -> dict:
    """Return the LOD's **quad** wireframe (positions + polygon edges).

    Reads the LOD's polygon OBJ (which preserves 4-vertex faces) and returns a
    compact `{positions: [x,y,z,...], edges: [i,j,...]}` payload the frontend
    draws as LineSegments — so the mesh shows real quad edges, not the triangle
    diagonals a glb would carry.
    """
    from ..core.lod import _cache_key

    manifest = _require_manifest(job_id)
    key = _cache_key(target_faces, baked)
    lod = manifest.lods.get(key)
    if lod is None or not lod.obj:
        raise HTTPException(404, "LOD not built; POST /jobs/{id}/lod first")
    # Prefer the projected polygon OBJ (kept next to model.obj in the LOD dir).
    lod_dir = (_job_dir(job_id) / lod.obj).parent
    poly = lod_dir / "project.obj"
    if not poly.exists():
        poly = _job_dir(job_id) / lod.obj
    if not poly.exists():
        raise HTTPException(404, "LOD mesh missing on disk")

    positions, edges = _obj_wire(poly)
    return {"positions": positions, "edges": edges}


def _obj_wire(path: Path) -> tuple[list[float], list[int]]:
    """Parse an OBJ into a flat vertex list and unique polygon-edge index list."""
    verts: list[float] = []
    edge_set: set[tuple[int, int]] = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.extend((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                n = len(idx)
                for i in range(n):
                    a, b = idx[i], idx[(i + 1) % n]
                    edge_set.add((a, b) if a < b else (b, a))
    edges: list[int] = []
    for a, b in edge_set:
        edges.extend((a, b))
    return verts, edges


@app.post("/jobs/{job_id}/export-lod")
def export_lod(job_id: str, req: ExportLodRequest):
    """Export a chosen LOD to the requested formats (glb/usd/obj).

    Builds the LOD with baking on (default) so exported formats carry textures,
    then zips the LOD's exported artifacts for download.
    """
    manifest = _require_manifest(job_id)
    if not _lod_source_ready(manifest):
        raise HTTPException(409, "job has no mesh to remesh yet")

    fut = _executor.submit(
        _build_lod_sync, job_id, req.target_faces, req.ratio, req.bake
    )
    result = fut.result()

    lod_dir = (_job_dir(job_id) / result.glb).parent

    # build_lod writes obj + glb. Produce any additionally requested formats
    # (e.g. usd) from the LOD mesh + baked textures.
    _export_lod_extra_formats(lod_dir, req.formats)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in ("model.*", "bake_*.png"):
            for f in lod_dir.glob(pattern):
                zf.write(f, arcname=f.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{job_id}_lod{result.actual_faces}.zip"'
            )
        },
    )


# --------------------------------------------------------------------------- #
# Triangle LOD (QEM) endpoints
# --------------------------------------------------------------------------- #


class TriLodRequest(BaseModel):
    target_faces: int | None = None
    ratio: float | None = None


def _build_tri_lod_sync(job_id: str, target_faces, ratio):
    manifest = JobManifest.load(_manifest_path(job_id))
    return build_tri_lod(manifest, target_faces=target_faces, ratio=ratio)


@app.post("/jobs/{job_id}/tri-lod")
def create_tri_lod(job_id: str, req: TriLodRequest) -> dict:
    """Build (or return cached) a triangle QEM LOD at a target face count."""
    manifest = _require_manifest(job_id)
    if not _lod_source_ready(manifest):
        raise HTTPException(409, "job has no mesh to decimate yet")
    if req.target_faces is None and req.ratio is None:
        raise HTTPException(422, "provide target_faces or ratio")
    result = _executor.submit(
        _build_tri_lod_sync, job_id, req.target_faces, req.ratio
    ).result()
    return {
        "job_id": job_id,
        "lod": result.model_dump(),
        "glb_url": f"/jobs/{job_id}/tri-lod/{result.target_faces}/model.glb",
    }


@app.get("/jobs/{job_id}/tri-lod/{target_faces}/model.glb")
def get_tri_lod_glb(job_id: str, target_faces: int):
    from ..core.lod import tri_cache_key

    manifest = _require_manifest(job_id)
    lod = manifest.tri_lods.get(tri_cache_key(target_faces))
    if lod is None or not lod.glb:
        raise HTTPException(404, "tri-LOD not built; POST /jobs/{id}/tri-lod first")
    path = _job_dir(job_id) / lod.glb
    if not path.exists():
        raise HTTPException(404, "tri-LOD glb missing on disk")
    return FileResponse(str(path), filename="model.glb", media_type="model/gltf-binary")


# --------------------------------------------------------------------------- #
# Point-cloud endpoints
# --------------------------------------------------------------------------- #


class DownsampleRequest(BaseModel):
    voxel_size: float | None = None
    target_points: int | None = None


def _points_file(job_id: str) -> Path | None:
    """The original/retained point cloud for a job, if any."""
    p = _job_dir(job_id) / "00_points.ply"
    return p if p.exists() else None


@app.get("/jobs/{job_id}/pointcloud.json")
def get_pointcloud_json(job_id: str) -> dict:
    """Return the original point cloud as flat positions for Three.js Points."""
    _require_manifest(job_id)
    pts = _points_file(job_id)
    if pts is None:
        raise HTTPException(404, "no point cloud for this job")
    return {"positions": points_to_positions(pts)}


@app.post("/jobs/{job_id}/pointcloud/downsample")
def downsample_pointcloud(job_id: str, req: DownsampleRequest) -> dict:
    """Voxel-downsample the job's point cloud; return reduced positions + stats."""
    _require_manifest(job_id)
    pts = _points_file(job_id)
    if pts is None:
        raise HTTPException(404, "no point cloud for this job")
    if req.voxel_size is None and req.target_points is None:
        raise HTTPException(422, "provide voxel_size or target_points")

    out = _job_dir(job_id) / "00_points_downsampled.ply"

    def _work():
        return voxel_downsample(
            pts, out, voxel_size=req.voxel_size, target_points=req.target_points
        )

    stats = _executor.submit(_work).result()
    return {
        "job_id": job_id,
        "stats": stats,
        "positions": points_to_positions(out),
    }


class GeneratePointsRequest(BaseModel):
    n: int = 50000


@app.post("/jobs/{job_id}/pointcloud/generate")
def generate_pointcloud(job_id: str, req: GeneratePointsRequest) -> dict:
    """Sample a point cloud from a mesh job's surface (opt-in).

    Lets a mesh-input job gain a point-cloud representation for the viewer /
    downsampling. Writes ``00_points.ply`` and flips ``has_pointcloud``.
    """
    manifest = _require_manifest(job_id)
    # Prefer the ingest (normalized) triangle mesh as the sampling source.
    src = manifest.artifact_path(Stage.INGEST)
    if src is None or not src.exists():
        raise HTTPException(409, "no mesh available to sample from")

    out = _job_dir(job_id) / "00_points.ply"

    def _work():
        stats = sample_points_from_mesh(src, out, n=req.n)
        m = JobManifest.load(_manifest_path(job_id))
        m.has_pointcloud = True
        m.save()
        return stats

    stats = _executor.submit(_work).result()
    return {
        "job_id": job_id,
        "stats": stats,
        "positions": points_to_positions(out),
    }


@app.get("/jobs/{job_id}/pointcloud/download")
def download_pointcloud(job_id: str, downsampled: bool = False):
    """Download the point cloud as a PLY (original or downsampled)."""
    _require_manifest(job_id)
    name = "00_points_downsampled.ply" if downsampled else "00_points.ply"
    path = _job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(404, "point cloud not available")
    return FileResponse(str(path), filename=name, media_type="application/octet-stream")
