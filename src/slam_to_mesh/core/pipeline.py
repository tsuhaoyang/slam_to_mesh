"""Pipeline orchestration.

Maps each :class:`Stage` to its implementation and runs them in canonical order.
This orchestrator is deliberately UI-agnostic so it is shared verbatim by the
CLI and the (future) FastAPI service.

Resumability: :func:`run_pipeline` accepts a ``start`` stage; earlier stages'
artifacts are expected to already exist on disk (recorded in the manifest). The
manifest is saved after every stage so a crash leaves a resumable state.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Iterable, Optional

from .context import StageContext
from .meshio import compute_stats, load_mesh
from .model import (
    JobManifest,
    PipelineConfig,
    Stage,
    StageResult,
    StageStatus,
    stages_from,
)
from .stages import (
    bake,
    clean,
    decimate,
    export,
    ingest,
    project,
    qc,
    remesh,
    unwrap,
)

#: Stage -> implementation function.
STAGE_FUNCS: dict[Stage, Callable[[StageContext], StageResult]] = {
    Stage.INGEST: ingest.run,
    Stage.CLEAN: clean.run,
    Stage.DECIMATE: decimate.run,
    Stage.REMESH: remesh.run,
    Stage.PROJECT: project.run,
    Stage.UNWRAP: unwrap.run,
    Stage.BAKE: bake.run,
    Stage.EXPORT: export.run,
    Stage.QC: qc.run,
}


def create_job(
    input_path: str | Path,
    job_dir: str | Path,
    config: Optional[PipelineConfig] = None,
    job_id: Optional[str] = None,
) -> JobManifest:
    """Create (or initialize) a job manifest for *input_path*."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = JobManifest(
        job_id=job_id or uuid.uuid4().hex[:12],
        input_path=str(Path(input_path).resolve()),
        job_dir=str(job_dir.resolve()),
        config=config or PipelineConfig(),
    )
    manifest.save()
    return manifest


def run_pipeline(
    manifest: JobManifest,
    start: Stage = Stage.INGEST,
    stages: Optional[Iterable[Stage]] = None,
    on_stage: Optional[Callable[[StageResult], None]] = None,
) -> JobManifest:
    """Run the pipeline on *manifest*, from *start* (or an explicit stage list).

    A callback ``on_stage`` is invoked after each stage completes for progress
    reporting. The manifest is persisted after every stage.
    """
    ctx = StageContext(manifest=manifest)
    to_run = list(stages) if stages is not None else stages_from(start)

    for stage in to_run:
        fn = STAGE_FUNCS[stage]
        result = fn(ctx)
        manifest.save()
        if on_stage is not None:
            on_stage(result)
        if result.status == StageStatus.FAILED:
            break

    return manifest
