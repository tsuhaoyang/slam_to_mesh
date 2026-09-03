"""Pipeline data model.

Everything the pipeline needs to run, resume, and report is captured here as
pydantic models. The :class:`JobManifest` is the single source of truth that is
serialized to ``job.json`` on disk, enabling:

* **Resumability** — each completed stage records its parameters and the path to
  its output artifact, so a user can tweak one stage and re-run from there.
* **Reproducibility** — the exact config used for every stage is persisted.
* **Shared core** — both the CLI and the (future) FastAPI service construct and
  consume the same manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Stage(str, Enum):
    """Ordered pipeline stages.

    The string values double as artifact filename prefixes (e.g.
    ``01_ingest``). Order is defined by :data:`STAGE_ORDER`.
    """

    INGEST = "ingest"
    CLEAN = "clean"
    DECIMATE = "decimate"
    REMESH = "remesh"
    PROJECT = "project"
    UNWRAP = "unwrap"
    BAKE = "bake"
    EXPORT = "export"
    QC = "qc"


#: Canonical execution order. Used for "resume-from" and dependency resolution.
STAGE_ORDER: list[Stage] = [
    Stage.INGEST,
    Stage.CLEAN,
    Stage.DECIMATE,
    Stage.REMESH,
    Stage.PROJECT,
    Stage.UNWRAP,
    Stage.BAKE,
    Stage.EXPORT,
    Stage.QC,
]


def stage_index(stage: Stage) -> int:
    """Return the position of *stage* in the canonical order."""
    return STAGE_ORDER.index(stage)


def stages_from(stage: Stage) -> list[Stage]:
    """Return *stage* and all stages that follow it, in order."""
    return STAGE_ORDER[stage_index(stage) :]


# --------------------------------------------------------------------------- #
# Per-stage configuration
# --------------------------------------------------------------------------- #


class CleanConfig(BaseModel):
    """Stage 2: cleaning parameters."""

    remove_islands: bool = True
    #: Components whose face count is below this fraction of the largest
    #: component are dropped as floating debris.
    min_component_face_ratio: float = 0.01
    fill_holes: bool = True
    #: Maximum hole boundary edge count to fill; larger holes are left alone.
    max_hole_edges: int = 300
    #: Max fill/repair passes. Fragmented SLAM meshes often need a repair pass
    #: to unblock hole closing, so we retry closing after re-repairing until no
    #: further boundary edges are closed (or this cap is hit).
    max_hole_fill_iterations: int = 3
    fix_non_manifold: bool = True
    unify_normals: bool = True


class DecimateConfig(BaseModel):
    """Stage 3: QEM decimation parameters."""

    #: Target absolute face count after decimation (pre-remesh budget).
    target_faces: int = 50000
    preserve_boundary: bool = True
    preserve_normals: bool = True
    #: Quality threshold for quadric edge collapse (0..1, higher = better).
    quality_threshold: float = 0.3


class RemeshConfig(BaseModel):
    """Stage 4: quad remesh parameters."""

    #: Backend id, e.g. "quadriflow_cpu", "instant_meshes_cpu", "gpu".
    backend: str = "quadriflow_cpu"
    #: Target face count for the quad mesh (final display budget).
    target_faces: int = 20000
    #: Prefer quad output where the backend supports it.
    quads: bool = True
    #: Preserve sharp features / hard edges during remeshing.
    preserve_sharp: bool = True
    #: Optional path to a feature-line file the user can supply to guide edge flow.
    feature_lines: Path | None = None


class ProjectConfig(BaseModel):
    """Stage 5: projection back onto the original high-res surface."""

    enabled: bool = True
    #: Max distance (as fraction of bbox diagonal) a vertex may move to snap.
    max_snap_ratio: float = 0.02


class UnwrapConfig(BaseModel):
    """Stage 6: UV unwrap parameters (xatlas)."""

    #: Padding between charts in texels.
    padding: int = 4
    #: Target texture resolution used for chart packing.
    resolution: int = 2048


class BakeConfig(BaseModel):
    """Stage 7: texture bake parameters (optional)."""

    enabled: bool = False
    bake_color: bool = True
    bake_normal: bool = True
    texture_size: int = 2048
    #: Ray cast max distance as fraction of bbox diagonal.
    max_ray_ratio: float = 0.05


class ExportConfig(BaseModel):
    """Stage 8: export parameters."""

    #: Output formats to write; subset of {"usd", "usdc", "gltf", "glb", "obj"}.
    formats: list[str] = Field(default_factory=lambda: ["glb", "obj"])
    #: Uniform scale applied on export (e.g. to convert SLAM units to meters).
    scale: float = 1.0


class PipelineConfig(BaseModel):
    """Aggregate configuration for a full pipeline run."""

    clean: CleanConfig = Field(default_factory=CleanConfig)
    decimate: DecimateConfig = Field(default_factory=DecimateConfig)
    remesh: RemeshConfig = Field(default_factory=RemeshConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    unwrap: UnwrapConfig = Field(default_factory=UnwrapConfig)
    bake: BakeConfig = Field(default_factory=BakeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


# --------------------------------------------------------------------------- #
# Results & manifest
# --------------------------------------------------------------------------- #


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class MeshStats(BaseModel):
    """Geometric statistics captured for analysis and QC."""

    vertices: int = 0
    faces: int = 0
    tris: int = 0
    quads: int = 0
    components: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    is_watertight: bool = False
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None


class StageResult(BaseModel):
    """Outcome of a single stage execution."""

    stage: Stage
    status: StageStatus = StageStatus.PENDING
    #: Path to the primary output artifact (mesh or report), relative to job dir.
    artifact: str | None = None
    #: Additional output paths (e.g. texture files), relative to job dir.
    extra_artifacts: list[str] = Field(default_factory=list)
    #: Effective parameters used, serialized for reproducibility.
    params: dict[str, Any] = Field(default_factory=dict)
    #: Free-form metrics (face counts, distances, timings).
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def mark_running(self) -> None:
        self.status = StageStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_done(self) -> None:
        self.status = StageStatus.DONE
        self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, message: str) -> None:
        self.status = StageStatus.FAILED
        self.message = message
        self.finished_at = datetime.now(timezone.utc)


class LodResult(BaseModel):
    """A single interactive level-of-detail built by re-remeshing.

    Produced by :func:`slam_to_mesh.core.lod.build_lod`. Paths are relative to
    the job directory. ``baked`` indicates whether ``glb`` carries color/normal
    textures.
    """

    target_faces: int
    actual_faces: int = 0
    quad_ratio: float = 0.0
    mean_dist_pct_bbox: float = 0.0
    hausdorff_pct_bbox: float = 0.0
    baked: bool = False
    glb: str | None = None
    obj: str | None = None


class JobManifest(BaseModel):
    """The persisted state of a pipeline job (``job.json``)."""

    job_id: str
    #: Absolute path to the original input mesh.
    input_path: str
    #: Job working directory (holds intermediate + final artifacts).
    job_dir: str
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    input_stats: MeshStats | None = None
    #: "mesh" or "pointcloud" — how the input was interpreted at ingest.
    input_kind: str = "mesh"
    #: True when a point cloud is available (original upload or generated).
    has_pointcloud: bool = False
    results: dict[Stage, StageResult] = Field(default_factory=dict)
    #: Interactive LOD cache, keyed by "<faces>" or "<faces>+baked".
    lods: dict[str, LodResult] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def result(self, stage: Stage) -> StageResult:
        """Return the (creating if needed) result record for *stage*."""
        if stage not in self.results:
            self.results[stage] = StageResult(stage=stage)
        return self.results[stage]

    def artifact_path(self, stage: Stage) -> Path | None:
        """Absolute path to a stage's primary artifact, if produced."""
        res = self.results.get(stage)
        if res and res.artifact:
            return Path(self.job_dir) / res.artifact
        return None

    def last_completed_stage(self) -> Stage | None:
        """The furthest stage (in canonical order) that finished successfully."""
        done = [s for s in STAGE_ORDER if self.results.get(s) and
                self.results[s].status == StageStatus.DONE]
        return done[-1] if done else None

    def save(self, path: Path | None = None) -> Path:
        """Persist the manifest to ``job.json`` (or *path*)."""
        self.updated_at = datetime.now(timezone.utc)
        target = path or (Path(self.job_dir) / "job.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))
        return target

    @classmethod
    def load(cls, path: Path) -> JobManifest:
        """Load a manifest from disk."""
        return cls.model_validate_json(Path(path).read_text())
