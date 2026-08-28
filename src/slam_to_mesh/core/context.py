"""Shared context passed to every stage.

A uniform stage signature ``fn(ctx) -> StageResult`` keeps the orchestrator
simple and makes stages trivially reusable from both the CLI and the service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import JobManifest, Stage


#: Fixed artifact filename prefixes, e.g. Stage.CLEAN -> "02_clean".
_STAGE_PREFIX = {
    Stage.INGEST: "01_ingest",
    Stage.CLEAN: "02_clean",
    Stage.DECIMATE: "03_decimate",
    Stage.REMESH: "04_remesh",
    Stage.PROJECT: "05_project",
    Stage.UNWRAP: "06_unwrap",
    Stage.BAKE: "07_bake",
    Stage.EXPORT: "08_export",
    Stage.QC: "09_qc",
}


def artifact_name(stage: Stage, ext: str) -> str:
    """Return the canonical artifact filename for *stage* with *ext*.

    ``ext`` may be given with or without a leading dot.
    """
    ext = ext.lstrip(".")
    return f"{_STAGE_PREFIX[stage]}.{ext}"


@dataclass
class StageContext:
    """Everything a stage needs to run."""

    manifest: JobManifest

    @property
    def job_dir(self) -> Path:
        return Path(self.manifest.job_dir)

    def out_path(self, stage: Stage, ext: str) -> Path:
        """Absolute path for a stage's primary output artifact."""
        return self.job_dir / artifact_name(stage, ext)

    def rel(self, path: Path) -> str:
        """Path relative to the job dir, for storing in the manifest."""
        return str(Path(path).relative_to(self.job_dir))

    def input_for(self, stage: Stage) -> Path:
        """Resolve the input mesh path for *stage*.

        Walks backwards from the previous stage to find the most recent
        produced mesh artifact, falling back to the original input.
        """
        from .model import STAGE_ORDER, stage_index

        idx = stage_index(stage)
        for prev in reversed(STAGE_ORDER[:idx]):
            p = self.manifest.artifact_path(prev)
            if p is not None and p.exists() and p.suffix.lower() in {
                ".ply", ".obj", ".stl", ".glb", ".off"
            }:
                return p
        return Path(self.manifest.input_path)
