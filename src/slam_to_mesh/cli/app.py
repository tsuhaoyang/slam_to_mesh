"""slam2mesh command-line interface.

Commands:
* ``run``           — run the full pipeline on an input mesh.
* ``resume-from``   — re-run from a given stage using an existing job.
* ``inspect``       — print a summary of a job manifest.
* ``list-backends`` — show available remesh backends.

All commands operate on a job directory containing ``job.json`` plus the
per-stage artifacts, so results are inspectable and resumable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..backends.remesh import available_backends
from ..core.model import (
    JobManifest,
    PipelineConfig,
    Stage,
    StageResult,
    StageStatus,
)
from ..core.pipeline import create_job, run_pipeline

app = typer.Typer(
    add_completion=False,
    help="Convert messy SLAM meshes into lightweight quad-dominant meshes for Omniverse.",
)
console = Console()


def _progress_printer(result: StageResult) -> None:
    status_style = {
        StageStatus.DONE: "green",
        StageStatus.SKIPPED: "yellow",
        StageStatus.FAILED: "red",
    }.get(result.status, "white")
    console.print(
        f"[{status_style}]{result.stage.value:>9}[/] "
        f"[{status_style}]{result.status.value:<8}[/] "
        f"{result.message or ''}"
    )


@app.command()
def run(
    input: Path = typer.Argument(..., exists=True, help="Input SLAM mesh (.ply/.obj/...)."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Job output directory."),
    target_faces: int = typer.Option(
        20000, "--target-faces", "-f", help="Target quad face budget."
    ),
    decimate_faces: Optional[int] = typer.Option(
        None, "--decimate-faces", help="Pre-remesh QEM face budget (default 2.5x target)."
    ),
    backend: str = typer.Option(
        "quadriflow_cpu", "--backend", "-b", help="Remesh backend id."
    ),
    bake: bool = typer.Option(False, "--bake", help="Bake original color into a texture."),
    formats: str = typer.Option(
        "glb,obj", "--formats", help="Comma-separated export formats (glb,gltf,obj,usd,...)."
    ),
    no_project: bool = typer.Option(
        False, "--no-project", help="Disable projection back onto the original surface."
    ),
) -> None:
    """Run the full retopology pipeline on INPUT."""
    config = PipelineConfig()
    config.remesh.target_faces = target_faces
    config.remesh.backend = backend
    config.decimate.target_faces = decimate_faces or int(target_faces * 2.5)
    config.bake.enabled = bake
    config.project.enabled = not no_project
    config.export.formats = [f.strip() for f in formats.split(",") if f.strip()]

    manifest = create_job(input, out, config=config)
    console.print(f"[bold]Job[/] {manifest.job_id}  →  {manifest.job_dir}")
    run_pipeline(manifest, start=Stage.INGEST, on_stage=_progress_printer)
    _print_summary(manifest)


@app.command(name="resume-from")
def resume_from(
    stage: str = typer.Argument(..., help="Stage to resume from (e.g. remesh)."),
    job: Path = typer.Option(..., "--job", "-j", exists=True, help="Path to job.json."),
) -> None:
    """Re-run the pipeline from STAGE using an existing job manifest."""
    try:
        target = Stage(stage)
    except ValueError:
        valid = ", ".join(s.value for s in Stage)
        console.print(f"[red]Unknown stage '{stage}'.[/] Valid: {valid}")
        raise typer.Exit(1)

    manifest = JobManifest.load(job)
    console.print(f"[bold]Resuming[/] {manifest.job_id} from '{target.value}'")
    run_pipeline(manifest, start=target, on_stage=_progress_printer)
    _print_summary(manifest)


@app.command()
def inspect(
    job: Path = typer.Argument(..., exists=True, help="Path to job.json."),
) -> None:
    """Print a summary of a job manifest."""
    manifest = JobManifest.load(job)
    _print_summary(manifest)


@app.command(name="list-backends")
def list_backends() -> None:
    """List remesh backends available in this environment."""
    backends = available_backends()
    if not backends:
        console.print("[yellow]No remesh backends available.[/]")
        return
    for b in backends:
        console.print(f"  • {b}")


def _print_summary(manifest: JobManifest) -> None:
    table = Table(title=f"Job {manifest.job_id}", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    table.add_column("Artifact")
    table.add_column("Notes", overflow="fold")

    from ..core.model import STAGE_ORDER

    for stage in STAGE_ORDER:
        res = manifest.results.get(stage)
        if res is None:
            table.add_row(stage.value, "-", "", "")
            continue
        style = {
            StageStatus.DONE: "green",
            StageStatus.SKIPPED: "yellow",
            StageStatus.FAILED: "red",
        }.get(res.status, "white")
        table.add_row(
            stage.value,
            f"[{style}]{res.status.value}[/]",
            res.artifact or "",
            res.message or "",
        )
    console.print(table)

    qc_res = manifest.results.get(Stage.QC)
    if qc_res and qc_res.metrics:
        m = qc_res.metrics
        console.print(
            f"\n[bold]QC:[/] faces={m.get('final_faces')} "
            f"reduction={m.get('face_reduction_ratio')} "
            f"quad_ratio={m.get('quad_ratio')} "
            f"meanDist={m.get('mean_dist_pct_bbox')}%bbox"
        )


if __name__ == "__main__":
    app()
