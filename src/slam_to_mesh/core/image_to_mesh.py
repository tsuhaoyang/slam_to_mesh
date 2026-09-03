"""Image → mesh bridge (TripoSR), run in an isolated venv via subprocess.

Single-image 3D reconstruction (TripoSR, MIT) is used as a **front-end** for the
retopology pipeline: it turns a 2D image into a dense triangle mesh, which the
normal pipeline then cleans, remeshes to quads, projects, etc.

TripoSR has heavy, version-pinned dependencies (and needs a Blackwell-capable
PyTorch nightly on this box), so it lives in its **own venv**. We call it out of
process — never importing it into the main environment — mirroring how the
QuadriFlow backend shells out to a binary.

Discovery order for the TripoSR checkout:
1. ``TRIPOSR_DIR`` environment variable.
2. A sibling ``TripoSR/`` directory next to this repo.

Availability requires the venv's python and ``run.py`` to exist; if absent,
:func:`is_available` is False and image input is simply unavailable (the service
returns a clear error), exactly like a missing remesh binary.

See ``docs/triposr_2d_to_3d.md``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: Image extensions accepted as pipeline input.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def find_triposr_dir() -> Path | None:
    """Locate the TripoSR checkout directory, or None."""
    env = os.environ.get("TRIPOSR_DIR")
    if env and (Path(env) / "run.py").is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "TripoSR"
        if (cand / "run.py").is_file():
            return cand
        cand2 = parent.parent / "TripoSR"
        if (cand2 / "run.py").is_file():
            return cand2
    return None


def _venv_python(triposr_dir: Path) -> Path | None:
    py = triposr_dir / ".venv" / "bin" / "python"
    return py if py.is_file() else None


def is_available() -> bool:
    """True if a runnable TripoSR (venv python + run.py) is present."""
    d = find_triposr_dir()
    if d is None:
        return False
    return _venv_python(d) is not None


def generate_mesh(
    image_path: str | Path,
    out_mesh: str | Path,
    mc_resolution: int = 256,
    timeout: int = 1200,
) -> dict:
    """Run TripoSR on *image_path*; write the resulting mesh to *out_mesh*.

    Executes ``run.py`` inside the TripoSR venv via subprocess, then moves its
    ``<out>/0/mesh.obj`` to *out_mesh*. Returns simple stats. Raises
    RuntimeError with a clear message when TripoSR is unavailable or fails.
    """
    triposr_dir = find_triposr_dir()
    if triposr_dir is None:
        raise RuntimeError(
            "TripoSR not found. Set TRIPOSR_DIR or place a TripoSR checkout "
            "beside this repo (see docs/triposr_2d_to_3d.md)."
        )
    py = _venv_python(triposr_dir)
    if py is None:
        raise RuntimeError(
            f"TripoSR venv python missing at {triposr_dir}/.venv; "
            "install per docs/triposr_2d_to_3d.md."
        )

    image_path = Path(image_path).resolve()
    out_mesh = Path(out_mesh)
    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    work = out_mesh.parent / "_triposr_out"

    # CUDA toolkit on PATH (nightly torch already targets the GPU; harmless if
    # already present). Inherit env so the venv's own libs resolve.
    env = dict(os.environ)
    env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"

    cmd = [
        str(py), "run.py",
        str(image_path),
        "--output-dir", str(work),
        "--mc-resolution", str(int(mc_resolution)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(triposr_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    produced = work / "0" / "mesh.obj"
    if proc.returncode != 0 or not produced.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"TripoSR failed (rc={proc.returncode}): {' | '.join(tail)}"
        )

    produced.replace(out_mesh)
    return {"source_image": str(image_path), "mesh": str(out_mesh)}
