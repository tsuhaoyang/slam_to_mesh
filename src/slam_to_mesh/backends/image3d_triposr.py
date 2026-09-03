"""TripoSR single-image 3D backend (subprocess into an isolated venv).

TripoSR (VAST-AI-Research, MIT — Tripo AI + Stability AI) reconstructs a mesh
from a single image via a feedforward network. It has heavy, version-pinned
dependencies (and needs a Blackwell-capable PyTorch nightly here), so it runs in
its own venv and is invoked out of process — never imported into the main
environment. Discovery: ``TRIPOSR_DIR`` env, else a sibling ``TripoSR/`` checkout.

See ``docs/triposr_2d_to_3d.md``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def find_triposr_dir() -> Path | None:
    """Locate the TripoSR checkout directory, or None."""
    env = os.environ.get("TRIPOSR_DIR")
    if env and (Path(env) / "run.py").is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "TripoSR", parent.parent / "TripoSR"):
            if (cand / "run.py").is_file():
                return cand
    return None


def _venv_python(triposr_dir: Path) -> Path | None:
    py = triposr_dir / ".venv" / "bin" / "python"
    return py if py.is_file() else None


class TripoSRBackend:
    """Single-image → mesh via the TripoSR CLI (subprocess, isolated venv)."""

    id = "triposr"

    def __init__(self, mc_resolution: int = 256, timeout: int = 1200) -> None:
        self.mc_resolution = mc_resolution
        self.timeout = timeout

    def is_available(self) -> bool:
        d = find_triposr_dir()
        return d is not None and _venv_python(d) is not None

    def generate_mesh(self, image_path: Path, out_mesh: Path) -> dict:
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

        env = dict(os.environ)
        env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"

        cmd = [
            str(py), "run.py",
            str(image_path),
            "--output-dir", str(work),
            "--mc-resolution", str(int(self.mc_resolution)),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(triposr_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
            check=False,
        )
        produced = work / "0" / "mesh.obj"
        if proc.returncode != 0 or not produced.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            raise RuntimeError(
                f"TripoSR failed (rc={proc.returncode}): {' | '.join(tail)}"
            )
        produced.replace(out_mesh)
        return {"backend": self.id, "source_image": str(image_path), "mesh": str(out_mesh)}
