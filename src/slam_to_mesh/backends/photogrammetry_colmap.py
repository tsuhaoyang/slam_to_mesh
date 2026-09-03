"""COLMAP photogrammetry backend (subprocess).

Runs COLMAP's automatic reconstruction over an image folder to produce a dense
point cloud (Structure-from-Motion + Multi-View Stereo), which the rest of the
pipeline consumes via the existing point-cloud path.

Availability: a ``colmap`` executable on PATH, or the ``COLMAP_BIN`` env var.
Dense MVS uses the GPU when COLMAP is built with CUDA; otherwise it runs on CPU
(slower). When COLMAP is absent, :meth:`is_available` is False and image-set /
video inputs are simply unavailable.

Install is deferred (see ``docs/spec_photogrammetry.md`` / ROADMAP). We shell out
rather than bind, matching the QuadriFlow backend's approach.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_colmap_bin() -> str | None:
    env = os.environ.get("COLMAP_BIN")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    return shutil.which("colmap")


class ColmapBackend:
    """Multi-view images → dense point cloud via COLMAP."""

    id = "colmap"

    def __init__(self, use_gpu: bool = True, timeout: int = 7200) -> None:
        self.use_gpu = use_gpu
        self.timeout = timeout

    def is_available(self) -> bool:
        return find_colmap_bin() is not None

    def reconstruct(self, images_dir: Path, out_points: Path) -> dict:
        binary = find_colmap_bin()
        if binary is None:
            raise RuntimeError(
                "COLMAP not found. Install it (see docs/spec_photogrammetry.md) "
                "or set COLMAP_BIN."
            )

        images_dir = Path(images_dir)
        imgs = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
        if len(imgs) < 3:
            raise RuntimeError(
                f"photogrammetry needs several images; found {len(imgs)} in "
                f"{images_dir}"
            )

        out_points = Path(out_points)
        out_points.parent.mkdir(parents=True, exist_ok=True)
        workspace = out_points.parent / "_colmap_ws"
        workspace.mkdir(parents=True, exist_ok=True)

        gpu = "1" if self.use_gpu else "0"
        cmd = [
            binary, "automatic_reconstructor",
            "--workspace_path", str(workspace),
            "--image_path", str(images_dir),
            "--dense", "1",
            "--use_gpu", gpu,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout, check=False
        )

        fused = self._find_fused_ply(workspace)
        if proc.returncode != 0 or fused is None:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(
                f"COLMAP failed (rc={proc.returncode}): {' | '.join(tail)}"
            )

        # Normalize the fused dense cloud to our expected output path.
        _copy_or_convert_points(fused, out_points)

        return {
            "backend": self.id,
            "images": len(imgs),
            "dense_ply": str(fused),
            "points": str(out_points),
        }

    @staticmethod
    def _find_fused_ply(workspace: Path) -> Path | None:
        """Locate the dense fused point cloud COLMAP produced."""
        # automatic_reconstructor writes dense/<i>/fused.ply
        candidates = sorted(workspace.glob("dense/*/fused.ply"))
        if candidates:
            return candidates[0]
        any_fused = sorted(workspace.rglob("fused.ply"))
        return any_fused[0] if any_fused else None


def _copy_or_convert_points(src_ply: Path, out_points: Path) -> None:
    """Put the dense cloud at *out_points* (re-encode via Open3D if needed)."""
    if src_ply.suffix.lower() == out_points.suffix.lower() == ".ply":
        shutil.copyfile(src_ply, out_points)
        return
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(src_ply))
    o3d.io.write_point_cloud(str(out_points), pcd)
