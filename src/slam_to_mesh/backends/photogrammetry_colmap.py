"""COLMAP photogrammetry backend (subprocess).

Runs COLMAP to reconstruct a point cloud from an image folder:

* Sparse SfM (``feature_extractor`` → ``exhaustive_matcher`` → ``mapper``) always
  runs — it works on CPU.
* Dense MVS (``image_undistorter`` → ``patch_match_stereo`` → ``stereo_fusion``)
  runs **only when the COLMAP build supports CUDA** (dense stereo requires it).
  When CUDA is unavailable we fall back to exporting the **sparse** SfM points
  (``model_converter`` → PLY) — fewer points, but the pipeline still completes.

The resulting point cloud flows into the existing point-cloud path. Availability:
a ``colmap`` executable on PATH, or the ``COLMAP_BIN`` env var. We shell out
rather than bind, matching the QuadriFlow backend.

Install is deferred (see ``docs/spec_photogrammetry.md`` / ROADMAP).
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

    def _has_cuda(self, binary: str) -> bool:
        """True if this COLMAP build supports CUDA dense stereo."""
        try:
            out = subprocess.run(
                [binary, "patch_match_stereo", "-h"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            text = (out.stdout + out.stderr).lower()
            return "requires cuda" not in text
        except Exception:  # noqa: BLE001
            return False

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
        ws = out_points.parent / "_colmap_ws"
        ws.mkdir(parents=True, exist_ok=True)

        dense_ok = self.use_gpu and self._has_cuda(binary)
        db = ws / "database.db"
        sparse = ws / "sparse"
        sparse.mkdir(exist_ok=True)
        gpu_flag = "1" if self.use_gpu else "0"

        def _run(args: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                [binary, *args], capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )

        # --- Sparse SfM (always; CPU-capable) ---
        steps = [
            ["feature_extractor", "--database_path", str(db),
             "--image_path", str(images_dir),
             "--SiftExtraction.use_gpu", gpu_flag],
            ["exhaustive_matcher", "--database_path", str(db),
             "--SiftMatching.use_gpu", gpu_flag],
            ["mapper", "--database_path", str(db),
             "--image_path", str(images_dir), "--output_path", str(sparse)],
        ]
        for args in steps:
            proc = _run(args)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
                raise RuntimeError(
                    f"COLMAP {args[0]} failed (rc={proc.returncode}): "
                    f"{' | '.join(tail)}"
                )

        model_dir = self._first_sparse_model(sparse)
        if model_dir is None:
            raise RuntimeError(
                "COLMAP sparse reconstruction produced no model (too few "
                "matches — capture more overlapping, sharp, textured views)"
            )

        used = "sparse"
        points_ply: Path | None = None

        if dense_ok:
            # --- Dense MVS (CUDA only) ---
            dense = ws / "dense"
            dense.mkdir(exist_ok=True)
            for args in [
                ["image_undistorter", "--image_path", str(images_dir),
                 "--input_path", str(model_dir), "--output_path", str(dense),
                 "--output_type", "COLMAP"],
                ["patch_match_stereo", "--workspace_path", str(dense)],
                ["stereo_fusion", "--workspace_path", str(dense),
                 "--output_path", str(dense / "fused.ply")],
            ]:
                proc = _run(args)
                if proc.returncode != 0:
                    break
            fused = dense / "fused.ply"
            if fused.exists():
                used = "dense"
                points_ply = fused

        if points_ply is None:
            # Sparse fallback: export the sparse SfM points as PLY.
            points_ply = ws / "sparse_points.ply"
            proc = _run([
                "model_converter", "--input_path", str(model_dir),
                "--output_path", str(points_ply), "--output_type", "PLY",
            ])
            if proc.returncode != 0 or not points_ply.exists():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
                raise RuntimeError(
                    f"COLMAP model_converter failed: {' | '.join(tail)}"
                )

        _copy_or_convert_points(points_ply, out_points)
        return {
            "backend": self.id,
            "images": len(imgs),
            "reconstruction": used,  # "dense" (CUDA) or "sparse" (CPU fallback)
            "points": str(out_points),
        }

    @staticmethod
    def _first_sparse_model(sparse_dir: Path) -> Path | None:
        """COLMAP writes sparse models under sparse/0, sparse/1, …"""
        for sub in sorted(sparse_dir.glob("*")):
            if (sub / "cameras.bin").exists() or (sub / "cameras.txt").exists():
                return sub
        # Some versions write directly into sparse/.
        if (sparse_dir / "cameras.bin").exists():
            return sparse_dir
        return None


def _copy_or_convert_points(src_ply: Path, out_points: Path) -> None:
    """Put the dense cloud at *out_points* (re-encode via Open3D if needed)."""
    if src_ply.suffix.lower() == out_points.suffix.lower() == ".ply":
        shutil.copyfile(src_ply, out_points)
        return
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(src_ply))
    o3d.io.write_point_cloud(str(out_points), pcd)
