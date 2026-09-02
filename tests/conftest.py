"""Shared pytest fixtures for slam_to_mesh.

Provides synthetic SLAM-like meshes (irregular triangulation + a floating
island, optionally vertex-colored) and helpers to build temporary jobs. Kept
CPU-only and small so the whole suite runs quickly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh


def _slam_like_mesh(subdivisions: int = 3, with_island: bool = True) -> trimesh.Trimesh:
    """An icosphere (irregular-ish tri surface) plus a small floating island.

    Mimics SLAM output: a main watertight-ish surface plus disconnected debris
    that the clean stage should drop.
    """
    main = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    if not with_island:
        return main
    island = trimesh.creation.box(extents=[0.08, 0.08, 0.08])
    island.apply_translation([3.0, 3.0, 3.0])
    return trimesh.util.concatenate([main, island])


def _colored(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attach deterministic per-vertex colors based on position."""
    v = np.asarray(mesh.vertices)
    # Map coordinates into 0..255 RGB.
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    rgb = ((v - lo) / span * 255).astype(np.uint8)
    colors = np.column_stack([rgb, np.full(len(v), 255, dtype=np.uint8)])
    mesh.visual.vertex_colors = colors
    return mesh


@pytest.fixture
def slam_mesh() -> trimesh.Trimesh:
    """A synthetic SLAM-like mesh with a floating island."""
    return _slam_like_mesh(subdivisions=3, with_island=True)


@pytest.fixture
def slam_mesh_path(tmp_path: Path, slam_mesh: trimesh.Trimesh) -> Path:
    """The synthetic SLAM mesh written to a PLY on disk."""
    p = tmp_path / "slam_in.ply"
    slam_mesh.export(str(p))
    return p


@pytest.fixture
def colored_slam_mesh_path(tmp_path: Path) -> Path:
    """A vertex-colored SLAM-like mesh (no island) for bake tests."""
    m = _colored(_slam_like_mesh(subdivisions=3, with_island=False))
    p = tmp_path / "slam_colored.ply"
    m.export(str(p))
    return p


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    """A fresh job working directory."""
    d = tmp_path / "job"
    d.mkdir()
    return d
