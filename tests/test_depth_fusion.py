"""Tests for depth fusion + Poisson hardening (no COLMAP/GPU needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest


def test_ply_vertex_count(tmp_path: Path):
    from slam_to_mesh.backends.photogrammetry_colmap import _ply_vertex_count

    # A real (tiny) PLY.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(42, 3))
    p = tmp_path / "pts.ply"
    o3d.io.write_point_cloud(str(p), pcd)
    assert _ply_vertex_count(p) == 42

    # Missing file → 0.
    assert _ply_vertex_count(tmp_path / "nope.ply") == 0

    # A zero-vertex PLY header (the COLMAP-fusion-failure case) → 0.
    zero = tmp_path / "zero.ply"
    zero.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n"
    )
    assert _ply_vertex_count(zero) == 0


def test_depth_fusion_camera_image_parsers(tmp_path: Path):
    """qvec2rot + intrinsics helpers behave sanely."""
    from slam_to_mesh.core.depth_fusion import _intrinsics, _qvec2rot

    # Identity quaternion → identity rotation.
    R = _qvec2rot(np.array([1.0, 0, 0, 0]))
    assert np.allclose(R, np.eye(3), atol=1e-9)

    # PINHOLE model (id 1): fx, fy, cx, cy.
    fx, fy, cx, cy = _intrinsics({"model": 1, "params": (500.0, 510.0, 320.0, 240.0)})
    assert (fx, fy, cx, cy) == (500.0, 510.0, 320.0, 240.0)
    # SIMPLE_PINHOLE (id 0): f, cx, cy → fx==fy==f.
    fx, fy, cx, cy = _intrinsics({"model": 0, "params": (500.0, 320.0, 240.0)})
    assert fx == fy == 500.0 and (cx, cy) == (320.0, 240.0)


def test_reconstruct_poisson_caps_and_survives(tmp_path: Path):
    """Poisson on a dense sphere cloud: no crash, capped at max_faces."""
    from slam_to_mesh.core.pointcloud import reconstruct_poisson

    n = 40000
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    src = tmp_path / "sphere.ply"
    o3d.io.write_point_cloud(str(src), pcd)

    out = tmp_path / "mesh.ply"
    stats = reconstruct_poisson(src, out, depth=8, max_faces=20000)
    assert out.exists()
    assert stats["mesh_faces"] <= 20000
    assert stats["input_points"] == n
    # Whether or not the cap triggered, the field must be present.
    assert "capped" in stats
