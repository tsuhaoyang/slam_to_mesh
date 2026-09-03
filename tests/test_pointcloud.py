"""Tests for core.pointcloud (Poisson reconstruct, voxel downsample, sampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from slam_to_mesh.core.pointcloud import (
    is_point_cloud_file,
    reconstruct_poisson,
    sample_points_from_mesh,
    voxel_downsample,
)


@pytest.fixture
def sphere_points(tmp_path: Path) -> Path:
    """A dense point cloud sampled on a unit sphere, with normals."""
    n = 8000
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)  # unit sphere
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    pcd.normals = o3d.utility.Vector3dVector(v)  # outward normals == positions
    p = tmp_path / "sphere.ply"
    o3d.io.write_point_cloud(str(p), pcd)
    return p


@pytest.fixture
def sphere_mesh(tmp_path: Path) -> Path:
    m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    m.compute_vertex_normals()
    p = tmp_path / "sphere_mesh.ply"
    o3d.io.write_triangle_mesh(str(p), m)
    return p


def test_is_point_cloud_file_by_extension(tmp_path: Path):
    assert is_point_cloud_file("a.pcd") is True
    assert is_point_cloud_file("a.xyz") is True
    assert is_point_cloud_file("a.obj") is False
    assert is_point_cloud_file("a.stl") is False
    assert is_point_cloud_file("a.glb") is False


def test_is_point_cloud_file_ply_points_vs_mesh(sphere_points: Path, sphere_mesh: Path):
    # A points-only PLY -> point cloud; a PLY with faces -> mesh.
    assert is_point_cloud_file(sphere_points) is True
    assert is_point_cloud_file(sphere_mesh) is False


def test_reconstruct_poisson_makes_a_mesh(sphere_points: Path, tmp_path: Path):
    out = tmp_path / "recon.ply"
    stats = reconstruct_poisson(sphere_points, out, depth=7)
    assert out.exists()
    assert stats["input_points"] == 8000
    assert stats["mesh_faces"] > 0
    assert stats["mesh_vertices"] > 0
    # The reconstructed mesh should load as a real triangle mesh.
    m = o3d.io.read_triangle_mesh(str(out))
    assert len(m.triangles) > 0


def test_reconstruct_poisson_estimates_missing_normals(tmp_path: Path):
    # Sphere points WITHOUT normals -> reconstruct should still work.
    n = 6000
    rng = np.random.default_rng(1)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    src = tmp_path / "nonormals.ply"
    o3d.io.write_point_cloud(str(src), pcd)

    out = tmp_path / "recon2.ply"
    stats = reconstruct_poisson(src, out, depth=7)
    assert stats["mesh_faces"] > 0


def test_voxel_downsample_by_size(sphere_points: Path, tmp_path: Path):
    out = tmp_path / "down.ply"
    stats = voxel_downsample(sphere_points, out, voxel_size=0.2)
    assert out.exists()
    assert stats["points_before"] == 8000
    assert stats["points_after"] < stats["points_before"]
    assert stats["points_after"] > 0


def test_voxel_downsample_by_target(sphere_points: Path, tmp_path: Path):
    out = tmp_path / "down_t.ply"
    stats = voxel_downsample(sphere_points, out, target_points=1000)
    assert out.exists()
    # Approximately the target (voxel binning is not exact).
    assert 300 <= stats["points_after"] <= 3000
    assert stats["points_after"] < stats["points_before"]


def test_voxel_downsample_requires_a_control(sphere_points: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        voxel_downsample(sphere_points, tmp_path / "x.ply")


def test_sample_points_from_mesh(sphere_mesh: Path, tmp_path: Path):
    out = tmp_path / "sampled.ply"
    stats = sample_points_from_mesh(sphere_mesh, out, n=2000)
    assert out.exists()
    assert stats["sampled_points"] > 0
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) > 0
