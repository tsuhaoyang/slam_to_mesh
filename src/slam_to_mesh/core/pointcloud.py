"""Point-cloud operations: reconstruction, downsampling, sampling.

Backs the flexible-input and point-cloud features (see
``docs/spec_flexible_io.md``):

* :func:`reconstruct_poisson` — turn a point cloud into a triangle mesh so the
  rest of the (surface-based) pipeline can run on point-cloud inputs.
* :func:`voxel_downsample` — reduce a point cloud's point count (independent of
  any mesh decimation).
* :func:`sample_points_from_mesh` — generate a point cloud from a mesh surface
  (for the point-cloud viewer when the input was a mesh).

All functions are thin, dependency-isolating wrappers over Open3D / trimesh so
stages and the service don't hard-code those calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a point cloud (.ply/.pcd/.xyz…) via Open3D."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"no points loaded from {path}")
    return pcd


def is_point_cloud_file(path: str | Path) -> bool:
    """Heuristically decide whether *path* holds a point cloud (no faces).

    - ``.pcd`` / ``.xyz`` are always point clouds.
    - ``.ply`` may be either; we peek at the header for a non-zero
      ``element face`` count. If faces exist it's a mesh, else a point cloud.
    - Mesh-only formats (.obj/.stl/.glb/.off) are never point clouds.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".pcd", ".xyz", ".xyzn", ".pts"}:
        return True
    if ext in {".obj", ".stl", ".glb", ".gltf", ".off"}:
        return False
    if ext == ".ply":
        return not _ply_has_faces(p)
    return False


def _ply_has_faces(path: Path) -> bool:
    """Parse a PLY header; return True if it declares a face element with >0."""
    try:
        with open(path, "rb") as fh:
            in_header = False
            current_face_count = 0
            for raw in fh:
                line = raw.decode("ascii", errors="ignore").strip()
                if line == "ply":
                    in_header = True
                    continue
                if not in_header:
                    break
                if line.startswith("element face"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[2].isdigit():
                        current_face_count = int(parts[2])
                if line == "end_header":
                    break
            return current_face_count > 0
    except OSError:
        return False


def reconstruct_poisson(
    points_path: str | Path,
    out_mesh: str | Path,
    depth: int = 9,
    density_quantile: float = 0.02,
    max_faces: int = 200000,
) -> dict:
    """Reconstruct a triangle mesh from a point cloud via Poisson.

    Estimates normals if the cloud lacks them (Poisson needs oriented normals),
    runs screened Poisson at *depth*, and trims the lowest-density vertices
    (thin, extrapolated web the algorithm adds beyond the samples). Writes the
    mesh to *out_mesh* and returns simple stats.
    """
    pcd = load_point_cloud(points_path)
    n_points = len(pcd.points)

    # Robust normals: KNN (always finds neighbors, unlike a fixed radius which
    # can leave isolated points with degenerate normals → Poisson can crash on
    # dense/noisy multi-view clouds). Always (re)estimate + orient consistently.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth)
    )
    densities = np.asarray(densities)
    if density_quantile > 0 and len(densities):
        thresh = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)
    mesh.remove_unreferenced_vertices()

    # Cap mesh size so downstream stages (which analyze/proximity-query the
    # whole mesh) stay fast and don't choke on a huge Poisson output.
    capped = False
    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(int(max_faces))
        capped = True

    mesh.compute_vertex_normals()
    out_mesh = Path(out_mesh)
    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)

    return {
        "input_points": int(n_points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_faces": len(mesh.triangles),
        "poisson_depth": int(depth),
        "capped": capped,
    }


def voxel_downsample(
    points_path: str | Path,
    out_points: str | Path,
    voxel_size: float | None = None,
    target_points: int | None = None,
) -> dict:
    """Voxel-downsample a point cloud.

    Provide either an explicit *voxel_size*, or a *target_points* count (the
    voxel size is then searched to approximately hit that count). Writes the
    reduced cloud to *out_points* and returns before/after counts.
    """
    pcd = load_point_cloud(points_path)
    before = len(pcd.points)

    if voxel_size is None:
        if target_points is None:
            raise ValueError("provide voxel_size or target_points")
        voxel_size = _voxel_for_target(pcd, int(target_points))

    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    out_points = Path(out_points)
    out_points.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_points), down)

    return {
        "points_before": int(before),
        "points_after": len(down.points),
        "voxel_size": round(float(voxel_size), 8),
    }


def _voxel_for_target(pcd: o3d.geometry.PointCloud, target: int) -> float:
    """Binary-search a voxel size that yields ~*target* points."""
    aabb = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(aabb.get_extent())) or 1.0
    lo, hi = diag * 1e-4, diag  # small voxel = many points; large = few
    target = max(1, target)
    best = diag * 1e-2
    for _ in range(24):
        mid = (lo + hi) / 2.0
        n = len(pcd.voxel_down_sample(voxel_size=mid).points)
        best = mid
        if n > target:
            lo = mid  # need bigger voxel to reduce further
        else:
            hi = mid
        if abs(n - target) <= max(1, target * 0.03):
            break
    return best


def points_to_positions(path: str | Path) -> list[float]:
    """Load a point cloud and return a flat [x,y,z,...] list (for the browser)."""
    pcd = load_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32).reshape(-1)
    return pts.tolist()


def sample_points_from_mesh(
    mesh_path: str | Path,
    out_points: str | Path,
    n: int = 50000,
) -> dict:
    """Sample *n* points uniformly from a mesh surface (Poisson-disk).

    Used to give mesh inputs a point-cloud representation for the viewer /
    downsampling when the user opts in.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no faces to sample")
    mesh.compute_vertex_normals()
    n = max(1, int(n))
    try:
        pcd = mesh.sample_points_poisson_disk(number_of_points=n)
    except Exception:  # noqa: BLE001 - fall back to uniform sampling
        pcd = mesh.sample_points_uniformly(number_of_points=n)

    out_points = Path(out_points)
    out_points.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_points), pcd)
    return {"sampled_points": len(pcd.points)}
