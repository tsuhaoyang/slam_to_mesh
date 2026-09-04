"""Fuse COLMAP photometric depth maps into a point cloud.

A fallback for COLMAP's own ``stereo_fusion``, which produces **zero points** on
some very new GPU/CUDA combinations (observed on sm_120 / CUDA 13.3): the
PatchMatch *geometric* consistency filter zeroes all depth, and fusion then has
nothing to fuse — even though the *photometric* depth maps are perfectly valid.

This module bypasses that: it reads COLMAP's sparse model (camera intrinsics +
per-image poses) and the valid photometric depth maps, back-projects each depth
map into world space, merges them, and voxel-downsamples + removes outliers to a
target point budget.

Parsing uses the documented COLMAP binary formats (cameras.bin / images.bin) and
the depth-map ``<w>&<h>&<c>&<float32...>`` layout — no pycolmap dependency.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import open3d as o3d

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# COLMAP camera model id -> number of params.
_CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read(f, n, fmt):
    return struct.unpack(fmt, f.read(n))


def read_cameras(path: Path) -> dict:
    cams = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "<Q")[0]
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, 24, "<iiQQ")
            k = _CAM_NPARAMS.get(model_id, 4)
            params = _read(f, 8 * k, "<" + "d" * k)
            cams[cam_id] = {"w": w, "h": h, "model": model_id, "params": params}
    return cams


def read_images(path: Path) -> dict:
    imgs = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "<Q")[0]
        for _ in range(n):
            img_id = _read(f, 4, "<i")[0]
            qvec = _read(f, 32, "<dddd")
            tvec = _read(f, 24, "<ddd")
            cam_id = _read(f, 4, "<i")[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = _read(f, 8, "<Q")[0]
            f.read(24 * npts)  # skip 2D observations
            imgs[img_id] = {
                "qvec": np.array(qvec), "tvec": np.array(tvec),
                "cam_id": cam_id, "name": name.decode(),
            }
    return imgs


def _qvec2rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def _read_depth_map(path: Path) -> np.ndarray:
    d = path.read_bytes()
    amp = [i for i, b in enumerate(d[:60]) if b == ord("&")]
    w, h, _c = int(d[:amp[0]]), int(d[amp[0]+1:amp[1]]), int(d[amp[1]+1:amp[2]])
    return np.frombuffer(d[amp[2]+1:], dtype=np.float32).reshape(h, w)


def _intrinsics(cam: dict):
    p, model = cam["params"], cam["model"]
    if model in (0, 2, 3, 7):  # SIMPLE_* : f, cx, cy
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]  # fx, fy, cx, cy


def fuse_depth_maps(
    dense_dir: str | Path,
    out_points: str | Path,
    target_points: int = 200000,
) -> dict:
    """Back-project photometric depth maps to a merged, downsampled point cloud.

    *dense_dir* is COLMAP's dense workspace (contains ``sparse/``, ``images/``,
    ``stereo/depth_maps/``). Writes a PLY to *out_points*; returns stats.
    """
    dense = Path(dense_dir)
    cams = read_cameras(dense / "sparse" / "cameras.bin")
    imgs = read_images(dense / "sparse" / "images.bin")
    from PIL import Image

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    used = 0
    for img in imgs.values():
        dm = dense / "stereo" / "depth_maps" / f"{img['name']}.photometric.bin"
        if not dm.exists():
            continue
        depth = _read_depth_map(dm)
        valid = depth > 0
        if not valid.any():
            continue
        cam = cams[img["cam_id"]]
        fx, fy, cx, cy = _intrinsics(cam)
        h, w = depth.shape
        sx, sy = w / cam["w"], h / cam["h"]
        fx, fy, cx, cy = fx*sx, fy*sy, cx*sx, cy*sy

        # Drop far outliers (background) beyond 1.5x the 95th percentile.
        hi = np.percentile(depth[valid], 95)
        valid &= depth < hi * 1.5

        ys, xs = np.nonzero(valid)
        z = depth[ys, xs]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)
        R, t = _qvec2rot(img["qvec"]), img["tvec"]
        pts_world = (pts_cam - t) @ R  # world = R^T (X_cam - t)
        all_pts.append(pts_world)

        rgb = np.asarray(
            Image.open(dense / "images" / img["name"]).convert("RGB").resize((w, h))
        )
        all_cols.append(rgb[ys, xs] / 255.0)
        used += 1

    if not all_pts:
        raise RuntimeError("depth fusion found no valid photometric depth maps")

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    raw = len(pcd.points)

    # Downsample to roughly target_points. Binary-search the voxel size, since
    # point count vs voxel is monotonic (bigger voxel → fewer points).
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    lo, hi = diag * 1e-4, diag * 0.2
    down = pcd
    for _ in range(18):
        mid = (lo + hi) / 2.0
        cand = pcd.voxel_down_sample(voxel_size=mid)
        n = len(cand.points)
        down = cand
        if n > target_points:
            lo = mid  # need bigger voxel to reduce
        else:
            hi = mid
        if abs(n - target_points) <= target_points * 0.1:
            break
    pcd = down
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    out_points = Path(out_points)
    out_points.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_points), pcd)
    return {
        "images_fused": used,
        "raw_points": int(raw),
        "points": len(pcd.points),
        "out": str(out_points),
    }
