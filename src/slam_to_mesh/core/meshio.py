"""Shared mesh I/O and statistics helpers.

Thin, dependency-isolating wrappers so stages don't each hard-code trimesh
calls. Kept intentionally small; heavy processing lives in the stage modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .model import MeshStats


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from disk as a single :class:`trimesh.Trimesh`.

    ``trimesh.load`` may return a ``Scene`` for multi-object files; we
    concatenate into one mesh so downstream stages always see one geometry.
    """
    loaded = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle geometry found in {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type loaded from {path}: {type(loaded)}")
    return loaded


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write a mesh to *path*, creating parent dirs. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(p))
    return p


def compute_stats(mesh: trimesh.Trimesh) -> MeshStats:
    """Compute geometric statistics used for analysis and QC."""
    faces = mesh.faces
    n_faces = int(len(faces))

    # trimesh triangulates on load, so faces are tris here; quad detection is
    # meaningful only after remesh where we track polygon faces separately.
    tris = n_faces
    quads = 0

    try:
        components = int(len(mesh.split(only_watertight=False)))
    except Exception:
        components = 1

    # Boundary edges: edges referenced by exactly one face.
    try:
        boundary_edges = int(len(mesh.edges[trimesh.grouping.group_rows(
            mesh.edges_sorted, require_count=1)]))
    except Exception:
        boundary_edges = 0

    # Non-manifold edges: edges shared by more than two faces.
    try:
        edges_sorted = mesh.edges_sorted
        unique, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        non_manifold_edges = int(np.count_nonzero(counts > 2))
    except Exception:
        non_manifold_edges = 0

    bbox_min = mesh.bounds[0].tolist() if mesh.bounds is not None else None
    bbox_max = mesh.bounds[1].tolist() if mesh.bounds is not None else None

    return MeshStats(
        vertices=int(len(mesh.vertices)),
        faces=n_faces,
        tris=tris,
        quads=quads,
        components=components,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        is_watertight=bool(mesh.is_watertight),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def bbox_diagonal(mesh: trimesh.Trimesh) -> float:
    """Length of the bounding-box diagonal (used to scale tolerances)."""
    if mesh.bounds is None:
        return 1.0
    return float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
