"""Photogrammetry backend registry (multi-view images → dense point cloud).

Measurement-based 3D from many overlapping views, mirroring the image3d / remesh
registries. Each backend wraps an external Structure-from-Motion + Multi-View
Stereo tool (run via subprocess) and reports availability.

First implementation: **COLMAP** (`photogrammetry_colmap.py`). Others (Meshroom /
AliceVision, or learning-based Dust3R/VGGT) can be registered later.

No universal fallback: if no backend is available, image-set / video inputs are
simply unavailable (ingest / service surface a clear error), gated like the
QuadriFlow binary and the image3d backends.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class PhotogrammetryBackend(Protocol):
    """Contract every photogrammetry backend must satisfy."""

    id: str

    def is_available(self) -> bool:
        """True if this backend can run in the current environment."""
        ...

    def reconstruct(self, images_dir: Path, out_points: Path) -> dict:
        """Reconstruct a dense point cloud from the images in *images_dir*.

        Writes a point cloud to *out_points* (PLY) and returns stats.
        """
        ...


_REGISTRY: dict[str, Callable[[], PhotogrammetryBackend]] = {}


def register_backend(
    backend_id: str, factory: Callable[[], PhotogrammetryBackend]
) -> None:
    _REGISTRY[backend_id] = factory


def get_backend(backend_id: str | None = None) -> PhotogrammetryBackend | None:
    """Return a usable backend (requested if available, else first available)."""
    if backend_id is not None:
        factory = _REGISTRY.get(backend_id)
        if factory is not None:
            b = factory()
            if b.is_available():
                return b
    for factory in _REGISTRY.values():
        try:
            b = factory()
            if b.is_available():
                return b
        except Exception:  # noqa: BLE001, S112
            continue
    return None


def available_backends() -> list[str]:
    out = []
    for bid, factory in _REGISTRY.items():
        try:
            if factory().is_available():
                out.append(bid)
        except Exception:  # noqa: BLE001, S112
            continue
    return out


def any_available() -> bool:
    return bool(available_backends())


# Register built-in backends at the bottom to avoid circular imports.
from .photogrammetry_colmap import ColmapBackend

register_backend("colmap", ColmapBackend)
