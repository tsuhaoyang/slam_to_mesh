"""Single-image → 3D backend registry.

Turning a 2D image into a 3D mesh is a swappable step, mirroring the remesh
backend registry (:mod:`slam_to_mesh.backends.remesh`). Each backend wraps an
external model — typically run in its **own isolated venv via subprocess** — and
reports whether it is available on this machine.

The first implementation is **TripoSR** (fast, low VRAM). A stronger model such
as **TRELLIS** (Microsoft, MIT) can be added later on a bigger-VRAM server by
registering another backend; nothing else in the pipeline changes.

Unlike remesh, there is no universally-available CPU fallback here: if no image
backend is available, image input is simply unavailable (the ingest stage /
service surface a clear error), gated like the QuadriFlow binary.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class Image3DBackend(Protocol):
    """Contract every single-image 3D backend must satisfy."""

    id: str

    def is_available(self) -> bool:
        """True if this backend can run in the current environment."""
        ...

    def generate_mesh(self, image_path: Path, out_mesh: Path) -> dict:
        """Reconstruct a mesh from *image_path*, writing to *out_mesh*."""
        ...


_REGISTRY: dict[str, Callable[[], Image3DBackend]] = {}


def register_backend(backend_id: str, factory: Callable[[], Image3DBackend]) -> None:
    """Register a backend factory under *backend_id*."""
    _REGISTRY[backend_id] = factory


def get_backend(backend_id: str | None = None) -> Image3DBackend | None:
    """Return a usable backend.

    If *backend_id* is given, return it when available. If it is ``None`` or the
    requested one is unavailable, return the first available registered backend.
    Returns ``None`` when no backend is available at all.
    """
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
    """List registered backend ids that report themselves available."""
    out = []
    for bid, factory in _REGISTRY.items():
        try:
            if factory().is_available():
                out.append(bid)
        except Exception:  # noqa: BLE001, S112
            continue
    return out


def any_available() -> bool:
    """True if at least one image3d backend can run here."""
    return bool(available_backends())


# Register built-in backends. Import at the bottom to avoid circular imports.
from .image3d_triposr import TripoSRBackend

register_backend("triposr", TripoSRBackend)
