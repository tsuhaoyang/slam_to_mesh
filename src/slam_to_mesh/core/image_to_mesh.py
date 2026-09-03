"""Image → mesh: thin compatibility layer over the image3d backend registry.

Single-image 3D reconstruction is a swappable backend
(:mod:`slam_to_mesh.backends.image3d`); this module keeps a small stable API used
by the ingest stage and service:

* :data:`IMAGE_EXTS`, :func:`is_image_file` — accepted image inputs.
* :func:`is_available` — whether any image backend can run here.
* :func:`generate_mesh` — run the selected (or first available) backend.

The concrete models (TripoSR now, TRELLIS later) live under
``slam_to_mesh.backends.image3d*``.
"""

from __future__ import annotations

from pathlib import Path

from ..backends import image3d
from ..backends.image3d_triposr import find_triposr_dir  # re-export (compat)

#: Image extensions accepted as pipeline input.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

__all__ = [
    "IMAGE_EXTS",
    "find_triposr_dir",
    "generate_mesh",
    "is_available",
    "is_image_file",
]


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_available(backend_id: str | None = None) -> bool:
    """True if the requested (or any) image3d backend can run here."""
    if backend_id is not None:
        return image3d.get_backend(backend_id) is not None
    return image3d.any_available()


def generate_mesh(
    image_path: str | Path,
    out_mesh: str | Path,
    backend_id: str | None = None,
) -> dict:
    """Reconstruct a mesh from *image_path* using an image3d backend.

    Uses *backend_id* when given and available, otherwise the first available
    backend. Raises RuntimeError with a clear message when none is available.
    """
    backend = image3d.get_backend(backend_id)
    if backend is None:
        raise RuntimeError(
            "no image-to-3D backend available; install TripoSR (see "
            "docs/triposr_2d_to_3d.md) or configure another backend"
        )
    return backend.generate_mesh(Path(image_path), Path(out_mesh))
