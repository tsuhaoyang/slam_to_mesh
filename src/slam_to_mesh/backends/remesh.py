"""Remesh backend abstraction.

Quad remeshing is the stage most likely to be GPU-accelerated later (e.g.
Instant Meshes / QuadriFlow on a GPU server). To keep the pipeline core stable
we hide the implementation behind a small protocol and a registry keyed by a
backend id stored in the job manifest (``RemeshConfig.backend``).

A backend takes an input mesh file and produces a quad-dominant mesh file,
returning metrics. It must run to completion on the target machine; the CPU
backend here uses pymeshlab and requires no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class RemeshRequest:
    """Inputs for a remesh operation."""

    input_path: Path
    output_path: Path
    target_faces: int
    quads: bool = True
    preserve_sharp: bool = True
    feature_lines: Path | None = None
    #: Precomputed bbox diagonal, used to derive absolute edge lengths.
    bbox_diagonal: float = 1.0


@dataclass
class RemeshResult:
    """Outputs from a remesh operation."""

    output_path: Path
    metrics: dict = field(default_factory=dict)


class RemeshBackend(Protocol):
    """Contract every remesh backend must satisfy."""

    id: str

    def is_available(self) -> bool:
        """Return True if this backend can run in the current environment."""
        ...

    def remesh(self, req: RemeshRequest) -> RemeshResult:
        """Perform the remesh, writing to ``req.output_path``."""
        ...


_REGISTRY: dict[str, Callable[[], RemeshBackend]] = {}


def register_backend(backend_id: str, factory: Callable[[], RemeshBackend]) -> None:
    """Register a backend factory under *backend_id*."""
    _REGISTRY[backend_id] = factory


def get_backend(backend_id: str) -> RemeshBackend:
    """Instantiate a registered backend, falling back to CPU if unavailable.

    If the requested backend id is unknown or reports itself unavailable (e.g. a
    GPU backend on a CPU-only box), we fall back to the CPU backend so the
    pipeline still completes.
    """
    factory = _REGISTRY.get(backend_id)
    if factory is not None:
        backend = factory()
        if backend.is_available():
            return backend
    # Fallback: CPU backend.
    cpu = _REGISTRY["quadriflow_cpu"]()
    return cpu


def available_backends() -> list[str]:
    """List registered backend ids that report themselves available."""
    out = []
    for bid, factory in _REGISTRY.items():
        try:
            if factory().is_available():
                out.append(bid)
        except Exception:  # noqa: BLE001
            continue
    return out


# Register built-in backends. Import here to avoid circulars at module top.
from .remesh_pymeshlab import PyMeshLabRemeshBackend  # noqa: E402

register_backend("pymeshlab_cpu", PyMeshLabRemeshBackend)
# "quadriflow_cpu" is the pymeshlab CPU remesher (isotropic + tri-to-quad
# pairing). It is the guaranteed-available fallback used by get_backend().
register_backend("quadriflow_cpu", PyMeshLabRemeshBackend)

# Field-aligned QuadriFlow backend (subprocess). Available only when the
# quadriflow binary is found; otherwise get_backend() falls back to CPU. The
# same backend is exposed under a "_gpu" alias — acceleration is a property of
# how the binary was built, not of the id.
from .quadriflow import QuadriFlowRemeshBackend

register_backend("quadriflow", QuadriFlowRemeshBackend)
register_backend("quadriflow_gpu", QuadriFlowRemeshBackend)
