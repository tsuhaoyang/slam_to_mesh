"""Tests for the remesh backend registry and fallback behavior."""

from __future__ import annotations

import pytest

from slam_to_mesh.backends import remesh as remesh_mod
from slam_to_mesh.backends.remesh import (
    RemeshRequest,
    RemeshResult,
    available_backends,
    get_backend,
    register_backend,
)


def test_builtin_backends_registered():
    backends = available_backends()
    assert "pymeshlab_cpu" in backends
    # The config default alias maps to the CPU backend.
    assert "quadriflow_cpu" in backends


def test_get_backend_returns_requested_when_available():
    b = get_backend("pymeshlab_cpu")
    assert b.id == "pymeshlab_cpu"


def test_get_backend_falls_back_to_cpu_for_unknown_id():
    b = get_backend("does_not_exist_gpu_9000")
    # Falls back to the registered CPU backend.
    assert b.id == "pymeshlab_cpu"


def test_get_backend_falls_back_when_unavailable():
    """A registered-but-unavailable backend triggers CPU fallback."""

    class _Unavailable:
        id = "fake_gpu"

        def is_available(self) -> bool:
            return False

        def remesh(self, req: RemeshRequest) -> RemeshResult:  # pragma: no cover
            raise AssertionError("should not be called")

    register_backend("fake_gpu", _Unavailable)
    try:
        b = get_backend("fake_gpu")
        assert b.id == "pymeshlab_cpu"
    finally:
        remesh_mod._REGISTRY.pop("fake_gpu", None)


def test_register_and_select_available_backend():
    """A custom available backend is selected and listed."""

    class _Custom:
        id = "custom_ok"

        def is_available(self) -> bool:
            return True

        def remesh(self, req: RemeshRequest) -> RemeshResult:  # pragma: no cover
            return RemeshResult(output_path=req.output_path, metrics={})

    register_backend("custom_ok", _Custom)
    try:
        assert "custom_ok" in available_backends()
        assert get_backend("custom_ok").id == "custom_ok"
    finally:
        remesh_mod._REGISTRY.pop("custom_ok", None)


def test_available_backends_swallows_faulty_factory():
    """A factory that raises must not break availability listing."""

    def _boom():
        raise RuntimeError("bad factory")

    register_backend("boom", _boom)
    try:
        # Should not raise; faulty backend simply omitted.
        backends = available_backends()
        assert "boom" not in backends
        assert "pymeshlab_cpu" in backends
    finally:
        remesh_mod._REGISTRY.pop("boom", None)


# --------------------------------------------------------------------------- #
# QuadriFlow backend
# --------------------------------------------------------------------------- #


def test_quadriflow_registered():
    """QuadriFlow is registered under both ids regardless of binary presence."""
    assert "quadriflow" in remesh_mod._REGISTRY
    assert "quadriflow_gpu" in remesh_mod._REGISTRY


def test_quadriflow_availability_matches_binary():
    """is_available() agrees with binary discovery, and drives the registry."""
    from slam_to_mesh.backends.quadriflow import (
        QuadriFlowRemeshBackend,
        find_quadriflow_binary,
    )

    has_binary = find_quadriflow_binary() is not None
    assert QuadriFlowRemeshBackend().is_available() == has_binary
    # When the binary is absent, requesting quadriflow falls back to CPU.
    if not has_binary:
        assert get_backend("quadriflow").id == "pymeshlab_cpu"
        assert "quadriflow" not in available_backends()
    else:
        assert get_backend("quadriflow").id == "quadriflow"
        assert "quadriflow" in available_backends()


def _quadriflow_available() -> bool:
    from slam_to_mesh.backends.quadriflow import find_quadriflow_binary

    return find_quadriflow_binary() is not None


@pytest.mark.skipif(
    not _quadriflow_available(), reason="quadriflow binary not built"
)
def test_quadriflow_remesh_produces_quads(tmp_path):
    """End-to-end: QuadriFlow turns a triangle mesh into a quad-dominant OBJ."""
    import trimesh

    from slam_to_mesh.backends.quadriflow import QuadriFlowRemeshBackend
    from slam_to_mesh.backends.remesh import RemeshRequest

    src = tmp_path / "in.obj"
    trimesh.creation.icosphere(subdivisions=3, radius=1.0).export(str(src))
    out = tmp_path / "out.obj"

    backend = QuadriFlowRemeshBackend()
    res = backend.remesh(
        RemeshRequest(
            input_path=src,
            output_path=out,
            target_faces=800,
            quads=True,
            preserve_sharp=True,
            bbox_diagonal=2.0 * (3 ** 0.5),
        )
    )
    assert out.exists()
    assert res.metrics["polygon_faces"] > 0
    assert res.metrics["quads"] > 0
    assert res.metrics["quad_ratio"] > 0.5
