"""Environment fixups that must run before third-party geometry libs import.

On headless Linux (e.g. WSL without a GPU/display), pymeshlab's ``meshing``
plugin fails to load because ``libOpenGL.so.0`` is absent, silently disabling
core filters such as quadric decimation and hole closing.

We preload a vendored copy of ``libOpenGL.so.0`` (extracted from the distro's
``libopengl0`` package) via ctypes with ``RTLD_GLOBAL`` so the symbol is
available when the pymeshlab plugins are dlopen'd. This works without root and
without relying on ``LD_LIBRARY_PATH`` being set before the interpreter starts.

Import this module *before* importing :mod:`pymeshlab`.
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

_LOADED = False


def _candidate_paths() -> list[str]:
    """Locations to look for a usable libOpenGL.so.0."""
    candidates: list[str] = []

    # 1. Vendored copy shipped inside the active venv.
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates.append(str(Path(venv) / "vendored_libs" / "libOpenGL.so.0"))

    # 2. Vendored copy relative to this file's venv (best effort).
    here = Path(__file__).resolve()
    for parent in here.parents:
        vlib = parent / "vendored_libs" / "libOpenGL.so.0"
        if vlib.exists():
            candidates.append(str(vlib))
            break

    # 3. System locations (present if libopengl0 is installed).
    candidates.extend(glob.glob("/usr/lib/*/libOpenGL.so.0"))
    candidates.append("libOpenGL.so.0")  # let the loader search its default path

    return candidates


def ensure_opengl_loaded() -> bool:
    """Best-effort preload of libOpenGL.so.0. Returns True if a load succeeded.

    Idempotent and never raises: if no library can be loaded we return False and
    let downstream code surface a clearer error.
    """
    global _LOADED
    if _LOADED:
        return True
    for path in _candidate_paths():
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            _LOADED = True
            return True
        except OSError:
            continue
    return False
