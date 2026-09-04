# slam_to_mesh — CPU image (pipeline + FastAPI service).
#
# Bundles the awkward native deps so the app "just runs": pymeshlab (needs
# OpenGL libs even headless), a source-built QuadriFlow (field-aligned quad
# remesh), COLMAP (CPU build, photogrammetry), Open3D, ffmpeg (video frames),
# xatlas, usd-core. GPU-only pieces (CUDA COLMAP, TripoSR) are intentionally
# left to host install scripts — see docs/deployment.md.
#
# Multi-stage: build QuadriFlow in a builder, copy the binary into a lean runtime.

# --------------------------------------------------------------------------- #
# Stage 1: build QuadriFlow from source
# --------------------------------------------------------------------------- #
FROM ubuntu:22.04 AS quadriflow-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libeigen3-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --recursive https://github.com/hjwdzh/QuadriFlow.git \
    && cd QuadriFlow \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=release -DBUILD_OPENMP=ON \
    && make -j"$(nproc)" \
    && test -f quadriflow

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # slam_to_mesh finds the QuadriFlow binary here (backends/quadriflow.py
    # checks QUADRIFLOW_BIN first).
    QUADRIFLOW_BIN=/opt/quadriflow/quadriflow \
    # Where the service stores jobs (mounted as a volume in compose).
    JOBS_ROOT=/data/service_jobs

# Runtime system deps:
# - python3 + venv/pip
# - OpenGL/GLU/gomp: pymeshlab meshing plugin needs libOpenGL/libGLU even headless
# - libgl1/libglib2/libsm/libxext/libxrender: Open3D + trimesh + rendering libs
# - colmap: photogrammetry (CPU build from Ubuntu repos)
# - ffmpeg: video → frames
# - libeigen/boost runtime: QuadriFlow binary's shared deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
        libgl1 libglu1-mesa libopengl0 libgomp1 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
        colmap ffmpeg \
        libboost-graph1.74.0 libboost-program-options1.74.0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# QuadriFlow binary from the builder stage.
COPY --from=quadriflow-builder /src/QuadriFlow/build/quadriflow /opt/quadriflow/quadriflow

# Python virtualenv (isolated from system python).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
# Install deps first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[service,usd]"

# Non-root user; data dir for jobs.
RUN useradd -m -u 1000 app \
    && mkdir -p /data/service_jobs \
    && chown -R app:app /app /data
USER app

EXPOSE 8000

# The service reads JOBS_ROOT via env (see service/app.py note); default CMD runs it.
CMD ["uvicorn", "slam_to_mesh.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
