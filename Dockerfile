FROM python:3.11-slim

# System dependencies via apt: these are standard, widely-available Debian
# packages (unlike calculix-ccx below, these reliably resolve).
# - libgl1 / libglu1-mesa: OpenGL libs that OCP (cadquery's OpenCASCADE binding)
#   and some trimesh code paths expect to find even when running headless.
# - libxrender1, libxext6, libsm6, libice6, libx11-6, libxi6, libxrandr1,
#   libxfixes3, libxcursor1, libxinerama1, libfontconfig1: X11/font runtime
#   libraries that OpenCASCADE (via OCP) dynamically loads at import time even
#   in a fully headless container with nothing ever displayed.
# - build-essential: a few pip packages compile small extensions on install.
# - wget/bzip2: needed to fetch and unpack the Miniforge installer below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libx11-6 \
    libxi6 \
    libxrandr1 \
    libxfixes3 \
    libxcursor1 \
    libxinerama1 \
    libfontconfig1 \
    build-essential \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# CalculiX (`ccx`) is NOT reliably available via Debian's apt repos — that was
# tried first and failed with "unable to locate package" on this base image.
# conda-forge does package it reliably, so install a minimal conda (Miniforge)
# just to pull `ccx` from there, then leave conda alone and use plain pip for
# everything else in requirements.txt.
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda install -y -c conda-forge calculix \
    && /opt/conda/bin/conda clean -afy \
    && ln -s /opt/conda/bin/ccx /usr/local/bin/ccx
# Deliberately NOT adding /opt/conda/bin to PATH — that would put conda's own
# Python ahead of this image's python:3.11-slim on PATH and could silently
# change which Python `pip install`/`uvicorn` below actually use. Only the
# `ccx` binary itself is symlinked out, so the rest of this image's Python
# environment is untouched.
#
# The symlink alone isn't enough, though: `ccx` dynamically links against
# shared libraries (gfortran, BLAS, etc.) that conda-forge installed into
# /opt/conda/lib, which a plain symlink doesn't carry along. LD_LIBRARY_PATH
# (not PATH) tells the dynamic linker where to find those at runtime, without
# affecting which `python`/`pip` gets resolved.
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render (and most PaaS Docker hosts) inject $PORT at runtime — bind to it
# rather than a hardcoded port, or the platform's health check will fail.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
