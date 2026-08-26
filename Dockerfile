FROM python:3.11-slim

# Split into isolated RUN steps (rather than one big chained command) so that
# if any step fails, Render's build log shows exactly WHICH step failed —
# the previous single-command version gave only a generic wrapper error with
# no way to tell whether `apt-get update` itself failed (network/mirror
# issue) or a specific package name was bad.

# Step 1: update package index alone — isolates a network/mirror-reachability
# failure from a package-resolution failure.
RUN apt-get update

# Step 2: OpenGL libs (small, minimal group)
RUN apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa

# Step 3: X11/font runtime libs that OCP (cadquery's OpenCASCADE binding)
# needs at import time even in a fully headless container.
RUN apt-get install -y --no-install-recommends \
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
    libfontconfig1

# Step 4: build tools + fetch tools for the CalculiX install below.
RUN apt-get install -y --no-install-recommends \
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

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
