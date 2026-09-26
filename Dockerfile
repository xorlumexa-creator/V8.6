FROM python:3.11-slim

# Generation service: LLM (Nemotron via NVIDIA NIM) + build123d script execution +
# STEP/STL export + SimScale cloud FEA client. No Gmsh/CalculiX here — that stays in the
# optional separate analysis service (ANALYSIS_SERVICE_URL).
#
# build123d uses the same OpenCASCADE binding (cadquery-ocp / "OCP") CadQuery did, which
# dynamically loads these X11/GL/font runtime libraries at import time even fully
# headless. This list is unchanged from the CadQuery image, where every entry was
# confirmed necessary against real crash logs. If a future build123d/OCP release logs
# "libXXX.so: cannot open shared object file", add that library's apt package here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libx11-6 \
    libxi6 \
    libxrandr2 \
    libxfixes3 \
    libxcursor1 \
    libxinerama1 \
    libfontconfig1 \
    libxft2 \
    libxt6 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb1 \
    libgomp1 \
    libegl1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# libegl1 + libosmesa6: software/off-screen OpenGL for VTK, used only when PYVISTA_RENDER=1.
#
# MALLOC_ARENA_MAX keeps glibc from fragmenting memory across threads (SimScale runs and
# LLM calls use worker threads) — matters on Render's 512 MB free instance.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .

# One worker on purpose: background jobs (JOB_STORE) and the SimScale lock live in process memory.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
