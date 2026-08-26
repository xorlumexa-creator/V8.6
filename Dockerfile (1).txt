FROM python:3.11-slim

# System dependencies:
# - calculix-ccx: provides the `ccx` binary this backend shells out to for real
#   FEM (both the solid-tet and shell paths in main.py). If this package name
#   isn't in your base image's apt repos, fall back to conda-forge's `calculix`
#   package (miniconda-based image) instead — untested from this environment,
#   verify on first build.
# - libgl1 / libglu1-mesa: OpenGL libs that OCP (cadquery's OpenCASCADE binding)
#   and some trimesh code paths expect to find even when running headless.
# - build-essential: a few pip packages compile small extensions on install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    calculix-ccx \
    libgl1 \
    libglu1-mesa \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render (and most PaaS Docker hosts) inject $PORT at runtime — bind to it
# rather than a hardcoded port, or the platform's health check will fail.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
