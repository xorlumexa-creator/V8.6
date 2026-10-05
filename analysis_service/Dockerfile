FROM python:3.11-slim
# CalculiX + the shared libraries the gmsh wheel loads
RUN apt-get update && apt-get install -y --no-install-recommends \
    calculix-ccx libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 libfontconfig1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py fem_core.py ./
ENV PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 CCX_THREADS=2
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
