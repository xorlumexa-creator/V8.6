FROM python:3.11-slim

# This service (generation: AI + CadQuery script execution + STEP/STL export)
# no longer needs Gmsh or CalculiX at all — that whole heavy dependency chain
# moved to the separate analysis service (see analysis_service.py / its own
# Dockerfile). Confirmed necessary via a real OOM crash on Render free tier:
# a Termux curl test showed the connection dying mid-request (0 bytes
# received), immediately followed by Render auto-restarting the container in
# the logs — the signature of the process being killed for memory.
#
# Still needed: X11/font runtime libraries, even fully headless — OCP
# (cadquery's OpenCASCADE binding) dynamically loads these at import time
# regardless of whether anything is ever displayed. This exact list was
# built the hard way, one missing library at a time, against real crash logs
# — kept as one block since every entry here was independently confirmed
# necessary.
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
