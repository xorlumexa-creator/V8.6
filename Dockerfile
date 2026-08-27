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

# Step 3: X11/font runtime libs, ONE PER STEP — deliberately not batched this
# time. After 2 rounds of guessing which package in a batch was the problem,
# splitting to one-per-step means the exact failing step tells us the exact
# bad package with certainty on the next attempt, instead of narrowing a list
# again. Once we know which one it is, these collapse back into one RUN line.
RUN apt-get install -y --no-install-recommends libxrender1
RUN apt-get install -y --no-install-recommends libxext6
RUN apt-get install -y --no-install-recommends libsm6
RUN apt-get install -y --no-install-recommends libice6
RUN apt-get install -y --no-install-recommends libx11-6
RUN apt-get install -y --no-install-recommends libxi6
RUN apt-get install -y --no-install-recommends libxrandr1
RUN apt-get install -y --no-install-recommends libxfixes3
RUN apt-get install -y --no-install-recommends libxcursor1
RUN apt-get install -y --no-install-recommends libxinerama1
RUN apt-get install -y --no-install-recommends libfontconfig1

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
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
