FROM python:3.11-slim

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa

RUN apt-get install -y --no-install-recommends libxrender1
RUN apt-get install -y --no-install-recommends libxext6
RUN apt-get install -y --no-install-recommends libsm6
RUN apt-get install -y --no-install-recommends libice6
RUN apt-get install -y --no-install-recommends libx11-6
RUN apt-get install -y --no-install-recommends libxi6
RUN apt-get install -y --no-install-recommends libxrandr2
RUN apt-get install -y --no-install-recommends libxfixes3
RUN apt-get install -y --no-install-recommends libxcursor1
RUN apt-get install -y --no-install-recommends libxinerama1
RUN apt-get install -y --no-install-recommends libfontconfig1

RUN apt-get install -y --no-install-recommends libxft2
RUN apt-get install -y --no-install-recommends libxt6
RUN apt-get install -y --no-install-recommends libxkbcommon0
RUN apt-get install -y --no-install-recommends libdbus-1-3
RUN apt-get install -y --no-install-recommends libxcb1
RUN apt-get install -y --no-install-recommends libgomp1

RUN apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

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
