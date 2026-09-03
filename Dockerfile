FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MOSS_TTS_COMMIT=c0880299e8b8d0f7119efab17e4e776fffe7b8fa

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    TMPDIR=/tmp \
    MODEL_ID=OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
    TORCHDYNAMO_DISABLE=1 \
    VIRTUAL_ENV=/opt/moss-venv \
    PATH=/opt/moss-venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git ffmpeg libsndfile1 python3.12 python3.12-dev \
    python3.12-venv build-essential && \
    python3.12 -m venv /opt/moss-venv && \
    /opt/moss-venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/OpenMOSS/MOSS-TTS.git && \
    cd MOSS-TTS && git checkout ${MOSS_TTS_COMMIT} && \
    /opt/moss-venv/bin/python -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -e "./moss_soundeffect_v2[torch-cu128]" && \
    /opt/moss-venv/bin/python -m pip install "runpod>=1.7,<2"

WORKDIR /app
COPY handler.py /app/handler.py
RUN /opt/moss-venv/bin/python -m py_compile /app/handler.py

CMD ["/opt/moss-venv/bin/python", "-u", "/app/handler.py"]
