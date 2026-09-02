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
    TORCHDYNAMO_DISABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git ffmpeg libsndfile1 python3.12 python3.12-dev \
    build-essential && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/OpenMOSS/MOSS-TTS.git && \
    cd MOSS-TTS && git checkout ${MOSS_TTS_COMMIT} && \
    python3.12 -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
      -e "./moss_soundeffect_v2[torch-cu128]" && \
    python3.12 -m pip install "runpod>=1.7,<2"

WORKDIR /app
COPY handler.py /app/handler.py
RUN mkdir -p /runpod-volume/huggingface /runpod-volume/torch /runpod-volume/tmp

CMD ["python3.12", "-u", "/app/handler.py"]
