FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CT2_FORCE_CPU=0

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip \
  && python3 -m pip install --no-cache-dir \
    faster-whisper \
    ctranslate2
