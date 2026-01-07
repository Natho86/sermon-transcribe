# sermon-transcribe
Sermon audio transcriptions using faster-whisper utilising the GPU

## Development shell
Run this before activating the virtualenv so native deps (zlib/ffmpeg) are available:

    nix-shell

## Docker (GPU)
This is the simplest way to avoid NixOS shared-library issues. Requires NVIDIA Container Toolkit.

  - Build the image:

    docker build -t sermon-transcribe .

  - Confirm CUDA is visible:

    docker run --rm --gpus all sermon-transcribe \
        python3 -c "import ctranslate2; print('cuda', ctranslate2.get_cuda_device_count())"

  - Mount a local folder (example):

    docker run --rm --gpus all -v "$PWD:/app" sermon-transcribe bash

If the CUDA count is still 0, the wheel is likely CPU-only; we can switch to a CUDA-specific wheel or build `ctranslate2` from source inside the image.
