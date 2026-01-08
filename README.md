# sermon-transcribe
Sermon audio transcriptions using faster-whisper utilising the GPU

## Project structure
  - `src/sermon_transcribe`: CLI and transcription pipeline

## Local usage
Run the CLI directly from source:

    PYTHONPATH=src python3 -m sermon_transcribe \
        --input /path/to/audio_or_dir

Common options:

  - `--convert-flac` to convert input WAV files to FLAC before transcription.
  - `--language en` to force a language, otherwise auto-detect.
  - `--model-cache ./model_cache` to control where the model is cached.

## Docker (GPU)
This is the simplest way to avoid NixOS shared-library issues. Requires NVIDIA Container Toolkit.

  - Build the image once:

    docker build -t sermon-transcribe .

  - Use the wrapper script for day-to-day runs:

    ./run.sh --input /app/path/to/audio_or_dir

The wrapper uses:
  - `./output` for transcripts
  - `./model_cache` for cached model downloads
  - `./.env` for your `HF_TOKEN`

  - Confirm CUDA is visible:

    docker run --rm --device nvidia.com/gpu=all sermon-transcribe \
        python3 -c "import ctranslate2; print('cuda', ctranslate2.get_cuda_device_count())"

  - Mount a local folder (example):

    docker run --rm --device nvidia.com/gpu=all -v "$PWD:/app" sermon-transcribe bash

  - Run the CLI inside the container (example):

    docker run --rm --device nvidia.com/gpu=all -v "$PWD:/app" sermon-transcribe \
        --input /app/path/to/audio_or_dir

  - Use your Hugging Face token from `.env` (avoids rate limits):

    docker run --rm --device nvidia.com/gpu=all --env-file .env -v "$PWD:/app" sermon-transcribe \
        --input /app/path/to/audio_or_dir

  - Persist the model cache between runs:

    mkdir -p model_cache
    docker run --rm --device nvidia.com/gpu=all --env-file .env \
        -v "$PWD:/app" \
        -v "$PWD/model_cache:/app/model_cache" \
        sermon-transcribe \
        --input /app/path/to/audio_or_dir \
        --model-cache /app/model_cache

If the CUDA count is still 0, the wheel is likely CPU-only; we can switch to a CUDA-specific wheel or build `ctranslate2` from source inside the image.
