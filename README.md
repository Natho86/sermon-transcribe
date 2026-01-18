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
  - `--raw-suffix _raw` to control the raw transcript filename suffix.
  - `--no-skip-existing` to re-run transcription even if outputs exist.
  - `--reprocess` to force reprocessing and overwrite outputs.

## Cleanup transcripts (Claude)
Clean up `.txt` transcripts into paragraphs and sentences with minimal edits, then generate a summary:

    ANTHROPIC_API_KEY=your_key_here \
    PYTHONPATH=src python3 -m sermon_transcribe.cleanup \
        --input /path/to/transcripts_or_dir

Outputs:

  - Raw transcript: `*_raw.txt` (original is renamed on first cleanup run)
  - Cleaned transcript: `*.txt` (same base name, no suffix)
  - Summary: `*_summary.txt`

Both files include an AI transcription disclaimer at the top.

Common options:

  - `--output /path/to/cleaned` to write cleaned files elsewhere.
  - `--suffix _cleaned` to add a suffix to cleaned filenames (default: none).
  - `--raw-suffix _raw` to set the suffix for original transcripts.
  - `--summary-suffix _summary` to change the summary filename suffix.
  - `--model claude-sonnet-4-5-20250929` to select a Claude model.
  - `--no-skip-existing` to re-run cleanup even if cleaned output exists.
  - `--reprocess` to re-run cleanup even if a cleaned file already exists.
  - `--recursive` to scan subdirectories.

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
  - Absolute paths like `/home/...` are supported; the wrapper will map them into the container.

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

## Wrapper cleanup
Use the Docker wrapper to clean up transcript text files with Claude and generate summaries:

    ./run.sh cleanup --input /app/path/to/transcripts_or_dir

The wrapper reads `ANTHROPIC_API_KEY` from `./.env`.

## Convert WAV to M4A (normalized)
Normalize WAV files to -20 LUFS and convert to M4A:

    ./run.sh convert --input /path/to/wav_or_dir --output /path/to/m4a_dir

Defaults:

  - `--lufs -20` to set the integrated loudness target.
  - `--true-peak -1.5` to set the true peak limit.
  - `--lra 11` to set the loudness range.
  - `--sample-rate 16000` to set the output sample rate.
  - `--channels 1` to set mono output.
  - `--jobs 2` to set concurrent conversions.
  - `--no-skip-existing` or `--reprocess` to overwrite existing outputs.

If the CUDA count is still 0, the wheel is likely CPU-only; we can switch to a CUDA-specific wheel or build `ctranslate2` from source inside the image.
