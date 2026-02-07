# Audio Transcription with Whisper

A powerful audio transcription application designed for processing sermon recordings, built with faster-whisper. Transcribe audio using CPU or GPU acceleration, then automatically clean and summarize transcripts using AI.

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper model via faster-whisper for accurate speech-to-text
- **CPU & GPU Support**: Runs on CPU for accessibility or GPU (CUDA) for speed
- **AI-Powered Cleanup**: Automatically formats raw transcripts into readable paragraphs and sentences
- **Intelligent Summaries**: Generates concise summaries of transcribed content
- **Web Interface**: User-friendly Flask web application for easy file uploads and management
- **Audio Conversion**: Normalize and convert audio files to M4A or FLAC formats
- **Batch Processing**: Process multiple files or entire directories at once
- **Docker Support**: Easy deployment with Docker and docker-compose

## Requirements

### For Transcription:
- Python 3.11+
- faster-whisper
- ffmpeg
- (Optional) NVIDIA GPU with CUDA support for faster processing

### For Cleanup & Summaries:
- **Anthropic API Key** (required) - Currently supports Claude models
- Support for additional LLM providers coming soon

## Quick Start

### Web Interface (Recommended)

1. **Copy `.env.example` to `.env` and configure:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

2. **Start with Docker Compose:**
   ```bash
   # CPU mode (default)
   docker compose up -d

   # GPU mode (requires NVIDIA GPU)
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   ```

3. **Access the web interface:**
   ```
   http://localhost:5000
   ```

## Configuration

### Environment Variables (.env)

```bash
# API Keys
HF_TOKEN=your_huggingface_token          # Optional, helps avoid rate limits
ANTHROPIC_API_KEY=your_anthropic_key     # Required for cleanup/summary features

# Whisper Model Settings
WHISPER_MODEL=base                        # Options: tiny, base, small, medium, large-v3
WHISPER_DEVICE=cpu                        # Options: cpu, cuda, auto
WHISPER_LANGUAGE=en                       # Or leave empty for auto-detection

# Claude Model for Cleanup
CLAUDE_MODEL=claude-sonnet-4-5-20250929  # Default Claude model

# Storage Directories (customize for your setup)
UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
MODEL_CACHE_DIR=./model_cache
```

### Model Selection

**CPU Mode** (slower, but works everywhere):
- `tiny` - Fastest, least accurate (~75MB)
- `base` - Good balance (default, ~145MB)
- `small` - Better accuracy (~465MB)

**GPU Mode** (requires NVIDIA GPU):
- `medium` - High quality (~1.5GB)
- `large-v3` - Best quality (~2.9GB, default for GPU)

## Command Line Usage

### Local Transcription

Run the CLI directly from source:

```bash
PYTHONPATH=src python3 -m sermon_transcribe \
    --input /path/to/audio_or_dir
```

Common options:
- `--convert-flac` - Convert input WAV files to FLAC before transcription
- `--language en` - Force a language, otherwise auto-detect
- `--model-cache ./model_cache` - Control where the model is cached
- `--raw-suffix _raw` - Control the raw transcript filename suffix
- `--no-skip-existing` - Re-run transcription even if outputs exist
- `--reprocess` - Force reprocessing and overwrite outputs

### Cleanup Transcripts

Clean up `.txt` transcripts into paragraphs and sentences with minimal edits, then generate a summary:

```bash
ANTHROPIC_API_KEY=your_key_here \
PYTHONPATH=src python3 -m sermon_transcribe.cleanup \
    --input /path/to/transcripts_or_dir
```

Outputs:
- Raw transcript: `*_raw.txt` (original is renamed on first cleanup run)
- Cleaned transcript: `*.txt` (same base name, no suffix)
- Summary: `*_summary.txt`

Both files include an AI transcription disclaimer at the top.

Common options:
- `--output /path/to/cleaned` - Write cleaned files elsewhere
- `--suffix _cleaned` - Add a suffix to cleaned filenames (default: none)
- `--raw-suffix _raw` - Set the suffix for original transcripts
- `--summary-suffix _summary` - Change the summary filename suffix
- `--model claude-sonnet-4-5-20250929` - Select a Claude model
- `--no-skip-existing` - Re-run cleanup even if cleaned output exists
- `--reprocess` - Re-run cleanup even if a cleaned file already exists
- `--recursive` - Scan subdirectories

### Audio Conversion

Normalize WAV files to -20 LUFS and convert to M4A:

```bash
./run.sh convert --input /path/to/wav_or_dir --output /path/to/m4a_dir
```

Options:
- `--lufs -20` - Set the integrated loudness target
- `--true-peak -1.5` - Set the true peak limit
- `--lra 11` - Set the loudness range
- `--sample-rate 16000` - Set the output sample rate
- `--channels 1` - Set mono output
- `--jobs 2` - Set concurrent conversions
- `--no-skip-existing` or `--reprocess` - Overwrite existing outputs

## Docker Usage

### Build the Image

```bash
# CPU version
docker build -f Dockerfile.cpu -t audio-transcribe:cpu .

# GPU version (requires NVIDIA Container Toolkit)
docker build -f Dockerfile.web -t audio-transcribe:gpu .
```

### Run with Docker

```bash
# Create directories
mkdir -p uploads output model_cache

# CPU mode
docker run --rm \
  -v ./uploads:/app/uploads \
  -v ./output:/app/output \
  -v ./model_cache:/app/model_cache \
  --env-file .env \
  -p 5000:5000 \
  audio-transcribe:cpu

# GPU mode (requires NVIDIA GPU + Container Toolkit)
docker run --rm \
  --gpus all \
  -v ./uploads:/app/uploads \
  -v ./output:/app/output \
  -v ./model_cache:/app/model_cache \
  --env-file .env \
  -p 5000:5000 \
  audio-transcribe:gpu
```

### Using the Helper Script

```bash
# Build and run
./run-web.sh start              # Auto-detect CPU/GPU
./run-web.sh start --cpu        # Force CPU mode
./run-web.sh start --gpu        # Force GPU mode

# View logs
./run-web.sh logs

# Stop
./run-web.sh stop
```

### CLI Operations with Docker

```bash
# Transcribe audio
docker run --rm --gpus all --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/model_cache:/app/model_cache" \
  audio-transcribe \
  --input /app/path/to/audio_or_dir \
  --model-cache /app/model_cache

# Cleanup transcripts
docker run --rm --env-file .env \
  -v "$PWD:/app" \
  --entrypoint python3 \
  audio-transcribe \
  -m sermon_transcribe.cleanup \
  --input /app/path/to/transcripts_or_dir

# Convert audio
docker run --rm \
  -v "$PWD:/app" \
  --entrypoint python3 \
  audio-transcribe \
  -m sermon_transcribe.convert \
  --input /app/path/to/wav_or_dir \
  --output /app/path/to/m4a_dir
```

## Project Structure

```
.
├── src/sermon_transcribe/     # Main application code
│   ├── webapp.py              # Flask web interface
│   ├── transcription.py       # Whisper transcription logic
│   ├── cleanup.py             # Claude-powered cleanup & summary
│   ├── convert.py             # Audio format conversion
│   └── templates/             # Web UI templates
├── docker-compose.yml         # Docker Compose configuration
├── docker-compose.gpu.yml     # GPU-specific overrides
├── Dockerfile.cpu             # CPU-only Docker image
├── Dockerfile.web             # GPU-enabled Docker image
├── .env.example               # Environment configuration template
├── run-web.sh                 # Helper script for web deployment
└── run.sh                     # Helper script for CLI operations
```

## API Key Setup

### Anthropic API Key

1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Create an API key
3. Add to `.env`:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-api03-...
   ```

### Hugging Face Token (Optional)

1. Sign up at [Hugging Face](https://huggingface.co/)
2. Create an access token
3. Add to `.env`:
   ```bash
   HF_TOKEN=hf_...
   ```

This helps avoid rate limits when downloading models.

## Supported Audio Formats

Input formats:
- WAV
- MP3
- M4A
- FLAC
- Any format supported by ffmpeg

Output formats:
- TXT (transcripts)
- JSON (transcripts with timestamps)
- M4A (converted audio)
- FLAC (converted audio)

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA GPU is available
nvidia-smi

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check ctranslate2 CUDA support
docker run --rm --gpus all audio-transcribe \
  python3 -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

### Model Download Issues

- Ensure internet connectivity
- Add HuggingFace token to `.env` to avoid rate limits
- Check disk space (models can be 500MB - 3GB)
- Models are cached in `MODEL_CACHE_DIR` (default: `./model_cache`)

### Memory Issues

For CPU mode with limited RAM, use smaller models:
```bash
WHISPER_MODEL=tiny  # or base
```

### Web Interface Not Loading

- Check logs: `docker compose logs -f`
- Verify port 5000 is not in use: `netstat -tulpn | grep 5000`
- Ensure `.env` file exists and is configured
- Check that templates directory exists: `ls -la src/sermon_transcribe/templates/`

## Performance

Typical transcription speeds (1 hour of audio):

| Hardware | Model | Time |
|----------|-------|------|
| CPU (8 cores) | base | ~30-60 min |
| CPU (8 cores) | small | ~60-120 min |
| GPU (RTX 3080) | large-v3 | ~5-10 min |
| GPU (RTX 3080) | medium | ~3-5 min |

*Actual performance varies based on audio quality, language, and hardware.*

## Roadmap

- [ ] Support for additional LLM providers (OpenAI, local models)
- [ ] Speaker diarization (identify different speakers)
- [ ] Multiple output formats (SRT, VTT subtitles)
- [ ] REST API for programmatic access
- [ ] Batch job queuing and management
- [ ] Audio preprocessing options
- [ ] Custom vocabulary/terminology support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add your license here]

## Acknowledgments

- Built with [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Powered by [OpenAI Whisper](https://github.com/openai/whisper)
- AI cleanup by [Anthropic Claude](https://www.anthropic.com/)
