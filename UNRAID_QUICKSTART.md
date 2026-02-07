# Unraid Quick Start

## TL;DR - Minimum Required Steps

For Unraid deployment, you only need 3 files and 3 steps:

### Required Files:
1. `docker-compose.yml`
2. `.env` (copy from `.env.example`)
3. Complete `src/` directory

### Steps:

```bash
# 1. Copy to Unraid
# Place files in: /mnt/user/appdata/audio-transcribe/

# 2. Configure .env
cd /mnt/user/appdata/audio-transcribe
cp .env.example .env
nano .env
# Add your API keys and configure storage paths

# 3. Start with Docker Compose
docker compose up -d

# Access at: http://your-unraid-ip:5000
```

## That's It!

Docker will automatically:
- ✅ Build the image
- ✅ Create storage directories
- ✅ Start the web application
- ✅ Download models on first use

## GPU Support (Optional)

If you have an NVIDIA GPU and want to use it:

```bash
# Install NVIDIA GPU Drivers plugin first
# Then use both compose files:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Configuration Options

Edit `.env` to customize:

```bash
# Storage locations (can point to Unraid shares)
UPLOAD_DIR=/mnt/user/audio-transcribe/uploads
OUTPUT_DIR=/mnt/user/audio-transcribe/output
MODEL_CACHE_DIR=/mnt/user/audio-transcribe/model_cache

# API Keys
HF_TOKEN=your_token_here
ANTHROPIC_API_KEY=your_key_here

# Model settings
WHISPER_MODEL=base          # CPU: tiny, base, small
WHISPER_DEVICE=cpu          # CPU: cpu, GPU: auto or cuda
WHISPER_LANGUAGE=en         # or leave empty for auto-detect
```

## No Script Required!

The `run-web.sh` script is optional - it's just a convenience wrapper. For Unraid, use Docker Compose directly via:
- Docker Compose Manager plugin (GUI)
- `docker compose` commands (CLI)

## Full Documentation

See [UNRAID_DEPLOYMENT.md](UNRAID_DEPLOYMENT.md) for detailed instructions, troubleshooting, and advanced configuration.
