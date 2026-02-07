# Unraid Deployment Guide

This guide explains how to deploy the audio transcription application on Unraid using the Docker Compose Manager plugin.

## Prerequisites

1. **Unraid 6.9+** with Docker support enabled
2. **Docker Compose Manager plugin** installed from Community Applications
3. For GPU support: **NVIDIA GPU Drivers plugin** (if using NVIDIA GPU)

## Required Files for Deployment

Only these files are required for Unraid deployment:
- ✅ `docker-compose.yml` (required)
- ✅ `docker-compose.gpu.yml` (only if using GPU)
- ✅ `.env` (required - create from `.env.example`)
- ✅ `src/` directory with application code (required)
- ✅ `Dockerfile.cpu` and/or `Dockerfile.web` (required)

Optional convenience files:
- ⚠️ `run-web.sh` - Helper script (not needed for Unraid)
- ⚠️ `run.sh` - CLI wrapper (not needed for web deployment)

## Installation Steps

### 1. Prepare the Application Directory

SSH into your Unraid server and create a directory for the application:

```bash
cd /mnt/user/appdata
mkdir audio-transcribe
cd audio-transcribe
```

### 2. Clone or Copy the Application

Option A - Using git:
```bash
git clone <repository-url> .
```

Option B - Copy files manually via SMB/NFS to `/mnt/user/appdata/audio-transcribe/`

### 3. Configure Environment Variables

Copy the example environment file and edit it with your settings:

```bash
cp .env.example .env
nano .env
```

Required configuration:
```bash
# Hugging Face token (optional but recommended)
HF_TOKEN=your_huggingface_token_here

# Anthropic API key (required for cleanup/summary features)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model settings for CPU mode (default)
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_LANGUAGE=en

# Claude model
CLAUDE_MODEL=claude-sonnet-4-5-20250929

# Storage directories (customize for your Unraid setup)
UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
MODEL_CACHE_DIR=./model_cache
```

**For Unraid Users**: You can point these to your preferred storage locations:
```bash
# Example: Store data on your Unraid array
UPLOAD_DIR=/mnt/user/audio-transcribe/uploads
OUTPUT_DIR=/mnt/user/audio-transcribe/output
MODEL_CACHE_DIR=/mnt/user/audio-transcribe/model_cache
```

### 4. Choose CPU or GPU Mode

#### CPU Mode (Default)
No additional configuration needed. The default `docker-compose.yml` uses CPU mode.

#### GPU Mode (NVIDIA GPU Required)
1. Install the **NVIDIA GPU Drivers plugin** from Community Applications
2. Verify GPU is accessible: `nvidia-smi`
3. The `run-web.sh` script will auto-detect GPU and use the appropriate configuration

### 5. Create Required Directories (Optional)

Docker will automatically create directories for volume mounts. However, if you're using custom paths or want to set specific permissions, you can create them manually:

```bash
# If using default paths (optional - Docker will create these)
mkdir -p uploads output model_cache
chmod -R 755 uploads output model_cache

# If using custom Unraid paths (example)
mkdir -p /mnt/user/audio-transcribe/{uploads,output,model_cache}
chmod -R 755 /mnt/user/audio-transcribe
```

### 6. Deploy with Docker Compose Manager

> **Note**: The `run-web.sh` script is a convenience wrapper for local development. For Unraid deployment, you can use Docker Compose Manager directly or the command line - the script is optional.

#### Option A: Using Docker Compose Manager UI (Recommended for Unraid)
1. Open Unraid web interface
2. Navigate to **Docker** tab
3. Click **Compose Manager** (or add via Apps)
4. Add new stack:
   - **Stack Name**: `audio-transcribe`
   - **Compose File Path**: `/mnt/user/appdata/audio-transcribe/docker-compose.yml`
   - For GPU support, also add: `/mnt/user/appdata/audio-transcribe/docker-compose.gpu.yml`
5. Click **Compose Up**

#### Option B: Using Command Line
```bash
cd /mnt/user/appdata/audio-transcribe

# CPU mode
docker compose up -d

# GPU mode
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

#### Option C: Using the Helper Script (Optional)
The `run-web.sh` script is a convenience wrapper that auto-detects GPU and creates directories:

```bash
cd /mnt/user/appdata/audio-transcribe

# Auto-detect and start (will use GPU if available)
./run-web.sh start

# Force CPU mode
./run-web.sh start --cpu

# Force GPU mode
./run-web.sh start --gpu
```

**Note**: This script is not required for Unraid deployment. Options A or B above are sufficient.

## Accessing the Application

Once deployed, access the web interface at:
```
http://your-unraid-ip:5000
```

## Volume Mappings

The following directories are mounted as persistent volumes and can be customized via `.env` file:

| Container Path | Default Host Path | Environment Variable | Purpose |
|---------------|-------------------|---------------------|---------|
| `/app/uploads` | `./uploads` | `UPLOAD_DIR` | Uploaded audio files |
| `/app/output` | `./output` | `OUTPUT_DIR` | Transcription outputs |
| `/app/model_cache` | `./model_cache` | `MODEL_CACHE_DIR` | Cached Whisper models |

**Example Unraid Configuration in `.env`:**
```bash
UPLOAD_DIR=/mnt/user/audio-transcribe/uploads
OUTPUT_DIR=/mnt/user/audio-transcribe/output
MODEL_CACHE_DIR=/mnt/user/audio-transcribe/model_cache
```

## Port Configuration

Default port: `5000`

To change the port, edit `.env`:
```bash
FLASK_PORT=8080
```

Then update the port mapping in `docker-compose.yml` or restart:
```bash
docker compose down
docker compose up -d
```

## Model Selection

### CPU Mode Models
For CPU, use smaller models for better performance:
- `tiny` - Fastest, least accurate (~75MB)
- `base` - Good balance (default, ~145MB)
- `small` - Better accuracy (~465MB)

### GPU Mode Models
With NVIDIA GPU, you can use larger models:
- `medium` - High quality (~1.5GB)
- `large-v3` - Best quality (~2.9GB, default for GPU)

To change model, edit `.env`:
```bash
WHISPER_MODEL=small
```

Then restart:
```bash
docker compose restart
```

## Troubleshooting

### Check Container Logs
```bash
docker compose logs -f
```

### Container Won't Start
1. Check logs: `docker compose logs`
2. Verify `.env` file exists and has valid API keys
3. Ensure directories exist and have proper permissions
4. Check port 5000 is not already in use: `netstat -tulpn | grep 5000`

### GPU Not Detected
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check docker can see GPU: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
3. Ensure using GPU compose file: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`

### Model Download Issues
If model downloads fail or are slow:
1. Add HuggingFace token to `.env`
2. Check internet connectivity
3. Check disk space: `df -h`
4. Manually download model cache will be preserved in `./model_cache`

### Templates Not Found Error
If you get a template error, verify the source code includes the templates:
```bash
ls -la src/audio_transcribe/templates/
# Should show: index.html, review.html
```

If missing, ensure you have the complete source code.

## Updating

To update to the latest version:

```bash
cd /mnt/user/appdata/audio-transcribe

# Pull latest code (if using git)
git pull

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Backup

Important directories to backup:
- `./output` - All transcription outputs
- `.env` - Your configuration
- `./model_cache` - Pre-downloaded models (optional, can be re-downloaded)

## Uninstalling

To remove the application:

```bash
cd /mnt/user/appdata/audio-transcribe

# Stop and remove containers
docker compose down

# Remove images (optional)
docker compose down --rmi all

# Remove application directory (WARNING: deletes all data)
cd /mnt/user/appdata
rm -rf audio-transcribe
```

## Support

For issues:
1. Check application logs: `docker compose logs`
2. Verify configuration in `.env`
3. Check Unraid system logs
4. Open an issue on the project repository

## Advanced Configuration

### Custom Model Cache Location
Edit `docker-compose.yml` to change cache location:
```yaml
volumes:
  - /mnt/user/ai-models/whisper:/app/model_cache
```

### Reverse Proxy Setup
If using a reverse proxy (nginx, traefik):
1. Set up proxy to forward to `http://unraid-ip:5000`
2. Ensure WebSocket support is enabled for real-time updates
3. Example nginx config:
```nginx
location / {
    proxy_pass http://unraid-ip:5000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### Resource Limits
To limit CPU/memory usage, edit `docker-compose.yml`:
```yaml
services:
  audio-transcribe-web:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          memory: 4G
```

## Performance Tips

1. **CPU Mode**: Use `base` or `small` models for reasonable speed
2. **GPU Mode**: Use `large-v3` for best quality
3. **Model Cache**: Keep `model_cache` directory to avoid re-downloading
4. **Disk Space**: Ensure adequate space - models can be 1-3GB each
5. **Concurrent Jobs**: Application processes one job at a time by default

## Security Considerations

1. **API Keys**: Keep `.env` file secure, don't commit to git
2. **Network Access**: Consider using Unraid's VPN or firewall rules if exposing externally
3. **File Uploads**: Max file size is 4GB
4. **HTTPS**: Use a reverse proxy with SSL for external access
