#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

show_help() {
  cat <<'EOF'
Sermon Transcription Web Application

Usage:
  ./run-web.sh [command] [--gpu]

Commands:
  start     Build and start the web application (default)
  stop      Stop the web application
  restart   Restart the web application
  logs      Show application logs
  build     Build the Docker image
  help      Show this help message

Options:
  --gpu     Force GPU mode (requires NVIDIA GPU + nvidia-docker)
  --cpu     Force CPU mode (default if GPU not detected)

The web interface will be available at:
  http://localhost:5000

GPU Detection:
  The script auto-detects NVIDIA GPU availability.
  - GPU mode: Uses CUDA acceleration with Dockerfile.web
  - CPU mode: Uses CPU-only processing with Dockerfile.cpu

Environment variables (set in .env):
  HF_TOKEN              - Hugging Face API token
  ANTHROPIC_API_KEY     - Anthropic API key (for cleanup)
  WHISPER_MODEL         - Model to use
                          CPU: base, small (default: base)
                          GPU: medium, large-v3 (default: large-v3)
  WHISPER_DEVICE        - Device: auto, cpu, cuda
                          CPU mode default: cpu
                          GPU mode default: auto
  CLAUDE_MODEL          - Claude model for cleanup (default: claude-sonnet-4-5-20250929)

Data directories:
  ./uploads      - Uploaded audio files
  ./output       - Transcription outputs
  ./model_cache  - Cached Whisper models
EOF
}

# Detect GPU availability
detect_gpu() {
  # Check if nvidia-smi exists and can detect GPUs
  if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
      # Check if docker supports nvidia runtime
      if docker info 2>/dev/null | grep -q "Runtimes.*nvidia" || \
         docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        return 0  # GPU available
      fi
    fi
  fi
  return 1  # No GPU
}

cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p uploads output model_cache

# Check for .env file
if [[ ! -f .env ]]; then
  echo "Warning: .env file not found. Creating template..."

  # Detect GPU for default template
  if detect_gpu; then
    cat > .env <<'ENVFILE'
# Hugging Face token (optional, helps avoid rate limits)
HF_TOKEN=

# Anthropic API key (required for cleanup/summary features)
ANTHROPIC_API_KEY=

# Whisper model settings (GPU detected)
WHISPER_MODEL=large-v3
WHISPER_DEVICE=auto

# Claude model for cleanup
CLAUDE_MODEL=claude-sonnet-4-5-20250929
ENVFILE
    echo "Created .env file with GPU defaults. Please edit it with your API keys."
  else
    cat > .env <<'ENVFILE'
# Hugging Face token (optional, helps avoid rate limits)
HF_TOKEN=

# Anthropic API key (required for cleanup/summary features)
ANTHROPIC_API_KEY=

# Whisper model settings (CPU mode)
# For CPU, use smaller models: tiny, base, small
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Claude model for cleanup
CLAUDE_MODEL=claude-sonnet-4-5-20250929
ENVFILE
    echo "Created .env file with CPU defaults. Please edit it with your API keys."
  fi
fi

# Parse GPU mode from arguments
USE_GPU=""
COMMAND=""
for arg in "$@"; do
  case "$arg" in
    --gpu)
      USE_GPU="true"
      ;;
    --cpu)
      USE_GPU="false"
      ;;
    *)
      if [[ -z "$COMMAND" ]]; then
        COMMAND="$arg"
      fi
      ;;
  esac
done

# Auto-detect GPU if not explicitly set
if [[ -z "$USE_GPU" ]]; then
  if detect_gpu; then
    USE_GPU="true"
  else
    USE_GPU="false"
  fi
fi

# Set up docker compose command
if [[ "$USE_GPU" == "true" ]]; then
  COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
  MODE_MSG="GPU mode"
else
  COMPOSE_CMD="docker compose"
  MODE_MSG="CPU mode"
fi

case "${COMMAND:-start}" in
  start)
    echo "Starting Sermon Transcription Web Application ($MODE_MSG)..."
    $COMPOSE_CMD up -d
    echo ""
    echo "✅ Web application started in $MODE_MSG!"
    echo "🌐 Open http://localhost:5000 in your browser"
    echo ""
    echo "To view logs: ./run-web.sh logs"
    echo "To stop: ./run-web.sh stop"
    ;;

  stop)
    echo "Stopping web application..."
    # Stop both possible configurations
    docker compose down 2>/dev/null || true
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml down 2>/dev/null || true
    echo "✅ Stopped"
    ;;

  restart)
    echo "Restarting web application ($MODE_MSG)..."
    $COMPOSE_CMD restart
    echo "✅ Restarted"
    ;;

  logs)
    $COMPOSE_CMD logs -f
    ;;

  build)
    echo "Building Docker image ($MODE_MSG)..."
    $COMPOSE_CMD build
    echo "✅ Build complete"
    ;;

  help|--help|-h)
    show_help
    ;;

  *)
    echo "Unknown command: ${COMMAND}"
    echo "Run './run-web.sh help' for usage information"
    exit 1
    ;;
esac
