#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

show_help() {
  cat <<'EOF'
Sermon Transcription Web Application

Usage:
  ./run-web.sh [command]

Commands:
  start     Build and start the web application (default)
  stop      Stop the web application
  restart   Restart the web application
  logs      Show application logs
  build     Build the Docker image
  help      Show this help message

The web interface will be available at:
  http://localhost:5000

Environment variables (set in .env):
  HF_TOKEN              - Hugging Face API token
  ANTHROPIC_API_KEY     - Anthropic API key (for cleanup)
  WHISPER_MODEL         - Model to use (default: large-v3)
  WHISPER_DEVICE        - Device: auto, cpu, cuda (default: auto)
  CLAUDE_MODEL          - Claude model for cleanup (default: claude-sonnet-4-5-20250929)

Data directories:
  ./uploads      - Uploaded audio files
  ./output       - Transcription outputs
  ./model_cache  - Cached Whisper models
EOF
}

cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p uploads output model_cache

# Check for .env file
if [[ ! -f .env ]]; then
  echo "Warning: .env file not found. Creating template..."
  cat > .env <<'ENVFILE'
# Hugging Face token (optional, helps avoid rate limits)
HF_TOKEN=

# Anthropic API key (required for cleanup/summary features)
ANTHROPIC_API_KEY=

# Whisper model settings
WHISPER_MODEL=large-v3
WHISPER_DEVICE=auto

# Claude model for cleanup
CLAUDE_MODEL=claude-sonnet-4-5-20250929
ENVFILE
  echo "Created .env file. Please edit it with your API keys."
fi

case "${1:-start}" in
  start)
    echo "Starting Sermon Transcription Web Application..."
    docker compose up -d
    echo ""
    echo "✅ Web application started!"
    echo "🌐 Open http://localhost:5000 in your browser"
    echo ""
    echo "To view logs: ./run-web.sh logs"
    echo "To stop: ./run-web.sh stop"
    ;;

  stop)
    echo "Stopping web application..."
    docker compose down
    echo "✅ Stopped"
    ;;

  restart)
    echo "Restarting web application..."
    docker compose restart
    echo "✅ Restarted"
    ;;

  logs)
    docker compose logs -f
    ;;

  build)
    echo "Building Docker image..."
    docker compose build
    echo "✅ Build complete"
    ;;

  help|--help|-h)
    show_help
    ;;

  *)
    echo "Unknown command: $1"
    echo "Run './run-web.sh help' for usage information"
    exit 1
    ;;
esac
