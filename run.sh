#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$SCRIPT_DIR/output" "$SCRIPT_DIR/model_cache"

if [[ $# -eq 0 ]]; then
  set -- --help
fi

exec docker run --rm --device nvidia.com/gpu=all \
  --env-file "$SCRIPT_DIR/.env" \
  -v "$SCRIPT_DIR:/app" \
  -v "$SCRIPT_DIR/model_cache:/app/model_cache" \
  -v "$SCRIPT_DIR/output:/app/output" \
  sermon-transcribe \
  --model-cache /app/model_cache \
  "$@"
