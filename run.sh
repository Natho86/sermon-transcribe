#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$SCRIPT_DIR/output" "$SCRIPT_DIR/model_cache"

show_help() {
  cat <<'EOF'
Usage:
  ./run.sh [transcribe] --input /path/to/audio_or_dir [options]
  ./run.sh cleanup --input /path/to/transcripts_or_dir [options]
  ./run.sh convert --input /path/to/wav_or_dir --output /path/to/m4a_dir [options]

Notes:
  - The wrapper mounts .env for HF_TOKEN and ANTHROPIC_API_KEY.
  - Absolute paths are mapped into the container under /host/...
EOF
}

mode="transcribe"
if [[ $# -gt 0 ]]; then
  case "$1" in
    cleanup|transcribe|convert)
      mode="$1"
      shift
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
  esac
fi

if [[ $# -eq 0 ]]; then
  show_help
  exit 0
fi

map_path() {
  local raw_path="$1"
  if [[ "$raw_path" == "~"* ]]; then
    raw_path="${HOME}${raw_path:1}"
  fi
  if [[ "$raw_path" == /* ]]; then
    echo "/host${raw_path}"
    return 0
  fi
  echo "$raw_path"
}

args=()
mount_host=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      mapped=$(map_path "$2")
      [[ "$mapped" == /host/* ]] && mount_host=true
      args+=("--input" "$mapped")
      shift 2
      ;;
    --input=*)
      mapped=$(map_path "${1#*=}")
      [[ "$mapped" == /host/* ]] && mount_host=true
      args+=("--input=$mapped")
      shift 1
      ;;
    --output|--out)
      mapped=$(map_path "$2")
      [[ "$mapped" == /host/* ]] && mount_host=true
      args+=("--output" "$mapped")
      shift 2
      ;;
    --output=*|--out=*)
      mapped=$(map_path "${1#*=}")
      [[ "$mapped" == /host/* ]] && mount_host=true
      args+=("--output=$mapped")
      shift 1
      ;;
    *)
      args+=("$1")
      shift 1
      ;;
  esac
done

docker_args=(
  --rm
  --device nvidia.com/gpu=all
  --env-file "$SCRIPT_DIR/.env"
  -v "$SCRIPT_DIR:/app"
  -v "$SCRIPT_DIR/model_cache:/app/model_cache"
  -v "$SCRIPT_DIR/output:/app/output"
)

if [[ "$mount_host" == true ]]; then
  docker_args+=(-v "/:/host")
fi

if [[ "$mode" == "cleanup" ]]; then
  exec docker run "${docker_args[@]}" \
    --entrypoint python3 \
    sermon-transcribe \
    -m sermon_transcribe.cleanup \
    "${args[@]}"
fi

if [[ "$mode" == "convert" ]]; then
  exec docker run "${docker_args[@]}" \
    --entrypoint python3 \
    sermon-transcribe \
    -m sermon_transcribe.convert \
    "${args[@]}"
fi

exec docker run "${docker_args[@]}" \
  sermon-transcribe \
  --model-cache /app/model_cache \
  "${args[@]}"
