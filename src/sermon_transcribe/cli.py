import argparse
import os
import sys
from pathlib import Path

from sermon_transcribe.io_utils import collect_audio_files, ensure_dir, normalize_extensions
from sermon_transcribe.transcription import build_config, build_model, output_paths, transcribe_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe sermon audio with faster-whisper.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input audio file or directory of audio files.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default="output",
        help="Output directory for transcription files (default: ./output).",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (default: large-v3).",
    )
    parser.add_argument(
        "--model-cache",
        default="model_cache",
        help="Model cache directory (default: ./model_cache).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Compute type for faster-whisper (e.g. float16, int8).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (leave empty for auto-detect).",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Transcription task (default: transcribe).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5).",
    )
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable VAD filtering (default: enabled).",
    )
    parser.add_argument(
        "--convert-flac",
        action="store_true",
        help="Convert input audio to FLAC before transcription.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory.",
    )
    parser.add_argument(
        "--extensions",
        default=".wav,.flac,.mp3,.m4a,.aac",
        help="Comma-separated list of extensions to scan.",
    )
    parser.add_argument(
        "--raw-suffix",
        default="_raw",
        help="Suffix for raw transcript outputs (default: _raw).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip audio files with existing transcript outputs (default: enabled).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-process audio even if outputs already exist.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    ensure_dir(output_dir)

    extensions = normalize_extensions(args.extensions)
    audio_files = collect_audio_files(input_path, args.recursive, extensions)

    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    config = build_config(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language,
        task=args.task,
        vad_filter=args.vad_filter,
        cache_dir=Path(args.model_cache).expanduser(),
        hf_token=args.hf_token,
    )

    print(f"Found {len(audio_files)} file(s) to process.", flush=True)
    print(
        f"Model: {config.model} | Device: {config.device} | Compute: {config.compute_type}",
        flush=True,
    )
    model = build_model(config)
    print("Model loaded. Starting transcription...", flush=True)

    failures = 0
    for idx, audio_path in enumerate(audio_files, start=1):
        print(f"[{idx}/{len(audio_files)}] Transcribing {audio_path}", flush=True)
        text_path, json_path = output_paths(
            output_dir=output_dir,
            base_name=audio_path.stem,
            raw_suffix=args.raw_suffix,
        )
        if args.skip_existing and not args.reprocess and (
            text_path.exists() or json_path.exists()
        ):
            print(
                f"Skipping existing outputs: {text_path.name}, {json_path.name}",
                flush=True,
            )
            continue
        try:
            result = transcribe_file(
                model=model,
                source_path=audio_path,
                output_dir=output_dir,
                config=config,
                convert_flac=args.convert_flac,
                raw_suffix=args.raw_suffix,
            )
        except Exception as exc:  # pragma: no cover - CLI guardrail
            failures += 1
            print(f"Failed to transcribe {audio_path}: {exc}", file=sys.stderr)
            continue

        print(f"Wrote: {result.text_path}")
        print(f"Wrote: {result.json_path}")

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
