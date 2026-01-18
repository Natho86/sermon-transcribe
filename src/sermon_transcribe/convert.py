import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from sermon_transcribe.io_utils import ensure_dir, normalize_extensions, readable_path


DEFAULT_EXTENSIONS = ".wav"
DEFAULT_LUFS = -20.0
DEFAULT_TRUE_PEAK = -1.5
DEFAULT_LRA = 11.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert WAV files to normalized M4A.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input WAV file or directory.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        required=True,
        help="Output directory for converted M4A files.",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated list of extensions to scan (default: .wav).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory.",
    )
    parser.add_argument(
        "--lufs",
        type=float,
        default=DEFAULT_LUFS,
        help=f"Integrated LUFS target (default: {DEFAULT_LUFS}).",
    )
    parser.add_argument(
        "--true-peak",
        type=float,
        default=DEFAULT_TRUE_PEAK,
        help=f"True peak target (default: {DEFAULT_TRUE_PEAK}).",
    )
    parser.add_argument(
        "--lra",
        type=float,
        default=DEFAULT_LRA,
        help=f"Loudness range target (default: {DEFAULT_LRA}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Output sample rate (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=f"Output channel count (default: {DEFAULT_CHANNELS}).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that already have outputs (default: enabled).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-process files even if outputs already exist.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of concurrent conversions (default: 2).",
    )
    return parser


def collect_audio_files(
    input_path: Path, recursive: bool, extensions: Iterable[str]
) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    ext_set = {ext.lower() for ext in extensions}
    candidates = input_path.rglob("*") if recursive else input_path.glob("*")
    files = [path for path in candidates if path.is_file() and path.suffix.lower() in ext_set]
    return sorted(files)


def output_path_for(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{source.stem}.m4a"


def run_ffmpeg(
    source: Path,
    target: Path,
    lufs: float,
    true_peak: float,
    lra: float,
    sample_rate: int,
    channels: int,
    show_stats: bool,
) -> None:
    loudnorm = f"loudnorm=I={lufs}:TP={true_peak}:LRA={lra}"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if show_stats:
        command.append("-stats")
    command.extend(
        [
            "-i",
            readable_path(source),
            "-af",
            loudnorm,
            "-c:a",
            "aac",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            readable_path(target),
        ]
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    ensure_dir(output_dir)

    extensions = normalize_extensions(args.extensions)
    files = collect_audio_files(input_path, args.recursive, extensions)

    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    failures = 0
    show_stats = args.jobs <= 1

    def convert_one(index: int, total: int, source_path: Path, target_path: Path) -> Optional[str]:
        try:
            run_ffmpeg(
                source=source_path,
                target=target_path,
                lufs=args.lufs,
                true_peak=args.true_peak,
                lra=args.lra,
                sample_rate=args.sample_rate,
                channels=args.channels,
                show_stats=show_stats,
            )
            return None
        except subprocess.CalledProcessError as exc:
            return f"[{index}/{total}] Conversion failed: {exc}"

    if args.jobs <= 1:
        for idx, source_path in enumerate(files, start=1):
            target_path = output_path_for(source_path, output_dir)
            if args.skip_existing and not args.reprocess and target_path.exists():
                print(f"[{idx}/{len(files)}] Skipping {readable_path(target_path)}", flush=True)
                continue

            print(f"[{idx}/{len(files)}] Converting {readable_path(source_path)}", flush=True)
            error = convert_one(idx, len(files), source_path, target_path)
            if error:
                failures += 1
                print(f"  {error}", file=sys.stderr)
            else:
                print(f"  Wrote {readable_path(target_path)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for idx, source_path in enumerate(files, start=1):
                target_path = output_path_for(source_path, output_dir)
                if args.skip_existing and not args.reprocess and target_path.exists():
                    print(f"[{idx}/{len(files)}] Skipping {readable_path(target_path)}", flush=True)
                    continue

                print(f"[{idx}/{len(files)}] Converting {readable_path(source_path)}", flush=True)
                future = executor.submit(convert_one, idx, len(files), source_path, target_path)
                futures[future] = target_path

            for future in as_completed(futures):
                target_path = futures[future]
                error = future.result()
                if error:
                    failures += 1
                    print(f"  {error}", file=sys.stderr)
                else:
                    print(f"  Wrote {readable_path(target_path)}", flush=True)

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
