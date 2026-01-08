import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ctranslate2
from faster_whisper import WhisperModel

from sermon_transcribe.io_utils import ensure_dir, readable_path, unique_path


@dataclass
class TranscriptionConfig:
    model: str
    device: str
    compute_type: str
    beam_size: int
    language: Optional[str]
    task: str
    vad_filter: bool
    cache_dir: Path
    hf_token: Optional[str]


@dataclass
class TranscriptionResult:
    text_path: Path
    json_path: Path
    audio_path: Path


def detect_device(requested: str) -> str:
    if requested in {"cpu", "cuda"}:
        return requested

    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda"

    return "cpu"


def default_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "int8"


def convert_to_flac(source: Path, target_dir: Path) -> Path:
    ensure_dir(target_dir)
    output_path = unique_path(target_dir / f"{source.stem}.flac")

    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        readable_path(source),
        "-vn",
        "-acodec",
        "flac",
        readable_path(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def transcribe_file(
    model: WhisperModel,
    source_path: Path,
    output_dir: Path,
    config: TranscriptionConfig,
    convert_flac: bool,
) -> TranscriptionResult:
    ensure_dir(output_dir)

    audio_path = source_path
    if convert_flac and source_path.suffix.lower() != ".flac":
        audio_path = convert_to_flac(source_path, output_dir / "_converted")

    segments, info = model.transcribe(
        readable_path(audio_path),
        beam_size=config.beam_size,
        language=config.language,
        task=config.task,
        vad_filter=config.vad_filter,
    )

    segment_list = [
        {
            "id": idx,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        }
        for idx, segment in enumerate(segments)
    ]

    base_name = source_path.stem
    text_path, json_path = _unique_output_paths(output_dir, base_name)

    transcript_text = _format_transcript(segment_list)
    text_path.write_text(transcript_text, encoding="utf-8")

    payload = {
        "source": readable_path(source_path),
        "audio": readable_path(audio_path),
        "model": config.model,
        "device": config.device,
        "compute_type": config.compute_type,
        "language": info.language,
        "duration": info.duration,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "segments": segment_list,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return TranscriptionResult(text_path=text_path, json_path=json_path, audio_path=audio_path)


def build_model(config: TranscriptionConfig) -> WhisperModel:
    ensure_dir(config.cache_dir)
    os.environ.setdefault("HF_HOME", str(config.cache_dir))
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE", str(config.cache_dir / "huggingface")
    )
    if config.hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", config.hf_token)
    return WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
        download_root=str(config.cache_dir),
    )


def build_config(
    model: str,
    device: str,
    compute_type: Optional[str],
    beam_size: int,
    language: Optional[str],
    task: str,
    vad_filter: bool,
    cache_dir: Path,
    hf_token: Optional[str],
) -> TranscriptionConfig:
    resolved_device = detect_device(device)
    resolved_compute = compute_type or default_compute_type(resolved_device)

    return TranscriptionConfig(
        model=model,
        device=resolved_device,
        compute_type=resolved_compute,
        beam_size=beam_size,
        language=language,
        task=task,
        vad_filter=vad_filter,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )


def _format_transcript(segments: Iterable[dict]) -> str:
    lines: List[str] = []
    for segment in segments:
        speaker = segment.get("speaker")
        text = segment["text"].strip()
        if not text:
            continue
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines) + "\n"


def _unique_output_paths(output_dir: Path, base_name: str) -> Tuple[Path, Path]:
    candidate_text = output_dir / f"{base_name}.txt"
    candidate_json = output_dir / f"{base_name}.json"
    if not candidate_text.exists() and not candidate_json.exists():
        return candidate_text, candidate_json

    for idx in range(1, 10_000):
        text_path = output_dir / f"{base_name}_{idx}.txt"
        json_path = output_dir / f"{base_name}_{idx}.json"
        if not text_path.exists() and not json_path.exists():
            return text_path, json_path

    text_path = unique_path(output_dir / f"{base_name}.txt")
    json_path = text_path.with_suffix(".json")
    return text_path, json_path
