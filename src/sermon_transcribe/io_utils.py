import os
from pathlib import Path
from typing import Iterable, List


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac"}


def collect_audio_files(input_path: Path, recursive: bool, extensions: Iterable[str]) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    ext_set = {ext.lower() for ext in extensions}
    if recursive:
        candidates = input_path.rglob("*")
    else:
        candidates = input_path.glob("*")

    files = [path for path in candidates if path.is_file() and path.suffix.lower() in ext_set]
    return sorted(files)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    for idx in range(1, 10_000):
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Unable to find unique path for {base_path}")


def normalize_extensions(raw: str) -> List[str]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    normalized = []
    for part in parts:
        if not part.startswith("."):
            part = f".{part}"
        normalized.append(part)
    return normalized


def readable_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return os.fspath(path)
