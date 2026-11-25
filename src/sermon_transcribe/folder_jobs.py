from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISLNK, S_ISREG


FILE_SUFFIXES = {
    "summary": "_summary.txt",
    "timestamps_text": "_timestamps.txt",
    "timestamps": "_timestamps.json",
    "legacy_raw": "_raw.txt",
    "wav": ".wav",
    "transcript": ".txt",
    "m4a": ".m4a",
    "flac": ".flac",
}


@dataclass
class FolderSermon:
    stem: str
    files: dict[str, list[Path]] = field(default_factory=dict)

    def has_source(self, kind: str) -> bool:
        files = self.files.get(kind, [])
        return len(files) == 1 and files[0].stat().st_size > 0

    def has_wav(self) -> bool:
        reserved = ("_summary", "_timestamps", "_raw")
        return not self.stem.lower().endswith(reserved) and self.has_source("wav")

    def destination(self, folder: Path, kind: str) -> Path:
        files = self.files.get(kind, [])
        if len(files) > 1:
            raise ValueError(f"{self.stem} has multiple files for {kind}")
        return files[0] if files else folder / f"{self.stem}{FILE_SUFFIXES[kind]}"


@dataclass
class FolderJob:
    key: str
    source: Path
    destinations: dict[str, Path]
    versions: dict[Path, tuple[int, int, int, int] | None]


def resolve_folder(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Choose a folder inside the configured sermon directory")

    folder = root
    for part in path.parts:
        folder = folder / part
        if folder.is_symlink():
            raise ValueError("Choose a folder inside the configured sermon directory")
    if not folder.is_dir():
        raise FileNotFoundError("Folder not found")
    return folder


def scan_folder(folder: Path, *, include_outputs_only: bool = False) -> tuple[list[str], dict[str, FolderSermon]]:
    directories = []
    sermons: dict[str, FolderSermon] = {}
    for path in sorted(folder.iterdir(), key=lambda path: (path.name.casefold(), path.name)):
        if path.name.startswith(".") or path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path.name)
            continue
        if not path.is_file():
            continue
        for kind, suffix in FILE_SUFFIXES.items():
            if path.name.lower().endswith(suffix):
                stem = path.name[:-len(suffix)]
                sermon = sermons.setdefault(stem, FolderSermon(stem))
                sermon.files.setdefault(kind, []).append(path)
                break

    sermons = {
        stem: sermon
        for stem, sermon in sorted(sermons.items())
        if include_outputs_only or "wav" in sermon.files or "transcript" in sermon.files
    }
    return directories, sermons


def file_version(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if S_ISLNK(info.st_mode):
        raise ValueError(f"{path.name} is a symbolic link")
    if not S_ISREG(info.st_mode):
        raise ValueError(f"{path.name} is not a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
