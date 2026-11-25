import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List

from sermon_transcribe.ai import AIClient, CreditBalanceError, GenerationError, configure_ai_logging
from sermon_transcribe.io_utils import ensure_dir, readable_path


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_EXTENSIONS = ".txt"
DEFAULT_CLEANED_SUFFIX = ""
DEFAULT_RAW_SUFFIX = "_raw"
DEFAULT_SUMMARY_SUFFIX = "_summary"
DEFAULT_SUMMARY_MAX_CHARS = 16000
SUMMARY_MAX_WORDS = 120
SUMMARY_MAX_POINTS = 3
DISCLAIMER_TEXT = (
    "Please note: This transcription and summary were created with the assistance of "
    "artificial intelligence technology. While we've used automated tools to help make "
    "the sermon content more accessible, there may be occasional errors or inaccuracies. "
    "If you notice anything that doesn't seem quite right, please feel free to let us know. "
    "For the most accurate representation of the message, we recommend listening to the "
    "original audio recording."
)
SUMMARY_DISCLAIMER_TEXT = "Please note: This summary was generated with AI. We welcome any feedback"
LEGACY_SUMMARY_DISCLAIMER_TEXT = (
    "Please note: This summary text was created with the assistance of artificial "
    "intelligence technology and may contain occasional errors or inaccuracies. "
    "If you notice anything that doesn't seem quite right, please feel free to let us know."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean up transcript text files with Anthropic Claude.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input transcript file or directory.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default=None,
        help="Output directory for cleaned transcripts (default: input dir).",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_CLEANED_SUFFIX,
        help="Suffix for cleaned files (default: none).",
    )
    parser.add_argument(
        "--raw-suffix",
        default=DEFAULT_RAW_SUFFIX,
        help=f"Suffix to add to original transcripts (default: {DEFAULT_RAW_SUFFIX}).",
    )
    parser.add_argument(
        "--summary-suffix",
        default=DEFAULT_SUMMARY_SUFFIX,
        help=f"Suffix for summary files (default: {DEFAULT_SUMMARY_SUFFIX}).",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated list of extensions to scan (default: .txt).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override the automatic output limit for each cleanup call.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Approximate max characters per chunk (default: 8000).",
    )
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=DEFAULT_SUMMARY_MAX_CHARS,
        help=f"Approximate max characters for summary chunks (default: {DEFAULT_SUMMARY_MAX_CHARS}).",
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        help="Override the automatic output limit for each summary call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override the task-specific sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that already have cleaned output (default: enabled).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-process files even if cleaned output already exists.",
    )
    return parser


def normalize_extensions(raw: str) -> List[str]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    normalized = []
    for part in parts:
        if not part.startswith("."):
            part = f".{part}"
        normalized.append(part)
    return normalized


def collect_text_files(
    input_path: Path,
    recursive: bool,
    extensions: Iterable[str],
    cleaned_suffix: str,
    summary_suffix: str,
    raw_suffix: str,
) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    ext_set = {ext.lower() for ext in extensions}
    candidates = input_path.rglob("*") if recursive else input_path.glob("*")
    files = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() not in ext_set:
            continue
        if cleaned_suffix and path.stem.endswith(cleaned_suffix):
            continue
        if path.stem.endswith(summary_suffix):
            continue
        if path.stem.endswith(raw_suffix):
            files.append(path)
            continue
        raw_candidate = path.with_name(f"{path.stem}{raw_suffix}{path.suffix}")
        if raw_candidate.exists():
            continue
        files.append(path)
    return sorted(files)


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        para_len = len(paragraph)
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(split_long_paragraph(paragraph, max_chars))
            continue

        if current_len + para_len + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = para_len
        else:
            current.append(paragraph)
            current_len += para_len + (2 if len(current) > 1 else 0)

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text]


def split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    chunks: List[str] = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(hard_wrap(sentence, max_chars))
            continue

        if current_len + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks or [paragraph]


def hard_wrap(text: str, max_chars: int) -> List[str]:
    chunks = []
    while len(text) > max_chars:
        limit = max_chars
        for index in range(max_chars, max_chars // 2, -1):
            if text[index].isspace():
                limit = index
                break
        chunks.append(text[:limit].rstrip())
        text = text[limit:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def build_prompt(chunk: str) -> str:
    return (
        "Clean up the transcript text below.\n"
        "- Insert paragraph breaks and sentence punctuation where it is obviously missing.\n"
        "- Fix obvious grammar and spelling mistakes.\n"
        "- Keep wording, tone, and meaning as close to the original as possible.\n"
        "- IMPORTANT: Do not remove or omit any content. Keep all the text, even if repetitive.\n"
        "- Only remove extreme filler words like 'um', 'uh', 'you know' if excessive.\n"
        "- Do not add commentary or labels. Return only the cleaned text.\n\n"
        f"Transcript:\n{chunk.strip()}"
    )


def summary_format_instructions() -> str:
    return (
        f"Write one summary paragraph of around 100 words, with a maximum of {SUMMARY_MAX_WORDS} words.\n"
        f"List one to {SUMMARY_MAX_POINTS} main points, one bullet per line. "
        "The word limit applies only to the summary paragraph.\n"
        "Use plain text with these exact headings and hyphen bullets:\n"
        "Summary:\n"
        "<single paragraph>\n\n"
        "Main points:\n"
        "- point 1\n"
        "- point 2\n\n"
        "Bible references:\n"
        "- reference 1\n"
        "- reference 2\n\n"
        "Preserve the preacher's central message and theological distinctions. Retain important "
        "illustrations, biblical parallels and practical applications where they support that message.\n"
        "List only scripture references explicitly mentioned in the source. A person's name is "
        "not a reference. If none were mentioned, write 'None noted.' Return only the formatted summary.\n\n"
    )


def build_summary_prompt(text: str) -> str:
    return (
        "Create a sermon summary from the cleaned transcript below.\n"
        f"{summary_format_instructions()}"
        f"Cleaned transcript:\n{text.strip()}"
    )


def build_summary_chunk_prompt(text: str) -> str:
    return (
        "From the text below, extract the key ideas and any Bible "
        "references. Keep the output short and factual. Preserve important theological "
        "distinctions, illustrations, and applications. List only references in the source.\n"
        "Format:\n"
        "Key ideas:\n"
        "- ...\n"
        "Bible references:\n"
        "- ...\n\n"
        f"Chunk:\n{text.strip()}"
    )


def build_summary_merge_prompt(text: str) -> str:
    return (
        "Combine the notes below into a single sermon summary.\n"
        f"{summary_format_instructions()}"
        f"Notes:\n{text.strip()}"
    )


def summary_validation_error(summary: str) -> str | None:
    sections = re.fullmatch(
        r"Summary:[ \t]*\n(.+?)\n\s*Main points:[ \t]*\n(.+?)\n\s*Bible references:[ \t]*\n(.+)",
        summary.replace("\r\n", "\n").strip(),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not sections:
        return "Use the exact headings Summary, Main points, and Bible references, with content under each."
    paragraph, points, references = sections.groups()
    count = len(paragraph.split())
    if not 0 < count <= SUMMARY_MAX_WORDS:
        return f"The summary paragraph has {count} words. Use 1 to {SUMMARY_MAX_WORDS} words."
    if re.search(r"\n\s*\n", paragraph.strip()):
        return "Write the summary as one paragraph."
    point_lines = [line.strip() for line in points.splitlines() if line.strip()]
    if not 1 <= len(point_lines) <= SUMMARY_MAX_POINTS or not all(re.match(r"-\s+\S", line) for line in point_lines):
        return f"Use one to {SUMMARY_MAX_POINTS} main points with one hyphen bullet per line."
    if not references.strip():
        return "Give the scripture references from the source or write 'None noted.'"
    return None


def summary_matches_format(summary: str) -> bool:
    return summary_validation_error(summary) is None


def cleanup_validation_error(source: str, cleaned: str) -> str | None:
    source_length = len(re.sub(r"\s+", "", source))
    cleaned_length = len(re.sub(r"\s+", "", cleaned))
    if source_length >= 80 and cleaned_length < source_length * 0.8:
        return "The cleaned text is substantially shorter than its source. Retain the full transcript in order."
    source_words = Counter(re.findall(r"[^\W_]+", source.casefold()))
    count = sum(source_words.values())
    if count >= 20:
        cleaned_words = Counter(re.findall(r"[^\W_]+", cleaned.casefold()))
        if sum((source_words & cleaned_words).values()) < count * 0.7:
            return "The cleanup replaced too much of the original wording. Limit changes to corrections and formatting."
    return None


def clean_transcript(text, client, *, max_chars=8000, max_tokens=None, temperature=None, progress=None):
    def clean_chunk(chunk, splits=0):
        try:
            return client.generate(
                build_prompt(chunk), "cleanup", source=chunk,
                validate=lambda cleaned: cleanup_validation_error(chunk, cleaned),
                max_tokens=max_tokens, temperature=temperature,
            )
        except GenerationError as exc:
            if exc.kind not in ("context", "length") or splits >= 2 or len(chunk) < 1000:
                raise
            return "\n\n".join(
                clean_chunk(part, splits + 1) for part in split_into_chunks(chunk, len(chunk) // 2)
            )

    with client.report("cleanup"):
        chunks = split_into_chunks(text, client.chunk_chars(max_chars, cleanup=True))
        cleaned = []
        for index, chunk in enumerate(chunks, 1):
            if progress:
                progress(index, len(chunks))
            cleaned.append(clean_chunk(chunk))
        return "\n\n".join(cleaned)


def summarize_transcript(
    text, client, *, max_chars=DEFAULT_SUMMARY_MAX_CHARS, max_tokens=None, temperature=None, progress=None,
):
    def generate(prompt, task, validate=None):
        return client.generate(prompt, task, validate=validate, max_tokens=max_tokens, temperature=temperature)

    def extract_notes(text, chunk_size):
        notes = []
        chunks = split_into_chunks(text, chunk_size)
        for index, chunk in enumerate(chunks, 1):
            if progress:
                progress(index, len(chunks))
            try:
                notes.append(generate(build_summary_chunk_prompt(chunk), "summary_notes"))
            except GenerationError as exc:
                if exc.kind != "context" or len(chunk) < 500:
                    raise
                notes.append(extract_notes(chunk, len(chunk) // 2))
        return "\n\n".join(notes)

    with client.report("summary"):
        chunk_size = client.chunk_chars(max_chars)
        if len(text) <= chunk_size:
            try:
                return generate(build_summary_prompt(text), "summary", summary_validation_error)
            except GenerationError as exc:
                if exc.kind != "context" or len(text) < 500:
                    raise
                chunk_size = len(text) // 2

        notes = extract_notes(text, chunk_size)
        for _ in range(4):
            if len(notes) <= chunk_size:
                try:
                    return generate(build_summary_merge_prompt(notes), "summary", summary_validation_error)
                except GenerationError as exc:
                    if exc.kind != "context" or len(notes) < 500:
                        raise
                    chunk_size = len(notes) // 2
            reduced = extract_notes(notes, chunk_size)
            if len(reduced) >= len(notes):
                break
            notes = reduced
        raise GenerationError("context", "The summary notes could not fit the model's context window.")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(content, encoding="utf-8")
        staged.replace(path)


def apply_disclaimer(text: str, disclaimer: str = DISCLAIMER_TEXT) -> str:
    cleaned = text.strip()
    if cleaned.startswith(disclaimer):
        return cleaned + "\n"
    return f"{disclaimer}\n\n{cleaned}\n"


def strip_disclaimer(text: str) -> str:
    cleaned = text.strip()
    for disclaimer in (DISCLAIMER_TEXT, SUMMARY_DISCLAIMER_TEXT, LEGACY_SUMMARY_DISCLAIMER_TEXT):
        if cleaned.startswith(disclaimer):
            return cleaned[len(disclaimer) :].lstrip()
    return cleaned


def base_stem(input_path: Path, raw_suffix: str) -> str:
    if raw_suffix and input_path.stem.endswith(raw_suffix):
        return input_path.stem[: -len(raw_suffix)]
    return input_path.stem


def cleaned_output_path(input_path: Path, output_dir: Path, suffix: str, raw_suffix: str) -> Path:
    stem = base_stem(input_path, raw_suffix)
    return output_dir / f"{stem}{suffix}{input_path.suffix}"


def summary_output_path(input_path: Path, output_dir: Path, suffix: str, raw_suffix: str) -> Path:
    stem = base_stem(input_path, raw_suffix)
    return output_dir / f"{stem}{suffix}.txt"


def process_transcript_file(raw_path, output_path, summary_path, args, client):
    skip_existing = args.skip_existing and not args.reprocess
    if output_path.exists() and skip_existing:
        existing_cleaned = load_text(output_path)
        cleaned = strip_disclaimer(existing_cleaned)
        if not existing_cleaned.strip().startswith(DISCLAIMER_TEXT):
            write_text(output_path, apply_disclaimer(cleaned))
    else:
        text = strip_disclaimer(load_text(raw_path))
        if not text:
            print("  Skipping empty file.", flush=True)
            return
        cleaned = clean_transcript(
            text, client, max_chars=args.max_chars, max_tokens=args.max_tokens, temperature=args.temperature,
        )
        write_text(output_path, apply_disclaimer(cleaned))
        print(f"  Wrote {readable_path(output_path)}", flush=True)

    if not cleaned:
        print("  Skipping summary for empty transcript.", flush=True)
        return
    if summary_path.exists() and skip_existing:
        existing_summary = load_text(summary_path)
        if not existing_summary.strip().startswith(SUMMARY_DISCLAIMER_TEXT):
            write_text(summary_path, apply_disclaimer(strip_disclaimer(existing_summary), SUMMARY_DISCLAIMER_TEXT))
            print("  Updated summary disclaimer.", flush=True)
        else:
            print(f"  Skipping existing {readable_path(summary_path)}", flush=True)
        return

    summary = summarize_transcript(
        cleaned, client, max_chars=args.summary_max_chars,
        max_tokens=args.summary_max_tokens, temperature=args.temperature,
    )
    write_text(summary_path, apply_disclaimer(summary, SUMMARY_DISCLAIMER_TEXT))
    print(f"  Wrote {readable_path(summary_path)}", flush=True)


def main() -> None:
    configure_ai_logging()
    parser = build_parser()
    args = parser.parse_args()
    for name in ("max_tokens", "summary_max_tokens", "max_chars", "summary_max_chars"):
        if (value := getattr(args, name)) is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else None
    extensions = normalize_extensions(args.extensions)

    files = collect_text_files(
        input_path,
        args.recursive,
        extensions,
        args.suffix,
        args.summary_suffix,
        args.raw_suffix,
    )
    if not files:
        print("No transcript text files found.", file=sys.stderr)
        sys.exit(1)

    failures = 0
    for idx, file_path in enumerate(files, start=1):
        client = AIClient("anthropic", args.model, api_key, job_id=file_path.name, timeout=args.timeout)
        target_dir = output_dir if output_dir else file_path.parent
        ensure_dir(target_dir)
        raw_path = file_path
        if args.raw_suffix and not file_path.stem.endswith(args.raw_suffix):
            candidate = file_path.with_name(f"{file_path.stem}{args.raw_suffix}{file_path.suffix}")
            if candidate.exists():
                raw_path = candidate
            else:
                file_path.rename(candidate)
                raw_path = candidate
                print(f"  Renamed raw transcript to {readable_path(raw_path)}", flush=True)

        output_path = cleaned_output_path(raw_path, target_dir, args.suffix, args.raw_suffix)
        summary_path = summary_output_path(raw_path, target_dir, args.summary_suffix, args.raw_suffix)
        print(f"[{idx}/{len(files)}] Cleaning {readable_path(raw_path)}", flush=True)
        try:
            with client.report():
                process_transcript_file(raw_path, output_path, summary_path, args, client)
        except CreditBalanceError as exc:
            print(f"AI processing failed: {exc}", file=sys.stderr)
            sys.exit(3)
        except RuntimeError as exc:
            failures += 1
            print(f"AI processing failed: {exc}", file=sys.stderr)

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
