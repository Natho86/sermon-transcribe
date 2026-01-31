import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

from sermon_transcribe.io_utils import ensure_dir, readable_path


API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_EXTENSIONS = ".txt"
DEFAULT_CLEANED_SUFFIX = ""
DEFAULT_RAW_SUFFIX = "_raw"
DEFAULT_SUMMARY_SUFFIX = "_summary"
DEFAULT_SUMMARY_MAX_CHARS = 16000
DEFAULT_SUMMARY_MAX_TOKENS = 600
DISCLAIMER_TEXT = (
    "Please note: This transcription and summary were created with the assistance of "
    "artificial intelligence technology. While we've used automated tools to help make "
    "the sermon content more accessible, there may be occasional errors or inaccuracies. "
    "If you notice anything that doesn't seem quite right, please feel free to let us know. "
    "For the most accurate representation of the message, we recommend listening to the "
    "original audio recording."
)


class CreditBalanceError(RuntimeError):
    pass


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
        default=1200,
        help="Max tokens per cleanup call (default: 1200).",
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
        default=DEFAULT_SUMMARY_MAX_TOKENS,
        help=f"Max tokens per summary call (default: {DEFAULT_SUMMARY_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
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
    return [text[idx : idx + max_chars] for idx in range(0, len(text), max_chars)]


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


def build_summary_prompt(text: str) -> str:
    return (
        "Create a concise sermon summary from the cleaned transcript below.\n"
        "Format your response exactly like this:\n"
        "Summary:\n"
        "<single paragraph>\n\n"
        "Main points:\n"
        "- point 1\n"
        "- point 2\n\n"
        "Bible references:\n"
        "- reference 1\n"
        "- reference 2\n\n"
        "Only include references explicitly mentioned. If none are mentioned, write "
        "'None noted.' Return only the formatted summary.\n\n"
        f"Cleaned transcript:\n{text.strip()}"
    )


def build_summary_chunk_prompt(text: str) -> str:
    return (
        "From the cleaned transcript chunk below, extract the key ideas and any Bible "
        "references. Keep the output short and factual.\n"
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
        "Format your response exactly like this:\n"
        "Summary:\n"
        "<single paragraph>\n\n"
        "Main points:\n"
        "- point 1\n"
        "- point 2\n\n"
        "Bible references:\n"
        "- reference 1\n"
        "- reference 2\n\n"
        "Only include references explicitly mentioned. If none are mentioned, write "
        "'None noted.' Return only the formatted summary.\n\n"
        f"Notes:\n{text.strip()}"
    )


def call_claude(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int = 3,
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                return extract_text(parsed)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(2**attempt)
                continue
            error_body = exc.read().decode("utf-8") if exc.fp else str(exc)
            if exc.code == 400 and "credit balance is too low" in error_body.lower():
                raise CreditBalanceError(
                    "Anthropic API credits are exhausted. Please top up and retry."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    "Anthropic API returned 404. The model name may be invalid; "
                    "set a valid model with --model."
                ) from exc
            raise RuntimeError(f"Anthropic API error {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Anthropic API connection error: {exc}") from exc

    if last_error:
        raise RuntimeError(f"Anthropic API request failed: {last_error}") from last_error
    raise RuntimeError("Anthropic API request failed with unknown error.")


def extract_text(payload: dict) -> str:
    content = payload.get("content", [])
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def apply_disclaimer(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith(DISCLAIMER_TEXT):
        return cleaned + "\n"
    return f"{DISCLAIMER_TEXT}\n\n{cleaned}\n"


def strip_disclaimer(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith(DISCLAIMER_TEXT):
        remainder = cleaned[len(DISCLAIMER_TEXT) :]
        return remainder.lstrip("\n").lstrip()
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


def summarize_transcript(
    cleaned_text: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    max_chars: int,
) -> str:
    if len(cleaned_text) <= max_chars:
        prompt = build_summary_prompt(cleaned_text)
        return call_claude(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    chunks = split_into_chunks(cleaned_text, max_chars)
    chunk_notes = []
    for chunk_idx, chunk in enumerate(chunks, start=1):
        print(f"  Summary chunk {chunk_idx}/{len(chunks)}", flush=True)
        prompt = build_summary_chunk_prompt(chunk)
        notes = call_claude(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        chunk_notes.append(notes.strip())

    combined_notes = "\n\n".join(note for note in chunk_notes if note)
    merge_prompt = build_summary_merge_prompt(combined_notes)
    return call_claude(
        prompt=merge_prompt,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
        has_cleaned = output_path.exists()
        has_summary = summary_path.exists()
        should_skip_cleaned = has_cleaned and args.skip_existing and not args.reprocess
        should_skip_summary = has_summary and args.skip_existing and not args.reprocess

        print(f"[{idx}/{len(files)}] Cleaning {readable_path(raw_path)}", flush=True)
        try:
            if should_skip_cleaned:
                existing_cleaned = load_text(output_path)
                cleaned_text = strip_disclaimer(existing_cleaned)
                if not existing_cleaned.strip().startswith(DISCLAIMER_TEXT):
                    write_text(output_path, apply_disclaimer(existing_cleaned))
                    print("  Added disclaimer to cleaned transcript.", flush=True)
            else:
                text = load_text(raw_path).strip()
                if not text:
                    print("  Skipping empty file.", flush=True)
                    continue

                chunks = split_into_chunks(text, args.max_chars)
                cleaned_chunks = []
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    if len(chunks) > 1:
                        print(f"  Chunk {chunk_idx}/{len(chunks)}", flush=True)
                    prompt = build_prompt(chunk)
                    cleaned = call_claude(
                        prompt=prompt,
                        api_key=api_key,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                    )
                    cleaned_chunks.append(cleaned.strip())

                cleaned_text = "\n\n".join(chunk for chunk in cleaned_chunks if chunk)
                write_text(output_path, apply_disclaimer(cleaned_text))
                print(f"  Wrote {readable_path(output_path)}", flush=True)
        except CreditBalanceError as exc:
            print(f"  Cleanup failed: {exc}", file=sys.stderr)
            sys.exit(3)
        except RuntimeError as exc:
            failures += 1
            print(f"  Cleanup failed: {exc}", file=sys.stderr)
            continue

        if not cleaned_text.strip():
            print("  Skipping summary for empty transcript.", flush=True)
            continue

        if should_skip_summary:
            existing_summary = load_text(summary_path)
            if not existing_summary.strip().startswith(DISCLAIMER_TEXT):
                write_text(summary_path, apply_disclaimer(existing_summary))
                print("  Added disclaimer to summary.", flush=True)
            else:
                print(f"  Skipping existing {readable_path(summary_path)}", flush=True)
            continue

        try:
            summary = summarize_transcript(
                cleaned_text=cleaned_text,
                api_key=api_key,
                model=args.model,
                max_tokens=args.summary_max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                max_chars=args.summary_max_chars,
            )
            write_text(summary_path, apply_disclaimer(summary))
            print(f"  Wrote {readable_path(summary_path)}", flush=True)
        except CreditBalanceError as exc:
            print(f"  Summary failed: {exc}", file=sys.stderr)
            sys.exit(3)
        except RuntimeError as exc:
            failures += 1
            print(f"  Summary failed: {exc}", file=sys.stderr)
            continue

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
