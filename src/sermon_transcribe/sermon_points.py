"""Verify proposed sermon points against the saved transcript."""

import re

NO_POINTS = "None noted."
MAX_POINTS = 3
QUOTE_SEPARATOR = "||"

BOUNDARY = re.compile(r"(?<=[.!?])[\"'”’\)\]]*\s+|\n[ \t]*\n+")
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
APOSTROPHES = str.maketrans("‘’", "''")
ORDINAL = r"(?:first|second|third|fourth|next|final|last)"
OUTLINE = re.compile(
    rf"^[\"'“‘(\s]*(?:(?:and|now|so)\s+)*"
    rf"(?:(?:firstly|secondly|thirdly|finally)\b\s*[,:]?\s*"
    rf"|(?:{ORDINAL}|(?:point|number) (?:one|two|three|[1-3]))\s*[,:]\s*"
    rf"|(?:(?:the|my|our)\s+)?{ORDINAL}(?:\s+(?:point|heading|question|thing))?"
    rf"(?:\s+(?:is|will be)\b\s*[:,]?\s*|\s*[,:]\s*))",
    re.I,
)
ABBREVIATION = re.compile(r"\b(?:mr|mrs|ms|dr|rev|st|cf|v|vv|e\.g|i\.e)\.(?=\s)", re.I)


def sentences(text: str) -> list[str]:
    # Shield common abbreviations from the sentence splitter, retaining their
    # original spelling and punctuation in the returned sentences.
    masked = ABBREVIATION.sub(lambda match: match[0][:-1] + "·", text)
    start = 0
    parts = []
    for boundary in BOUNDARY.finditer(masked):
        if part := text[start:boundary.start()].strip():
            parts.append(part)
        start = boundary.end()
    if part := text[start:].strip():
        parts.append(part)
    return parts


def words(text: str) -> str:
    return " ".join(WORD.findall(text.translate(APOSTROPHES).casefold()))


def _outline_points(transcript: str):
    """Yield wording from complete outline statements and its source position."""
    parts = sentences(transcript)
    for index, sentence in enumerate(parts):
        body = OUTLINE.sub("", sentence, count=1)
        announced = body != sentence
        heading, colon, statement = body.partition(":")
        labelled = (
            colon and (not statement or statement[:1].isspace())
            and heading.rstrip()[-1:].isalpha() and 1 <= len(words(heading).split()) <= 8
        )
        if announced or labelled:
            if point := words(body):
                yield point, index
        is_heading = (announced and not colon and 1 <= len(words(body).split()) <= 8) or (labelled and not statement)
        if is_heading and index + 1 < len(parts):
            # Keep an announced heading and its adjacent statement together.
            following = parts[index + 1]
            if not OUTLINE.match(following):
                yield words(f"{body} {following}"), index


def verified_main_points(bullets: list[str], transcript: str) -> list[str]:
    proposed = []
    for bullet in bullets:
        point, separator, quote = bullet.partition(QUOTE_SEPARATOR)
        if separator and point.strip() and quote.strip() and QUOTE_SEPARATOR not in quote:
            proposed.append((point.strip(), words(quote)))
    if not proposed:
        return []

    source = f" {words(transcript)} "
    candidates = {}
    for wording, position in _outline_points(transcript):
        candidates.setdefault(wording, position)
    verified = {}
    for point, quote in proposed:
        key = words(point)
        position = candidates.get(key)
        if position is None or f" {key} " not in f" {quote} " or f" {quote} " not in source:
            continue
        if position not in verified or len(key) > len(words(verified[position])):
            verified[position] = point

    return [verified[position] for position in sorted(verified)[:MAX_POINTS]]
