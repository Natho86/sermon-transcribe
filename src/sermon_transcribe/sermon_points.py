"""Verify proposed sermon points against the saved transcript."""

import logging
import re
from collections import Counter

NO_POINTS = "None noted."
MAX_POINTS = 5
QUOTE_SEPARATOR = "||"
logger = logging.getLogger("sermon_transcribe.ai.points")

BOUNDARY = re.compile(r"(?<=[.!?])[\"'”’\)\]]*\s+|\n[ \t]*\n+")
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
APOSTROPHES = str.maketrans("‘’", "''")
ORDINAL = r"(?:first|second|third|fourth|fifth|next|final|last)"
OUTLINE = re.compile(
    rf"^[\"'“‘(\s]*(?:(?:and|but|now|so|that['’]s)\b[:,]?\s+)*"
    rf"(?:(?:firstly|secondly|thirdly|fourthly|fifthly|finally|first of all)\b\s*[,:]?\s*"
    rf"|(?:{ORDINAL}|(?:point|number) (?:one|two|three|four|five|[1-5]))\s*[,:]\s*"
    rf"|(?:(?:the|my|our)\s+)?{ORDINAL}(?:\s+(?:point|heading|question|thing|one))?"
    rf"(?:\s+(?:(?:we|we['’]re|we are) (?:going to )?look at )?(?:is|will be|relates to)\b\s*[:,]?\s*|\s*[,:]\s*))",
    re.I,
)
ABBREVIATION = re.compile(r"\b(?:mr|mrs|ms|dr|rev|st|cf|v|vv|e\.g|i\.e)\.(?=\s)", re.I)
LEAD_IN = re.compile(r"^(?:(?:and|now|so)\b[:,]?\s+)+", re.I)
QUESTION_LEAD_IN = re.compile(r"^(?:we['’]ll|we will) (?:think|talk) about (?=(?:the )?question of\s+)", re.I)
HEADING_LEAD_IN = re.compile(
    r"^(?:(?:we['’]ll|we will) (?:think|talk) about )?(?:the )?question of\s+", re.I,
)
POINT_LEAD_IN = re.compile(r"^(?:is )?that\s+", re.I)
CONFIRMATION_TAG = re.compile(r",\s*okay\?\s*$", re.I)
HEADING_LINK = re.compile(r"^([^,]+),\s+which is that\s+(.+)$", re.I)
LIST_MARKER = re.compile(r"^(?:[-*•–—]|[0-9]+[.)])\s+")
FINAL_TRANSITION = re.compile(r"^(?:and |so )?that (?:just )?(?:kind of )?brings us to (?:a|the) final point:\s*", re.I)
EMPHASIS_TRANSITION = re.compile(
    r"^(?:you see,\s*)?what is being emphasi[sz]ed in this passage,?\s+first of all,?\s+is that\s+", re.I,
)
CONTRAST = re.compile(r"(?:;\s*(?=[\w'’]+\s+also\b)|,\s*and yet,?\s+at the same time,\s*)", re.I)
RESTART = re.compile(r"^([\w'’]+\s+\w+) also\s*[—–]\s*and this is the main point of the passage\s*[—–]\s*(.+)$", re.I)


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


def _parallel_points(text: str) -> list[str]:
    clauses = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", LEAD_IN.sub("", text), flags=re.I)
    keys = [words(clause).split() for clause in clauses]
    # Require a repeated opening across the whole list. Never split off a
    # fragment of a claim or discard a qualifier to make a point fit.
    if len(keys) < 2 or not all(len(key) >= 3 and key[:2] == keys[0][:2] for key in keys):
        return []
    return [" ".join(key) for key in keys]


def _outline_points(transcript: str):
    """Yield wording from complete outline statements and its source position."""
    parts = [" ".join(part.split()) for part in sentences(transcript)]
    for index, sentence in enumerate(parts):
        body = OUTLINE.sub("", sentence, count=1)
        body = FINAL_TRANSITION.sub("", body, count=1)
        body = EMPHASIS_TRANSITION.sub("", body, count=1)
        announced = body != sentence
        if re.search(r",\s*(?:first|second|third) of all,\s*", body, re.I):
            body = LEAD_IN.sub("", body)
            announced = True
        heading, colon, statement = body.partition(":")
        labelled = (
            colon and (not statement or statement[:1].isspace())
            and heading.rstrip()[-1:].isalpha() and 1 <= len(words(heading).split()) <= 8
        )
        variants = [body]
        if announced:
            point_body = POINT_LEAD_IN.sub("", body)
            variants.append(HEADING_LEAD_IN.sub("", point_body))
            if QUESTION_LEAD_IN.match(point_body):
                variants.append(QUESTION_LEAD_IN.sub("", point_body, count=1))
            if CONFIRMATION_TAG.search(point_body):
                variants.append(CONFIRMATION_TAG.sub("", point_body, count=1))
        if announced or labelled:
            for variant in variants:
                if point := words(variant):
                    yield point, (index, 0), point
        if announced and (link := HEADING_LINK.fullmatch(body)):
            heading, statement = link.groups()
            for variant in dict.fromkeys((heading, HEADING_LEAD_IN.sub("", heading))):
                if 1 <= len(words(variant).split()) <= 8:
                    yield words(f"{variant} {statement}"), (index, 0), words(body)
        if announced and (restart := RESTART.fullmatch(body)) and words(restart[2]).split()[:2] == words(restart[1]).split():
            point = words(restart[2])
            yield point, (index, 0), point
        for clause_index, match in enumerate(CONTRAST.finditer(body), 1):
            point = words(body[match.end():])
            if point:
                # A complete contrasting clause can stand alone, but its
                # quotation must retain the source contrast and qualifications.
                yield point, (index, clause_index), words(body)
        is_heading = (announced and not colon and 1 <= len(words(body).split()) <= 8) or (labelled and not statement)
        if is_heading and index + 1 < len(parts):
            # Keep an announced heading and its adjacent statement together.
            following = parts[index + 1]
            if not OUTLINE.match(following):
                for variant in variants:
                    point = words(f"{variant} {following}")
                    yield point, (index, 0), point
        for clause_index, point in enumerate(_parallel_points(body), 1):
            yield point, (index, clause_index), point


def _point_entries(section: str) -> list[str]:
    entries = []
    if section.strip() == NO_POINTS:
        return entries
    for raw in section.splitlines():
        if not (line := raw.strip()):
            continue
        if marker := LIST_MARKER.match(line):
            entries.append(line[marker.end():])
        elif entries:
            entries[-1] += f" {line}"
        else:
            entries.append(line)
    return entries


def log_point_notes(notes: str, job_id, stage: str):
    section = re.search(r"(?im)^Announced points and headings:\s*\n(.*?)(?=^Bible passages:|\Z)", notes, re.S)
    entries = _point_entries(section[1]) if section else []
    logger.info("AI point notes job=%s stage=%s generated=%d with_evidence=%d", job_id, stage, len(entries),
                sum(bool(entry.partition(QUOTE_SEPARATOR)[2].strip()) for entry in entries))


def verified_main_points(section: str, transcript: str, *, job_id=None) -> list[str]:
    entries = _point_entries(section)
    if not entries:
        logger.info("AI main points job=%s generated=0 parsed=0 verified=0 rejected=0 reasons={}", job_id)
        return []
    source = f" {words(transcript)} "
    candidates = {}
    for wording, position, proof in _outline_points(transcript):
        candidates.setdefault(wording, []).append((position, proof))
    verified = {}
    reasons = Counter()
    parsed = 0
    for entry in entries:
        point, separator, quote = (part.strip() for part in entry.partition(QUOTE_SEPARATOR))
        if not separator or not quote:
            reasons["missing_quotation"] += 1
            continue
        if not point or QUOTE_SEPARATOR in quote or not words(quote):
            reasons["malformed_evidence"] += 1
            continue
        parsed += 1
        quote = words(quote)
        key = words(point)
        matches = candidates.get(key, [])
        # Preserve the first announced position while accepting an exact
        # supporting quotation from a later repetition of the same point.
        position = matches[0][0] if matches else None
        if f" {quote} " not in source:
            reasons["quotation_not_in_transcript"] += 1
        elif not key or not (
            any(f" {proof} " in f" {quote} " for _, proof in matches)
            if matches else f" {key} " in f" {quote} "
        ):
            reasons["point_not_in_quotation"] += 1
        elif position is None:
            reasons["no_matching_outline_or_parallel_clause"] += 1
        elif position not in verified:
            verified[position] = point
        else:
            reasons["duplicate_outline_position"] += 1
            if len(key) > len(words(verified[position])):
                verified[position] = point

    reasons["point_limit"] += max(0, len(verified) - MAX_POINTS)
    logger.info(
        "AI main points job=%s generated=%d parsed=%d verified=%d rejected=%d reasons=%s",
        job_id, len(entries), parsed, min(len(verified), MAX_POINTS), sum(reasons.values()),
        {reason: count for reason, count in reasons.items() if count},
    )
    return [verified[position] for position in sorted(verified)[:MAX_POINTS]]
