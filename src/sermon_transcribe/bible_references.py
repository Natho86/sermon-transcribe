"""Validate the structure of model-generated Bible references without inferring citations."""

import re
from dataclasses import dataclass
from math import inf

MAX_REFERENCES = 5
NO_REFERENCES = "None noted."
BOOKS = {
    book.casefold(): book
    for book in (
        "Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|1 Samuel|2 Samuel|"
        "1 Kings|2 Kings|1 Chronicles|2 Chronicles|Ezra|Nehemiah|Esther|Job|Psalms|Proverbs|"
        "Ecclesiastes|Song of Songs|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|"
        "Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Mark|"
        "Luke|John|Acts|Romans|1 Corinthians|2 Corinthians|Galatians|Ephesians|Philippians|"
        "Colossians|1 Thessalonians|2 Thessalonians|1 Timothy|2 Timothy|Titus|Philemon|Hebrews|"
        "James|1 Peter|2 Peter|1 John|2 John|3 John|Jude|Revelation"
    ).split("|")
}
BOOKS.update({"psalm": "Psalms", "song of solomon": "Song of Songs"})
DASHES = str.maketrans("‐‑‒–—−", "------")
NUMBER = r"[1-9][0-9]{0,2}"
CITATION = re.compile(
    rf"(?P<book>.+?) (?P<chapter>{NUMBER})(?:\s*:\s*(?P<verse>{NUMBER}))?"
    rf"(?:\s*-\s*(?P<end>{NUMBER})(?:\s*:\s*(?P<end_verse>{NUMBER}))?)?"
)
REDUNDANT_VERSES = re.compile(r"(?:specifically |including )?(?:vv?\.|verses?)\s+(.+)", re.I)
LIST_MARKER = re.compile(r"^(?:[-*•‣◦▪–—]|[0-9]+[.)])\s*")
EMPHASIS = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__|\*([^*]+)\*|_([^_]+)_")


@dataclass(frozen=True)
class Reference:
    book: str
    start: tuple[int, int]
    end: tuple[int, int | float]

    def __str__(self):
        book = "Psalm" if self.book == "Psalms" else self.book
        start_chapter, start_verse = self.start
        end_chapter, end_verse = self.end
        if end_verse == inf:
            span = str(start_chapter) if start_chapter == end_chapter else f"{start_chapter}–{end_chapter}"
        elif self.start == self.end:
            span = f"{start_chapter}:{start_verse}"
        elif start_chapter == end_chapter:
            span = f"{start_chapter}:{start_verse}–{end_verse}"
        else:
            span = f"{start_chapter}:{start_verse}–{end_chapter}:{end_verse}"
        return f"{book} {span}"

    def contains(self, other):
        return self.book == other.book and self.start <= other.start and self.end >= other.end


def parse_citation(text: str) -> Reference | None:
    match = CITATION.fullmatch(text)
    if not match:
        return None
    book = BOOKS.get(match["book"].casefold())
    if not book:
        return None
    chapter, verse, end, end_verse = (int(match[name]) if match[name] else None for name in ("chapter", "verse", "end", "end_verse"))
    if verse is None and end_verse is not None:
        return None
    start = (chapter, verse or 1)
    if end_verse is not None:
        finish = (end, end_verse)
    elif verse is None:
        finish = (end or chapter, inf)
    else:
        finish = (chapter, end or verse)
    return Reference(book, start, finish) if start <= finish else None


def parse_reference(text: str) -> Reference | None:
    text = " ".join(text.translate(DASHES).split())
    text = re.sub(r"(?<=[0-9])\s+to\s+(?=[0-9])", "-", text, flags=re.I)
    if match := re.fullmatch(r"([^()]+?)\s*\(([^()]+)\)", text):
        outer, annotation = (part.strip() for part in match.groups())
        if book := BOOKS.get(outer.casefold()):
            reference = parse_citation(annotation)
            return reference if reference and reference.book == book else None
        reference = parse_citation(outer)
        verses = REDUNDANT_VERSES.fullmatch(annotation)
        if not reference or not verses or reference.start[0] != reference.end[0]:
            return None
        for item in verses[1].split(","):
            verse = parse_citation(f"{reference.book} {reference.start[0]}:{item.strip()}")
            if not verse or verse.start[0] != verse.end[0] or not reference.contains(verse):
                return None
        return reference
    return parse_citation(text)


def merge_references(first: Reference, second: Reference) -> Reference | None:
    if first.book != second.book:
        return None
    if first.contains(second):
        return first
    if second.contains(first):
        return second
    earlier, later = sorted((first, second), key=lambda reference: reference.start)
    overlap = later.start <= earlier.end
    adjacent = later.start[0] == earlier.end[0] and later.start[1] == earlier.end[1] + 1
    if not overlap and not adjacent:
        return None
    end = max(earlier.end, later.end)
    if end[1] == inf and earlier.start[1] != 1:
        # A range ending at an unspecified last verse cannot be written without
        # knowing that verse. Ask for simpler citations instead of guessing.
        raise ValueError("Combine overlapping Bible references using a clear chapter or verse range.")
    return Reference(first.book, earlier.start, end)


def _reference_entry(line: str) -> str:
    plain = EMPHASIS.sub(lambda match: next(part for part in match.groups() if part), line.strip())
    return " ".join(LIST_MARKER.sub("", plain).split())


def normalize_bible_references(section: str) -> str:
    entries = [_reference_entry(line) for line in section.splitlines() if line.strip()]
    if len(entries) == 1 and entries[0].casefold() in ("none noted", "none noted."):
        return NO_REFERENCES

    references = []
    for entry in entries:
        reference = parse_reference(entry)
        if reference is None:
            raise ValueError("Use a Bible book plus a chapter or verse range in each reference. Remove names and nonessential annotations.")

        # A new entry can bridge two earlier ranges. Restart the scan after a
        # merge and keep the earliest position of the entries being combined.
        index = 0
        position = len(references)
        while index < len(references):
            merged = merge_references(references[index], reference)
            if merged is None:
                index += 1
                continue
            reference = merged
            references.pop(index)
            position = min(position, index)
            index = 0
        references.insert(position, reference)

    if not 1 <= len(references) <= MAX_REFERENCES:
        raise ValueError(f"Use one to {MAX_REFERENCES} distinct Bible references, with the main passage first.")
    return "\n".join(f"- {reference}" for reference in references)
