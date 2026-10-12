"""Find and replace personal values in plain text."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .anonymize import Faker, label
from .core import Entity, rule_matches

MAX_TEXT = 30_000


@dataclass
class Span:
    start: int
    end: int
    category: str
    source: str


def clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def rule_spans(text: str) -> list[Span]:
    spans = []
    for start, end, category in rule_matches(text):
        start, end = _trim(text, start, end)
        if end > start:
            spans.append(Span(start, end, category, "pattern"))
    return spans


def entity_spans(text: str, entities: list[Entity]) -> tuple[list[Span], int, int]:
    """Locate model values in the text. Returns (spans, unmatched count, whole-line fallbacks)."""
    lines, offset, line_ranges = text.split("\n"), 0, {}
    for i, line in enumerate(lines, 1):
        line_ranges[i] = _trim(text, offset, offset + len(line))
        offset += len(line) + 1
    lowered = [c.lower() if len(c.lower()) == 1 else c for c in text]
    compact_indices = [i for i, c in enumerate(text) if c.isalnum()]
    compact = "".join(lowered[i] for i in compact_indices)
    spans, unmatched, whole_lines = [], 0, 0
    for entity in entities:
        parts = entity.text.split()
        if not parts:
            unmatched += 1
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(p) for p in parts) + r"(?!\w)"
        found = [m.span() for m in re.finditer(pattern, text, re.I)]
        if not found:
            # Tolerate punctuation and spacing differences without changing letters.
            needle = "".join(c.lower() for c in entity.text if c.isalnum())
            if len(needle) >= 5:
                for match in re.finditer(re.escape(needle), compact):
                    start, end = compact_indices[match.start()], compact_indices[match.end() - 1] + 1
                    if (start == 0 or not text[start - 1].isalnum()) and (end == len(text) or not text[end].isalnum()):
                        found.append((start, end))
        if found:
            spans.extend(Span(start, end, entity.category, "llama") for start, end in found)
        elif entity.line_ids and all(i in line_ranges for i in entity.line_ids):
            # The model changed the value; cover its whole source line rather than leak it.
            for line_id in dict.fromkeys(entity.line_ids):
                start, end = line_ranges[line_id]
                if end > start:
                    spans.append(Span(start, end, entity.category, "llama whole line"))
                    whole_lines += 1
        else:
            unmatched += 1
    return spans, unmatched, whole_lines


def merge_spans(spans: list[Span]) -> list[Span]:
    """Join overlapping spans; the merged span keeps the category of its longest part."""
    merged: list[tuple[Span, Span]] = []  # (merged span, longest member)
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if merged and span.start < merged[-1][0].end:
            current, longest = merged[-1]
            current.end = max(current.end, span.end)
            if span.source != current.source:
                current.source = "llama + pattern"
            if span.end - span.start > longest.end - longest.start:
                longest = span
                current.category = span.category
            merged[-1] = (current, longest)
        else:
            merged.append((Span(span.start, span.end, span.category, span.source), span))
    return [span for span, _ in merged]


def replace_spans(text: str, spans: list[Span], mode: str, faker: Faker | None = None) -> str:
    faker = faker or Faker()
    out, position = [], 0
    for span in merge_spans(spans):
        out.append(text[position:span.start])
        value = text[span.start:span.end]
        out.append(faker.value(span.category, value) if mode == "anonymize" else label(span.category))
        position = span.end
    out.append(text[position:])
    return "".join(out)
