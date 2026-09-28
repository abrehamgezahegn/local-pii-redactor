from __future__ import annotations

import csv
import io
import math
import re
import shutil
import subprocess
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageOps, ImageStat, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from pillow_heif import register_heif_opener

register_heif_opener()

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_SIDE = 30_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
Category = Literal["name", "email", "phone", "address", "date_of_birth", "id", "financial", "other"]


class RedactionError(ValueError):
    pass


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


@dataclass
class Word:
    text: str
    box: Box
    line: tuple[int, int, int]
    confidence: float = 100


class Region(BaseModel):
    id: str
    category: str
    text: str
    source: str
    boxes: list[Box]


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    category: Category
    line_ids: list[int] = Field(default_factory=list, max_length=12)


class Entities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[Entity] = Field(max_length=500)


def load_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES:
        raise RedactionError("Choose an image smaller than 20 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as src:
                if src.format not in {"PNG", "JPEG", "WEBP", "TIFF", "BMP", "GIF", "AVIF", "HEIF"}:
                    raise RedactionError("Use PNG, JPEG, HEIC, WebP, AVIF, BMP, static GIF or single-page TIFF.")
                if getattr(src, "n_frames", 1) != 1:
                    raise RedactionError("Animated or multi-page images are not supported. Export each page separately.")
                if src.width * src.height > MAX_PIXELS:
                    raise RedactionError("Choose an image with no more than 24 million pixels.")
                if max(src.size) > MAX_SIDE:
                    raise RedactionError("Each image side must be at most 30,000 pixels. Crop long images into sections.")
                src.load()
                oriented = ImageOps.exif_transpose(src)
                if oriented.mode in {"I", "F", "I;16", "I;16B", "I;16L"}:
                    low, high = oriented.getextrema()
                    if not math.isfinite(low) or not math.isfinite(high):
                        raise RedactionError("The image contains invalid pixel values.")
                    if high != low:
                        oriented = oriented.convert("F").point(lambda p: (p - low) * 255 / (high - low)).convert("L")
                    else:
                        oriented = Image.new("L", oriented.size, 255 if high > 0 else 0)
                oriented = oriented.convert("RGBA")
                clean = Image.new("RGB", oriented.size, "white")
                clean.paste(oriented, mask=oriented.getchannel("A"))
                return clean
    except RedactionError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise RedactionError("This image is damaged, unsupported, or too large.") from None


def png_bytes(image: Image.Image) -> bytes:
    # Copy pixels onto a fresh image so EXIF, comments, profiles and text chunks are absent.
    clean = Image.new("RGB", image.size, "white")
    clean.paste(image)
    out = io.BytesIO()
    clean.save(out, format="PNG")
    return out.getvalue()


def _ocr_pass(image: Image.Image, language: str, timeout: float) -> list[Word]:
    try:
        executable = shutil.which("tesseract")
        if not executable:
            # Finder does not inherit Homebrew's PATH from the user's shell.
            executable = next((p for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract") if Path(p).is_file()), "tesseract")
        result = subprocess.run(
            [executable, "stdin", "stdout", "-l", language, "--psm", "11", "tsv"],
            input=png_bytes(image), capture_output=True, timeout=max(1, timeout), check=False,
        )
    except FileNotFoundError:
        raise RedactionError("Tesseract is missing. Install it, then restart the app.") from None
    except subprocess.TimeoutExpired:
        raise RedactionError("Text recognition took too long. Try a smaller or clearer image.") from None
    if result.returncode:
        raise RedactionError("Text recognition failed. Check that the selected Tesseract language is installed.")
    words = []
    for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8", errors="replace")), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        try:
            x, y = int(row["left"]), int(row["top"])
            width, height = int(row["width"]), int(row["height"])
            confidence = float(row["conf"])
            line = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0 or x < 0 or y < 0 or not math.isfinite(confidence):
            continue
        width, height = min(width, image.width - x), min(height, image.height - y)
        if width > 0 and height > 0:
            words.append(Word(text, Box(x=x, y=y, width=width, height=height), line, confidence))
    return words


def _deskew_angle(image: Image.Image) -> int:
    small = image.copy()
    small.thumbnail((800, 800))
    mask = small.point(lambda p: 255 if p < 160 else 0)
    def score(angle):
        rotated = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        raw, width = rotated.tobytes(), rotated.width
        return sum(sum(raw[i:i + width]) ** 2 for i in range(0, len(raw), width))
    scores = {angle: score(angle) for angle in range(-6, 7)}
    best = max(scores, key=scores.get)
    return best if scores[best] > scores[0] * 1.12 else 0


def _ocr_score(words: list[Word]) -> float:
    return sum(min(len(w.text), 25) * (max(0, min(w.confidence, 100)) / 100) ** 3 for w in words if any(c.isalnum() for c in w.text))


def _restore_words(words: list[Word], original_size: tuple[int, int], prepared_size: tuple[int, int], rotated_size: tuple[int, int], angle: float) -> list[Word]:
    """Map all four box corners back to original pixels after scale and rotation."""
    width, height = original_size
    pw, ph = prepared_size
    rw, rh = rotated_size
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    restored = []
    for word in words:
        b = word.box
        points = []
        for x, y in ((b.x, b.y), (b.x + b.width, b.y), (b.x, b.y + b.height), (b.x + b.width, b.y + b.height)):
            dx, dy = x - rw / 2, y - rh / 2
            points.append(((cosine * dx - sine * dy + pw / 2) * width / pw,
                           (sine * dx + cosine * dy + ph / 2) * height / ph))
        x0 = max(0, math.floor(min(x for x, _ in points)))
        y0 = max(0, math.floor(min(y for _, y in points)))
        x1 = min(width, math.ceil(max(x for x, _ in points)))
        y1 = min(height, math.ceil(max(y for _, y in points)))
        if x1 > x0 and y1 > y0:
            restored.append(Word(word.text, Box(x=x0, y=y0, width=x1-x0, height=y1-y0), word.line, word.confidence))
    return restored


def snap_to_ink(words: list[Word], ink: Image.Image) -> list[Word]:
    """Widen word boxes over adjacent glyph pixels that Tesseract left out.

    Tesseract can report a word's text in full while its box skips a thin leading or trailing
    glyph, such as the "1" in "1992-04-18", which then stays visible beside the marker.
    `ink` is a grayscale image of the same size with dark text on a light background.
    Only unclaimed ink is absorbed: growth stops at any other word's box.
    """
    snapped = []
    for i, word in enumerate(words):
        b = word.box
        x0, y0 = math.floor(b.x), math.floor(b.y)
        x1, y1 = min(ink.width, math.ceil(b.x + b.width)), min(ink.height, math.ceil(b.y + b.height))
        height = y1 - y0
        reach, max_gap = max(3, round(height * 0.8)), max(2, round(height * 0.5))
        others = [(o.box.x, o.box.x + o.box.width) for j, o in enumerate(words)
                  if j != i and o.box.y < y1 and o.box.y + o.box.height > y0]
        claimed = lambda x: any(a <= x < c for a, c in others)  # noqa: E731
        band_x0, band_x1 = max(0, x0 - reach), min(ink.width, x1 + reach)
        band = ink.crop((band_x0, y0, band_x1, y1))
        raw, width = band.tobytes(), band.width
        dark = [any(raw[row * width + col] < 128 for row in range(band.height)) for col in range(width)]
        new_x0, new_x1 = x0, x1
        for direction, start, stop in ((-1, x0 - 1, band_x0 - 1), (1, x1, band_x1)):
            gap = 0
            for x in range(start, stop, direction):
                if claimed(x):
                    break
                if dark[x - band_x0]:
                    gap = 0
                    if direction < 0:
                        new_x0 = x
                    else:
                        new_x1 = x + 1
                else:
                    gap += 1
                    if gap > max_gap:
                        break
        if (new_x0, new_x1) != (x0, x1):
            word = Word(word.text, Box(x=new_x0, y=b.y, width=new_x1 - new_x0, height=b.height), word.line, word.confidence)
        snapped.append(word)
    return snapped


def ocr(image: Image.Image, language: str = "eng") -> list[Word]:
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language):
        raise RedactionError("Use a Tesseract language code, such as eng or eng+amh.")
    started = time.monotonic()
    prepared = ImageOps.autocontrast(ImageOps.grayscale(image))
    if ImageStat.Stat(prepared).mean[0] < 127:
        prepared = ImageOps.invert(prepared)
    ink = prepared
    # Improve small screenshot text without reducing original pixels or exceeding
    # a bounded working canvas. Pixel coordinates are restored before redaction.
    scale = min(3 if max(image.size) < 1200 else 2, math.sqrt(12_000_000 / (image.width * image.height)), MAX_SIDE / max(image.size))
    scale = max(1, scale)
    if scale > 1:
        prepared = prepared.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    best_words, best_score, best_angle, best_size = [], -1, 0, prepared.size
    for orientation in (0, 90, 180, 270):
        rotated = prepared.rotate(orientation, expand=True, fillcolor=255)
        correction = _deskew_angle(rotated)
        for angle in dict.fromkeys((orientation, orientation + correction)):
            view = prepared.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=255)
            remaining = 60 - (time.monotonic() - started)
            if remaining <= 1:
                raise RedactionError("Text recognition took too long. Crop the image into smaller sections.")
            words = _ocr_pass(view, language, remaining)
            quality = _ocr_score(words)
            if quality > best_score:
                best_words, best_score, best_angle, best_size = words, quality, angle, view.size
        weight = sum(len(w.text) for w in best_words)
        confidence = sum(len(w.text) * max(0, w.confidence) for w in best_words) / max(1, weight)
        if orientation == 0 and len(best_words) >= 3 and confidence >= 85:
            break
    return snap_to_ink(_restore_words(best_words, image.size, prepared.size, best_size, best_angle), ink)


def text_map(words: list[Word]) -> tuple[str, list[tuple[int, int]]]:
    text = ""
    spans = []
    last_line = None
    for word in words:
        if text:
            text += " " if last_line == word.line else "\n"
        start = len(text)
        text += unicodedata.normalize("NFKC", word.text).casefold()
        spans.append((start, len(text)))
        last_line = word.line
    return text, spans


def document_text(words: list[Word]) -> str:
    """Preserve capitalization for the model; case folding is only for matching."""
    groups: dict[tuple[int, int, int], list[str]] = {}
    for word in words:
        groups.setdefault(word.line, []).append(word.text)
    return "\n".join(" ".join(values) for values in groups.values())


def span_region(words: list[Word], spans: list[tuple[int, int]], start: int, end: int, category: str, source: str) -> Region:
    selected = [w for w, (a, b) in zip(words, spans) if a < end and b > start]
    lines: dict[tuple[int, int, int], list[Box]] = {}
    for word in selected:
        lines.setdefault(word.line, []).append(word.box)
    boxes = []
    for group in lines.values():
        x, y = min(b.x for b in group), min(b.y for b in group)
        boxes.append(Box(x=x, y=y, width=max(b.x + b.width for b in group) - x,
                         height=max(b.y + b.height for b in group) - y))
    return Region(id="", category=category, text=" ".join(w.text for w in selected), source=source, boxes=boxes)


PATTERNS = [
    ("email", re.compile(r"\b[A-Z0-9._%+\-]+\s*@\s*[A-Z0-9.\-]+\s*\.\s*[A-Z]{2,}\b", re.I)),
    ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d ().\-]{5,}\d)(?!\w)")),
    ("id", re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b")),
    ("financial", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.I)),
    # Later space-separated groups must contain a digit, so "X1234567 and DOB" stops after the number.
    ("id", re.compile(r"\b(?:passport|national\s+id|id\s*(?:number|no\.?|#)|license|licence|ssn|tax\s+id)\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{1,30}(?:[ ](?=[A-Z0-9-]*\d)[A-Z0-9-]{1,30}){0,5})", re.I)),
    ("date_of_birth", re.compile(r"\b(?:dob|date\s+of\s+birth|birth\s*date)\s*[:#-]?\s*(\d{1,4}[./-]\d{1,2}[./-]\d{1,4})", re.I)),
]


def rule_matches(text: str):
    """Yield (start, end, category) for every pattern match in text."""
    for category, pattern in PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span(1 if match.lastindex else 0)
            if category == "phone":
                if re.fullmatch(r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}", match.group()):
                    continue
                digits = re.sub(r"\D", "", match.group())
                if not 7 <= len(digits) <= 19:
                    continue
                category_for_match = "financial" if len(digits) >= 13 else "phone"
            else:
                category_for_match = category
            if category == "id" and match.lastindex and not any(c.isdigit() for c in match.group(1)):
                continue
            yield start, end, category_for_match


def rule_regions(words: list[Word]) -> list[Region]:
    text, spans = text_map(words)
    return [span_region(words, spans, start, end, category, "pattern") for start, end, category in rule_matches(text)]


def entity_regions(words: list[Word], entities: list[Entity]) -> tuple[list[Region], int]:
    text, spans = text_map(words)
    line_numbers = {line: i + 1 for i, line in enumerate(dict.fromkeys(w.line for w in words))}
    compact_indices = [i for i, char in enumerate(text) if char.isalnum()]
    compact = "".join(text[i] for i in compact_indices)
    regions, unmatched = [], 0
    for entity in entities:
        value = unicodedata.normalize("NFKC", entity.text).casefold().strip()
        pattern = r"\s+".join(re.escape(part) for part in value.split())
        if not pattern:
            unmatched += 1
            continue
        matches = list(re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text))
        matched_spans = [match.span() for match in matches]
        if not matched_spans:
            # Match punctuation/spacing differences without changing any letters.
            needle = "".join(char for char in value if char.isalnum())
            if len(needle) >= 5:
                for match in re.finditer(re.escape(needle), compact):
                    start = compact_indices[match.start()]
                    end = compact_indices[match.end() - 1] + 1
                    if (start == 0 or not text[start - 1].isalnum()) and (end == len(text) or not text[end].isalnum()):
                        matched_spans.append((start, end))
        if matched_spans:
            for start, end in matched_spans:
                regions.append(span_region(words, spans, start, end, entity.category, "llama"))
        elif entity.line_ids and all(i in line_numbers.values() for i in entity.line_ids):
            # A copied value can contain corrected OCR or an invented suffix.
            # Its schema-constrained source lines still have known pixel positions.
            for line_id in dict.fromkeys(entity.line_ids):
                selected = [span for w, span in zip(words, spans) if line_numbers[w.line] == line_id]
                regions.append(span_region(words, spans, selected[0][0], selected[-1][1], entity.category, "llama whole line"))
        else:
            unmatched += 1
    return regions, unmatched


def deduplicate(regions: list[Region]) -> list[Region]:
    found: dict[tuple, Region] = {}
    for region in regions:
        if not region.boxes:
            continue
        key = tuple((b.x, b.y, b.width, b.height) for b in region.boxes)
        if key in found:
            if found[key].category == "phone" and region.category in {"id", "date_of_birth", "financial"}:
                found[key].category = region.category
            if found[key].source != region.source:
                found[key].source = "llama + pattern"
        else:
            found[key] = region
    result = sorted(found.values(), key=lambda r: (r.boxes[0].y, r.boxes[0].x))
    for i, region in enumerate(result):
        region.id = f"r{i + 1}"
    return result


def marker_rect(image: Image.Image, box: Box, padding: int) -> tuple[int, int, int, int]:
    """Return the padded (x0, y0, x1, y1) pixel rectangle a marker covers; x1/y1 are exclusive."""
    if box.x >= image.width or box.y >= image.height or box.x + box.width > image.width + 0.01 or box.y + box.height > image.height + 0.01:
        raise RedactionError("A redaction falls outside the image. Reload the image and try again.")
    return (max(0, math.floor(box.x) - padding), max(0, math.floor(box.y) - padding),
            min(image.width, math.ceil(box.x + box.width) + padding), min(image.height, math.ceil(box.y + box.height) + padding))


def redact(image: Image.Image, boxes: list[Box], padding: int = 4) -> Image.Image:
    if not 0 <= padding <= 40:
        raise RedactionError("Padding must be between 0 and 40 pixels.")
    if not boxes:
        raise RedactionError("Select at least one redaction before exporting.")
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box in boxes:
        x0, y0, x1, y1 = marker_rect(image, box, padding)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(0, 0, 0))
    return result
