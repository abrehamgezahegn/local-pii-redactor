"""Replace detected personal values with consistent fictional ones."""
from __future__ import annotations

import random
import re
import string
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from .core import Region, RedactionError, marker_rect

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Jamie", "Robin", "Elena", "Marco",
               "Priya", "Kenji", "Amara", "Lucas", "Sofia", "Daniel", "Nadia", "Omar", "Leila", "Hugo", "Ines", "Mateo",
               "Zara", "Felix", "Noor", "Tomas", "Yara", "Ivan", "Lena", "Rafael"]
LAST_NAMES = ["Rivera", "Chen", "Okafor", "Novak", "Silva", "Larsen", "Moreau", "Kowalski", "Tanaka", "Mensah", "Reyes",
              "Lindqvist", "Duarte", "Bianchi", "Fischer", "Ramos", "Osei", "Petrov", "Sato", "Walsh", "Ibrahim",
              "Keller", "Varga", "Nakamura", "Costa", "Andersen", "Popescu", "Nguyen", "Herrera", "Brandt"]
STREETS = ["Maple", "Oak", "Willow", "Harbor", "Elm", "Juniper", "Linden", "Birch", "Aspen", "Meadow", "Orchard", "Sycamore"]
STREET_TYPES = ["Street", "Avenue", "Road", "Lane", "Drive", "Court"]
CITIES = ["Springfield", "Riverton", "Fairview", "Lakeside", "Brookfield", "Ashford", "Millbrook", "Westbury"]
STATES = ["CA", "NY", "TX", "WA", "IL", "OR", "CO", "MA"]
LABELS = {"name": "NAME", "email": "EMAIL", "phone": "PHONE", "address": "ADDRESS", "date_of_birth": "DATE OF BIRTH",
          "id": "ID", "financial": "ACCOUNT", "other": "PERSONAL INFO"}


def label(category: str) -> str:
    return f"[{LABELS.get(category, 'PERSONAL INFO')}]"


class Faker:
    """Generates fictional values; the same real value always gets the same fake within one Faker."""

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)
        self.values: dict[tuple[str, str], str] = {}
        self.name_tokens: dict[str, str] = {}

    def value(self, category: str, original: str) -> str:
        key = (category, " ".join(original.casefold().split()))
        if key not in self.values:
            # Keep surrounding punctuation that OCR attaches to words, such as "Tesfaye," or "(202)".
            match = re.fullmatch(r"([^\w@+]*)(.*?)(\W*)", original.strip(), re.S)
            prefix, core, suffix = match.groups()
            fake = core
            for _ in range(10):
                fake = self._make(category, core) if core else ""
                if fake.casefold() != core.casefold():
                    break
            self.values[key] = prefix + fake + suffix
        return self.values[key]

    def _make(self, category: str, core: str) -> str:
        maker = {"name": self._name, "email": self._email, "phone": self._phone, "address": self._address,
                 "date_of_birth": self._date, "id": self._shape, "financial": self._shape, "other": self._other}
        return maker.get(category, self._shape)(core)

    def _pick(self, pool: list[str], used: set[str]) -> str:
        options = [p for p in pool if p not in used] or pool
        return self.random.choice(options)

    def _name(self, core: str) -> str:
        tokens = core.split()[:6]
        used = set(self.name_tokens.values())
        fake = []
        for i, token in enumerate(tokens):
            key = token.casefold().strip(".,")
            if key not in self.name_tokens:
                if len(key) <= 2:  # Initials stay initials.
                    self.name_tokens[key] = self.random.choice(string.ascii_uppercase) + ("." if token.endswith(".") else "")
                else:
                    pool = LAST_NAMES if i == len(tokens) - 1 and len(tokens) > 1 else FIRST_NAMES
                    self.name_tokens[key] = self._pick(pool, used)
                used.add(self.name_tokens[key])
            fake.append(self.name_tokens[key])
        result = " ".join(fake)
        if core.isupper():
            return result.upper()
        if core.islower():
            return result.lower()
        return result

    def _email(self, core: str) -> str:
        first, last = self.random.choice(FIRST_NAMES), self.random.choice(LAST_NAMES)
        return f"{first}.{last}@{self.random.choice(['example.com', 'example.org', 'example.net'])}".lower()

    def _phone(self, core: str) -> str:
        # Keep a "+1 " style country code so the number still reads as the same region.
        country = re.match(r"\+\d{1,3}[ -]", core)
        if country and len(re.sub(r"\D", "", core[country.end():])) >= 6:
            return country.group() + self._shape(core[country.end():])
        return self._shape(core)

    def _address(self, core: str) -> str:
        parts = []
        number = re.match(r"\d+", core)
        if number:
            digits = len(number.group())
            parts.append(f"{self.random.randint(10 ** (digits - 1), 10 ** digits - 1)} {self.random.choice(STREETS)} {self.random.choice(STREET_TYPES)}")
        # A value without a house number is usually the city/state/ZIP part of an address.
        if not number or len(core.split()) > 4:
            parts.append(self.random.choice(CITIES))
            if not number or re.search(r"\b[A-Z]{2}\b", core):
                parts[-1] += " " + self.random.choice(STATES)
        joined = ", ".join(parts) if "," in core else " ".join(parts)
        if re.search(r"\b\d{5}\b", core):
            joined += f" {self.random.randint(10000, 99999)}"
        return joined

    def _date(self, core: str) -> str:
        match = re.fullmatch(r"(\d{1,4})([./-])(\d{1,2})\2(\d{1,4})", core)
        if not match:
            return self._shape(core)
        parts, separator = [match.group(1), match.group(3), match.group(4)], match.group(2)
        year_index = next((i for i, p in enumerate(parts) if len(p) == 4), 2)
        values = []
        for i, part in enumerate(parts):
            if i == year_index:
                year = self.random.randint(1950, 2004)
                values.append(str(year) if len(part) == 4 else f"{year % 100:02d}")
            else:
                # Day and month both stay at 12 or below so either date order is valid.
                values.append(str(self.random.randint(1, 12)).zfill(len(part)))
        return separator.join(values)

    def _other(self, core: str) -> str:
        handle = f"{self.random.choice(FIRST_NAMES).lower()}{self.random.randint(10, 99)}"
        if core.startswith("@"):
            return "@" + handle
        if "@" in core:
            return self._email(core)
        url = re.match(r"(https?://)?(www\.)?", core, re.I)
        if re.search(r"[a-z0-9]\.[a-z]{2,}", core, re.I):
            return (url.group(1) or "") + (url.group(2) or "") + f"{handle}.example.org"
        return handle

    def _shape(self, core: str) -> str:
        """Same length and punctuation, random digits and letters."""
        out = []
        first_digit = True
        for char in core:
            if char.isdigit():
                out.append(str(self.random.randint(1 if first_digit and char != "0" else 0, 9)))
                first_digit = False
            elif char.isalpha():
                letter = self.random.choice(string.ascii_uppercase)
                out.append(letter if char.isupper() else letter.lower())
            else:
                out.append(char)
        return "".join(out)


@lru_cache(maxsize=64)
def _font(size: int):
    return ImageFont.load_default(size=size)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, width: int, height: int):
    size = max(6, min(int(height * 0.8), 200))
    while size > 6 and (draw.textlength(text, font=_font(size)) > width or _font(size).getbbox(text)[3] - _font(size).getbbox(text)[1] > height):
        size -= 1
    font = _font(size)
    while len(text) > 1 and draw.textlength(text, font=font) > width:
        text = text[:-2] + "…"
    return text, font


def _split_words(value: str, widths: list[int]) -> list[str]:
    """Spread a fake value over a region's line boxes, proportional to each line's width."""
    words = value.split()
    if len(widths) == 1 or not words:
        return [value] + [""] * (len(widths) - 1)
    total, parts, taken, running = sum(widths), [], 0, 0
    for i, width in enumerate(widths):
        running += width
        end = len(words) if i == len(widths) - 1 else max(taken + 1, round(len(words) * running / total))
        parts.append(" ".join(words[taken:end]))
        taken = min(end, len(words))
    return parts


def anonymize_image(image: Image.Image, regions: list[Region], faker: Faker, padding: int = 4) -> Image.Image:
    """Cover every region with a black marker, then write a fictional value on top in white."""
    if not 0 <= padding <= 40:
        raise RedactionError("Padding must be between 0 and 40 pixels.")
    if not any(region.boxes for region in regions):
        raise RedactionError("Select at least one redaction before exporting.")
    result = image.copy()
    draw = ImageDraw.Draw(result)
    labelled = []
    for region in regions:
        rects = [marker_rect(image, box, padding) for box in region.boxes]
        fake = faker.value(region.category, region.text)
        for rect, text in zip(rects, _split_words(fake, [r[2] - r[0] for r in rects])):
            # Every marker is drawn before any text, so original pixels stay covered even where markers overlap.
            draw.rectangle((rect[0], rect[1], rect[2] - 1, rect[3] - 1), fill=(0, 0, 0))
            labelled.append((rect, text))
    written: list[tuple[int, int, int, int]] = []
    for rect, text in sorted(labelled, key=lambda item: -(item[0][2] - item[0][0]) * (item[0][3] - item[0][1])):
        x0, y0, x1, y1 = rect
        if not text or any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in written):
            continue
        inset = max(1, min(4, (y1 - y0) // 8))
        fitted, font = _fit_text(draw, text, x1 - x0 - 2 * inset, y1 - y0 - 2 * inset)
        draw.text((x0 + inset, (y0 + y1) / 2), fitted, font=font, fill=(255, 255, 255), anchor="lm")
        written.append(rect)
    return result
