import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image

from pii_redactor.anonymize import Faker, anonymize_image
from pii_redactor.core import Box, Entity, RedactionError, Region
from pii_redactor.pipeline import analyze_text
from pii_redactor.text import Span, entity_spans, merge_spans, replace_spans, rule_spans


class FakeModel:
    def __init__(self, entities):
        self.entities = entities

    def detect(self, text, model):
        return self.entities


class BrokenModel:
    def detect(self, text, model):
        raise RedactionError("Ollama could not complete the request.")


def test_faker_is_consistent_and_never_returns_the_original():
    faker = Faker(seed=1)
    first = faker.value("name", "Maya Tesfaye")
    assert first != "Maya Tesfaye" and len(first.split()) == 2
    assert faker.value("name", "maya  tesfaye") == first
    # A later first-name-only mention gets the same fake first name.
    assert faker.value("name", "Maya") == first.split()[0]
    assert faker.value("name", "Samir Haddad") != first


@pytest.mark.parametrize("category, value, shape", [
    ("email", "maya.tesfaye@company.com", r"^[a-z]+\.[a-z]+@example\.(com|org|net)$"),
    ("phone", "+1 (202) 555-0147", r"^\+1 \(\d{3}\) \d{3}-\d{4}$"),
    ("phone", "0911 234 567", r"^\d{4} \d{3} \d{3}$"),
    ("address", "Austin TX 73301", r"^[A-Z][a-z]+ [A-Z]{2} \d{5}$"),
    ("address", "184 Cedar Lane, Portland, OR 97205", r"^\d{3} \w+ \w+, [A-Z][a-z]+ [A-Z]{2} \d{5}$"),
    ("id", "AB1234567", r"^[A-Z]{2}\d{7}$"),
    ("financial", "4111 1111 1111 1111", r"^\d{4} \d{4} \d{4} \d{4}$"),
    ("date_of_birth", "1992-04-18", r"^(19[5-9]\d|200[0-4])-(0[1-9]|1[0-2])-(0[1-9]|1[0-2])$"),
    ("address", "184 Cedar Lane", r"^\d{3} \w+ \w+$"),
    ("other", "@mayat", r"^@[a-z]+\d{2}$"),
])
def test_fake_values_keep_the_shape_of_each_category(category, value, shape):
    import re
    fake = Faker(seed=3).value(category, value)
    assert fake != value
    assert re.match(shape, fake), fake


@pytest.mark.parametrize("text, expected", [
    ("Her passport: X12345678 and DOB: 1988-03-14.", "X12345678"),
    ("ID number: 12 345 678 issued today", "12 345 678"),
    ("License: D-1234-5678", "D-1234-5678"),
])
def test_id_rule_stops_before_following_words(text, expected):
    assert [text[s.start:s.end] for s in rule_spans(text) if s.category == "id"] == [expected]


def test_trailing_ocr_punctuation_is_kept():
    assert Faker(seed=2).value("name", "Tesfaye,").endswith(",")


def test_anonymized_image_covers_original_pixels_and_writes_light_text():
    image = Image.new("RGB", (300, 80), "white")
    image.paste((200, 30, 30), (20, 20, 160, 50))  # stand-in for the original text
    region = Region(id="r1", category="name", text="Maya Tesfaye", source="pattern", boxes=[Box(x=20, y=20, width=140, height=30)])
    output = anonymize_image(image, [region], Faker(seed=1), padding=4)
    marker = output.crop((16, 16, 164, 54))
    colors = {color for _, color in marker.getcolors(maxcolors=100_000)}
    assert (200, 30, 30) not in colors  # nothing of the original survives
    assert (0, 0, 0) in colors and any(min(c) > 200 for c in colors)  # black marker with white writing
    assert output.getpixel((200, 60)) == (255, 255, 255)
    assert image.getpixel((30, 30)) == (200, 30, 30)  # source untouched


def test_overlapping_regions_are_all_covered():
    image = Image.new("RGB", (300, 80), (200, 30, 30))
    regions = [Region(id="a", category="name", text="Maya Tesfaye", source="llama", boxes=[Box(x=10, y=10, width=200, height=30)]),
               Region(id="b", category="id", text="AB123", source="pattern", boxes=[Box(x=150, y=10, width=120, height=30)])]
    output = anonymize_image(image, regions, Faker(seed=1), padding=0)
    colors = {color for _, color in output.crop((10, 10, 270, 40)).getcolors(maxcolors=100_000)}
    assert (200, 30, 30) not in colors


def test_anonymize_requires_regions():
    with pytest.raises(RedactionError):
        anonymize_image(Image.new("RGB", (10, 10)), [], Faker())


SAMPLE = "Hi, I'm Maya Tesfaye.\nEmail me at maya@example.com or call +1 (202) 555-0147.\nPassport: AB1234567\nThanks, Maya Tesfaye"


def test_text_redaction_replaces_every_value_with_labels():
    result = analyze_text(SAMPLE, "test", FakeModel([Entity(text="Maya Tesfaye", category="name", line_ids=[1, 4])]))
    assert result["complete"]
    output = replace_spans(result["text"], result["spans"], "redact")
    assert output == "Hi, I'm [NAME].\nEmail me at [EMAIL] or call [PHONE].\nPassport: [ID]\nThanks, [NAME]"


def test_text_anonymization_is_consistent_and_removes_originals():
    result = analyze_text(SAMPLE, "test", FakeModel([Entity(text="Maya Tesfaye", category="name", line_ids=[1, 4])]))
    output = replace_spans(result["text"], result["spans"], "anonymize", Faker(seed=4))
    for original in ["Maya", "Tesfaye", "maya@example.com", "555-0147", "AB1234567"]:
        assert original not in output
    fake_name = output.split("I'm ")[1].split(".")[0]
    assert output.endswith("Thanks, " + fake_name)


def test_text_falls_back_to_whole_line_and_reports_missing_values():
    text = "Patient: Mya Tesfay\nNotes follow"
    spans, unmatched, whole = entity_spans(text, [Entity(text="Maya Tesfaye", category="name", line_ids=[1]),
                                                   Entity(text="Nobody", category="name")])
    assert unmatched == 1 and whole == 1
    assert replace_spans(text, spans, "redact") == "[NAME]\nNotes follow"


def test_text_scan_is_incomplete_when_model_fails_but_keeps_pattern_hits():
    result = analyze_text("Write to maya@example.com", "test", BrokenModel())
    assert not result["complete"]
    assert [text for text in (result["text"][s.start:s.end] for s in result["spans"])] == ["maya@example.com"]


def test_text_limits():
    with pytest.raises(RedactionError, match="Type or paste"):
        analyze_text("   \n ", "test", FakeModel([]))
    with pytest.raises(RedactionError, match="at most"):
        analyze_text("a" * 30_001, "test", FakeModel([]))


def test_merge_keeps_longest_category():
    merged = merge_spans([Span(0, 5, "phone", "pattern"), Span(0, 12, "id", "llama"), Span(20, 25, "name", "llama")])
    assert [(s.start, s.end, s.category) for s in merged] == [(0, 12, "id"), (20, 25, "name")]
    assert merged[0].source == "llama + pattern"


def test_merge_handles_nested_and_adjacent_spans():
    # Fully nested: inner span absorbed, outer category kept if it is the longest.
    nested = merge_spans([Span(0, 20, "name", "llama"), Span(5, 10, "email", "pattern")])
    assert [(s.start, s.end, s.category) for s in nested] == [(0, 20, "name")]
    assert nested[0].source == "llama + pattern"
    # Adjacent (touching, not overlapping) spans stay separate.
    adjacent = merge_spans([Span(0, 5, "name", "llama"), Span(5, 12, "id", "pattern")])
    assert [(s.start, s.end) for s in adjacent] == [(0, 5), (5, 12)]
    # Disjoint spans stay separate.
    disjoint = merge_spans([Span(0, 5, "name", "llama"), Span(10, 15, "id", "pattern")])
    assert [(s.start, s.end) for s in disjoint] == [(0, 5), (10, 15)]


def test_rule_spans_trim_whitespace():
    text = "Passport: AB1234567   "
    assert [text[s.start:s.end] for s in rule_spans(text)] == ["AB1234567"]


def test_faker_never_returns_original_across_all_categories():
    faker = Faker(seed=42)
    values = [
        ("name", "Maya Tesfaye"),
        ("email", "maya.tesfaye@company.com"),
        ("phone", "+1 (202) 555-0147"),
        ("address", "184 Cedar Lane, Portland, OR 97205"),
        ("date_of_birth", "1992-04-18"),
        ("id", "AB1234567"),
        ("financial", "4111 1111 1111 1111"),
        ("other", "@mayat"),
    ]
    fakes = []
    for category, original in values:
        fake = faker.value(category, original)
        assert fake != original, f"{category}: fake matched original"
        fakes.append(fake)
    # No duplicate fakes across categories.
    assert len(set(fakes)) == len(fakes)


def test_faker_preserves_surrounding_punctuation():
    faker = Faker(seed=7)
    assert faker.value("name", "(Maya)").startswith("(")
    assert faker.value("name", "(Maya)").endswith(")")
    assert faker.value("name", "Tesfaye,").endswith(",")
    assert faker.value("name", "'Samir'").startswith("'")
    assert faker.value("name", "'Samir'").endswith("'")
