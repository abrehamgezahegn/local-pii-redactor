import io
import json

import pytest
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from pii_redactor.core import Box, Entity, RedactionError, Word, deduplicate, entity_regions, load_image, ocr, png_bytes, redact, rule_regions
from pii_redactor.model import Ollama, local_url
from pii_redactor.pipeline import analyze


def sample_image():
    image = Image.new("RGB", (1000, 300), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=29)
    for y, value in enumerate(["Name: Maya Tesfaye", "Email: maya.tesfaye@example.com", "Passport: AB1234567"]):
        draw.text((40, 40 + y * 70), value, font=font, fill="black")
    return image


def make_words(lines):
    words = []
    for y, line in enumerate(lines):
        x = 10
        for word in line.split():
            words.append(Word(word, Box(x=x, y=y * 40 + 10, width=len(word) * 9, height=20), (1, 1, y)))
            x += len(word) * 9 + 7
    return words


def test_opaque_pixels_and_metadata_removal():
    source = Image.new("RGB", (100, 80), "white")
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Private Person")
    stream = io.BytesIO()
    source.save(stream, format="PNG", pnginfo=info)
    clean = load_image(stream.getvalue())
    output = redact(clean, [Box(x=10.8, y=20.2, width=20.1, height=9.9)], padding=3)
    decoded = Image.open(io.BytesIO(png_bytes(output)))
    assert decoded.info == {}
    assert decoded.crop((7, 17, 34, 34)).getextrema() == ((0, 0), (0, 0), (0, 0))
    assert decoded.getpixel((6, 17)) == (255, 255, 255)
    assert source.getpixel((15, 25)) == (255, 255, 255)


def test_exif_rotation_and_transparency():
    source = Image.new("RGB", (60, 30), "white")
    exif = source.getexif()
    exif[274] = 6
    data = io.BytesIO()
    source.save(data, format="JPEG", exif=exif)
    loaded = load_image(data.getvalue())
    assert loaded.size == (30, 60)
    assert not loaded.getexif()
    alpha = Image.new("RGBA", (10, 10), (12, 34, 56, 0))
    data = io.BytesIO()
    alpha.save(data, format="PNG")
    assert load_image(data.getvalue()).getpixel((0, 0)) == (255, 255, 255)


def test_multiline_entity_and_repeated_values_map_to_real_words():
    words = make_words(["Name: Maya Tesfaye", "Address: 184 Cedar Lane", "Portland OR 97205", "Contact Maya Tesfaye"])
    regions, missing = entity_regions(words, [Entity(text="Maya Tesfaye", category="name"), Entity(text="184 Cedar Lane Portland OR 97205", category="address"), Entity(text="Invented Person", category="name")])
    assert missing == 1
    assert len(regions) == 3
    address = next(r for r in regions if r.category == "address")
    assert len(address.boxes) == 2
    assert address.boxes[0].x > words[3].box.x
    assert [r.text for r in regions if r.category == "name"] == ["Maya Tesfaye", "Maya Tesfaye"]


def test_common_rules_cover_personal_values():
    words = make_words(["Email: maya@example.com", "Phone: +1 (202) 555-0147", "SSN: 123-45-6789", "Passport: AB1234567", "DOB: 1992-04-18", "Card: 4111 1111 1111 1111"])
    regions = deduplicate(rule_regions(words))
    for expected in ["maya@example.com", "+1 (202) 555-0147", "123-45-6789", "AB1234567", "1992-04-18", "4111 1111 1111 1111"]:
        assert any(expected in r.text for r in regions)
    assert next(r.category for r in regions if r.text == "1992-04-18") == "date_of_birth"
    assert next(r.category for r in regions if r.text == "123-45-6789") == "id"


def test_ordinary_date_is_not_a_phone_number():
    assert not rule_regions(make_words(["Meeting: 2026-11-17"]))


def test_model_chunks_overlap_and_cover_whole_document(monkeypatch):
    client = Ollama()
    monkeypatch.setattr(client, "verify_local", lambda model: None)
    chunks = []
    def request(method, path, body=None, timeout=15):
        chunks.append(json.loads(body["messages"][-1]["content"].split("\n", 1)[1]))
        return {"message": {"content": '{"entities": []}'}}
    monkeypatch.setattr(client, "_request", request)
    text = "\n".join(f"Line {i}: This is a fictional test document with ordinary words." for i in range(120))
    client.detect(text, "llama3.2:latest")
    assert len(chunks) >= 3
    assert chunks[0][0]["line_id"] == 1 and chunks[-1][-1]["line_id"] == 120
    for a, b in zip(chunks, chunks[1:]):
        assert a[-1] == b[0]
    assert {r["line_id"] for chunk in chunks for r in chunk} == set(range(1, 121))


@pytest.mark.parametrize("url", ["https://127.0.0.1:11434", "http://example.com", "http://192.168.1.4:11434", "http://127.0.0.1.example.com", "http://user:pass@localhost:11434", "http://localhost:11434/proxy", "http://localhost:11434?target=cloud"])
def test_remote_ollama_addresses_are_rejected(url):
    with pytest.raises(RedactionError):
        local_url(url)


def test_local_url_ignores_proxy_environment():
    assert local_url("http://localhost:11434") == "http://127.0.0.1:11434"
    assert local_url("http://[::1]:11434/") == "http://[::1]:11434"


def test_remote_model_alias_never_receives_document(monkeypatch):
    client = Ollama()
    calls = []
    def request(method, path, body=None, timeout=15):
        calls.append(path)
        return {"models": [{"name": "alias:latest"}]} if path == "/api/tags" else {"remote_host": "https://ollama.com", "remote_model": "llama"}
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RedactionError):
        client.detect("Private Name", "alias:latest")
    assert "/api/chat" not in calls


def test_model_invalid_output_is_not_treated_as_empty(monkeypatch):
    client = Ollama()
    monkeypatch.setattr(client, "verify_local", lambda model: None)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {"message": {"content": "{bad json}"}})
    with pytest.raises(RedactionError, match="invalid"):
        client.detect("Name: Maya Tesfaye", "llama3.2:latest")


def test_model_failure_returns_explicit_partial_result(monkeypatch):
    monkeypatch.setattr("pii_redactor.pipeline.ocr", lambda image, language: make_words(["Email: maya@example.com"]))
    class Offline:
        def detect(self, text, model):
            raise RedactionError("Ollama is unavailable.")
    result = analyze(Image.new("RGB", (300, 100)), "llama3.2:latest", client=Offline())
    assert not result["complete"]
    assert result["warnings"]
    assert result["regions"][0]["category"] == "email"


def test_no_ocr_text_is_incomplete(monkeypatch):
    monkeypatch.setattr("pii_redactor.pipeline.ocr", lambda image, language: [])
    result = analyze(Image.new("RGB", (100, 100)), "llama3.2:latest")
    assert not result["complete"]
    assert result["word_count"] == 0


def test_reject_empty_out_of_bounds_and_animated_images():
    image = Image.new("RGB", (10, 10))
    with pytest.raises(RedactionError):
        redact(image, [])
    with pytest.raises(RedactionError):
        redact(image, [Box(x=9, y=0, width=2, height=1)])
    data = io.BytesIO()
    image.save(data, format="TIFF", save_all=True, append_images=[image])
    with pytest.raises(RedactionError, match="multi-page"):
        load_image(data.getvalue())


@pytest.mark.parametrize("padding", [-1, 41, 100])
def test_redact_rejects_padding_out_of_range(padding):
    image = Image.new("RGB", (50, 50))
    box = Box(x=0, y=0, width=10, height=10)
    with pytest.raises(RedactionError, match="Padding"):
        redact(image, [box], padding=padding)


def test_redact_with_zero_padding_covers_exact_box():
    image = Image.new("RGB", (50, 50), "white")
    box = Box(x=10, y=10, width=20, height=20)
    output = redact(image, [box], padding=0)
    assert output.getpixel((10, 10)) == (0, 0, 0)
    assert output.getpixel((29, 29)) == (0, 0, 0)
    assert output.getpixel((9, 10)) == (255, 255, 255)
    assert output.getpixel((30, 10)) == (255, 255, 255)


def glyph_line():
    """'birth:' at x 171-234, a thin '1' at 246-254, then '992-04-18' at 261-392 (the sample's real ink layout)."""
    image = Image.new("L", (420, 40), 255)
    draw = ImageDraw.Draw(image)
    for x0, x1 in [(171, 185), (189, 191), (196, 211), (215, 227), (231, 234), (246, 254), (261, 274), (278, 291), (296, 392)]:
        draw.rectangle((x0, 10, x1, 29), fill=0)
    return image


def test_snap_to_ink_recovers_glyph_left_out_of_word_box():
    from pii_redactor.core import snap_to_ink
    label = Word("birth:", Box(x=171, y=10, width=64, height=20), (1, 1, 1))
    date = Word("1992-04-18", Box(x=261, y=10, width=132, height=20), (1, 1, 1))  # Tesseract's box skips the "1"
    snapped = snap_to_ink([label, date], glyph_line())
    assert snapped[1].box.x == 246
    assert snapped[1].box.x + snapped[1].box.width == 393
    assert snapped[0].box == label.box  # the neighbouring label is never swallowed
    redacted = redact(glyph_line().convert("RGB"), [snapped[1].box], padding=0)
    assert redacted.crop((246, 10, 255, 30)).getextrema() == ((0, 0), (0, 0), (0, 0))


def test_snap_to_ink_leaves_exact_boxes_and_far_ink_alone():
    from pii_redactor.core import snap_to_ink
    image = Image.new("L", (300, 40), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 10, 120, 29), fill=0)
    draw.rectangle((200, 10, 210, 29), fill=0)  # separate ink well beyond a word gap
    word = Word("value", Box(x=50, y=10, width=71, height=20), (1, 1, 1))
    assert snap_to_ink([word], image)[0].box == word.box


def test_real_tesseract_finds_fixture_text():
    words = ocr(sample_image())
    text = " ".join(w.text for w in words)
    assert "Maya Tesfaye" in text
    assert "maya.tesfaye@example.com" in text
    assert "AB1234567" in text
