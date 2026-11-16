"""Image decoding, OCR geometry and safe export across varied real encodings."""
import io
import math
import random
import re

import pytest
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from pydantic import ValidationError

from pii_redactor.core import Box, Entity, RedactionError, Word, document_text, entity_regions, load_image, ocr, png_bytes, redact, _restore_words


FORMATS = [("PNG", mode) for mode in ["RGB", "RGBA", "L", "LA", "P", "1", "I;16"]] + [("JPEG", mode) for mode in ["RGB", "L", "CMYK"]] + [("TIFF", mode) for mode in ["RGB", "RGBA", "L", "I;16", "F", "CMYK"]] + [(fmt, "RGB") for fmt in ["BMP", "GIF", "WEBP", "AVIF", "HEIF"]] + [(fmt, "RGBA") for fmt in ["WEBP", "AVIF", "HEIF"]]


@pytest.mark.parametrize("fmt,mode", FORMATS)
def test_decode_encode_matrix(fmt, mode):
    image = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 30, 40), fill="black")
    image = image.convert(mode)
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    decoded = load_image(stream.getvalue())
    assert decoded.mode == "RGB" and decoded.size == (80, 60)
    assert decoded.getpixel((20, 20))[0] < 30
    assert decoded.getpixel((50, 50))[0] > 220
    output = Image.open(io.BytesIO(png_bytes(redact(decoded, [Box(x=10,y=10,width=21,height=31)], 0))))
    assert output.info == {}
    assert output.crop((10,10,31,41)).getextrema() == ((0,0),(0,0),(0,0))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_orientations(orientation):
    image = Image.new("RGB", (60, 30), "white")
    ImageDraw.Draw(image).rectangle((0,0,14,14), fill="black")
    exif = image.getexif(); exif[274] = orientation
    stream = io.BytesIO(); image.save(stream, format="PNG", exif=exif)
    expected = ImageOps.exif_transpose(Image.open(io.BytesIO(stream.getvalue())))
    actual = load_image(stream.getvalue())
    assert actual.size == expected.size
    assert actual.tobytes() == expected.convert("RGB").tobytes()
    assert not actual.info


@pytest.mark.parametrize("fmt", ["GIF", "PNG", "WEBP", "TIFF", "AVIF", "HEIF"])
def test_multiframe_images_are_never_silently_reduced_to_one_frame(fmt):
    first, second = Image.new("RGB", (32,32), "white"), Image.new("RGB",(32,32),"black")
    stream = io.BytesIO(); first.save(stream,format=fmt,save_all=True,append_images=[second],duration=100)
    with pytest.raises(RedactionError, match="multi-page"):
        load_image(stream.getvalue())


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF89a", b"%PDF-1.4", bytes(range(256))])
def test_invalid_files_fail_cleanly(data):
    with pytest.raises(RedactionError):
        load_image(data)


def test_truncated_valid_image_fails_cleanly():
    data = png_bytes(Image.effect_noise((300,300),50).convert("RGB"))
    with pytest.raises(RedactionError):
        load_image(data[:len(data)//2])


def test_limits_checked_before_loading_pixels(monkeypatch):
    image = Image.new("RGB", (101,100), "white")
    data=png_bytes(image)
    monkeypatch.setattr("pii_redactor.core.MAX_PIXELS", 10_000)
    with pytest.raises(RedactionError,match="million pixels"):
        load_image(data)
    monkeypatch.setattr("pii_redactor.core.MAX_PIXELS", 24_000_000)
    monkeypatch.setattr("pii_redactor.core.MAX_SIDE", 100)
    with pytest.raises(RedactionError,match="30,000"):
        load_image(data)
    monkeypatch.setattr("pii_redactor.core.MAX_BYTES",len(data)-1)
    with pytest.raises(RedactionError,match="20 MB"):
        load_image(data)


def test_high_bit_depth_contrast_and_transparent_hidden_pixels():
    high = Image.new("I",(20,20),50_000)
    ImageDraw.Draw(high).rectangle((5,5,10,10),fill=30_000)
    data=io.BytesIO();high.save(data,format="TIFF")
    loaded=load_image(data.getvalue())
    assert loaded.getpixel((0,0)) == (255,255,255)
    assert loaded.getpixel((6,6)) == (0,0,0)
    transparent=Image.new("RGBA",(20,20),(255,0,0,0))
    data=io.BytesIO();transparent.save(data,format="PNG")
    assert load_image(data.getvalue()).getextrema() == ((255,255),)*3


@pytest.mark.parametrize("angle", [0,90,180,270,-4,4])
def test_rotation_coordinates_remain_in_original_bounds(angle):
    original=(900,340);prepared=(1800,680)
    rotated=Image.new("L",prepared).rotate(angle,expand=True)
    words=[Word("test",Box(x=rotated.width/3,y=rotated.height/3,width=20,height=15),(1,1,1))]
    restored=_restore_words(words,original,prepared,rotated.size,angle)
    assert restored
    b=restored[0].box
    assert 0 <= b.x < original[0] and b.x+b.width <= original[0]
    assert 0 <= b.y < original[1] and b.y+b.height <= original[1]


@pytest.mark.parametrize("bad", [math.inf,-math.inf,math.nan,-1])
def test_invalid_coordinates_rejected(bad):
    with pytest.raises(ValidationError):
        Box(x=bad,y=0,width=10,height=10)


def test_random_redactions_cover_every_requested_pixel():
    rng=random.Random(1947)
    for _ in range(100):
        width,height=rng.randint(1,500),rng.randint(1,500)
        x,y=rng.randrange(width),rng.randrange(height)
        w,h=rng.randint(1,width-x),rng.randint(1,height-y)
        result=redact(Image.new("RGB",(width,height),"white"),[Box(x=x,y=y,width=w,height=h)],rng.randint(0,20))
        assert result.crop((x,y,x+w,y+h)).getextrema() == ((0,0),)*3


def text_fixture(size=30, foreground="black", background="white"):
    image=Image.new("RGB",(900,340),background)
    draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=size)
    for i,line in enumerate(["Name: Maya Tesfaye","Email: maya@example.com","Phone: +1 (202) 555-0147","Passport: AB1234567"]):
        draw.text((30,30+i*65),line,font=font,fill=foreground)
    return image


@pytest.mark.parametrize("variant", ["standard","dark","contrast","gray","90","180","270","tilt4","tilt-4","blur","jpeg","small","tiny"])
def test_real_ocr_variations_and_pixel_alignment(variant):
    image=text_fixture(12 if variant=="small" else 9 if variant=="tiny" else 30,
        "white" if variant=="dark" else "#c5c5c5" if variant=="contrast" else "black",
        "black" if variant=="dark" else "#dddddd" if variant=="contrast" else "white")
    if variant in {"90","180","270"}: image=image.rotate(int(variant),expand=True)
    if variant.startswith("tilt"): image=image.rotate(int(variant[4:]),expand=True,fillcolor="white")
    if variant=="gray": image=image.convert("L").convert("RGB")
    if variant=="blur": image=image.filter(ImageFilter.GaussianBlur(.7))
    if variant=="jpeg":
        data=io.BytesIO();image.save(data,format="JPEG",quality=30);image=Image.open(io.BytesIO(data.getvalue())).convert("RGB")
    words=ocr(image)
    assert len(words)>=8
    text=document_text(words).casefold()
    assert "maya" in text
    assert "ab1234567" in re.sub(r"\s", "", text)
    if variant not in {"small","tiny"}:
        assert "maya@example.com" in text
        assert "555-0147" in text
    # Geometry matters more than exact OCR spelling. Cover all recognized words
    # and verify that the rendered text's dark pixels are actually removed.
    coverage=redact(Image.new("RGB",image.size,"white"),[w.box for w in words],padding=5).convert("L")
    if variant not in {"dark","contrast"}:
        uncovered=sum(p<80 and mask!=0 for p,mask in zip(image.convert("L").get_flattened_data(),coverage.get_flattened_data()))
        assert uncovered == 0


def test_grounded_fallback_covers_full_line_and_not_unrelated_lines():
    words=[Word("Name:",Box(x=0,y=0,width=30,height=15),(1,1,1)),Word("Maya",Box(x=40,y=0,width=30,height=15),(1,1,1)),Word("Lunch",Box(x=0,y=40,width=40,height=15),(1,1,2))]
    regions,missing=entity_regions(words,[Entity(text="Maya corrected",category="name",line_ids=[1])])
    assert missing==0 and len(regions)==1
    assert regions[0].source=="llama whole line"
    assert regions[0].boxes[0].y==0 and regions[0].boxes[0].height==15
    assert entity_regions(words,[Entity(text="unknown",category="name",line_ids=[99])])[1]==1


def test_unicode_punctuation_and_spacing_match_real_text():
    words=[Word("Fei—Fei",Box(x=0,y=0,width=50,height=15),(1,1,1)),Word("Li",Box(x=60,y=0,width=15,height=15),(1,1,1))]
    regions,missing=entity_regions(words,[Entity(text="Fei-Fei Li",category="name")])
    assert len(regions)==1 and missing==0
    assert document_text(words)=="Fei—Fei Li"
