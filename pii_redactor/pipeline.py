from PIL import Image

from .core import RedactionError, deduplicate, document_text, entity_regions, ocr, rule_regions
from .model import Ollama
from .text import MAX_TEXT, clean_text, entity_spans, merge_spans, rule_spans


def analyze_text(text: str, model: str, client: Ollama | None = None) -> dict:
    text = clean_text(text)
    if not text.strip():
        raise RedactionError("Type or paste some text first.")
    if len(text) > MAX_TEXT:
        raise RedactionError(f"Text can be at most {MAX_TEXT:,} characters. Split it into smaller parts.")
    spans = rule_spans(text)
    notices = []
    complete = True
    try:
        found, unmatched, whole_lines = entity_spans(text, (client or Ollama()).detect(text, model))
        spans.extend(found)
        if unmatched:
            complete = False
            notices.append(f"{unmatched} detection(s) could not be found in the text.")
        if whole_lines:
            notices.append(f"{whole_lines} uncertain value(s) were replaced along with their full lines.")
    except RedactionError as error:
        complete = False
        notices.append(str(error) + " Only pattern checks completed.")
    return {"text": text, "spans": merge_spans(spans), "model": model, "complete": complete, "warnings": notices}


def analyze(image: Image.Image, model: str, language: str = "eng", client: Ollama | None = None) -> dict:
    words = ocr(image, language)
    notices = []
    complete = True
    regions = rule_regions(words)
    if not words:
        complete = False
        notices.append("No text was recognized. Try a clearer image.")
    else:
        try:
            entities = (client or Ollama()).detect(document_text(words), model)
            model_regions, unmatched = entity_regions(words, entities)
            regions.extend(model_regions)
            if unmatched:
                complete = False
                notices.append(f"{unmatched} detection(s) could not be linked to image text. Try a clearer image or a smaller crop.")
            whole_lines = sum(r.source == "llama whole line" for r in model_regions)
            if whole_lines:
                notices.append(f"{whole_lines} uncertain value(s) were covered across their full text lines.")
        except RedactionError as error:
            complete = False
            notices.append(str(error) + " Only pattern checks completed.")
    low_confidence = sum(w.confidence < 60 for w in words)
    if low_confidence:
        notices.append(f"{low_confidence} word(s) were hard to read. Check those areas closely.")
    return {
        "regions": [r.model_dump() for r in deduplicate(regions)],
        "word_boxes": [w.box.model_dump() for w in words],
        "word_count": len(words), "low_confidence_words": low_confidence,
        "model": model, "complete": complete, "warnings": notices,
    }
