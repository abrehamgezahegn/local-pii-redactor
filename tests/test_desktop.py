import io
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QSettings, QUrl, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from pii_redactor.core import RedactionError, png_bytes
from pii_redactor.desktop import RedactorWindow, process_image, scan_text
from pii_redactor.text import Span


def fake_analyze(image, *args):
    """Detect one region covering the top-left quarter of any image."""
    return {"complete": True, "warnings": [], "regions": [{"id": "r1", "category": "name", "text": "Maya Tesfaye", "source": "llama",
            "boxes": [{"x": 0, "y": 0, "width": max(1, image.width // 2), "height": max(1, image.height // 2)}]}]}


def fake_analyze_text(text, model):
    spans = [Span(m.start(), m.end(), "name", "llama") for m in re.finditer("Maya Tesfaye", text)]
    return {"text": text, "complete": True, "warnings": [], "spans": spans}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(app, settings, monkeypatch):
    monkeypatch.setattr("pii_redactor.desktop.analyze", fake_analyze)
    monkeypatch.setattr("pii_redactor.desktop.analyze_text", fake_analyze_text)
    widget = RedactorWindow(settings)
    widget.watch_clipboard = False
    widget.launch = lambda work: work()  # run scans synchronously
    yield widget
    widget.close()
    app.clipboard().clear()


def white_png(tmp_path, name="test.png", size=(100, 100)):
    path = tmp_path / name
    path.write_bytes(png_bytes(Image.new("RGB", size, "white")))
    return path


def visible(widget, window):
    return widget.isVisibleTo(window)


# ---------- scan helpers ----------

def test_incomplete_detection_never_produces_export(monkeypatch):
    monkeypatch.setattr("pii_redactor.desktop.analyze", lambda *args: {"complete": False, "warnings": ["Model unavailable"], "regions": []})
    with pytest.raises(RedactionError, match="Nothing was saved"):
        process_image(Image.new("RGB", (100, 100)))


def test_empty_detection_does_not_produce_unchanged_image(monkeypatch):
    monkeypatch.setattr("pii_redactor.desktop.analyze", lambda *args: {"complete": True, "warnings": [], "regions": []})
    with pytest.raises(RedactionError, match="No personal information"):
        process_image(Image.new("RGB", (100, 100)))


def test_text_scan_requires_detections(monkeypatch):
    monkeypatch.setattr("pii_redactor.desktop.analyze_text", lambda text, model: {"text": text, "complete": True, "warnings": [], "spans": []})
    with pytest.raises(RedactionError, match="No personal information"):
        scan_text("nothing here")
    monkeypatch.setattr("pii_redactor.desktop.analyze_text", lambda text, model: {"text": text, "complete": False, "warnings": ["Model down."], "spans": []})
    with pytest.raises(RedactionError, match="not changed"):
        scan_text("nothing here")


# ---------- one-step flow ----------

def test_start_screen_has_no_result_actions(window):
    assert window.kind == "empty"
    assert not visible(window.copy_button, window) and not visible(window.save_button, window)
    assert not visible(window.new_button, window) and not visible(window.retry_button, window)
    assert "Review results before sharing" in window.status.text()


def test_loading_an_image_processes_and_copies_it_without_more_clicks(window, app, tmp_path):
    window.load_file(str(white_png(tmp_path)))
    assert window.kind == "image"
    assert window.result_data is not None
    assert visible(window.copy_button, window) and visible(window.save_button, window) and visible(window.new_button, window)
    assert app.clipboard().image().pixelColor(10, 10).name() == "#000000"
    assert app.clipboard().image().pixelColor(90, 90).name() == "#ffffff"
    assert "Copied, ready to paste" in window.status.text()


def test_auto_copy_and_mode_are_remembered(window, app, tmp_path, settings):
    window.auto_copy_box.setChecked(False)
    window.set_mode("anonymize")
    app.clipboard().setText("keep me")
    window.load_file(str(white_png(tmp_path)))
    assert app.clipboard().text() == "keep me"
    assert "Copied" not in window.status.text()
    window.copy_result()
    assert "Copied the anonymized image" in window.status.text()
    reopened = RedactorWindow(settings)
    assert reopened.mode == "anonymize" and reopened.mode_buttons["anonymize"].isChecked()
    assert not reopened.auto_copy and not reopened.auto_copy_box.isChecked()
    reopened.close()


def test_mode_switch_rerenders_without_rescanning_and_recopies(window, app, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("pii_redactor.desktop.analyze", lambda image, *a: calls.append(1) or fake_analyze(image))
    window.load_file(str(white_png(tmp_path, size=(200, 100))))
    redacted = window.result_data
    window.set_mode("anonymize")
    assert calls == [1]
    assert window.result_data != redacted
    assert window.filename.text().startswith("Anonymized preview")
    assert "fictional" in window.status.text()
    anonymized = Image.open(io.BytesIO(window.result_data))
    assert any(min(color) > 200 for _, color in anonymized.crop((0, 0, 100, 50)).getcolors(maxcolors=100_000))
    assert app.clipboard().image().pixelColor(90, 45) != app.clipboard().image().pixelColor(150, 80)
    window.set_mode("redact")
    assert window.result_data == redacted


def test_new_input_while_scanning_discards_the_stale_result(window, tmp_path):
    pending = []
    window.launch = pending.append
    window.load_file(str(white_png(tmp_path, "first.png", (100, 100))))
    window.load_file(str(white_png(tmp_path, "second.png", (300, 60))))
    assert window.busy and visible(window.progress, window)
    pending[1]()
    pending[0]()  # the first scan finishes last and must be ignored
    assert Image.open(io.BytesIO(window.result_data)).size == (300, 60)
    assert not window.busy and not visible(window.progress, window)


def test_scan_failure_offers_retry(window, tmp_path, monkeypatch):
    def broken(*args):
        raise RedactionError("Ollama could not complete the request.")
    monkeypatch.setattr("pii_redactor.desktop.analyze", broken)
    window.load_file(str(white_png(tmp_path)))
    assert "Ollama" in window.status.text() and window.status.property("tone") == "error"
    assert visible(window.retry_button, window)
    monkeypatch.setattr("pii_redactor.desktop.analyze", fake_analyze)
    window.retry_button.click()
    assert window.result_data is not None
    assert not visible(window.retry_button, window)


def test_new_returns_to_start_screen(window, tmp_path):
    window.load_file(str(white_png(tmp_path)))
    window.new()
    assert window.kind == "empty" and window.source is None and window.result_data is None
    assert not visible(window.copy_button, window)


# ---------- clipboard ----------

def test_paste_image_from_clipboard(window, app):
    image = QImage(120, 80, QImage.Format.Format_RGB32)
    image.fill("white")
    app.clipboard().setImage(image)
    window.paste()
    assert window.source.size == (120, 80)
    assert window.source_path is None
    assert window.result_data is not None
    assert window.filename.text().startswith("Redacted preview")


def test_paste_text_goes_straight_to_text_mode(window, app):
    app.clipboard().setText("Contact Maya Tesfaye today")
    window.paste()
    assert window.kind == "text"
    assert window.text_input.toPlainText() == "Contact Maya Tesfaye today"
    assert window.text_output.toPlainText() == "Contact [NAME] today"
    assert app.clipboard().text() == "Contact [NAME] today"


def test_finder_file_takes_priority_over_icon_thumbnail(window, app, tmp_path):
    path = white_png(tmp_path, "fullsize.png", (500, 300))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    thumbnail = QImage(16, 16, QImage.Format.Format_RGB32)
    thumbnail.fill("black")
    mime.setImageData(thumbnail)
    app.clipboard().setMimeData(mime)
    window.paste()
    assert window.source.size == (500, 300)
    assert window.source_path == path


def test_new_clipboard_image_is_picked_up_once_and_own_output_is_ignored(window, app):
    image = QImage(90, 60, QImage.Format.Format_RGB32)
    image.fill("white")
    app.clipboard().setImage(image)
    window.import_clipboard_if_new()
    first = window.source
    assert first is not None and window.result_data is not None
    assert window.filename.text().startswith("Redacted preview")
    # The clipboard now holds our own result; coming back to the app must not re-import it.
    window.import_clipboard_if_new()
    assert window.source is first
    other = QImage(40, 40, QImage.Format.Format_RGB32)
    other.fill("gray")
    app.clipboard().setImage(other)
    window.import_clipboard_if_new()
    assert window.source.size == (40, 40)


def test_clipboard_pickup_never_interrupts_typing(window, app):
    window.show_text_editor()
    window.text_input.setPlainText("Draft about Maya Tesfaye")
    image = QImage(50, 50, QImage.Format.Format_RGB32)
    image.fill("white")
    app.clipboard().setImage(image)
    window.import_clipboard_if_new()
    assert window.kind == "text" and window.source is None


# ---------- text ----------

def test_typing_is_processed_after_a_pause(window):
    window.show_text_editor()
    window.text_input.setPlainText("Contact Maya Tesfaye today")
    assert window.typing_timer.isActive() and window.result_text is None
    window.typing_timer.timeout.emit()
    assert window.result_text == "Contact [NAME] today"
    window.text_input.setPlainText("Something else")
    assert window.result_text is None and not visible(window.copy_button, window)


def test_typing_on_start_screen_opens_text_box(window):
    QTest.keyClick(window, Qt.Key.Key_H, Qt.KeyboardModifier.NoModifier)
    assert window.kind == "text"
    assert window.text_input.toPlainText() == "h"


def test_text_file_opens_in_text_mode(window, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Call Maya Tesfaye")
    window.load_file(str(path))
    assert window.kind == "text" and window.result_text == "Call [NAME]"


def test_text_save_and_copy(window, app, tmp_path, monkeypatch):
    window.load_text("Contact Maya Tesfaye today")
    window.set_mode("anonymize")
    assert "Maya" not in window.result_text
    assert app.clipboard().text() == window.result_text
    output = tmp_path / "out.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "Text"))
    window.save_current()
    assert output.read_text() == window.result_text
    assert "Saved out.txt" in window.status.text()


# ---------- review ----------

def test_hold_to_compare_shows_original(window, tmp_path):
    window.load_file(str(white_png(tmp_path)))
    window.upload.pressed.emit()
    assert window.upload.image.toImage().pixelColor(10, 10).name() == "#ffffff"
    assert window.filename.text().startswith("Original")
    window.upload.released.emit()
    assert window.upload.image.toImage().pixelColor(10, 10).name() == "#000000"


def test_double_click_fullscreen_and_escape_restore(window, app, tmp_path):
    window.load_file(str(white_png(tmp_path)))
    window.show()
    window.upload.double_clicked.emit()
    app.processEvents()
    assert window.isFullScreen()
    window.exit_fullscreen()
    app.processEvents()
    assert not window.isFullScreen()


@pytest.mark.parametrize("size", [(500, 600), (1024, 768), (1920, 1080), (2535, 1381)])
@pytest.mark.parametrize("image_size", [(2206, 1400), (400, 1600), (1800, 200), (1, 1)])
def test_preview_expands_with_window(window, app, size, image_size):
    window.set_source(Image.new("RGB", image_size, "white"), "Preview test")
    window.resize(*size)
    window.show()
    app.processEvents()
    assert window.upload.height() >= window.height() - 220
    pixmap = window.upload.picture.pixmap()
    shown = pixmap.deviceIndependentSize()
    assert shown.width() > 0 and shown.height() > 0
    assert shown.width() <= window.upload.width()
    assert shown.height() <= window.upload.height()
    expected = min((window.upload.width() - 40) / image_size[0], (window.upload.height() - 40) / image_size[1])
    assert abs(shown.height() - round(image_size[1] * expected)) <= 2
    # Sharp on Retina: the pixmap carries one pixel per device pixel, not per point.
    ratio = window.upload.devicePixelRatioF()
    assert pixmap.devicePixelRatio() == ratio
    if image_size[0] >= 1000:
        assert abs(pixmap.width() - shown.width() * ratio) <= 2


# ---------- saving and safety ----------

def test_save_keeps_original_and_writes_flattened_png(window, tmp_path, monkeypatch):
    original = white_png(tmp_path, "original.png")
    original_data = original.read_bytes()
    window.load_file(str(original))
    output = tmp_path / "redacted.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "PNG"))
    data = png_bytes(Image.new("RGB", (100, 100), "black"))
    window.save_result(data, 2)
    assert output.read_bytes() == data
    assert original.read_bytes() == original_data
    assert Image.open(io.BytesIO(output.read_bytes())).info == {}
    assert "2 redactions" in window.status.text()


def test_save_suggests_name_next_to_last_folder(window, tmp_path, monkeypatch):
    window.load_file(str(white_png(tmp_path, "receipt.png")))
    seen = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda parent, title, suggested, filters: seen.append(suggested) or (str(tmp_path / "a.png"), "PNG"))
    window.save_current()
    assert seen[0].endswith("receipt-redacted.png")
    window.save_current()
    assert seen[1] == str(tmp_path / "receipt-redacted.png")


def test_save_refuses_original_or_wrong_extension(window, tmp_path, monkeypatch):
    original = white_png(tmp_path, "original.png")
    data = original.read_bytes()
    window.load_file(str(original))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(original), "PNG"))
    window.save_result(b"different data", 1)
    assert original.read_bytes() == data
    assert "preserve" in window.status.text()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "wrong.jpg"), "PNG"))
    window.save_result(data, 1)
    assert not (tmp_path / "wrong.jpg").exists()
    assert not (tmp_path / "wrong.jpg.png").exists()


def test_cancel_save_writes_nothing(window, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    window.save_result(b"unused", 1)
    assert "Nothing was saved" in window.status.text()
    assert not (tmp_path / "redacted.png").exists()


def test_quality_warning_survives_copy_and_save(window, app, tmp_path, monkeypatch):
    window.set_source(Image.new("RGB", (100, 100), "white"), "Example", scan=False)
    window.show_result(png_bytes(Image.new("RGB", (100, 100), "black")), 1, "Some words were hard to read.")
    assert "hard to read" in window.status.text()
    window.copy_result()
    assert "hard to read" in window.status.text()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "saved.png"), "PNG"))
    window.save_current()
    assert "hard to read" in window.status.text()


def test_failed_import_preserves_current_image(window, tmp_path):
    window.load_file(str(white_png(tmp_path)))
    shown = window.filename.text()
    invalid = tmp_path / "corrupt.png"
    invalid.write_bytes(b"not a real image")
    window.load_file(str(invalid))
    assert window.source.size == (100, 100)
    assert window.filename.text() == shown
    assert "damaged" in window.status.text()
    assert not visible(window.retry_button, window)


def test_status_and_filename_never_interpret_html(window):
    window.set_source(Image.new("RGB", (100, 100), "white"), '<img src="file:///private.png">', scan=False)
    assert window.filename.textFormat() == Qt.TextFormat.PlainText
    assert window.status.textFormat() == Qt.TextFormat.PlainText


def test_closed_window_marks_inflight_worker_as_discarded(window):
    window.close()
    assert window.closed.is_set()
