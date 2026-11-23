from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QEvent, QIODevice, QObject, QSaveFile, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout,
                               QWidget)

from .anonymize import Faker, anonymize_image
from .core import MAX_BYTES, MAX_PIXELS, MAX_SIDE, Region, RedactionError, load_image, png_bytes, redact
from .pipeline import analyze, analyze_text
from .text import MAX_TEXT, clean_text, replace_spans

MODEL = os.environ.get("PII_MODEL", "llama3.2:latest")
LANGUAGE = os.environ.get("PII_LANGUAGE", "eng")
TEXT_FILES = {".txt", ".md", ".csv", ".log", ".json", ".eml"}
TYPING_PAUSE_MS = 1200
PAGES = {"empty": 0, "image": 1, "text": 2}


def scan_image(image: Image.Image) -> dict:
    result = analyze(image, MODEL, LANGUAGE)
    if not result["complete"]:
        raise RedactionError(" ".join(result["warnings"]) + " Nothing was saved.")
    if not any(region["boxes"] for region in result["regions"]):
        raise RedactionError("No personal information was detected. Check the image yourself or try a clearer image.")
    return result


def render_image(image: Image.Image, analysis: dict, mode: str, faker: Faker) -> tuple[bytes, int, str]:
    regions = [Region(**region) for region in analysis["regions"]]
    if mode == "anonymize":
        output = anonymize_image(image, regions, faker, padding=5)
    else:
        output = redact(image, [box for region in regions for box in region.boxes], padding=5)
    return png_bytes(output), len(regions), " ".join(analysis["warnings"])


def process_image(image: Image.Image, mode: str = "redact", faker: Faker | None = None) -> tuple[bytes, int, str]:
    return render_image(image, scan_image(image), mode, faker or Faker())


def scan_text(text: str) -> dict:
    result = analyze_text(text, MODEL)
    if not result["complete"]:
        raise RedactionError(" ".join(result["warnings"]) + " The text was not changed.")
    if not result["spans"]:
        raise RedactionError("No personal information was detected. Check the text yourself before sharing.")
    return result


def render_text(analysis: dict, mode: str, faker: Faker) -> tuple[str, int, str]:
    return replace_spans(analysis["text"], analysis["spans"], mode, faker), len(analysis["spans"]), " ".join(analysis["warnings"])


def image_fingerprint(image: QImage) -> str:
    converted = image.convertToFormat(QImage.Format.Format_RGB32)
    return hashlib.sha256(bytes(converted.constBits())).hexdigest()


class WorkerSignals(QObject):
    scanned = Signal(int, str, object)
    failed = Signal(int, str, str)


class UploadArea(QPushButton):
    """Image preview. Press and hold to see the original; double-click for full screen."""
    double_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("upload")
        self.setAccessibleName("Image preview. Press and hold to compare with the original.")
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.picture = QLabel(self)
        self.picture.setObjectName("picture")
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.picture.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image: QPixmap | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(self.picture)

    def show_image(self, data: bytes):
        self.image = QPixmap.fromImage(QImage.fromData(data, "PNG"))
        self.update_preview()

    def update_preview(self):
        if self.image:
            # Scale to device pixels so the preview stays sharp on Retina displays.
            ratio = self.devicePixelRatioF()
            scaled = self.image.scaled(round((self.width() - 40) * ratio), round((self.height() - 40) * ratio),
                                       Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(ratio)
            self.picture.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()


class TextInput(QPlainTextEdit):
    def canInsertFromMimeData(self, source):
        # Dropped or pasted files go to the window, which loads them, instead of inserting their paths.
        return not source.hasUrls() and super().canInsertFromMimeData(source)


def segmented(parent: QWidget, options: list[tuple[str, str, str]], on_change) -> dict[str, QPushButton]:
    box = QWidget(parent)
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    group = QButtonGroup(box)
    group.setExclusive(True)
    buttons = {}
    for i, (key, text, tip) in enumerate(options):
        button = QPushButton(text)
        button.setObjectName("segment")
        button.setProperty("position", "first" if i == 0 else "last")
        button.setCheckable(True)
        button.setToolTip(tip)
        button.setAccessibleName(tip)
        button.clicked.connect(lambda _checked=False, k=key: on_change(k))
        group.addButton(button)
        layout.addWidget(button)
        buttons[key] = button
    buttons["_box"] = box
    return buttons


class RedactorWindow(QWidget):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle("Local Redact")
        self.setAcceptDrops(True)
        self.settings = settings if settings is not None else QSettings("Local Redact", "Local Redact")
        mode = self.settings.value("mode", "redact")
        self.mode = mode if mode in {"redact", "anonymize"} else "redact"
        self.auto_copy = self.settings.value("auto_copy", True, type=bool)
        self.kind = "empty"
        self.source: Image.Image | None = None
        self.source_png: bytes | None = None
        self.source_path: Path | None = None
        self.image_analysis: dict | None = None
        self.image_faker = Faker()
        self.result_data: bytes | None = None
        self.result_count = 0
        self.result_warning = ""
        self.text_analysis: dict | None = None
        self.text_faker = Faker()
        self.result_text: str | None = None
        self.text_count = 0
        self.text_warning = ""
        self.generation = 0
        self.busy = False
        self.loading_text = False
        self.clipboard_seen: str | None = None
        self.watch_clipboard = True
        self.closed = threading.Event()
        self.signals = WorkerSignals(self)
        self.signals.scanned.connect(self.scanned)
        self.signals.failed.connect(self.failed)
        self.typing_timer = QTimer(self)
        self.typing_timer.setSingleShot(True)
        self.typing_timer.setInterval(TYPING_PAUSE_MS)
        self.typing_timer.timeout.connect(self.start_scan)
        self.build()
        self.restore_geometry()

    # ---------- layout ----------
    def build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.mode_buttons = segmented(self, [
            ("redact", "Redact", "Cover personal information with black markers"),
            ("anonymize", "Anonymize", "Replace personal information with fictional values"),
        ], self.set_mode)
        self.mode_buttons[self.mode].setChecked(True)
        top.addWidget(self.mode_buttons["_box"])
        top.addStretch()
        self.auto_copy_box = QCheckBox("Auto-copy result")
        self.auto_copy_box.setToolTip("Put each finished result on the clipboard, ready to paste")
        self.auto_copy_box.setChecked(self.auto_copy)
        self.auto_copy_box.toggled.connect(self.set_auto_copy)
        top.addWidget(self.auto_copy_box)
        top.addSpacing(6)
        self.new_button = QPushButton("New")
        self.new_button.setToolTip("Start over (⌘N)")
        self.new_button.clicked.connect(self.new)
        top.addWidget(self.new_button)
        layout.addLayout(top)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_empty_page())
        self.upload = UploadArea()
        self.upload.pressed.connect(self.show_original)
        self.upload.released.connect(self.show_processed)
        self.upload.double_clicked.connect(self.toggle_fullscreen)
        self.stack.addWidget(self.upload)
        self.stack.addWidget(self.build_text_page())
        layout.addWidget(self.stack, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.filename = QLabel("")
        self.filename.setObjectName("hint")
        self.filename.setWordWrap(True)
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.filename)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setMinimumHeight(36)
        self.status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom.addWidget(self.status, 1)
        self.retry_button = QPushButton("Try again")
        self.retry_button.clicked.connect(self.start_scan)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setObjectName("primary")
        self.copy_button.setToolTip("Copy the result (⌘C)")
        self.copy_button.clicked.connect(self.copy_result)
        self.save_button = QPushButton("Save…")
        self.save_button.setToolTip("Save the result (⌘S)")
        self.save_button.clicked.connect(self.save_current)
        for button in (self.retry_button, self.save_button, self.copy_button):
            bottom.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        self.retry_button.hide()
        layout.addLayout(bottom)

        self.setStyleSheet("""
            QWidget { background: #ffffff; color: #1c2022; font-size: 14px; }
            QFrame#dropzone { background: #f5f7f7; border: 1px dashed #a8b1b5; border-radius: 12px; }
            QFrame#dropzone[dragging="true"] { background: #e8f1f3; border: 2px solid #1b6c80; }
            QFrame#dropzone QLabel, QFrame#dropzone QPushButton { background: transparent; }
            QLabel#lead { font-size: 18px; font-weight: 600; color: #1c2022; }
            QLabel#sub { color: #596267; font-size: 13px; }
            QLabel#tip { color: #6b7579; font-size: 12px; }
            QPushButton#upload { background: #f5f7f7; border: 1px solid #e1e6e8; border-radius: 12px; }
            QPushButton#upload:focus { border: 2px solid #1b6c80; }
            QLabel#picture { background: transparent; }
            QLabel#hint { color: #6b7579; font-size: 12px; }
            QLabel#status { color: #3d4649; font-size: 13px; }
            QLabel#status[tone="error"] { color: #a1352b; }
            QPlainTextEdit { background: #f5f7f7; border: 1px solid #d4dbde; border-radius: 10px; padding: 8px; font-size: 14px; }
            QPlainTextEdit:focus { border: 2px solid #1b6c80; }
            QPlainTextEdit#output { background: #ffffff; }
            QPushButton { border: 1px solid #b8c1c5; border-radius: 7px; padding: 7px 14px; background: #ffffff; }
            QPushButton:hover { background: #eef2f3; }
            QPushButton:disabled { color: #9aa3a7; border-color: #e1e6e8; }
            QPushButton#primary { background: #20282c; color: white; border: none; font-weight: 600; }
            QPushButton#primary:hover { background: #35434a; }
            QPushButton#segment { border-radius: 0; padding: 6px 14px; }
            QPushButton#segment[position="first"] { border-top-left-radius: 7px; border-bottom-left-radius: 7px; }
            QPushButton#segment[position="last"] { border-top-right-radius: 7px; border-bottom-right-radius: 7px; border-left: none; }
            QPushButton#segment:checked { background: #20282c; border-color: #20282c; color: white; font-weight: 600; }
            QCheckBox { color: #3d4649; font-size: 13px; }
            QProgressBar { background: #e4e8e9; border: none; border-radius: 1px; }
            QProgressBar::chunk { background: #1b6c80; }
        """)
        shortcuts = [
            (QKeySequence.StandardKey.Paste, self.paste), (QKeySequence.StandardKey.Copy, self.copy_result),
            (QKeySequence.StandardKey.Save, self.save_current), (QKeySequence.StandardKey.Open, self.choose_image),
            (QKeySequence.StandardKey.New, self.new), (QKeySequence("Ctrl+Return"), self.start_scan),
            (QKeySequence("Ctrl+Meta+F" if sys.platform == "darwin" else "F11"), self.toggle_fullscreen),
            (QKeySequence("Esc"), self.exit_fullscreen),
        ]
        self.shortcuts = []
        for key, action in shortcuts:
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.show_page("empty")

    def build_empty_page(self) -> QWidget:
        self.dropzone = QFrame()
        self.dropzone.setObjectName("dropzone")
        box = QVBoxLayout(self.dropzone)
        box.setContentsMargins(24, 24, 24, 24)
        box.setSpacing(8)
        box.addStretch()
        for text, name in [("Paste, drop or type", "lead"),
                           ("Images and text are cleaned up as soon as they arrive.", "sub")]:
            label = QLabel(text)
            label.setObjectName(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            box.addWidget(label)
        buttons = QHBoxLayout()
        buttons.addStretch()
        choose = QPushButton("Choose image…")
        choose.clicked.connect(self.choose_image)
        type_text = QPushButton("Type text")
        type_text.clicked.connect(lambda: self.show_text_editor())
        buttons.addWidget(choose)
        buttons.addWidget(type_text)
        buttons.addStretch()
        box.addSpacing(8)
        box.addLayout(buttons)
        box.addSpacing(8)
        tip = QLabel("Tip: copy a screenshot (⌃⇧⌘4), then switch here. It is picked up automatically.")
        tip.setObjectName("tip")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setWordWrap(True)
        box.addWidget(tip)
        box.addStretch()
        return self.dropzone

    def build_text_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.text_input = TextInput()
        self.text_input.setPlaceholderText("Type or paste text. It is processed when you pause.")
        self.text_input.setAccessibleName("Text to process")
        self.text_input.textChanged.connect(self.text_changed)
        layout.addWidget(self.text_input, 1)
        self.text_output_label = QLabel("Result")
        self.text_output_label.setObjectName("hint")
        layout.addWidget(self.text_output_label)
        self.text_output = QPlainTextEdit()
        self.text_output.setObjectName("output")
        self.text_output.setReadOnly(True)
        self.text_output.setAccessibleName("Processed text")
        self.text_output.setPlaceholderText("The cleaned text appears here.")
        layout.addWidget(self.text_output, 1)
        return page

    def restore_geometry(self):
        geometry = self.settings.value("geometry")
        if geometry is None or not self.restoreGeometry(geometry):
            self.resize(560, 600)
        self.setMinimumSize(460, 440)

    # ---------- state helpers ----------
    def show_page(self, kind: str):
        self.kind = kind
        self.stack.setCurrentIndex(PAGES[kind])
        self.new_button.setVisible(kind != "empty")
        self.refresh()

    def has_result(self) -> bool:
        if self.kind == "image":
            return self.result_data is not None
        if self.kind == "text":
            return self.result_text is not None
        return False

    def refresh(self):
        result = self.has_result()
        self.copy_button.setVisible(result)
        self.save_button.setVisible(result)
        self.progress.setVisible(self.busy)
        if not self.busy and self.kind == "empty" and not self.status.property("tone"):
            self.status.setText("Processed on this computer with Llama. Review results before sharing.")

    def say(self, message: str, error: bool = False, retry: bool = False):
        self.status.setText(" ".join(message.split()))
        self.status.setProperty("tone", "error" if error else None)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.retry_button.setVisible(retry)

    def verb(self) -> str:
        return "anonymized" if self.mode == "anonymize" else "redacted"

    # ---------- input ----------
    def choose_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose an image or text file", "", "Images and text (*.png *.jpg *.jpeg *.heic *.heif *.avif *.webp *.gif *.bmp *.tif *.tiff *.txt *.md)")
        if filename:
            self.load_file(filename)

    def load_file(self, filename: str):
        try:
            path = Path(filename)
            size = path.stat().st_size
            if size > MAX_BYTES:
                raise RedactionError(f"Choose a file smaller than 20 MB. This one is {size / (1024 * 1024):.1f} MB.")
            if path.suffix.lower() in TEXT_FILES:
                self.load_text(path.read_text(encoding="utf-8", errors="replace"))
                return
            image = load_image(path.read_bytes())
            self.set_source(image, path.name, path)
        except (RedactionError, OSError) as error:
            self.say(str(error), error=True)

    def set_source(self, image: Image.Image, name: str, path: Path | None = None, scan: bool = True):
        self.source = image
        self.source_png = png_bytes(image)
        self.source_path = path
        self.image_analysis = None
        self.result_data = None
        self.result_count = 0
        self.result_warning = ""
        self.upload.show_image(self.source_png)
        self.filename.setText(name)
        self.show_page("image")
        if scan:
            self.start_scan()
        else:
            self.say("Ready.")

    def show_text_editor(self):
        self.show_page("text")
        self.update_text_hint()
        self.text_input.setFocus()
        if not self.result_text:
            self.say("Type or paste text. It is processed when you pause.")

    def load_text(self, text: str):
        self.loading_text = True
        self.text_input.setPlainText(text)
        self.loading_text = False
        self.show_text_editor()
        self.start_scan()

    def text_changed(self):
        self.text_analysis = None
        self.result_text = None
        self.text_output.clear()
        self.update_text_hint()
        if self.kind == "text":
            self.refresh()
        if self.loading_text:
            return
        # Any edit makes an in-flight scan stale; a new one starts once typing pauses.
        self.generation += 1
        self.busy = False
        self.refresh()
        if self.text_input.toPlainText().strip():
            self.typing_timer.start()
        else:
            self.typing_timer.stop()

    def update_text_hint(self):
        count = len(clean_text(self.text_input.toPlainText()))
        suffix = " · Too long, split it into smaller parts" if count > MAX_TEXT else " · ⌘↩ to process now"
        self.filename.setText(f"{count:,} / {MAX_TEXT:,} characters{suffix}" if self.kind == "text" else self.filename.text())

    def paste(self):
        mime = QApplication.clipboard().mimeData()
        if mime is None:
            self.say("The clipboard is empty. Copy an image or text first.")
            return
        if mime.hasUrls() and len(mime.urls()) == 1 and mime.urls()[0].isLocalFile():
            # Finder may expose a small file icon as an image alongside the full file.
            self.load_file(mime.urls()[0].toLocalFile())
        elif mime.hasImage():
            self.paste_image()
        elif mime.hasText() and mime.text().strip():
            self.load_text(mime.text())
        else:
            self.say("The clipboard has no image or text. Copy something, then paste again.")

    def paste_image(self):
        mime = QApplication.clipboard().mimeData()
        if mime is not None and mime.hasUrls() and len(mime.urls()) == 1 and mime.urls()[0].isLocalFile():
            self.load_file(mime.urls()[0].toLocalFile())
            return
        if mime is None or not mime.hasImage():
            self.say("The clipboard has no image. Copy an image or image file, then paste again.")
            return
        image = QApplication.clipboard().image()
        if image.isNull():
            self.say("The clipboard image could not be read.", error=True)
            return
        self.import_qimage(image, "Pasted image")

    def import_qimage(self, image: QImage, name: str):
        if image.width() * image.height() > MAX_PIXELS:
            self.say("Paste an image with no more than 24 million pixels.", error=True)
            return
        if max(image.width(), image.height()) > MAX_SIDE:
            self.say("Crop long images into sections of at most 30,000 pixels per side.", error=True)
            return
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        try:
            if not image.save(buffer, "PNG"):
                self.say("The clipboard image could not be decoded. Try choosing the image file.", error=True)
                return
            self.clipboard_seen = image_fingerprint(image)
            self.set_source(load_image(bytes(buffer.data())), name)
        except RedactionError as error:
            self.say(str(error), error=True)
        finally:
            buffer.close()

    def import_clipboard_if_new(self):
        """When the app comes forward, process a new clipboard image without being asked."""
        if self.kind == "text" and self.text_input.toPlainText().strip():
            return
        mime = QApplication.clipboard().mimeData()
        if mime is None or mime.hasUrls() or not mime.hasImage():
            return
        image = QApplication.clipboard().image()
        if image.isNull() or image.width() * image.height() > MAX_PIXELS:
            return
        fingerprint = image_fingerprint(image)
        if fingerprint == self.clipboard_seen:
            return
        self.import_qimage(image, "Image from clipboard")

    def keyPressEvent(self, event):
        # Typing on the start screen goes straight into a text box.
        if self.kind == "empty" and event.text() and event.text().isprintable() and not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self.show_text_editor()
            self.text_input.insertPlainText(event.text())
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        urls = mime.urls()
        if (len(urls) == 1 and urls[0].isLocalFile()) or (not urls and (mime.hasImage() or mime.hasText())):
            event.acceptProposedAction()
            self.set_dragging(True)

    def dragLeaveEvent(self, event):
        self.set_dragging(False)

    def dropEvent(self, event):
        self.set_dragging(False)
        mime = event.mimeData()
        urls = mime.urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.load_file(urls[0].toLocalFile())
        elif mime.hasImage():
            self.import_qimage(QImage(mime.imageData()), "Dropped image")
        elif mime.hasText():
            self.load_text(mime.text())
        else:
            return
        event.acceptProposedAction()

    def set_dragging(self, dragging: bool):
        self.dropzone.setProperty("dragging", dragging)
        self.dropzone.style().unpolish(self.dropzone)
        self.dropzone.style().polish(self.dropzone)

    # ---------- modes and settings ----------
    def set_mode(self, mode: str):
        self.mode_buttons[mode].setChecked(True)
        if mode == self.mode:
            return
        self.mode = mode
        self.settings.setValue("mode", mode)
        # Reuse the last scan: switching modes re-renders instantly without asking the model again.
        if self.image_analysis is not None and self.source is not None:
            self.show_result(*render_image(self.source, self.image_analysis, mode, self.image_faker))
        if self.text_analysis is not None:
            self.show_text_result(*render_text(self.text_analysis, mode, self.text_faker))

    def set_auto_copy(self, enabled: bool):
        self.auto_copy = enabled
        self.settings.setValue("auto_copy", enabled)
        if enabled and self.has_result():
            self.copy_result()

    def new(self):
        self.generation += 1
        self.busy = False
        self.typing_timer.stop()
        self.source = self.source_png = self.source_path = self.image_analysis = self.result_data = None
        self.loading_text = True
        self.text_input.clear()
        self.loading_text = False
        self.text_analysis = self.result_text = None
        self.exit_fullscreen()
        self.filename.setText("")
        self.status.setProperty("tone", None)
        self.retry_button.hide()
        self.show_page("empty")

    # ---------- scanning ----------
    def launch(self, work):
        # Closing the app exits without waiting for an in-flight model request.
        threading.Thread(target=work, daemon=True, name="local-redaction").start()

    def start_scan(self):
        self.typing_timer.stop()
        kind = self.kind
        if kind == "image" and self.source is not None:
            self.image_analysis = None
            self.result_data = None
            self.upload.show_image(self.source_png)
            payload, scan = self.source.copy(), scan_image
            self.say("Finding personal information on this computer…")
        elif kind == "text" and self.text_input.toPlainText().strip():
            if len(self.text_input.toPlainText()) > MAX_TEXT:
                self.say(f"Text can be at most {MAX_TEXT:,} characters. Split it into smaller parts.", error=True)
                return
            self.text_analysis = None
            self.result_text = None
            self.text_output.clear()
            payload, scan = self.text_input.toPlainText(), scan_text
            self.say("Finding personal information on this computer…")
        else:
            return
        self.generation += 1
        generation = self.generation
        self.busy = True
        self.refresh()
        signals, closed = self.signals, self.closed

        def work():
            try:
                analysis = scan(payload)
                if not closed.is_set():
                    signals.scanned.emit(generation, kind, analysis)
            except (RedactionError, OSError) as error:
                if not closed.is_set():
                    signals.failed.emit(generation, kind, str(error))
            except Exception:
                if not closed.is_set():
                    signals.failed.emit(generation, kind, f"The {kind} could not be processed. Check that Ollama is running.")

        self.launch(work)

    def scanned(self, generation: int, kind: str, analysis: dict):
        if generation != self.generation or kind != self.kind:
            return  # the input changed while this scan ran
        self.busy = False
        try:
            if kind == "image":
                self.image_analysis = analysis
                self.image_faker = Faker()
                self.show_result(*render_image(self.source, analysis, self.mode, self.image_faker))
            else:
                self.text_analysis = analysis
                self.text_faker = Faker()
                self.show_text_result(*render_text(analysis, self.mode, self.text_faker))
        except RedactionError as error:
            self.failed(generation, kind, str(error))

    def failed(self, generation: int | str, kind: str | None = None, message: str | None = None):
        if message is None:  # failed("message") form
            generation, kind, message = self.generation, self.kind, str(generation)
        if generation != self.generation:
            return
        self.busy = False
        self.refresh()
        self.say(message, error=True, retry=True)

    def closeEvent(self, event):
        self.closed.set()
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    # ---------- results ----------
    def summary(self, count: int, unit: str, warning: str) -> str:
        done = f"Replaced {count} {unit} with fictional ones." if self.mode == "anonymize" else f"{'Covered' if unit == 'regions' else 'Labelled'} {count} {unit}."
        copied = " Copied, ready to paste." if self.auto_copy else ""
        return f"{done}{copied} {warning} Review before sharing."

    def show_result(self, data: bytes, count: int, warning: str = ""):
        self.result_data = data
        self.result_count = count
        self.result_warning = warning
        self.upload.show_image(data)
        self.filename.setText(f"{self.verb().capitalize()} preview · Hold to compare with the original · Double-click for full screen")
        self.busy = False
        self.refresh()
        if self.auto_copy:
            self.copy_to_clipboard()
        self.say(self.summary(count, "regions", warning))

    def show_text_result(self, text: str, count: int, warning: str = ""):
        self.result_text = text
        self.text_count = count
        self.text_warning = warning
        self.text_output_label.setText(f"{self.verb().capitalize()} result")
        self.text_output.setPlainText(text)
        self.busy = False
        self.refresh()
        if self.auto_copy:
            self.copy_to_clipboard()
        self.say(self.summary(count, "values", warning))

    def show_original(self):
        if self.result_data is not None and self.source_png is not None:
            self.upload.show_image(self.source_png)
            self.filename.setText("Original · Release to see the result")

    def show_processed(self):
        if self.result_data is not None:
            self.upload.show_image(self.result_data)
            self.filename.setText(f"{self.verb().capitalize()} preview · Hold to compare with the original · Double-click for full screen")

    def copy_to_clipboard(self) -> bool:
        if self.kind == "image" and self.result_data is not None:
            image = QImage.fromData(self.result_data, "PNG")
            QApplication.clipboard().setImage(image)
            self.clipboard_seen = image_fingerprint(image)  # never re-import our own output
            return True
        if self.kind == "text" and self.result_text is not None:
            QApplication.clipboard().setText(self.result_text)
            return True
        return False

    def copy_result(self):
        if self.copy_to_clipboard():
            warning = self.result_warning if self.kind == "image" else self.text_warning
            self.say(f"Copied the {self.verb()} {self.kind}, ready to paste. {warning}")

    def toggle_fullscreen(self):
        if self.kind != "image":
            return
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.watch_clipboard and self.isActiveWindow() and not self.closed.is_set():
            # Let the window finish activating before reading the clipboard.
            QTimer.singleShot(0, self.import_clipboard_if_new)

    # ---------- saving ----------
    def save_current(self):
        if self.kind == "image" and self.result_data is not None:
            self.save_result(self.result_data, self.result_count)
        elif self.kind == "text" and self.result_text is not None:
            self.save_text(self.result_text, self.text_count)

    def _write(self, path: Path, data: bytes):
        # The native save dialog handles replacement confirmation; use an atomic save.
        output = QSaveFile(str(path))
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            raise RedactionError("The file could not be saved in this folder. Choose Save to try another location.")
        if output.write(data) != len(data) or not output.commit():
            output.cancelWriting()
            raise RedactionError("Saving failed. Choose Save to try another location.")

    def default_name(self, suffix: str) -> str:
        stem = self.source_path.stem if self.source_path else ("image" if suffix == ".png" else "text")
        return str(Path(self.settings.value("save_dir", str(Path.home() / "Downloads"))) / f"{stem}-{self.verb()}{suffix}")

    def save_result(self, data: bytes, count: int):
        filename, _ = QFileDialog.getSaveFileName(self, "Save image", self.default_name(".png"), "PNG image (*.png)")
        if not filename:
            self.say("Nothing was saved. The result is still here.")
            return
        path = Path(filename)
        try:
            if path.suffix.lower() != ".png":
                raise RedactionError("Use a filename ending in .png.")
            if self.source_path and path.resolve() == self.source_path.resolve():
                raise RedactionError("Choose a different filename to preserve your original image.")
            self._write(path, data)
            self.settings.setValue("save_dir", str(path.parent))
            changes = "replacements" if self.mode == "anonymize" else "redactions"
            self.say(f"Saved {path.name} with {count} {changes}. {self.result_warning}")
        except (RedactionError, OSError) as error:
            self.say(str(error), error=True)

    def save_text(self, text: str, count: int):
        filename, _ = QFileDialog.getSaveFileName(self, "Save text", self.default_name(".txt"), "Text file (*.txt)")
        if not filename:
            self.say("Nothing was saved. The result is still here.")
            return
        path = Path(filename)
        try:
            if path.suffix.lower() != ".txt":
                raise RedactionError("Use a filename ending in .txt.")
            self._write(path, text.encode("utf-8"))
            self.settings.setValue("save_dir", str(path.parent))
            self.say(f"Saved {path.name} with {count} replaced values. {self.text_warning}")
        except (RedactionError, OSError) as error:
            self.say(str(error), error=True)


def main():
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Local Redact")
    app.setOrganizationName("Local Redact")
    window = RedactorWindow()
    window.show()
    if len(sys.argv) == 2 and Path(sys.argv[1]).is_file():
        window.load_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
