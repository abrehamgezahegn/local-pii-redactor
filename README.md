# Local Redact

A small desktop app that removes personal information from images or text. It uses your installed `llama3.2:latest` model in Ollama, Tesseract for text locations, and Pillow to draw over the original pixels.

## Use the Mac app

Open **Local Redact.app** (built into `dist/` by PyInstaller, see below).

Give it an image or text and it does the rest. There is no Submit button.

- **Paste (⌘V), drop, or choose** an image or a `.txt` file. Processing starts immediately.
- **Start typing** anywhere on the start screen to get a text box. Text is processed when you pause, or right away with ⌘↩.
- **Screenshots:** copy one to the clipboard (⌃⇧⌘4) and switch to the app. A new clipboard image is picked up and processed automatically. The app's own copied results are never picked up again, and it never interrupts text you are typing.
- With **Auto-copy result** on (the default), every finished result goes on the clipboard, ready to paste where you were going. Otherwise use **Copy** (⌘C).
- **Save…** (⌘S) suggests `<original name>-redacted.png` (or `.txt`) in the folder you last saved to.
- **New** (⌘N) clears everything. Pasting something new also replaces the current item, even mid-scan.

Pick **Redact** or **Anonymize** at the top. The app remembers your choice, and switching after a scan re-renders instantly without scanning again.

- **Redact** covers each value with a black marker. In text, values become labels such as `[NAME]` or `[EMAIL]`.
- **Anonymize** replaces each value with a fictional one of the same kind. Images get a black marker with the fake value written on it in white. Text gets the fake value in place. The same real value always gets the same fake within one scan, so "Maya Tesfaye" and a later "Maya" stay consistent.

Review images by **pressing and holding** the preview to see the original, and **double-click** for full screen (Escape returns). If a scan fails, for example because Ollama is not running, **Try again** reruns it.

Fake values are clearly fictional (emails use `example.com`), but they look real. Tell readers the data was anonymized when that matters. No file is written until you choose to save. Your original file stays unchanged. Check the result before sharing, since automatic detection can miss personal information. Faces, signatures, QR codes and barcodes are not automatically detected. The simplified app does not include manual editing.

Ollama must be running. This computer already has the model and Tesseract installed. The app includes its Python runtime and desktop libraries. It does not start a web server or open a browser.

## Supported images

PNG, JPEG, WebP, BMP and single-page TIFF, up to 20 MB and 24 million pixels. Scanning defaults to English. Animated and multi-page images are rejected. PDFs and batch processing are outside this version's scope.

The app detects names, emails, phones, addresses, birth dates, IDs and financial details. Pattern checks supplement Llama. If text recognition or the model fails, the app does not save a partial result. If no personal information is found, it reports that and saves nothing.

## Local processing

Images stay on this computer. The app sends only extracted text to a loopback Ollama endpoint. It rejects remote-model metadata and cloud model names, and disables HTTP proxies and redirects. There are no analytics or external UI assets.

Tesseract receives image pixels through stdin and returns text through stdout. The app keeps no upload history or OCR files. It writes only the PNG you choose to save. Export produces a new RGB image without copied EXIF, GPS, comments or hidden source-image layers.

This does not guarantee secure memory erasure. OS swap and your local Ollama configuration remain outside the app's control. Use a trusted local Ollama installation.

## Run from source

Python 3.11 or later, [uv](https://docs.astral.sh/uv/getting-started/installation/), [Ollama](https://ollama.com/download), and [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) are required. Run these separately in this folder:

```sh
uv sync --locked
uv run local-redact
```

`Start Local Redact.command` also launches the desktop app from source. For a different installed local model or OCR language:

```sh
env PII_MODEL=llama3.2:latest PII_LANGUAGE=eng uv run local-redact
```

Use `PII_OLLAMA_URL` only for another local loopback port. No cloud endpoints are accepted.

## Tests and packaging

```sh
uv run pytest
uv run pyinstaller --noconfirm --windowed --name "Local Redact" --osx-bundle-identifier local.redact.app desktop_entry.py
```

Tests cover exact redaction pixels, metadata removal, orientation, multiline and repeated values, model failures, local-only model checks, the uploader, clipboard import/export, full-screen state and save behavior. A fictional image is used for the live Llama check. No user document is needed for testing.

The Mac bundle is built for Apple silicon and signed locally, not notarized for public distribution. It uses the existing Tesseract installation at `/opt/homebrew/bin/tesseract` or `/usr/local/bin/tesseract`. Run it on this computer; other computers need Ollama and Tesseract installed separately.

References: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [Tesseract command-line usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html), [Pillow orientation handling](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html), and [Qt native file dialogs](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QFileDialog.html).
