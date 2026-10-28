from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import Box, RedactionError, load_image, png_bytes, redact
from .pipeline import analyze


def main():
    parser = argparse.ArgumentParser(description="Redact image PII locally with Ollama, Tesseract and Pillow.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("desktop", help="Open the desktop app")
    run = commands.add_parser("redact", help="Automatically redact a single image; review the result before sharing")
    run.add_argument("input", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--model", default="llama3.2:latest")
    run.add_argument("--language", default="eng")
    run.add_argument("--padding", type=int, default=4)
    run.add_argument("--allow-partial", action="store_true", help="Allow export when Llama fails; only available matches will be redacted")
    args = parser.parse_args()
    if args.command == "desktop":
        from .desktop import main as desktop_main
        desktop_main()
        return
    try:
        if args.input.resolve() == args.output.resolve():
            raise RedactionError("Choose a separate output path to preserve the original image.")
        if args.output.suffix.lower() != ".png":
            raise RedactionError("The output filename must end in .png.")
        image = load_image(args.input.read_bytes())
        result = analyze(image, args.model, args.language)
        for notice in result["warnings"]:
            print(notice, file=sys.stderr)
        if not result["complete"] and not args.allow_partial:
            raise RedactionError("No file was saved because the scan is incomplete. Try a clearer image, or explicitly use --allow-partial.")
        boxes = [Box(**box) for region in result["regions"] for box in region["boxes"]]
        output = png_bytes(redact(image, boxes, args.padding))
        # Exclusive create prevents accidental replacement of an existing result.
        with args.output.open("xb") as file:
            file.write(output)
        print(f"Saved {len(result['regions'])} redaction region(s) to {args.output}. Review the image before sharing.")
    except (RedactionError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
