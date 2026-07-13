from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.dataset import discover_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible image-dataset profile.")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "val_test2020" / "test")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "dataset_profile.json")
    args = parser.parse_args()
    paths = discover_images(args.image_dir)
    widths, heights = [], []
    orientations = Counter()
    failures = []
    for path in paths:
        try:
            with Image.open(path) as raw:
                width, height = ImageOps.exif_transpose(raw).size
                raw.verify()
            widths.append(width)
            heights.append(height)
            orientations["portrait" if height > width else "landscape" if width > height else "square"] += 1
        except Exception as exc:  # profile corrupt images instead of aborting the whole run
            failures.append({"path": str(path), "error": str(exc)})
    payload = {
        "root": str(args.image_dir.resolve()),
        "image_count": len(paths),
        "valid_images": len(widths),
        "corrupt_images": failures,
        "formats": dict(Counter(path.suffix.lower() for path in paths)),
        "orientation": dict(orientations),
        "width": {"min": min(widths), "median": statistics.median(widths), "max": max(widths)},
        "height": {"min": min(heights), "median": statistics.median(heights), "max": max(heights)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

