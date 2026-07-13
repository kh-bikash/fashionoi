from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.dataset import deterministic_sample, discover_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=ROOT / "val_test2020" / "test")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "dataset_montage.jpg")
    args = parser.parse_args()
    paths = deterministic_sample(discover_images(args.image_dir), 9, seed=29)
    size, label_h = (300, 360), 26
    canvas = Image.new("RGB", (900, 3 * (360 + label_h)), "#f4efe7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, path in enumerate(paths):
        row, col = divmod(index, 3)
        with Image.open(path) as raw:
            image = ImageOps.fit(ImageOps.exif_transpose(raw).convert("RGB"), size, method=Image.Resampling.LANCZOS)
        x, y = col * size[0], row * (size[1] + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + size[1] + 7), path.name[:20], fill="#25282e", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=90)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

