from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "evaluation_final"
OUTPUT = ROOT / "reports" / "evaluation_summary.jpg"
ORDER = ("attribute", "context", "semantic", "style", "compositional")


def main() -> None:
    tile_w, tile_h, label_h = 250, 310, 58
    canvas = Image.new("RGB", (tile_w * len(ORDER), tile_h + label_h), "#f8f4ec")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, query_id in enumerate(ORDER):
        payload = json.loads((RESULTS / f"{query_id}.json").read_text(encoding="utf-8"))
        result = payload["results"][0]
        with Image.open(result["path"]) as raw:
            image = ImageOps.fit(ImageOps.exif_transpose(raw).convert("RGB"), (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        x = column * tile_w
        canvas.paste(image, (x, 0))
        draw.text((x + 8, tile_h + 8), query_id.upper(), fill="#a34532", font=font)
        draw.text((x + 8, tile_h + 30), f"rank 1 | {result['score']:.3f}", fill="#20242a", font=font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=92)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
