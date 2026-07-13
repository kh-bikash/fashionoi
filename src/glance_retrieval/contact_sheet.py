from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def create_contact_sheet(query: str, results, output: Path, columns: int = 3, thumb_size: tuple[int, int] = (320, 360)) -> None:
    rows = (len(results) + columns - 1) // columns
    header = 84
    canvas = Image.new("RGB", (columns * thumb_size[0], header + rows * (thumb_size[1] + 44)), "#f6f1e8")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((18, 18), "FASHION RETRIEVAL", fill="#9b3f2e", font=font)
    draw.text((18, 44), query[:130], fill="#20242a", font=font)
    for i, result in enumerate(results):
        row, col = divmod(i, columns)
        x, y = col * thumb_size[0], header + row * (thumb_size[1] + 44)
        with Image.open(result.path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            fitted = ImageOps.fit(image, thumb_size, method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x, y))
        draw.text((x + 8, y + thumb_size[1] + 10), f"#{result.rank}  score {result.score:.3f}", fill="#20242a", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)

