from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Region:
    name: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    width: int
    height: int
    regions: tuple[Region, ...]


def discover_images(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.as_posix().lower(),
    )


def deterministic_sample(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    if limit is None or limit <= 0 or limit >= len(paths):
        return paths

    def key(path: Path) -> str:
        payload = f"{seed}:{path.as_posix()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return sorted(sorted(paths, key=key)[:limit], key=lambda p: p.as_posix().lower())


def load_selection_file(selection_path: Path, image_root: Path) -> list[Path]:
    """Resolve a reproducible image selection written as paths relative to image_root."""
    image_root = image_root.expanduser().resolve()
    selected: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(selection_path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        path = (image_root / value).resolve()
        try:
            path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError(f"Selection line {line_number} escapes the image directory: {value}") from exc
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"Selection line {line_number} is not a supported image: {path}")
        if path not in seen:
            selected.append(path)
            seen.add(path)
    if not selected:
        raise ValueError(f"Selection file contains no images: {selection_path}")
    return selected


def heuristic_regions(width: int, height: int) -> tuple[Region, ...]:
    """Multi-scale crops tuned for mostly portrait outfit imagery.

    The full image is retained for scene/style evidence. Remaining crops offer local
    evidence for garment-attribute bindings without requiring a detector.
    """
    x0, y0, x1, y1 = 0, 0, width, height
    return (
        Region("full", (x0, y0, x1, y1)),
        Region("upper", (0, 0, width, max(1, round(height * 0.58)))),
        Region("torso", (round(width * 0.12), round(height * 0.12), round(width * 0.88), round(height * 0.68))),
        Region("lower", (0, round(height * 0.42), width, height)),
        Region("center", (round(width * 0.16), round(height * 0.16), round(width * 0.84), round(height * 0.84))),
        Region("left", (0, round(height * 0.08), round(width * 0.58), round(height * 0.92))),
        Region("right", (round(width * 0.42), round(height * 0.08), width, round(height * 0.92))),
    )


def load_annotation_boxes(annotation_path: Path | None) -> dict[str, list[Region]]:
    """Load optional Fashionpedia/COCO boxes keyed by file name.

    Annotation crops are appended to heuristic crops and improve small-item retrieval
    (ties, collars, shoes). Test images do not include labels, so this remains optional.
    """
    if annotation_path is None:
        return {}
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in data.get("images", [])}
    category_by_id = {int(item["id"]): item.get("name", "item") for item in data.get("categories", [])}
    result: dict[str, list[Region]] = {}
    for ann in data.get("annotations", []):
        image = image_by_id.get(int(ann["image_id"]))
        bbox = ann.get("bbox")
        if not image or not bbox or len(bbox) != 4:
            continue
        x, y, w, h = (float(v) for v in bbox)
        iw, ih = int(image.get("width", 0)), int(image.get("height", 0))
        if w <= 1 or h <= 1 or iw <= 0 or ih <= 0:
            continue
        category = category_by_id.get(int(ann.get("category_id", -1)), "item")
        box = (
            max(0, round(x)),
            max(0, round(y)),
            min(iw, round(x + w)),
            min(ih, round(y + h)),
        )
        result.setdefault(str(image["file_name"]), []).append(Region(f"annotation:{category}", box))
    return result


def iter_records(paths: Iterable[Path], annotation_regions: dict[str, list[Region]] | None = None) -> Iterator[ImageRecord]:
    annotation_regions = annotation_regions or {}
    for path in paths:
        with Image.open(path) as raw:
            width, height = ImageOps.exif_transpose(raw).size
        extra = annotation_regions.get(path.name, [])
        regions = heuristic_regions(width, height) + tuple(extra)
        image_id = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:16]
        yield ImageRecord(image_id, path.resolve(), width, height, regions)


def load_region_images(record: ImageRecord, max_annotation_regions: int = 12) -> list[Image.Image]:
    with Image.open(record.path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        selected = list(record.regions[:7])
        selected.extend(record.regions[7 : 7 + max_annotation_regions])
        return [image.crop(region.box).copy() for region in selected]
