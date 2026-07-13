from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.curation import DEFAULT_COVERAGE_FACETS, allocate_unique_by_quota, encode_facet_vectors, facet_report
from glance_retrieval.dataset import discover_images
from glance_retrieval.encoder import OpenClipEncoder, l2_normalize


def encode_full_images(paths: list[Path], encoder: OpenClipEncoder, cache_dir: Path, record_batch_size: int) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256("\n".join(path.as_posix() for path in paths).encode("utf-8")).hexdigest()
    vector_path = cache_dir / "global.npy"
    manifest_path = cache_dir / "manifest.json"
    if vector_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature and manifest.get("model_id") == encoder.model_id:
            vectors = np.load(vector_path)
            if vectors.shape[0] == len(paths):
                print(f"Reusing coverage cache: {vector_path}")
                return np.asarray(vectors, dtype=np.float32)

    batches: list[np.ndarray] = []
    for start in range(0, len(paths), record_batch_size):
        batch_paths = paths[start : start + record_batch_size]
        images: list[Image.Image] = []
        for path in batch_paths:
            with Image.open(path) as raw:
                images.append(ImageOps.exif_transpose(raw).convert("RGB"))
        batches.append(encoder.encode_images(images))
        for image in images:
            image.close()
        done = min(start + len(batch_paths), len(paths))
        print(f"[{done:>4}/{len(paths)}] full-image coverage vectors", flush=True)
    vectors = l2_normalize(np.concatenate(batches, axis=0))
    np.save(vector_path, vectors.astype(np.float32))
    manifest_path.write_text(
        json.dumps({"signature": signature, "model_id": encoder.model_id, "image_count": len(paths), "dimension": int(vectors.shape[1])}, indent=2),
        encoding="utf-8",
    )
    return vectors


def coverage_montage(paths: list[Path], scores: np.ndarray, output: Path, top_n: int = 6) -> None:
    facets = [facet for facet in DEFAULT_COVERAGE_FACETS if facet.axis == "environment"]
    cell_w, cell_h = 180, 220
    canvas = Image.new("RGB", (top_n * cell_w, len(facets) * cell_h), "#f4f0e8")
    draw = ImageDraw.Draw(canvas)
    for row, facet in enumerate(facets):
        column = DEFAULT_COVERAGE_FACETS.index(facet)
        order = np.argsort(-scores[:, column], kind="stable")[:top_n]
        for col, index in enumerate(order):
            with Image.open(paths[int(index)]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((cell_w - 12, cell_h - 42), Image.Resampling.LANCZOS)
                x = col * cell_w + (cell_w - image.width) // 2
                y = row * cell_h + 22 + (cell_h - 42 - image.height) // 2
                canvas.paste(image, (x, y))
            draw.text((col * cell_w + 6, row * cell_h + 5), f"{facet.name} #{col + 1}", fill="#1f2933")
            draw.text((col * cell_w + 6, (row + 1) * cell_h - 16), f"score {scores[int(index), column]:.3f}", fill="#486b68")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible, coverage-aware Fashionpedia subset.")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "val_test2020" / "test")
    parser.add_argument("--selection", type=Path, default=ROOT / "evaluation" / "curated_fashionpedia_1000.txt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "coverage_audit.json")
    parser.add_argument("--montage", type=Path, default=ROOT / "reports" / "coverage_context_montage.jpg")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts" / "coverage-all-3200")
    parser.add_argument("--model", default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--record-batch-size", type=int, default=32)
    args = parser.parse_args()

    image_root = args.image_dir.expanduser().resolve()
    paths = discover_images(image_root)
    encoder = OpenClipEncoder(args.model, args.device, args.batch_size)
    image_vectors = encode_full_images(paths, encoder, args.cache_dir, args.record_batch_size)
    facet_vectors = encode_facet_vectors(encoder, DEFAULT_COVERAGE_FACETS)
    scores = image_vectors @ facet_vectors.T
    selected = allocate_unique_by_quota(scores, DEFAULT_COVERAGE_FACETS)
    relative_paths = [path.relative_to(image_root).as_posix() for path in paths]

    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text("\n".join(relative_paths[item.index] for item in selected) + "\n", encoding="utf-8")
    report = {
        "purpose": "Zero-shot semantic coverage curation; scores are proxies, not human ground-truth labels.",
        "source": "Fashionpedia test images downloaded from the official CVDF archive",
        "model_id": encoder.model_id,
        "candidate_images": len(paths),
        "selected_images": len(selected),
        "selection_counts": {f"{axis}:{label}": count for (axis, label), count in sorted(Counter((item.axis, item.label) for item in selected).items())},
        "facets": facet_report(scores, DEFAULT_COVERAGE_FACETS, relative_paths),
        "selection": [dict(index=item.index, path=relative_paths[item.index], axis=item.axis, label=item.label, score=item.score, rank_within_label=item.rank_within_label) for item in selected],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    coverage_montage(paths, scores, args.montage)
    print(f"Wrote {len(selected)} paths to {args.selection}")
    print(f"Coverage audit: {args.report}")
    print(f"Context montage: {args.montage}")


if __name__ == "__main__":
    main()
