from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.dataset import deterministic_sample, discover_images, iter_records, load_annotation_boxes
from glance_retrieval.encoder import OpenClipEncoder
from glance_retrieval.indexing import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Index fashion images into region-aware multimodal vectors.")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "val_test2020" / "test")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "fashionpedia-1000")
    parser.add_argument("--annotations", type=Path, default=None, help="Optional Fashionpedia instances JSON")
    parser.add_argument("--max-images", type=int, default=1000, help="0 indexes all images")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto when omitted")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--record-batch-size", type=int, default=8, help="Images grouped into each encoder call")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    paths = deterministic_sample(discover_images(args.image_dir), args.max_images, args.seed)
    annotation_boxes = load_annotation_boxes(args.annotations)
    records = list(iter_records(paths, annotation_boxes))
    print(f"Selected {len(records)} of {len(discover_images(args.image_dir))} images")
    encoder = OpenClipEncoder(args.model, args.device, args.batch_size)

    def progress(done: int, total: int, path: Path) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"[{done:>4}/{total}] {path.name}", flush=True)

    manifest = build_index(records, encoder, args.output, not args.no_faiss, args.record_batch_size, progress)
    print(f"Index ready: {args.output.resolve()} ({manifest.image_count} images, {manifest.dimension} dimensions)")


if __name__ == "__main__":
    main()
