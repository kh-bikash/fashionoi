from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.contact_sheet import create_contact_sheet
from glance_retrieval.encoder import OpenClipEncoder
from glance_retrieval.index_store import IndexStore
from glance_retrieval.retrieval import FashionRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve fashion images from a natural-language query.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "fashionpedia-1000")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--strategy", choices=("global", "structured"), default="structured")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    args = parser.parse_args()

    manifest, *_ = IndexStore(args.index).load()
    encoder = OpenClipEncoder(manifest.model_id, args.device)
    results = FashionRetriever(args.index, encoder).search(args.query, args.k, args.candidate_k, args.strategy)
    payload = {"query": args.query, "strategy": args.strategy, "results": [asdict(result) for result in results]}
    print(json.dumps(payload, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.contact_sheet:
        create_contact_sheet(args.query, results, args.contact_sheet)


if __name__ == "__main__":
    main()
