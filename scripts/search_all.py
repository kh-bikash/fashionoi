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
    parser = argparse.ArgumentParser(description="Run a JSON list of evaluation queries with one model load.")
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation" / "queries.json")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "fashionpedia-1000")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    manifest, *_ = IndexStore(args.index).load()
    retriever = FashionRetriever(args.index, OpenClipEncoder(manifest.model_id, args.device))
    args.output.mkdir(parents=True, exist_ok=True)

    summary = []
    for item in queries:
        results = retriever.search(item["query"], args.k, args.candidate_k)
        payload = {"id": item["id"], "query": item["query"], "results": [asdict(result) for result in results]}
        (args.output / f"{item['id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        create_contact_sheet(item["query"], results, args.output / f"{item['id']}.jpg")
        summary.append(payload)
        print(f"{item['id']}: {results[0].path} (score={results[0].score:.4f})", flush=True)
    (args.output / "all_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()

