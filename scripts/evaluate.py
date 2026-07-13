from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.encoder import OpenClipEncoder
from glance_retrieval.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from glance_retrieval.index_store import IndexStore
from glance_retrieval.retrieval import FashionRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval from human relevance judgments.")
    parser.add_argument("--judgments", type=Path, default=ROOT / "evaluation" / "judgments.json")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "fashionpedia-1000")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = json.loads(args.judgments.read_text(encoding="utf-8"))
    if any(not row.get("relevant_image_ids") for row in data):
        raise SystemExit("Judgments contain empty relevance sets. Label them before reporting metrics.")
    manifest, *_ = IndexStore(args.index).load()
    retriever = FashionRetriever(args.index, OpenClipEncoder(manifest.model_id, args.device))
    rows = []
    for item in data:
        ranked = [result.image_id for result in retriever.search(item["query"], args.k, max(100, args.k))]
        relevant = set(item["relevant_image_ids"])
        rows.append(
            {
                "query": item["query"],
                "mrr": reciprocal_rank(ranked, relevant),
                f"recall@{args.k}": recall_at_k(ranked, relevant, args.k),
                f"ndcg@{args.k}": ndcg_at_k(ranked, relevant, args.k),
            }
        )
    print(json.dumps({"per_query": rows, "macro_average": {key: sum(r[key] for r in rows) / len(rows) for key in rows[0] if key != "query"}}, indent=2))


if __name__ == "__main__":
    main()

