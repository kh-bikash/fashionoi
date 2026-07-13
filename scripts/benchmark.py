from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.encoder import OpenClipEncoder
from glance_retrieval.index_store import IndexStore
from glance_retrieval.retrieval import FashionRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval stages on the five assignment queries.")
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation" / "queries.json")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "fashionpedia-1000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "latency_benchmark.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-k", type=int, default=100)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    manifest, *_ = IndexStore(args.index).load()
    encoder = OpenClipEncoder(manifest.model_id, args.device)
    retriever = FashionRetriever(args.index, encoder)
    # Warm up tokenizer/model kernels without including startup and weight loading.
    retriever.search(queries[0]["query"], k=1, candidate_k=args.candidate_k, strategy="global")

    rows: list[dict] = []
    for item in queries:
        timing: dict[str, float | int | str] = {}
        retriever.search(item["query"], k=10, candidate_k=args.candidate_k, strategy="structured", profile=timing)
        rows.append({"id": item["id"], "query": item["query"], **timing})
        print(f"{item['id']}: {timing['total_seconds']} s", flush=True)

    totals = [float(row["total_seconds"]) for row in rows]
    index_bytes = sum(path.stat().st_size for path in args.index.glob("*") if path.is_file())
    report = {
        "device": encoder.device,
        "model_id": encoder.model_id,
        "image_count": manifest.image_count,
        "candidate_k": args.candidate_k,
        "index_bytes": index_bytes,
        "startup_excluded": True,
        "summary": {
            "mean_total_seconds": round(statistics.mean(totals), 6),
            "median_total_seconds": round(statistics.median(totals), 6),
            "maximum_total_seconds": round(max(totals), 6),
        },
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
