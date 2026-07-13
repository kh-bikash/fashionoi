from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glance_retrieval.encoder import OpenClipEncoder
from glance_retrieval.index_store import IndexStore
from glance_retrieval.retrieval import FashionRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one-vector and structured retrieval on the same candidates.")
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation" / "queries.json")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts" / "fashionpedia-1000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "ablation_results.json")
    parser.add_argument("--benchmark-output", type=Path, default=ROOT / "reports" / "latency_benchmark.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-k", type=int, default=100)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    manifest, *_ = IndexStore(args.index).load()
    encoder = OpenClipEncoder(manifest.model_id, args.device)
    retriever = FashionRetriever(args.index, encoder)
    rows: list[dict] = []
    timing_rows: list[dict] = []
    for item in queries:
        global_results = retriever.search(item["query"], k=args.candidate_k, candidate_k=args.candidate_k, strategy="global")
        timing: dict[str, float | int | str] = {}
        structured_results = retriever.search(item["query"], k=10, candidate_k=args.candidate_k, strategy="structured", profile=timing)
        timing_rows.append({"id": item["id"], "query": item["query"], **timing})
        global_rank = {result.image_id: result.rank for result in global_results}
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "global_top_10": [asdict(result) for result in global_results[:10]],
                "structured_top_10": [asdict(result) for result in structured_results],
                "structured_top1_rank_in_global_baseline": global_rank.get(structured_results[0].image_id),
                "note": "This is an unlabeled ranking comparison, not an accuracy metric.",
            }
        )
        print(f"{item['id']}: structured top-1 was global rank {rows[-1]['structured_top1_rank_in_global_baseline']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"baseline": "full-image FashionSigLIP vector", "chosen": "region-aware structured reranker", "queries": rows}, indent=2), encoding="utf-8")
    totals = [float(row["total_seconds"]) for row in timing_rows]
    index_bytes = sum(path.stat().st_size for path in args.index.glob("*") if path.is_file())
    benchmark = {
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
        "queries": timing_rows,
    }
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.benchmark_output}")


if __name__ == "__main__":
    main()
