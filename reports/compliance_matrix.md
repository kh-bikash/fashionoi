# Assignment compliance matrix

| Requirement | Implementation | Evidence |
|---|---|---|
| 500-1,000 images | Coverage-aware selection of exactly 1,000 from 3,200 audited Fashionpedia test images | `evaluation/curated_fashionpedia_1000.txt`, `reports/coverage_audit.json` |
| Environment variation | Explicit office, urban street, park, and home curation quotas with inspectable top examples | `scripts/curate_dataset.py`, `reports/coverage_context_montage.jpg` |
| Clothing variation | Formal, casual, and outerwear quotas plus FashionSigLIP semantics | `src/glance_retrieval/curation.py` |
| Color variation | Eleven explicit color quotas; color remains bound to garment during retrieval | `src/glance_retrieval/curation.py`, `src/glance_retrieval/query.py` |
| Part A: feature extraction | Full image, six heuristic crops, and optional Fashionpedia annotation boxes | `scripts/index.py`, `src/glance_retrieval/indexing.py` |
| Part A: vector storage | Memory-mapped NumPy vectors, JSONL metadata, versioned manifest, optional FAISS HNSW | `src/glance_retrieval/index_store.py` |
| Part B: natural-language top-k | Query CLI returns ranked results and per-facet score breakdown | `scripts/search.py` |
| Context awareness | Separate binding, scene, style, action, and negative facets | `src/glance_retrieval/query.py`, `src/glance_retrieval/retrieval.py` |
| Beyond one-vector CLIP | Local evidence, same-region counterfactuals, and smooth-AND reranking | `scripts/compare_ablation.py`, `reports/ablation_results.json` |
| Five required prompts | Versioned queries and real contact-sheet outputs | `evaluation/queries.json`, `outputs/evaluation/` |
| Modular code | Data, curation, encoding, storage, parsing, ranking, and evaluation are independent modules | `src/glance_retrieval/` |
| Scalability | Offline encoding, mmap, FAISS, candidate reranking, stage-level benchmark | `scripts/benchmark.py`, PDF section 9 |
| Zero-shot behavior | FashionSigLIP plus deterministic query structure; no task-specific label training | `src/glance_retrieval/encoder.py`, `src/glance_retrieval/query.py` |
| Single PDF | Approaches, chosen architecture, code link, evidence, limitations, and future work | `output/pdf/glance_fashion_retrieval_submission.pdf` |

Semantic curation assignments and qualitative retrieval inspection are not presented as ground-truth accuracy. MRR, Recall@10, and nDCG@10 remain gated on independent relevance judgments.
