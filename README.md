# Glance: Fashion & Context Retrieval

I built this repository as my submission for the Glance ML internship assignment. My system retrieves outfit images from natural-language queries while explicitly modeling garment attributes, scene, style, and actions.

## Why I went beyond vanilla CLIP

A one-line CLIP baseline embeds the whole sentence and the whole image once. It often recognizes the right words but loses which color belongs to which garment. I added three ML ideas:

1. **Fashion-domain encoder.** The default model is `Marqo/marqo-fashionSigLIP`, a SigLIP model fine-tuned for categories, styles, colors, materials, and fashion details.
2. **Multi-granularity image representation.** Every image has a full-scene vector plus upper-body, torso, lower-body, center, left, and right crop vectors. Optional Fashionpedia bounding boxes add true apparel-instance crops when annotation JSON is available.
3. **Structured conjunctive reranking.** A deterministic parser keeps bound phrases such as `red tie` and `white shirt`, scores same-region counterfactuals such as `red pants` and `white tie`, and combines required facets using a smooth minimum. One missing requirement therefore cannot be hidden by one very strong match.

```text
Image -> full + local crops -> FashionSigLIP -> normalized vectors -> FAISS HNSW
Query -> full embedding ---------------------------> candidate retrieval
      -> bound fashion facets + scene/style/action -> conjunctive reranking -> top k
```

## Repository layout

```text
scripts/index.py             Part A: image indexing CLI
scripts/search.py            Part B: natural-language retrieval CLI
scripts/search_all.py        Batch runner for the five assignment queries
scripts/benchmark.py         Stage-level retrieval latency benchmark
scripts/evaluate.py          MRR, Recall@k, nDCG@k from human judgments
src/glance_retrieval/        Reusable ML, storage, parsing, and scoring modules
evaluation/                  Five assignment queries and judgment template
tests/                       Offline unit tests for parser, scoring, and storage
output/pdf/                  Final assignment write-up
DATASET.md                   Source, split, annotation, coverage, and license notes
```

I intentionally exclude the downloaded dataset from Git. My workspace has 3,200 Fashionpedia test JPGs in `val_test2020/test`. The official test archive contains images but no public garment annotations, so the default system works annotation-free. When Fashionpedia train/validation instance JSON is available, the same indexer adds the annotated apparel boxes as higher-quality local regions.

Because the assignment also requires office, street, park, and home variation, I do not assume that a random fashion subset is context-balanced. `scripts/curate_dataset.py` audits all 3,200 images with zero-shot environment, clothing, and color prompts and writes a reproducible 1,000-image selection. Its scores are curation proxies, not ground-truth scene labels.

## Submission status

The source code, real model run, evaluation contact sheets, and PDF are complete. I have already:

1. Audited all 3,200 images with zero corrupt files.
2. Indexed a deterministic 1,000-image subset into 1,000 global and 7,000 local FashionSigLIP vectors.
3. Ran all five required queries and inspected their contact sheets.
4. Generated a coverage-aware 1,000-image selection with explicit environment, clothing, and color quotas.
5. Verified 11/11 offline tests and the index integrity checks.

The public codebase is:

[https://github.com/kh-bikash/fashionoi](https://github.com/kh-bikash/fashionoi)

Independent human relevance labels are optional but required before reporting MRR, Recall@10, or nDCG@10. To reproduce the full model run from scratch, use `powershell -ExecutionPolicy Bypass -File scripts/run_assignment.ps1`.

## Setup

Python 3.10+ is recommended. The first model-backed command downloads model weights from Hugging Face.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```

For tests and report generation, use `pip install -e ".[test,report]"`.

Optional FAISS acceleration:

```bash
pip install -r requirements-faiss.txt
```

If FAISS is unavailable, the same index format uses exact NumPy cosine search. That is convenient for 1,000 images; use the generated HNSW index for large collections.

## Part A - index images

First create the coverage-aware 1,000-image selection from the downloaded test archive:

```bash
python scripts/curate_dataset.py --image-dir val_test2020/test
```

This writes `evaluation/curated_fashionpedia_1000.txt`, `reports/coverage_audit.json`, and a context montage. Then index that exact selection:

```bash
python scripts/index.py \
  --image-dir val_test2020/test \
  --selection-file evaluation/curated_fashionpedia_1000.txt \
  --output artifacts/fashionpedia-1000 \
  --max-images 1000
```

With Fashionpedia validation annotations, add:

```bash
python scripts/index.py --image-dir path/to/val --annotations path/to/instances_attributes_val2020.json
```

Outputs are intentionally simple and inspectable:

- `global.npy`: one normalized scene/outfit vector per image;
- `regions.npy` and `region_mask.npy`: local vectors and valid-region mask;
- `metadata.jsonl`: paths, dimensions, crop types, and boxes;
- `index.faiss`: optional HNSW approximate-nearest-neighbor index;
- `manifest.json`: version, model, dimension, count, and metric.

## Part B - query

```bash
python scripts/search.py "A red tie and a white shirt in a formal setting." \
  --index artifacts/fashionpedia-1000 \
  -k 5 \
  --contact-sheet outputs/red-tie-white-shirt.jpg \
  --json outputs/red-tie-white-shirt.json
```

Each result includes the final score, global score, conjunction score, and every facet score. This makes failure analysis possible instead of returning an opaque cosine number.

Run all five assignment queries by invoking `scripts/search.py` for each entry in `evaluation/queries.json`.

For an auditable ablation, compare full-image one-vector retrieval with the chosen structured reranker:

```bash
python scripts/compare_ablation.py
```

The generated JSON reports how far the structured top result moved relative to the one-vector baseline. It is deliberately described as an unlabeled ranking comparison until relevance judgments are added.

Measure model-warm retrieval stages separately with `python scripts/benchmark.py`. The report excludes model download/startup and records full-query encoding, ANN/exact candidate search, and structured reranking time for each assignment query.

## Evaluation without label leakage

Fashionpedia's test split has no public ground-truth attributes. Do not claim a fabricated accuracy. Instead:

1. Retrieve the top 20-50 images for every query and make contact sheets.
2. Have at least two people independently mark relevant images.
3. Resolve disagreements, copy `evaluation/judgments.template.json` to `evaluation/judgments.json`, and add relevant `image_id` values.
4. Run `python scripts/evaluate.py --judgments evaluation/judgments.json -k 10`.

Report MRR, Recall@10, nDCG@10, and inter-annotator agreement. Compare three ablations: full-image FashionSigLIP; + region max pooling; + query decomposition and conjunctive reranking.

## Tests and data audit

The unit tests do not download a model:

```bash
python -m unittest discover -s tests -v
python scripts/profile_dataset.py
```

The editable install in Setup makes `glance_retrieval` importable for the test command.

## Scalability to one million images

Image embeddings are computed offline and memory-mapped at query time. FAISS HNSW works well into the low millions when recall and latency matter more than memory. At larger scale, replace HNSW with IVF-PQ or a managed vector service, shard by catalog/region, store float16 or product-quantized vectors, batch GPU inference, and rerank only 100-500 candidates. The query parser and reranker are independent of the candidate index.

## Important limitations

- The supplied images are the test split and do not include attribute labels; quantitative relevance requires manual judgments.
- Heuristic crops are not object masks. Tiny ties/accessories benefit significantly from validation annotations or a fashion detector.
- Fashion-domain tuning helps clothing semantics but can weaken rare location cues; a generic scene encoder can be fused in future work.
- Deterministic lexicons are transparent and fast but do not cover every paraphrase. A constrained JSON-producing LLM parser is a future option, not a runtime dependency.

## Reproducibility

Sampling uses SHA-256 over `seed:path`, not Python's process-randomized hash. Vectors are L2-normalized, cosine similarity is an inner product, the model ID is stored in the manifest, and a mismatched query model is rejected.

## References

- [Official Fashionpedia dataset repository and downloads](https://github.com/cvdfoundation/fashionpedia)
- [Fashionpedia project](https://fashionpedia.github.io/home/index.html)
- [Fashionpedia paper](https://arxiv.org/abs/2004.12276)
- [Marqo-FashionSigLIP model card](https://huggingface.co/Marqo/marqo-fashionSigLIP)
- [SigLIP paper](https://arxiv.org/abs/2303.15343)
- [FAISS](https://github.com/facebookresearch/faiss)
