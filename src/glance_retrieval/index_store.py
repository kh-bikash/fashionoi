from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Manifest:
    version: int
    model_id: str
    dimension: int
    image_count: int
    max_regions: int
    region_strategy: str = "full+heuristic+optional-fashionpedia-boxes"
    metric: str = "cosine"


class IndexStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def write(
        self,
        manifest: Manifest,
        metadata: list[dict[str, Any]],
        global_vectors: np.ndarray,
        region_vectors: np.ndarray,
        region_mask: np.ndarray,
        build_faiss: bool = True,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.root / "global.npy", np.asarray(global_vectors, dtype=np.float32))
        np.save(self.root / "regions.npy", np.asarray(region_vectors, dtype=np.float32))
        np.save(self.root / "region_mask.npy", np.asarray(region_mask, dtype=np.bool_))
        (self.root / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
        with (self.root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for row in metadata:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        if build_faiss:
            self._write_faiss(global_vectors)

    def _write_faiss(self, vectors: np.ndarray) -> None:
        try:
            import faiss
        except ImportError:
            return
        dimension = int(vectors.shape[1])
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(self.root / "index.faiss"))

    def load(self, mmap: bool = True):
        mode = "r" if mmap else None
        manifest = Manifest(**json.loads((self.root / "manifest.json").read_text(encoding="utf-8")))
        metadata = [json.loads(line) for line in (self.root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
        global_vectors = np.load(self.root / "global.npy", mmap_mode=mode)
        region_vectors = np.load(self.root / "regions.npy", mmap_mode=mode)
        region_mask = np.load(self.root / "region_mask.npy", mmap_mode=mode)
        if global_vectors.shape != (manifest.image_count, manifest.dimension):
            raise ValueError("Corrupt index: global vector shape does not match manifest")
        if len(metadata) != manifest.image_count:
            raise ValueError("Corrupt index: metadata count does not match manifest")
        return manifest, metadata, global_vectors, region_vectors, region_mask

    def candidate_search(self, query: np.ndarray, k: int, global_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = min(max(1, int(k)), len(global_vectors))
        faiss_path = self.root / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss
                index = faiss.read_index(str(faiss_path))
                if hasattr(index, "hnsw"):
                    index.hnsw.efSearch = max(64, k)
                scores, ids = index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), k)
                return ids[0], scores[0]
            except ImportError:
                pass
        scores = np.asarray(global_vectors) @ np.asarray(query, dtype=np.float32)
        ids = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[ids], kind="stable")
        return ids[order], scores[ids][order]

