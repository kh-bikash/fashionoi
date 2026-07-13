import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from glance_retrieval.index_store import IndexStore, Manifest


class StoreTests(unittest.TestCase):
    def test_store_round_trip_and_exact_search(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            store = IndexStore(tmp_path)
            vectors = np.eye(3, dtype=np.float32)
            regions = vectors[:, None, :]
            mask = np.ones((3, 1), dtype=np.bool_)
            metadata = [{"image_id": str(i), "path": f"{i}.jpg"} for i in range(3)]
            manifest = Manifest(1, "fake", 3, 3, 1)
            store.write(manifest, metadata, vectors, regions, mask, build_faiss=False)
            loaded, loaded_meta, global_vectors, *_ = store.load(mmap=False)
            self.assertEqual(loaded, manifest)
            self.assertEqual(loaded_meta, metadata)
            ids, scores = store.candidate_search(np.array([0, 1, 0], dtype=np.float32), 2, global_vectors)
            self.assertEqual(ids[0], 1)
            self.assertEqual(scores[0], 1.0)
            json.loads((tmp_path / "manifest.json").read_text())
