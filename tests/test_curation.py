import unittest

import numpy as np

from glance_retrieval.curation import CoverageFacet, allocate_unique_by_quota


class CurationTests(unittest.TestCase):
    def test_quota_allocator_is_unique_and_balanced(self):
        facets = (
            CoverageFacet("scene", "office", ("office",), 2),
            CoverageFacet("scene", "park", ("park",), 2),
        )
        scores = np.array(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.8, 0.9],
                [0.7, 0.7],
            ],
            dtype=np.float32,
        )
        selected = allocate_unique_by_quota(scores, facets)
        self.assertEqual(len({item.index for item in selected}), 4)
        self.assertEqual(sum(item.label == "office" for item in selected), 2)
        self.assertEqual(sum(item.label == "park" for item in selected), 2)
