import numpy as np
import unittest

from glance_retrieval.retrieval import _soft_min, _standardize


class RetrievalTests(unittest.TestCase):
    def test_soft_min_penalizes_a_missing_required_facet(self):
        scores = np.array([[1.0, 1.0], [2.0, -1.0]], dtype=np.float32)
        conjunction = _soft_min(scores)
        self.assertGreater(conjunction[0], conjunction[1])

    def test_standardization_handles_constant_scores(self):
        self.assertTrue(np.array_equal(_standardize(np.ones(4, dtype=np.float32)), np.zeros(4, dtype=np.float32)))
