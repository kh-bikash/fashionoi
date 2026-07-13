from pathlib import Path
import unittest

from glance_retrieval.dataset import deterministic_sample, heuristic_regions


class DatasetTests(unittest.TestCase):
    def test_sample_is_repeatable_and_sorted(self):
        paths = [Path(f"{i}.jpg") for i in range(30)]
        first = deterministic_sample(paths, 10, 17)
        second = deterministic_sample(list(reversed(paths)), 10, 17)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=lambda p: p.as_posix().lower()))


    def test_regions_are_valid(self):
        regions = heuristic_regions(852, 1024)
        self.assertEqual(regions[0].name, "full")
        self.assertEqual(regions[0].box, (0, 0, 852, 1024))
        self.assertEqual(len(regions), 7)
        self.assertTrue(all(x1 > x0 and y1 > y0 for _, (x0, y0, x1, y1) in ((r.name, r.box) for r in regions)))
