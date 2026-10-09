import json
import tempfile
import unittest
from pathlib import Path

from nnsmith.cli.fuzz import (
    load_fuzz_seeds,
    normalize_fuzz_seeds,
    resolve_fuzz_seeds,
)


class FuzzSeedConfigTest(unittest.TestCase):
    def test_normalize_preserves_exact_order(self):
        self.assertEqual(
            normalize_fuzz_seeds([20260954, 20260944, 20260954]),
            [20260954, 20260944, 20260954],
        )

    def test_normalize_accepts_one_seed(self):
        self.assertEqual(normalize_fuzz_seeds(20260944), [20260944])

    def test_normalize_rejects_invalid_values(self):
        for seeds in ([], [-1], [2**32]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                normalize_fuzz_seeds(seeds)

    def test_loads_json_seed_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seeds.json"
            path.write_text(json.dumps([11, 7, 11]))
            self.assertEqual(load_fuzz_seeds(path), [11, 7, 11])

    def test_loads_seeds_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seeds.json"
            path.write_text(json.dumps({"seeds": [13, 17]}))
            self.assertEqual(load_fuzz_seeds(path), [13, 17])

    def test_loads_grouped_known_bug_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "known_bug_seeds.json"
            path.write_text(
                json.dumps(
                    {
                        "patterns": [
                            {"pattern_id": "p0", "seeds": [3, 5]},
                            {"pattern_id": "p1", "seeds": [8, 13]},
                        ]
                    }
                )
            )
            self.assertEqual(load_fuzz_seeds(path), [3, 5, 8, 13])

    def test_rejects_explicit_seeds_with_seed_file(self):
        with self.assertRaises(ValueError):
            resolve_fuzz_seeds([1], "seeds.json")


if __name__ == "__main__":
    unittest.main()
