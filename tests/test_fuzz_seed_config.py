import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nnsmith.cli.fuzz import (
    FuzzingLoop,
    StatusCollect,
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

    @staticmethod
    def make_mock_loop(seeds):
        loop = FuzzingLoop.__new__(FuzzingLoop)
        loop.specific_seeds = seeds
        loop.timeout_s = 60
        loop.max_testcases = None
        loop.status = StatusCollect.__new__(StatusCollect)
        loop.status.n_testcases = 0
        loop.status.n_bugs = 0
        loop.status.n_fail_make_test = 0
        loop.cfg = {"model": {"type": "mock"}}
        loop.save_test = None
        loop.n_dedup_rewrite = 0
        loop.n_dedup_rewrite_fail = 0
        return loop

    def test_explicit_seed_finished_for_handled_outcomes(self):
        loop = self.make_mock_loop([11, 22, 33])
        loop.make_testcase = Mock(
            side_effect=[ValueError("bad testcase"), object(), object()]
        )
        loop.validate_and_report = Mock(side_effect=[False, True])

        with patch("nnsmith.cli.fuzz.set_seed"), patch(
            "nnsmith.cli.fuzz.FUZZ_LOG.info"
        ) as info:
            loop.run()

        messages = [call.args[0] for call in info.call_args_list]
        failed = "Explicit seed finished: index=1/3 seed=11 status=make_testcase_failed"
        bug = "Explicit seed finished: index=2/3 seed=22 status=bug"
        completed = "Explicit seed finished: index=3/3 seed=33 status=completed"
        timing_indices = [
            i for i, message in enumerate(messages) if message.startswith("Timing:")
        ]
        self.assertIn(failed, messages)
        self.assertIn(bug, messages)
        self.assertIn(completed, messages)
        self.assertLess(timing_indices[0], messages.index(bug))
        self.assertLess(timing_indices[1], messages.index(completed))

    def test_explicit_seed_not_finished_for_unhandled_exception(self):
        loop = self.make_mock_loop([33])
        loop.make_testcase = Mock(return_value=object())
        loop.validate_and_report = Mock(side_effect=RuntimeError("native boundary failed"))

        with patch("nnsmith.cli.fuzz.set_seed"), patch(
            "nnsmith.cli.fuzz.FUZZ_LOG.info"
        ) as info, self.assertRaises(RuntimeError):
            loop.run()

        messages = [call.args[0] for call in info.call_args_list]
        self.assertFalse(any(message.startswith("Explicit seed finished:") for message in messages))


if __name__ == "__main__":
    unittest.main()
