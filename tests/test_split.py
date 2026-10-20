from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ranking.data import load_bundle
from split import build_mock_candidates, build_sequences, generate


class TemporalSplitTests(unittest.TestCase):
    def test_unsorted_events_and_equal_timestamps_keep_source_order(self):
        sequences = build_sequences([
            (1, 7, 10), (1, 2, 10), (1, 5, 9), (1, 9, 20), (1, 1, 20),
            (3, 4, 4), (3, 8, 1), (3, 2, 3), (3, 9, 2),
        ])
        self.assertEqual(sequences, {
            "1": {"train": [5, 7, 2], "validation": 9, "test": 1},
            "3": {"train": [8, 9], "validation": 2, "test": 4},
        })

    def test_invalid_histories_fail_instead_of_silently_dropping_users(self):
        for events in ([], [(1, 1, 1)] * 4,
                       [(1, 1, 1), (1, 2, 2), (1, 3, 3)]):
            with self.assertRaises(ValueError):
                build_sequences(events)

    def test_popularity_uses_train_counts_and_masks_each_splits_history(self):
        rows = {
            "1": {"train": [1, 2], "validation": 4, "test": 8},
            "2": {"train": [3, 4], "validation": 8, "test": 7},
            "3": {"train": [3, 5], "validation": 8, "test": 7},
        }
        candidates = build_mock_candidates(rows, [1, 2, 3, 4, 5, 7, 8], k=2)
        self.assertEqual(candidates["validation"]["1"], [3, 4])
        self.assertEqual(candidates["test"]["1"], [3, 5])
        self.assertNotIn(8, candidates["test"]["1"])
        self.assertEqual(set(candidates["validation"]), set(rows))
        self.assertEqual(set(candidates["test"]), set(rows))
        rows["2"]["test"] = 8
        self.assertEqual(candidates, build_mock_candidates(rows, [1, 2, 3, 4, 5, 7, 8], k=2))
        with self.assertRaises(ValueError):
            build_mock_candidates(rows, [1, 2, 3, 4, 5, 7, 8], k=10)

    def test_generation_is_reproducible_and_compatible_with_cs_existing_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "users.dat").write_text("1::F::1::1::0\n3::M::25::2::0\n")
            (root / "movies.dat").write_text(
                "".join(f"{i}::Movie {i}::Comedy\n" for i in range(1, 21)), encoding="latin-1"
            )
            (root / "ratings.dat").write_text(
                "1::1::1::10\n1::2::2::20\n1::3::3::30\n1::4::4::40\n"
                "3::3::5::10\n3::4::1::20\n3::5::2::30\n3::6::3::40\n"
            )
            paths = [root / name for name in ("sequences.json", "mock_candidates.json", "manifest.json")]
            with redirect_stdout(io.StringIO()):
                generate(root, *paths)
                first = [path.read_bytes() for path in paths]
                for artifact in first:
                    self.assertNotIn(b"\r\n", artifact)
                for name in ("users.dat", "movies.dat", "ratings.dat"):
                    source = root / name
                    source.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
                sequences, candidates, manifest = generate(root, *paths)
            self.assertEqual(first, [path.read_bytes() for path in paths])
            self.assertEqual(manifest["counts"], {"train": 4, "validation": 2, "test": 2})
            self.assertEqual(sequences["1"]["train"], [1, 2])
            self.assertEqual(set(json.loads(paths[1].read_text())), {"validation", "test"})
            bundle = load_bundle(*paths[:2], root / "movies.dat", val_ratio=0)
            self.assertTrue(bundle.train)
            for split in ("validation", "test"):
                for user, items in candidates[split].items():
                    self.assertEqual(len(items), 10)
                    self.assertEqual(len(set(items)), 10)
                    seen = set(sequences[user]["train"])
                    if split == "test":
                        seen.add(sequences[user]["validation"])
                    self.assertTrue(seen.isdisjoint(items))


if __name__ == "__main__":
    unittest.main()
