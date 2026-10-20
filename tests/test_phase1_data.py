"""Validate committed Phase-1 artifacts against all raw MovieLens interactions."""
from collections import Counter
import json
from pathlib import Path
import unittest

from metrics import evaluate_rankings
from ranking.data import load_bundle
from split import sha256

ROOT = Path(__file__).resolve().parents[1]


class Phase1ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequences = json.loads((ROOT / "data/sequences.json").read_text(encoding="utf-8"))
        cls.candidates = json.loads((ROOT / "data/mock_candidates.json").read_text(encoding="utf-8"))

    def test_all_raw_interactions_are_preserved_in_stable_time_order(self):
        events = {}
        with (ROOT / "ml-1m/ratings.dat").open(encoding="ascii") as stream:
            for row_number, line in enumerate(stream):
                user, movie, _, timestamp = map(int, line.split("::"))
                key = (str(user), movie)
                self.assertNotIn(key, events)
                events[key] = (timestamp, row_number)
        self.assertEqual(len(self.sequences), 6040)
        self.assertEqual(len(events), 1000209)
        recovered = set()
        for user, record in self.sequences.items():
            self.assertEqual(set(record), {"train", "validation", "test"})
            movies = record["train"] + [record["validation"], record["test"]]
            self.assertGreaterEqual(len(movies), 20)
            self.assertEqual(len(movies), len(set(movies)))
            keys = [(user, movie) for movie in movies]
            times = [events[key] for key in keys]
            self.assertEqual(times, sorted(times))
            recovered.update(keys)
        self.assertEqual(recovered, set(events))

    def test_every_user_has_ten_valid_unseen_movies_in_popularity_order(self):
        with (ROOT / "ml-1m/movies.dat").open(encoding="latin-1") as stream:
            valid_ids = {int(line.split("::", 1)[0]) for line in stream}
        counts = Counter(movie for row in self.sequences.values() for movie in row["train"])
        global_order = sorted(valid_ids, key=lambda movie: (-counts[movie], movie))
        positions = {movie: index for index, movie in enumerate(global_order)}
        self.assertEqual(set(self.candidates), {"validation", "test"})
        for split, users in self.candidates.items():
            self.assertEqual(set(users), set(self.sequences))
            for user, movies in users.items():
                self.assertEqual(len(movies), 10)
                self.assertEqual(len(set(movies)), 10)
                self.assertTrue(set(movies) <= valid_ids)
                seen = set(self.sequences[user]["train"])
                if split == "test":
                    seen.add(self.sequences[user]["validation"])
                self.assertTrue(seen.isdisjoint(movies))
                ranks = [positions[movie] for movie in movies]
                self.assertEqual(ranks, sorted(ranks))
                self.assertTrue(set(global_order[:ranks[-1] + 1]) <= seen | set(movies))

    def test_manifest_metrics_hashes_and_existing_c_loader(self):
        manifest = json.loads((ROOT / "data/split_manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["artifact_sha256"].items():
            self.assertEqual(sha256(ROOT / "data" / name), digest)
        for name, digest in manifest["source_text_sha256_lf"].items():
            self.assertEqual(sha256(ROOT / "ml-1m" / name, normalize_newlines=True), digest)
        for split in ("validation", "test"):
            targets = {user: row[split] for user, row in self.sequences.items()}
            actual = evaluate_rankings(self.candidates[split], targets)
            expected = manifest["mock_debug_metrics"][split]
            for metric in ("hr10", "ndcg10"):
                self.assertAlmostEqual(actual[metric], expected[metric], places=12)
        bundle = load_bundle(ROOT / "data/sequences.json", ROOT / "data/mock_candidates.json",
                             ROOT / "ml-1m/movies.dat")
        self.assertTrue(bundle.train)
        self.assertTrue(bundle.select)
        self.assertTrue(bundle.test)


if __name__ == "__main__":
    unittest.main()
