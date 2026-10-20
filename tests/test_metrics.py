import math
import unittest

from metrics import evaluate_rankings, hr_at_k, ndcg_at_k


class TemporalMetricsTests(unittest.TestCase):
    def test_rank_cutoff_and_discount(self):
        ranking = list(range(1, 12))
        self.assertEqual(hr_at_k(ranking, 1), 1.0)
        self.assertEqual(ndcg_at_k(ranking, 1), 1.0)
        self.assertAlmostEqual(ndcg_at_k(ranking, 10), 1 / math.log2(11))
        self.assertEqual(hr_at_k(ranking, 11), 0.0)
        self.assertEqual(ndcg_at_k(ranking, 11), 0.0)
        self.assertEqual(hr_at_k([], 1), 0.0)

    def test_all_users_including_candidate_misses_are_averaged(self):
        result = evaluate_rankings({"1": [1, 2], "2": [2, 3]}, {"1": 2, "2": 5})
        self.assertEqual(result["hr10"], 0.5)
        self.assertAlmostEqual(result["ndcg10"], 0.5 / math.log2(3))
        with self.assertRaises(ValueError):
            evaluate_rankings({"1": [1]}, {"1": 1, "2": 2})
        with self.assertRaises(ValueError):
            evaluate_rankings({}, {})

    def test_invalid_or_duplicate_movie_ids_are_rejected(self):
        for ranking, target, k in [([1, 1], 1, 10), ([0], 1, 10),
                                   ([True], 1, 10), ([1.5], 1, 10),
                                   ([1], False, 10), ([1], 1, 0)]:
            with self.subTest(ranking=ranking, target=target, k=k):
                with self.assertRaises(ValueError):
                    hr_at_k(ranking, target, k)


if __name__ == "__main__":
    unittest.main()
