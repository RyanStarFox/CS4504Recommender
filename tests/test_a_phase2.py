"""Check that temporal evaluation excludes the correct observed history."""

import unittest

import torch

from mf import BPRMF, evaluate


class TemporalMFEvaluationTests(unittest.TestCase):
    def test_test_history_includes_validation_item(self):
        model = BPRMF(2, 13, embed_dim=1)
        with torch.no_grad():
            model.user_embedding.weight.fill_(1)
            model.item_embedding.weight[:, 0] = torch.arange(13, 0, -1)
        sequences = {"1": {"train": [1], "validation": 2, "test": 12}}
        movie_ids = set(range(1, 13))
        self.assertEqual(evaluate(model, sequences, movie_ids, "validation", "cpu")["hr10"], 1.0)
        self.assertEqual(evaluate(model, sequences, movie_ids, "test", "cpu")["hr10"], 1.0)


if __name__ == "__main__":
    unittest.main()
