import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from run_sasrec import (
    SASRecTrainingDataset,
    UserSequence,
    evaluate_split,
    load_sequences,
    train_model,
    write_candidates,
)
from sasrec import SASRec


class AddTextFusion(nn.Module):
    def __init__(self, text_size, hidden_size):
        super().__init__()
        self.projection = nn.Linear(text_size, hidden_size, bias=False)

    def fuse_items(self, id_vectors, text_vectors):
        return id_vectors + self.projection(text_vectors)


def tiny_sequences():
    return {
        1: UserSequence((1, 2, 3, 4), 5, 6),
        2: UserSequence((2, 4, 6, 8), 10, 12),
        3: UserSequence((3, 6, 9, 12), 15, 18),
    }


class SASRecTests(unittest.TestCase):
    def test_causal_mask_padding_and_feature_contract(self):
        torch.manual_seed(3)
        model = SASRec(
            item_num=21,
            rec_hidden_size=8,
            max_seq_len=4,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
        ).eval()
        first = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))
        changed_future = model(torch.tensor([[0, 1, 7, 8]], dtype=torch.long))
        torch.testing.assert_close(first[:, 1], changed_future[:, 1])
        torch.testing.assert_close(first[:, 0], torch.zeros_like(first[:, 0]))

        features = torch.randn(21, 5)
        features[0].zero_()
        fused = SASRec(
            item_num=21,
            rec_hidden_size=8,
            max_seq_len=4,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
            movie_features=features,
            fusion_module=AddTextFusion(5, 8),
        )
        self.assertFalse(fused.movie_features.requires_grad)
        self.assertEqual(fused(torch.tensor([[0, 1, 2, 3]])).shape, (1, 4, 8))
        self.assertIn("fusion_module.projection.weight", fused.state_dict())
        self.assertNotIn("movie_features", fused.state_dict())

    def test_negative_sampling_uses_only_valid_unobserved_ids(self):
        data = SASRecTrainingDataset(
            tiny_sequences(), list(range(1, 21)), max_seq_len=3, seed=11
        )
        for index, (_, _, positives, observed) in enumerate(data.rows):
            negatives = data.negatives[index]
            self.assertTrue(torch.equal(negatives.eq(0), positives.eq(0)))
            for negative in negatives[negatives.ne(0)].tolist():
                self.assertNotIn(negative, observed)
                self.assertIn(negative, data.valid_movie_ids)

    def test_sequence_contract_and_candidate_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sequences.json"
            source.write_text(
                json.dumps({
                    "1": {"train": [1, 2], "validation": 3, "test": 4},
                    "2": {"train": [2, 3], "validation": 4, "test": 5},
                }),
                encoding="utf-8",
            )
            loaded = load_sequences(source, set(range(1, 21)))
            self.assertEqual(loaded[1].validation, 3)

            model = SASRec(
                item_num=21,
                rec_hidden_size=8,
                max_seq_len=3,
                num_blocks=1,
                num_heads=2,
                dropout=0.0,
            )
            _, validation = evaluate_split(
                model, loaded, list(range(1, 21)), "validation",
                torch.device("cpu"), batch_size=2,
            )
            _, test = evaluate_split(
                model, loaded, list(range(1, 21)), "test",
                torch.device("cpu"), batch_size=2,
            )
            self.assertEqual(len(validation["1"]), 10)
            self.assertTrue(set(validation["1"]).isdisjoint({1, 2}))
            self.assertTrue(set(test["1"]).isdisjoint({1, 2, 3}))

            output = Path(directory) / "candidates.json"
            write_candidates(output, validation, test)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"validation", "test"})
            self.assertEqual(payload["validation"]["1"], validation["1"])

    def test_tiny_training_loop_runs_end_to_end(self):
        torch.manual_seed(7)
        sequences = tiny_sequences()
        movie_ids = list(range(1, 21))
        data = SASRecTrainingDataset(sequences, movie_ids, max_seq_len=3, seed=7)
        model = SASRec(
            item_num=21,
            rec_hidden_size=8,
            max_seq_len=3,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
        )
        validation, test = train_model(
            model=model,
            training_data=data,
            sequences=sequences,
            valid_movie_ids=movie_ids,
            device=torch.device("cpu"),
            epochs=1,
            batch_size=3,
            eval_batch_size=3,
            learning_rate=1e-3,
            weight_decay=0.0,
            patience=1,
            seed=7,
        )
        for result in (validation, test):
            self.assertGreaterEqual(result["hr10"], 0.0)
            self.assertLessEqual(result["hr10"], 1.0)
            self.assertGreaterEqual(result["ndcg10"], 0.0)
            self.assertLessEqual(result["ndcg10"], 1.0)


if __name__ == "__main__":
    unittest.main()
