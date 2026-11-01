import copy
import unittest

import torch

from run_sasrec import SASRecTrainingDataset, UserSequence, train_model
from run_text_sasrec import build_model
from text_training import batch_alignment_loss, train_aligned


class AlignmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.sequences = {
            1: UserSequence((1, 2, 3, 4), 5, 6),
            2: UserSequence((2, 4, 6, 8), 10, 12),
            3: UserSequence((3, 6, 9, 12), 15, 18),
        }
        self.movies = list(range(1, 21))
        torch.manual_seed(42)
        features = torch.randn(21, 5)
        features[0].zero_()
        self.model = build_model(
            dict(
                item_num=21,
                rec_hidden_size=8,
                max_seq_len=3,
                num_blocks=1,
                num_heads=2,
                dropout=0.2,
            ),
            features,
            42,
        )
        self.options = dict(weight=0.01, temperature=0.07, max_items=4)

    def test_zero_weight_matches_d_training_and_rng_exactly(self):
        snapshots = []
        for aligned in [False, True]:
            torch.manual_seed(91)
            model = copy.deepcopy(self.model)
            data = SASRecTrainingDataset(self.sequences, self.movies, 3, 42)
            kwargs = dict(
                epochs=2,
                batch_size=2,
                eval_batch_size=3,
                learning_rate=0.001,
                weight_decay=0.0,
                patience=5,
                seed=42,
            )
            if aligned:
                result = train_aligned(
                    model,
                    data,
                    self.sequences,
                    self.movies,
                    torch.device("cpu"),
                    alignment=dict(self.options, weight=0),
                    **kwargs,
                )[:2]
            else:
                result = train_model(
                    model,
                    data,
                    self.sequences,
                    self.movies,
                    torch.device("cpu"),
                    **kwargs,
                )
            snapshots.append((result, model.state_dict(), torch.get_rng_state()))
        left, right = snapshots
        self.assertEqual(left[0], right[0])
        for key in left[1]:
            torch.testing.assert_close(left[1][key], right[1][key], rtol=0, atol=0)
        torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)

    def test_alignment_update_has_finite_gradients_and_leaves_cache_frozen(self):
        valid = torch.ones(21, dtype=torch.bool)
        valid[0] = False
        before_rng = torch.get_rng_state().clone()
        loss = batch_alignment_loss(
            self.model,
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[0, 2, 3, 4]]),
            valid,
            torch.Generator().manual_seed(42),
            self.options,
        )
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in [
            self.model.item_embedding.weight,
            self.model.fusion_module.projection.weight,
        ]:
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertIsNone(self.model.movie_features.grad)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
