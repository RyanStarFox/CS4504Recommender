import io
import unittest

import torch
from torch.nn import functional as F

from fusion import GatedMovieFusion, alignment_loss
from run_text_sasrec import build_model, load_features
from run_sasrec import read_movie_ids


class FusionTests(unittest.TestCase):
    def test_shapes_zero_rows_switch_and_gradients(self):
        fusion = GatedMovieFusion(5, 8)
        ids = torch.randn(2, 3, 8, requires_grad=True)
        text = torch.randn(2, 3, 5, requires_grad=True)
        with torch.no_grad():
            text[0, 0].zero_()
            ids[0, 0].zero_()
            text[1, 1].zero_()
        result = fusion.fuse_items(ids, text)
        self.assertEqual(result.shape, ids.shape)
        torch.testing.assert_close(result[0, 0], torch.zeros(8))
        torch.testing.assert_close(result[1, 1], ids[1, 1], rtol=0, atol=0)
        result.square().sum().backward()
        for parameter in fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(text.grad)
        self.assertGreater(float(ids.grad.abs().sum()), 0)
        fusion.use_movie_text = False
        torch.testing.assert_close(fusion.fuse_items(ids, text), ids, rtol=0, atol=0)

    def test_sasrec_registration_scoring_and_checkpoint(self):
        features = torch.randn(12, 5, requires_grad=True)
        with torch.no_grad():
            features[0].zero_()
            features[7].zero_()
        config = dict(
            item_num=12,
            rec_hidden_size=8,
            max_seq_len=4,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
        )
        model = build_model(config, features, 42)
        history = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]])
        candidates = torch.tensor([1, 2, 3, 4])
        model.score_candidates(history, candidates).square().sum().backward()
        self.assertIsNone(features.grad)
        self.assertFalse(model.movie_features.requires_grad)
        self.assertGreater(
            float(model.fusion_module.projection.weight.grad.abs().sum()), 0
        )
        self.assertGreater(float(model.fusion_module.gate.weight.grad.abs().sum()), 0)
        state = model.state_dict()
        self.assertNotIn("movie_features", state)
        self.assertIn("fusion_module.gate.weight", state)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = build_model(config, features, 42)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model.score_candidates(history, candidates),
                restored.score_candidates(history, candidates),
                rtol=0,
                atol=0,
            )
            users = model.sequence_vectors(history)
            expected = users @ model.item_vectors(candidates).T
            torch.testing.assert_close(
                model.score_candidates(history, candidates), expected
            )

    def test_encoder_dimensions_do_not_change_id_initialization_or_rng(self):
        config = dict(
            item_num=12,
            rec_hidden_size=8,
            max_seq_len=4,
            num_blocks=1,
            num_heads=2,
            dropout=0.2,
        )
        snapshots = []
        for width in [5, 9]:
            torch.manual_seed(42)
            model = build_model(config, torch.ones(12, width), 42)
            state = {
                key: value.clone()
                for key, value in model.state_dict().items()
                if not key.startswith("fusion_module.")
            }
            snapshots.append((state, torch.get_rng_state()))
        left, right = snapshots
        for key in left[0]:
            torch.testing.assert_close(left[0][key], right[0][key], rtol=0, atol=0)
        torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)

    def test_alignment_excludes_padding_gaps_and_duplicates(self):
        ids = torch.tensor([0, 1, 1, 2, 3])
        valid = torch.tensor([False, True, False, True])
        left = torch.randn(5, 8, requires_grad=True)
        right = torch.randn(5, 8, requires_grad=True)
        loss = alignment_loss(left, right, ids, valid, temperature=0.2)
        rows = torch.tensor([1, 4])
        logits = (
            F.normalize(left[rows], dim=-1) @ F.normalize(right[rows], dim=-1).T / 0.2
        )
        labels = torch.arange(2)
        expected = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertEqual(float(left.grad[[0, 2, 3]].abs().sum()), 0)
        self.assertGreater(float(right.grad[rows].abs().sum()), 0)
        with self.assertRaises(ValueError):
            alignment_loss(left, right, ids, valid, temperature=0)

    def test_empty_and_single_movie_alignment_have_finite_zero_gradient(self):
        for ids in [torch.tensor([0, 0]), torch.tensor([1, 1])]:
            left = torch.randn(2, 8, requires_grad=True)
            right = torch.randn(2, 8, requires_grad=True)
            loss = alignment_loss(left, right, ids, torch.tensor([False, True]))
            self.assertEqual(float(loss), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(left.grad).all())

    def test_real_encoders_integrate_with_shared_sasrec(self):
        movie_ids = read_movie_ids("ml-1m/movies.dat")
        for encoder in ["gpt2", "qwen"]:
            features, _ = load_features(f"data/{encoder}_movie_features.pt", movie_ids)
            config = dict(
                item_num=len(features),
                rec_hidden_size=8,
                max_seq_len=4,
                num_blocks=1,
                num_heads=2,
                dropout=0.0,
            )
            model = build_model(config, features, 42)
            scores = model.score_candidates(
                torch.tensor([[0, 1, 2, 3]]), torch.tensor(movie_ids[:10])
            )
            self.assertTrue(torch.isfinite(scores).all())
            scores.square().sum().backward()
            self.assertGreater(
                float(model.fusion_module.projection.weight.grad.abs().sum()), 0
            )


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
