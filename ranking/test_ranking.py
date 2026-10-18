"""Contract tests for the ranking data, losses and llama.cpp sampler boundary."""
import json
import math
import stat
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ranking.batching import collate
from ranking.data import dpo_pairs, held_out, load_bundle, load_sample_file, write_json
from ranking.lora import LoRALinear, apply_lora, lora_fingerprint
from ranking.objective import dpo_loss, grpo_loss, sft_loss
from ranking.prompts import assert_non_thinking, place_first, ranking_text, user_prompt
from ranking.reward import group_advantages, hr_at_k, ndcg_at_k, parse_ranking
from ranking.sample import llama_completion
from ranking.train import _write_fixture


class RankingContractTest(unittest.TestCase):
    def test_ndcg_matches_course_formula(self):
        self.assertAlmostEqual(ndcg_at_k([5, 1, 2], 5), 1.0)
        self.assertAlmostEqual(ndcg_at_k([1, 5, 2], 5), 1 / math.log2(3))
        self.assertEqual(ndcg_at_k([1, 2, 3], 5), 0.0)
        self.assertEqual(hr_at_k([1, 2, 5], 5), 1.0)
        self.assertEqual(hr_at_k([1, 2, 3], 5), 0.0)

    def test_parse_ranking_keeps_candidate_order_of_appearance(self):
        parsed = parse_ranking("year 1995 then 1 and 6 again 1", (6, 1, 2))
        self.assertEqual(parsed, (1, 6, 2))

    def test_advantages_are_centered(self):
        advantages = group_advantages([1.0, 0.0])
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertGreater(advantages[0], 0)
        self.assertEqual(group_advantages([0.2, 0.2]), [0.0, 0.0])

    def test_prompt_and_preference_order(self):
        example = _example()
        titles = {1: "Title: A. Genres: Comedy.", 2: "Title: B. Genres: Drama.", 3: "Title: C. Genres: Action."}
        prompt = user_prompt(example, titles)
        self.assertIn("看过：", prompt)
        self.assertIn("（1）", prompt)
        self.assertIn("（3）", prompt)
        self.assertNotIn("（9）", prompt)
        chosen = place_first(example.candidates, example.ground_truth)
        rejected = place_first(example.candidates, 1)
        self.assertEqual(chosen[0], 3)
        self.assertEqual(rejected[0], 1)
        self.assertEqual(set(chosen), set(example.candidates))
        self.assertEqual(ranking_text(chosen), "3 2 1")
        assert_non_thinking("answer\n</think>\n\n")
        with self.assertRaises(RuntimeError):
            assert_non_thinking("<think>\nstill thinking")

    def test_loader_keeps_only_hits_and_holds_out_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_raw(root)
            bundle = load_bundle(
                root / "sequences.json", root / "candidates.json", root / "movies.dat",
                candidate_k=3, max_history=2, val_ratio=0, seed=1)
            user1 = next(row for row in bundle.train if row.user_id == "1")
            self.assertEqual(user1.history, (2, 3))
            self.assertEqual(user1.ground_truth, 4)
            self.assertEqual([row.ground_truth for row in bundle.test], [5])
            self.assertFalse(any(row.split == "test" for row in bundle.train))
            pairs = dpo_pairs([user1], max_rejected=2)
            self.assertEqual([rejected for _, rejected in pairs], [2, 1])
            seed = next(
                item for item in range(30)
                if held_out("1", 0.5, item) != held_out("2", 0.5, item))
            held = load_bundle(
                root / "sequences.json", root / "candidates.json", root / "movies.dat",
                candidate_k=3, val_ratio=0.5, seed=seed)
            self.assertEqual(len(held.train), 1)
            self.assertEqual(len(held.select), 1)
            with self.assertRaises(ValueError):
                load_bundle(
                    root / "sequences.json", root / "candidates.json", root / "movies.dat",
                    candidate_k=3, val_ratio=1, seed=1)

    def test_mock_and_formal_files_use_one_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_raw(root)
            formal = root / "candidates.json"
            mock = root / "mock_candidates.json"
            mock.write_text(formal.read_text(encoding="utf-8"), encoding="utf-8")
            left = load_bundle(root / "sequences.json", formal, root / "movies.dat", candidate_k=3, val_ratio=0)
            right = load_bundle(root / "sequences.json", mock, root / "movies.dat", candidate_k=3, val_ratio=0)
            self.assertEqual(left.train, right.train)

    def test_smoke_fixture_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            movies = root / "movies.dat"
            lines = [
                f"{movie_id}::Movie {movie_id} (1995)::Comedy"
                for movie_id in range(1, 21)
            ]
            movies.write_text("\n".join(lines) + "\n", encoding="latin-1")
            _write_fixture(root, movies)
            bundle = load_bundle(
                root / "sequences.json", root / "candidates.json", movies,
                candidate_k=10, val_ratio=0, max_history=5)
            self.assertEqual(len(bundle.train), 2)
            self.assertTrue(all(row.ground_truth in row.candidates for row in bundle.train + bundle.test))

    def test_sample_file_rejects_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            write_json(path, {
                "format_version": 1,
                "split": "test",
                "groups": [{
                    "user_id": "1",
                    "prompt": "rank",
                    "ground_truth": 3,
                    "candidates": [3, 2, 1],
                    "completions": ["3 2 1", "1 2 3"],
                }],
            })
            with self.assertRaises(ValueError):
                load_sample_file(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["split"] = "validation"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, groups = load_sample_file(path)
            self.assertEqual(groups[0]["ground_truth"], 3)

    def test_losses_and_lora_step(self):
        tokens = torch.tensor([[-1.0, -1.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        self.assertAlmostEqual(float(sft_loss(tokens, mask)), 1.0)
        chosen = torch.tensor([0.0, 2.0])
        rejected = torch.tensor([0.0, 0.0])
        close = dpo_loss(chosen[:1], rejected[:1], chosen[:1], rejected[:1], beta=1.0)
        better = dpo_loss(chosen[1:], rejected[1:], chosen[:1], rejected[:1], beta=1.0)
        self.assertAlmostEqual(float(close), -math.log(0.5))
        self.assertLess(float(better), float(close))
        policy = torch.zeros(2, 1)
        reference = torch.zeros(2, 1)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 1, dtype=torch.bool)
        loss = grpo_loss(policy, reference, mask, advantages, beta=0.0)
        self.assertAlmostEqual(float(loss), 0.0)

        linear = nn.Linear(4, 3, bias=False)
        wrapped = LoRALinear(linear, rank=2, alpha=4, dropout=0)
        inputs = torch.randn(2, 4)
        self.assertTrue(torch.allclose(wrapped(inputs), linear(inputs)))
        wrapped(inputs).sum().backward()
        self.assertGreater(float(wrapped.lora_B.grad.abs().sum()), 0)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)

        model = Tiny()
        apply_lora(model, rank=2, alpha=4, dropout=0)
        before = lora_fingerprint(model).clone()
        model.q_proj(torch.randn(2, 4)).sum().backward()
        model.q_proj.lora_B.data.add_(model.q_proj.lora_B.grad)
        self.assertFalse(torch.equal(before, lora_fingerprint(model)))

    def test_collate_pads_on_the_right(self):
        batch = collate([([1, 2], [0, 1]), ([3], [1])], pad_id=0)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["completion_mask"].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_held_out_is_stable(self):
        self.assertFalse(held_out("7", 0, seed=1))
        self.assertTrue(held_out("7", 1, seed=1))
        self.assertEqual(held_out("7", 0.5, seed=3), held_out("7", 0.5, seed=3))

    def test_llama_backend_reports_a_missing_binary_and_strips_the_prompt(self):
        with self.assertRaises(FileNotFoundError):
            llama_completion("llama-cli-missing", "model.gguf", "prompt", 8, 0.8, 128)
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "llama-cli"
            binary.write_text("#!/bin/sh\nprintf '%s' \"prompt 4 5\"\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
            self.assertEqual(llama_completion(binary, "model.gguf", "prompt", 8, 0.8, 128), "4 5")


def _example():
    from ranking.data import Example
    return Example("1", "validation", (1, 2), (3, 2, 1), 3)


def _write_raw(root):
    movies = "\n".join([
        "1::Toy Story (1995)::Animation|Children's|Comedy",
        "2::Jumanji (1995)::Adventure|Children's|Fantasy",
        "3::Grumpier Old Men (1995)::Comedy|Romance",
        "4::Waiting to Exhale (1995)::Comedy|Drama",
        "5::Father of the Bride Part II (1995)::Comedy",
    ]) + "\n"
    (root / "movies.dat").write_text(movies, encoding="latin-1")
    write_json(root / "sequences.json", {
        "1": {"train": [1, 2, 3], "validation": 4, "test": 5},
        "2": {"train": [1, 2, 3], "validation": 5, "test": 4},
    })
    write_json(root / "candidates.json", {
        "validation": {"1": [4, 2, 1], "2": [5, 1, 2]},
        "test": {"1": [5, 1, 2], "2": [1, 2, 3]},
    })


if __name__ == "__main__":
    unittest.main()
