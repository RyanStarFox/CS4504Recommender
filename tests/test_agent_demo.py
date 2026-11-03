import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from agent_demo import (
    DEFAULT_ADAPTER_REVISION,
    Movie,
    download_adapter,
    extract_excluded_genres,
    filtered_recall,
    genre_filter,
    print_demo,
    run_demo,
)
from sasrec import SASRec


class AgentDemoTests(unittest.TestCase):
    def test_cached_readme_adapter_metadata_is_validated_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "grpo"
            adapter.mkdir()
            adapter.joinpath("adapter.json").write_text(json.dumps({
                "format_version": 1,
                "stage": "grpo",
                "model_id": "Qwen/Qwen3.5-0.8B",
                "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            }), encoding="utf-8")
            adapter.joinpath("adapter.pt").write_bytes(b"cached")
            self.assertEqual(
                download_adapter(
                    "grpo", repo_id="unused", revision=DEFAULT_ADAPTER_REVISION,
                    destination=directory,
                ),
                adapter,
            )

    def test_extracts_only_negated_genres(self):
        request = "我不喜欢恐怖片，但喜欢喜剧和科幻片"
        self.assertEqual(extract_excluded_genres(request), {"Horror"})
        self.assertEqual(
            extract_excluded_genres("No horror, without thriller; comedy is fine"),
            {"Horror", "Thriller"},
        )

    def test_genre_filter_and_recall_remove_genres_and_history(self):
        movies = {
            1: Movie(1, "A", ("Comedy",)),
            2: Movie(2, "B", ("Horror",)),
            3: Movie(3, "C", ("Drama",)),
            4: Movie(4, "D", ("Horror", "Comedy")),
            5: Movie(5, "E", ("Action",)),
        }
        eligible = genre_filter([1, 2, 3, 4, 5], movies, {"Horror"})
        self.assertEqual(eligible, [1, 3, 5])
        model = SASRec(
            item_num=6, rec_hidden_size=4, max_seq_len=2,
            num_blocks=1, num_heads=1, dropout=0,
        ).eval()
        recalled = filtered_recall(
            model, history=[1], eligible_movie_ids=eligible,
            device=torch.device("cpu"), topk=2,
        )
        self.assertEqual(set(recalled), {3, 5})

    def test_tracked_artifacts_only_path_falls_back_to_sasrec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            movies = root / "movies.dat"
            rows = [
                "1::Seen (2000)::Comedy",
                "2::Scary (2000)::Horror",
                "3::Drama A (2000)::Drama",
                "4::Action A (2000)::Action",
                "5::Drama B (2000)::Drama",
                "6::Action B (2000)::Action",
            ]
            movies.write_text("\n".join(rows) + "\n", encoding="latin-1")
            sequences = root / "sequences.json"
            sequences.write_text(json.dumps({
                "1": {"train": [1, 3], "validation": 4, "test": 5}
            }), encoding="utf-8")

            torch.manual_seed(1)
            model = SASRec(
                item_num=7, rec_hidden_size=4, max_seq_len=2,
                num_blocks=1, num_heads=1, dropout=0,
            )
            checkpoint = root / "sasrec.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {
                    "item_num": 7, "rec_hidden_size": 4, "max_seq_len": 2,
                    "num_blocks": 1, "num_heads": 1, "dropout": 0,
                },
                "valid_movie_ids": list(range(1, 7)),
                "validation_metrics": {"hr10": 0.1, "ndcg10": 0.05},
                "test_metrics": {"hr10": 0.1, "ndcg10": 0.05},
            }, checkpoint)
            args = Namespace(
                movies=str(movies), sequences=str(sequences), checkpoint=str(checkpoint),
                user_id=1, request="我不喜欢恐怖片", exclude_genre=[],
                history_split="validation", top_k=3, device="cpu",
                gguf=str(root / "missing.gguf"), tokenizer=str(root / "tokenizer"),
                llama_bin="missing-llama", require_reranker=False,
                reranker_backend="none", reranker_stage="grpo",
                adapter_repo="unused", adapter_revision="main",
                adapter_root=str(root / "adapters"),
                max_history=10, history_limit=10, max_new_tokens=32, context=2048,
            )
            result = run_demo(args)

            self.assertEqual(result["filter"]["excluded_genres"], ["Horror"])
            self.assertFalse(result["reranker"]["used"])
            final_ids = [row["movie_id"] for row in result["final_recommendations"]]
            self.assertNotIn(2, final_ids)
            self.assertTrue(set(final_ids).isdisjoint({1, 3}))
            self.assertEqual(
                final_ids,
                [row["movie_id"] for row in result["sasrec"]["recommendations"]],
            )
            self.assertEqual(
                [row["movie_id"] for row in result["watched_history"]["movies"]],
                [1, 3],
            )
            self.assertEqual(result["held_out_evaluation"]["target"]["movie_id"], 4)
            self.assertTrue(result["held_out_evaluation"]["hit_at_k"])

            output = StringIO()
            with redirect_stdout(output):
                print_demo(result)
            rendered = output.getvalue()
            self.assertIn("已观看历史", rendered)
            self.assertIn("Seen (2000)", rendered)
            self.assertIn("留出的真实下一部观看", rendered)
            self.assertIn("Action A (2000)", rendered)
            self.assertIn("Top-K 命中", rendered)


if __name__ == "__main__":
    unittest.main()
