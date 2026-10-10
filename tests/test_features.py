import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from device import get_device, get_dtype
import torch
from features import masked_mean_pool, read_movies


class FeatureContractTests(unittest.TestCase):
    def test_device_priority_and_precision(self):
        for cuda, mps, expected in [(True, True, "cuda"), (False, True, "mps"),
                                    (False, False, "cpu")]:
            with patch("torch.cuda.is_available", return_value=cuda), \
                    patch("torch.backends.mps.is_available", return_value=mps):
                self.assertEqual(get_device().type, expected)
        for supported, expected in [(True, torch.bfloat16), (False, torch.float16)]:
            with patch("torch.cuda.is_bf16_supported", return_value=supported):
                self.assertEqual(get_dtype(torch.device("cuda")), expected)
        self.assertEqual(get_dtype(torch.device("mps")), torch.float16)
        self.assertEqual(get_dtype(torch.device("cpu")), torch.float32)

    def test_pooling_ignores_padding_and_accumulates_float32(self):
        hidden = torch.tensor([[[60000., 2.], [60000., 4.], [-60000., 100.]]],
                              dtype=torch.float16)
        result = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor([[60000., 3.]]))
        with self.assertRaises(ValueError):
            masked_mean_pool(hidden, torch.zeros(1, 3))

    def test_original_ids_and_readable_movie_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "movies.dat"
            path.write_text("9::Amélie (2001)::Comedy|Romance\n2::Toy (1995)::Animation\n",
                            encoding="latin-1")
            records = read_movies(path)
            self.assertEqual([r[0] for r in records], [2, 9])
            self.assertEqual(records[1][1], "Title: Amélie (2001). Genres: Comedy, Romance.")
            path.write_text("1::A::Comedy\n1::B::Comedy\n")
            with self.assertRaises(ValueError):
                read_movies(path)


if __name__ == "__main__":
    unittest.main()
