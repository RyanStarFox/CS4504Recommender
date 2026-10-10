"""Course-compatible entry point for frozen GPT-2 movie features."""
import argparse
from pathlib import Path

from features import ROOT, extract_movie_features as _extract


def choose_model_source(model_path=None):
    return str(Path(model_path or ROOT / "models/gpt2")), True


def extract_movie_features(data_path, output_path, batch_size=32, model_path=None):
    return _extract(encoder="gpt2", data_path=data_path, output=output_path,
                    batch_size=batch_size, model_path=model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 MovieLens 电影的 GPT-2 特征")
    parser.add_argument("--data_path", default="ml-1m")
    parser.add_argument("--output", default="movie_features.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()
    extract_movie_features(args.data_path, args.output, args.batch_size, args.model_path)
