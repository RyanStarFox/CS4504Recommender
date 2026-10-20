"""Run A's three baselines; only validation chooses training epochs."""

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np

from device import get_device
import torch

from dataloader import EvalDataLoader, TrainDataLoader
from dataset import ML1MDataset
from metrics import evaluate_rankings
from mf import fit as fit_mf
from recommender import HybridRecommender
from split import read_ids
from trainer import Trainer


FIELDS = ("model", "hr10", "ndcg10", "split")


def record(path, model, metrics, split):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected columns in {path}")
            rows = [row for row in reader if (row["model"], row["split"]) != (model, split)]
    rows.append({"model": model, "hr10": f"{metrics['hr10']:.8f}",
                 "ndcg10": f"{metrics['ndcg10']:.8f}", "split": split})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_ncf(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = ML1MDataset(args.data_path)
    train = dataset.get_split_data("train")
    validation = dataset.get_split_data("validation")
    test = dataset.get_split_data("test")
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    if features.shape[0] != dataset.get_item_num() or features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("GPT-2 feature tensor has invalid shape or values")
    model = HybridRecommender(dataset.get_user_num(), dataset.get_item_num(), features,
                              embed_dim=args.ncf_dim)
    train_loader = TrainDataLoader(train, batch_size=args.ncf_batch_size, device=device)
    valid_loader = EvalDataLoader(validation, train, batch_size=args.ncf_eval_batch_size, device=device)
    test_loader = EvalDataLoader(test, train, batch_size=args.ncf_eval_batch_size, device=device)
    trainer = Trainer(model, train_loader, valid_loader, test_loader, device=device,
                      epochs=args.ncf_epochs, lr=args.ncf_lr)
    best_validation, test_result = trainer.fit(save_model=False)
    print(f"NCF validation={best_validation} test={test_result}", flush=True)
    return {"hr10": test_result["HR@10"], "ndcg10": test_result["NDCG@10"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("all", "popularity", "mf", "ncf"), default="all")
    parser.add_argument("--data-path", default="ml-1m")
    parser.add_argument("--sequences", default="data/sequences.json")
    parser.add_argument("--mock-candidates", default="data/mock_candidates.json")
    parser.add_argument("--features", default="data/gpt2_movie_features.pt")
    parser.add_argument("--results", default="data/results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mf-epochs", type=int, default=10)
    parser.add_argument("--mf-dim", type=int, default=64)
    parser.add_argument("--mf-batch-size", type=int, default=8192)
    parser.add_argument("--mf-lr", type=float, default=0.005)
    parser.add_argument("--mf-reg", type=float, default=1e-5)
    parser.add_argument("--ncf-epochs", type=int, default=3)
    parser.add_argument("--ncf-dim", type=int, default=32)
    parser.add_argument("--ncf-batch-size", type=int, default=2048)
    parser.add_argument("--ncf-eval-batch-size", type=int, default=32)
    parser.add_argument("--ncf-lr", type=float, default=0.001)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)
    sequences = json.loads(Path(args.sequences).read_text())
    selected = ("popularity", "mf", "ncf") if args.models == "all" else (args.models,)
    if "popularity" in selected:
        candidates = json.loads(Path(args.mock_candidates).read_text())
        targets = {user: row["test"] for user, row in sequences.items()}
        result = evaluate_rankings(candidates["test"], targets)
        print(f"Popularity test={result}", flush=True)
        record(args.results, "popularity", result, "temporal")
    if "mf" in selected:
        valid_items = read_ids(Path(args.data_path) / "movies.dat")
        best_epoch, validation, result = fit_mf(
            sequences, valid_items, device, epochs=args.mf_epochs,
            embed_dim=args.mf_dim, batch_size=args.mf_batch_size,
            lr=args.mf_lr, reg=args.mf_reg, seed=args.seed)
        print(f"BPR-MF best_epoch={best_epoch} validation={validation} test={result}", flush=True)
        record(args.results, "bpr_mf", result, "temporal")
    if "ncf" in selected:
        result = run_ncf(args, device)
        record(args.results, "ncf_gpt2", result, "random")


if __name__ == "__main__":
    main()
