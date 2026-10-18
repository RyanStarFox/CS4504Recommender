"""Train pure-ID SASRec on A's temporal split and export ordered Top-10.

Expected input: ``data/sequences.json`` as specified in
``docs/division-of-labor.md``.  Validation uses the training history; test uses
training history plus the validation interaction.  Seen movies are excluded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from device import get_device
from sasrec import SASRec


@dataclass(frozen=True)
class UserSequence:
    train: tuple[int, ...]
    validation: int
    test: int


def read_movie_ids(path: str | Path) -> list[int]:
    """Read valid original IDs without treating gaps as real movies."""
    movie_ids = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="latin-1").splitlines(), start=1
    ):
        try:
            movie_id = int(line.split("::", 1)[0])
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid movie row at line {line_number}") from error
        movie_ids.append(movie_id)
    if not movie_ids or min(movie_ids) <= 0 or len(movie_ids) != len(set(movie_ids)):
        raise ValueError("Movie IDs must be unique positive integers")
    return sorted(movie_ids)


def load_sequences(
    path: str | Path, valid_movie_ids: set[int]
) -> dict[int, UserSequence]:
    """Load and strictly validate A's shared temporal-split file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("sequences.json must be a non-empty object")
    sequences: dict[int, UserSequence] = {}
    for raw_user_id, value in raw.items():
        try:
            user_id = int(raw_user_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid user ID: {raw_user_id!r}") from error
        if user_id <= 0 or str(user_id) != str(raw_user_id):
            raise ValueError(f"User ID must be a canonical positive integer: {raw_user_id!r}")
        if user_id in sequences:
            raise ValueError(f"Duplicate user ID after parsing: {user_id}")
        if not isinstance(value, dict) or set(value) != {"train", "validation", "test"}:
            raise ValueError(
                f"User {user_id} must contain exactly train, validation and test"
            )
        train = value["train"]
        if not isinstance(train, list) or len(train) < 2:
            raise ValueError(f"User {user_id} needs at least two training interactions")
        values = train + [value["validation"], value["test"]]
        if any(type(item) is not int or item <= 0 for item in values):
            raise ValueError(f"User {user_id} contains a non-positive or non-integer movie ID")
        unknown = set(values) - valid_movie_ids
        if unknown:
            raise ValueError(f"User {user_id} contains unknown movie IDs: {sorted(unknown)}")
        sequences[user_id] = UserSequence(
            train=tuple(train),
            validation=value["validation"],
            test=value["test"],
        )
    return sequences


class SASRecTrainingDataset(Dataset):
    """One recent, left-padded SASRec training sequence per user."""

    def __init__(
        self,
        sequences: dict[int, UserSequence],
        valid_movie_ids: list[int],
        max_seq_len: int,
        seed: int,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        self.valid_movie_ids = tuple(valid_movie_ids)
        self.rows: list[tuple[int, torch.Tensor, torch.Tensor, frozenset[int]]] = []
        for user_id in sorted(sequences):
            train = sequences[user_id].train[-(max_seq_len + 1):]
            inputs = torch.zeros(max_seq_len, dtype=torch.long)
            positives = torch.zeros(max_seq_len, dtype=torch.long)
            length = len(train) - 1
            inputs[-length:] = torch.tensor(train[:-1], dtype=torch.long)
            positives[-length:] = torch.tensor(train[1:], dtype=torch.long)
            self.rows.append((user_id, inputs, positives, frozenset(sequences[user_id].train)))
        self.negatives = torch.zeros(len(self.rows), max_seq_len, dtype=torch.long)
        self.resample_negatives(seed)

    def resample_negatives(self, seed: int) -> None:
        generator = random.Random(seed)
        for row_index, (_, _, positives, observed) in enumerate(self.rows):
            available = [item for item in self.valid_movie_ids if item not in observed]
            if not available:
                raise ValueError("A user has no unobserved movie available for negative sampling")
            target_indices = positives.nonzero(as_tuple=False).flatten().tolist()
            self.negatives[row_index].zero_()
            for position in target_indices:
                self.negatives[row_index, position] = generator.choice(available)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        user_id, inputs, positives, _ = self.rows[index]
        return (
            torch.tensor(user_id, dtype=torch.long),
            inputs,
            positives,
            self.negatives[index],
        )


def sampled_bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if not valid.any():
        raise ValueError("A training batch contains no next-item targets")
    return -F.logsigmoid(positive_scores[valid] - negative_scores[valid]).mean()


def pad_histories(histories: list[list[int]], max_seq_len: int) -> torch.Tensor:
    result = torch.zeros(len(histories), max_seq_len, dtype=torch.long)
    for row, history in enumerate(histories):
        if not history:
            raise ValueError("Evaluation histories cannot be empty")
        retained = history[-max_seq_len:]
        result[row, -len(retained):] = torch.tensor(retained, dtype=torch.long)
    return result


@torch.inference_mode()
def evaluate_split(
    model: SASRec,
    sequences: dict[int, UserSequence],
    valid_movie_ids: list[int],
    split: str,
    device: torch.device,
    batch_size: int,
    topk: int = 10,
) -> tuple[dict[str, float], dict[str, list[int]]]:
    """Full-sort one held-out next item per user and return ordered candidates."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if topk <= 0 or batch_size <= 0:
        raise ValueError("topk and batch_size must be positive")
    model.eval()
    candidate_ids = torch.tensor(valid_movie_ids, dtype=torch.long, device=device)
    candidate_columns = {item_id: column for column, item_id in enumerate(valid_movie_ids)}
    user_ids = sorted(sequences)
    ranks: list[int | None] = []
    recommendations: dict[str, list[int]] = {}

    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]
        histories: list[list[int]] = []
        targets: list[int] = []
        for user_id in batch_users:
            record = sequences[user_id]
            history = list(record.train)
            target = record.validation
            if split == "test":
                history.append(record.validation)
                target = record.test
            histories.append(history)
            targets.append(target)

        inputs = pad_histories(histories, model.max_seq_len).to(device)
        users = torch.tensor(batch_users, dtype=torch.long, device=device)
        scores = model.score_candidates(inputs, candidate_ids, user_ids=users)
        for row, history in enumerate(histories):
            seen_columns = [candidate_columns[item] for item in set(history)]
            scores[row, seen_columns] = -torch.inf
            available_count = len(valid_movie_ids) - len(set(history))
            if available_count < topk:
                raise ValueError(
                    f"User {batch_users[row]} has only {available_count} unseen movies"
                )

        top_columns = scores.topk(topk, dim=1).indices.cpu().tolist()
        for row, user_id in enumerate(batch_users):
            ranked_items = [valid_movie_ids[column] for column in top_columns[row]]
            recommendations[str(user_id)] = ranked_items
            try:
                ranks.append(ranked_items.index(targets[row]) + 1)
            except ValueError:
                ranks.append(None)

    hit_rate = sum(rank is not None for rank in ranks) / len(ranks)
    ndcg = sum(
        0.0 if rank is None else 1.0 / math.log2(rank + 1) for rank in ranks
    ) / len(ranks)
    return {"hr10": hit_rate, "ndcg10": ndcg}, recommendations


def train_model(
    model: SASRec,
    training_data: SASRecTrainingDataset,
    sequences: dict[int, UserSequence],
    valid_movie_ids: list[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_state = None
    best_metrics = None
    best_ndcg = -math.inf
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        training_data.resample_negatives(seed + epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        model.train()
        total_loss = 0.0
        total_targets = 0
        for users, inputs, positives, negatives in loader:
            users = users.to(device)
            inputs = inputs.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            positive_scores, negative_scores, valid = model.training_scores(
                inputs, positives, negatives, user_ids=users
            )
            loss = sampled_bpr_loss(positive_scores, negative_scores, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            target_count = int(valid.sum())
            total_loss += float(loss) * target_count
            total_targets += target_count

        valid_metrics, _ = evaluate_split(
            model, sequences, valid_movie_ids, "validation", device,
            eval_batch_size,
        )
        print(
            f"Epoch {epoch:03d}: loss={total_loss / total_targets:.4f}, "
            f"validation HR@10={valid_metrics['hr10']:.4f}, "
            f"NDCG@10={valid_metrics['ndcg10']:.4f}",
            flush=True,
        )
        if valid_metrics["ndcg10"] > best_ndcg:
            best_ndcg = valid_metrics["ndcg10"]
            best_metrics = valid_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    test_metrics, _ = evaluate_split(
        model, sequences, valid_movie_ids, "test", device, eval_batch_size
    )
    return best_metrics, test_metrics


def write_candidates(
    path: str | Path,
    validation: dict[str, list[int]],
    test: dict[str, list[int]],
) -> None:
    if set(validation) != set(test):
        raise ValueError("Validation and test candidates cover different users")
    payload = {"validation": validation, "test": test}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def upsert_result(
    path: str | Path, model_name: str, metrics: dict[str, float]
) -> None:
    """Add or replace this model's temporal test result without touching others."""
    path = Path(path)
    fieldnames = ["model", "hr10", "ndcg10", "split"]
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fieldnames:
                raise ValueError(f"Unexpected results.csv columns: {reader.fieldnames}")
            rows = list(reader)
    replacement = {
        "model": model_name,
        "hr10": f"{metrics['hr10']:.8f}",
        "ndcg10": f"{metrics['ndcg10']:.8f}",
        "split": "temporal",
    }
    rows = [
        row for row in rows
        if not (row["model"] == model_name and row["split"] == "temporal")
    ]
    rows.append(replacement)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", default="data/sequences.json")
    parser.add_argument("--movies", default="ml-1m/movies.dat")
    parser.add_argument("--candidates", default="data/candidates.json")
    parser.add_argument("--results", default="data/results.csv")
    parser.add_argument("--checkpoint", default="checkpoints/sasrec_id.pt")
    parser.add_argument("--model-name", default="sasrec_id")
    parser.add_argument("--rec-hidden-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.patience <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer or early-stopping settings")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    valid_movie_ids = read_movie_ids(args.movies)
    sequences = load_sequences(args.sequences, set(valid_movie_ids))
    item_num = max(valid_movie_ids) + 1
    training_data = SASRecTrainingDataset(
        sequences, valid_movie_ids, args.max_seq_len, args.seed
    )
    model = SASRec(
        item_num=item_num,
        rec_hidden_size=args.rec_hidden_size,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        movie_features=None,
        user_features=None,
    )
    print(
        f"Training pure-ID SASRec on {device}: {len(sequences)} users, "
        f"{len(valid_movie_ids)} valid movies",
        flush=True,
    )
    valid_metrics, test_metrics = train_model(
        model=model,
        training_data=training_data,
        sequences=sequences,
        valid_movie_ids=valid_movie_ids,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "item_num": item_num,
                "rec_hidden_size": args.rec_hidden_size,
                "max_seq_len": args.max_seq_len,
                "num_blocks": args.num_blocks,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
            },
            "valid_movie_ids": valid_movie_ids,
            "seed": args.seed,
            "validation_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    _, validation_candidates = evaluate_split(
        model, sequences, valid_movie_ids, "validation", device,
        args.eval_batch_size,
    )
    _, test_candidates = evaluate_split(
        model, sequences, valid_movie_ids, "test", device,
        args.eval_batch_size,
    )
    write_candidates(args.candidates, validation_candidates, test_candidates)
    upsert_result(args.results, args.model_name, test_metrics)
    print(
        f"Best validation: HR@10={valid_metrics['hr10']:.4f}, "
        f"NDCG@10={valid_metrics['ndcg10']:.4f}\n"
        f"Test: HR@10={test_metrics['hr10']:.4f}, "
        f"NDCG@10={test_metrics['ndcg10']:.4f}\n"
        f"Saved {checkpoint_path} and {args.candidates}",
        flush=True,
    )


if __name__ == "__main__":
    main()
