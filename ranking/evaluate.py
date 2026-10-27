"""Test-set reranking for results.csv rows 7-9.

llama.cpp produces one ranking per user. Users whose ground truth sits
outside the Top-10 stay in the average, so HR@10 matches the recall list
and only NDCG can move.
"""
import csv
import json
from pathlib import Path

from metrics import evaluate_rankings
from ranking.reward import parse_ranking

FIELDS = ("model", "hr10", "ndcg10", "split")


def rankings_from_completions(examples, completions):
    if len(examples) != len(completions):
        raise ValueError("Each eval user needs one completion")
    rankings = {}
    truths = {}
    for example, text in zip(examples, completions):
        if example.user_id in rankings:
            raise ValueError(f"Duplicate eval user {example.user_id}")
        rankings[example.user_id] = parse_ranking(text, example.candidates)
        truths[example.user_id] = example.ground_truth
    return evaluate_rankings(rankings, truths)


def write_result(path, model, metrics, split):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected columns in {path}")
            rows = [row for row in reader if (row["model"], row["split"]) != (model, split)]
    rows.append({
        "model": model,
        "hr10": f"{metrics['hr10']:.8f}",
        "ndcg10": f"{metrics['ndcg10']:.8f}",
        "split": split,
    })
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_completions(path):
    done = {}
    path = Path(path)
    if not path.exists():
        return done
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        user_id = str(row["user_id"])
        if user_id in done:
            raise ValueError(f"{path}:{line_number} repeats user {user_id}")
        done[user_id] = row["completion"]
    return done


def append_completion(path, user_id, completion):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"user_id": user_id, "completion": completion}, ensure_ascii=False) + "\n")
