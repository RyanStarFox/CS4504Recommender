"""Generate A's temporal split and popularity mock candidates (standard library).

All ratings are implicit interactions. Equal timestamps retain ratings.dat row
order. Validation uses train history; test uses train plus validation history.
Popularity counts only train interactions, and ties use ascending movie ID.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from metrics import evaluate_rankings


def read_ids(path):
    """Read positive, unique original IDs from users.dat or movies.dat."""
    ids = set()
    with Path(path).open(encoding="latin-1") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = int(line.split("::", 1)[0])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid ID") from error
            if value <= 0 or value in ids:
                raise ValueError(f"{path}:{line_number}: non-positive or duplicate ID")
            ids.add(value)
    if not ids:
        raise ValueError(f"{path} contains no IDs")
    return ids


def read_interactions(path, user_ids, movie_ids):
    """Yield (user, movie, timestamp) in source order, without rating filtering."""
    with Path(path).open(encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                user, movie, rating, timestamp = map(int, line.rstrip().split("::"))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid rating record") from error
            if user not in user_ids or movie not in movie_ids:
                raise ValueError(f"{path}:{line_number}: unknown user or movie ID")
            if not 1 <= rating <= 5 or timestamp < 0:
                raise ValueError(f"{path}:{line_number}: invalid rating or timestamp")
            yield user, movie, timestamp


def build_sequences(interactions):
    """Hold out the last two interactions per user; never silently drop users."""
    histories = defaultdict(list)
    for source_row, (user, movie, timestamp) in enumerate(interactions):
        histories[user].append((timestamp, source_row, movie))
    if not histories:
        raise ValueError("No interactions to split")
    sequences = {}
    for user in sorted(histories):
        rows = sorted(histories[user])
        if len(rows) < 4:
            raise ValueError(f"User {user} needs at least four interactions for D's interface")
        movies = [row[2] for row in rows]
        if len(set(movies)) != len(movies):
            raise ValueError(f"User {user} has repeated movie interactions")
        sequences[str(user)] = {
            "train": movies[:-2], "validation": movies[-2], "test": movies[-1]
        }
    return sequences


def build_mock_candidates(sequences, movie_ids, k=10):
    """Return all users' unseen popularity Top-k, without injecting targets."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    counts = Counter(movie for row in sequences.values() for movie in row["train"])
    valid_ids = set(movie_ids)
    if not set(counts) <= valid_ids:
        raise ValueError("Training interactions contain unknown movies")
    ranked = sorted(valid_ids, key=lambda movie: (-counts[movie], movie))
    result = {"validation": {}, "test": {}}
    for user, row in sequences.items():
        seen = set(row["train"])
        for split in ("validation", "test"):
            if split == "test":
                seen.add(row["validation"])
            candidates = []
            for movie in ranked:
                if movie not in seen:
                    candidates.append(movie)
                    if len(candidates) == k:
                        break
            if len(candidates) != k:
                raise ValueError(f"User {user} has fewer than {k} unseen movies")
            result[split][user] = candidates
    return result


def write_user_json(path, payload, nested=False):
    """Atomic UTF-8 JSON, one user per line for readable dataset diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    def user_lines(users, indent):
        rows = list(users.items())
        for index, (user, row) in enumerate(rows):
            suffix = "," if index + 1 < len(rows) else ""
            yield " " * indent + json.dumps(user) + ": " + json.dumps(row) + suffix + "\n"

    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("{\n")
        if nested:
            splits = list(payload.items())
            for index, (split, users) in enumerate(splits):
                stream.write("  " + json.dumps(split) + ": {\n")
                stream.writelines(user_lines(users, 4))
                stream.write("  }" + ("," if index + 1 < len(splits) else "") + "\n")
        else:
            stream.writelines(user_lines(payload, 2))
        stream.write("}\n")
    temporary.replace(path)


def sha256(path, normalize_newlines=False):
    if normalize_newlines:
        # Git may check out .dat files as CRLF on Windows. Hash the same LF
        # text as on Linux/macOS, since line endings do not affect the split.
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(data_path, output, mock_output, manifest_output, k=10):
    data_path = Path(data_path)
    sources = [data_path / name for name in ("users.dat", "movies.dat", "ratings.dat")]
    outputs = [Path(output), Path(mock_output), Path(manifest_output)]
    if len({path.resolve() for path in sources + outputs}) != 6:
        raise ValueError("Input and output paths must be distinct")
    user_ids, movie_ids = read_ids(sources[0]), read_ids(sources[1])
    sequences = build_sequences(read_interactions(sources[2], user_ids, movie_ids))
    if {int(user) for user in sequences} != user_ids:
        raise ValueError("Some users have no interactions; refusing to drop them")
    candidates = build_mock_candidates(sequences, movie_ids, k)
    write_user_json(output, sequences)
    write_user_json(mock_output, candidates, nested=True)
    manifest = {
        "format_version": 1,
        "split": "temporal",
        "interaction_policy": "all ratings; no rating threshold",
        "sort_order": ["timestamp ascending", "ratings.dat source row ascending"],
        "holdout": "last=test; second last=validation; all earlier=train",
        "history": {"validation": "train", "test": "train + validation"},
        "history_mask": "exclude all movies in the current history, before truncation",
        "popularity": "train counts descending; movie ID ascending for ties",
        "candidate_kind": "mock; not SASRec; no ground-truth injection or user filtering",
        "candidate_k": k,
        "user_count": len(sequences),
        "movie_count": len(movie_ids),
        "user_tensor_rows": max(user_ids) + 1,
        "movie_tensor_rows": max(movie_ids) + 1,
        "counts": {
            "train": sum(len(row["train"]) for row in sequences.values()),
            "validation": len(sequences), "test": len(sequences),
        },
        "source_text_sha256_lf": {
            path.name: sha256(path, normalize_newlines=True) for path in sources
        },
        "artifact_sha256": {path.name: sha256(path) for path in outputs[:2]},
        "mock_debug_metrics": {
            split: evaluate_rankings(
                candidates[split], {user: row[split] for user, row in sequences.items()}, k
            ) for split in ("validation", "test")
        },
    }
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    return sequences, candidates, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="ml-1m")
    parser.add_argument("--output", default="data/sequences.json")
    parser.add_argument("--mock-output", default="data/mock_candidates.json")
    parser.add_argument("--manifest-output", default="data/split_manifest.json")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    generate(args.data_path, args.output, args.mock_output, args.manifest_output, args.topk)
