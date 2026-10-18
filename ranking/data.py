"""Load the shared sequence and candidate files.

Movie ids stay as MovieLens ids. A user enters training only when the
ground-truth next movie is already inside that split's candidate list.
The test next-movie is never a training label. A fraction of validation
users is held out for parameter selection and is not put in the loss.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Same sentence B encodes in features.py. The ranker prompt uses this text
# so both sides describe a movie the same way.
MOVIE_TEMPLATE = "Title: {title}. Genres: {genres}."


@dataclass(frozen=True)
class Example:
    user_id: str
    split: str
    history: tuple
    candidates: tuple
    ground_truth: int


@dataclass(frozen=True)
class Bundle:
    titles: dict
    train: tuple
    select: tuple
    test: tuple


def read_movies(path):
    titles = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="latin-1").splitlines(), start=1):
        if not line:
            continue
        parts = line.split("::")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number} is not movie_id::title::genres")
        movie_id, title, genres = parts
        movie_id = int(movie_id)
        if movie_id <= 0 or movie_id in titles:
            raise ValueError(f"Movie id {movie_id} is missing or duplicated")
        titles[movie_id] = MOVIE_TEMPLATE.format(title=title, genres=genres.replace("|", ", "))
    if not titles:
        raise ValueError(f"No movies in {path}")
    return titles


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc


def _movie_list(value, path, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} field {label} must be a non-empty list of movie ids")
    movies = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{path} field {label} contains a non-integer movie id")
        if item <= 0:
            raise ValueError(f"{path} field {label} contains movie id {item}")
        movies.append(item)
    if len(set(movies)) != len(movies):
        raise ValueError(f"{path} field {label} contains duplicate movie ids")
    return tuple(movies)


def held_out(user_id, val_ratio, seed):
    """Stable user split. val_ratio <= 0 keeps everyone for training."""
    if val_ratio <= 0:
        return False
    if val_ratio >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
    value = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return value < val_ratio


def load_bundle(sequences_path, candidates_path, movies_path, candidate_k=10,
                max_history=10, val_ratio=0.1, seed=42, limit=None):
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    titles = read_movies(movies_path)
    sequences = _load_json(sequences_path)
    candidates = _load_json(candidates_path)
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError(f"{sequences_path} must be a non-empty object keyed by user id")
    if not isinstance(candidates, dict) or set(candidates) != {"validation", "test"}:
        raise ValueError(f"{candidates_path} must contain exactly the keys validation and test")

    grouped = {"train": [], "select": [], "test": []}
    skipped = 0
    for split in ("validation", "test"):
        rows = candidates[split]
        if not isinstance(rows, dict):
            raise ValueError(f"{candidates_path} {split} must be an object keyed by user id")
        for user_id, movie_ids in rows.items():
            user_id = str(user_id)
            record = sequences.get(user_id)
            if record is None:
                raise ValueError(f"User {user_id} is in {candidates_path} but not in {sequences_path}")
            if not isinstance(record, dict) or not {"train", "validation", "test"} <= set(record):
                raise ValueError(f"User {user_id} in {sequences_path} needs train, validation and test")
            history = _movie_list(record["train"], sequences_path, f"{user_id}.train")
            label = record[split]
            if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
                raise ValueError(f"User {user_id} {split} label must be a positive movie id")
            options = _movie_list(movie_ids, candidates_path, f"{split}.{user_id}")
            if candidate_k is not None and len(options) != candidate_k:
                raise ValueError(
                    f"User {user_id} {split} candidates have length {len(options)}, expected {candidate_k}"
                )
            unknown = [movie_id for movie_id in history + options + (label,) if movie_id not in titles]
            if unknown:
                raise ValueError(f"User {user_id} refers to unknown movie ids {unknown[:5]}")
            if label not in options:
                skipped += 1
                continue
            example = Example(
                user_id=user_id,
                split=split,
                history=history[-max_history:],
                candidates=options,
                ground_truth=label,
            )
            if split == "test":
                grouped["test"].append(example)
            elif held_out(user_id, val_ratio, seed):
                grouped["select"].append(example)
            else:
                grouped["train"].append(example)

    if limit is not None:
        for key in grouped:
            grouped[key] = sorted(grouped[key], key=lambda row: int(row.user_id))[:limit]
    if not grouped["train"]:
        raise ValueError(
            "No training users remain. The ground-truth next movie must sit inside the "
            f"validation candidates. Skipped {skipped} users whose label was outside the list."
        )
    return Bundle(
        titles=titles,
        train=tuple(grouped["train"]),
        select=tuple(grouped["select"]),
        test=tuple(grouped["test"]),
    )


def dpo_pairs(examples, max_rejected=1):
    """chosen is the ground-truth movie; rejected is another movie in the same list."""
    if max_rejected <= 0:
        raise ValueError("max_rejected must be positive")
    pairs = []
    for example in examples:
        rejected = [movie_id for movie_id in example.candidates if movie_id != example.ground_truth]
        rejected = rejected[:max_rejected]
        if not rejected:
            raise ValueError(f"User {example.user_id} has no rejected movie")
        for movie_id in rejected:
            pairs.append((example, movie_id))
    if not pairs:
        raise ValueError("No DPO pairs")
    return pairs


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_sample_file(path):
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"{path} is not a ranking sample file")
    if payload.get("split") != "validation":
        raise ValueError("GRPO samples must be drawn from the validation split, not the test split")
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{path} has no sample groups")
    checked = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"Sample group {index} is not an object")
        completions = group.get("completions")
        candidates = group.get("candidates")
        if not isinstance(completions, list) or len(completions) < 2:
            raise ValueError(f"Sample group {index} needs at least two completions")
        if not all(isinstance(text, str) and text.strip() for text in completions):
            raise ValueError(f"Sample group {index} has an empty completion")
        options = _movie_list(candidates, path, f"groups[{index}].candidates")
        label = group.get("ground_truth")
        if isinstance(label, bool) or not isinstance(label, int) or label not in options:
            raise ValueError(f"Sample group {index} ground_truth is not in its candidates")
        prompt = group.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Sample group {index} is missing its prompt")
        checked.append({
            "user_id": str(group.get("user_id", index)),
            "prompt": prompt,
            "ground_truth": label,
            "candidates": options,
            "completions": tuple(completions),
        })
    return payload, checked
