"""NDCG of a parsed ranking. The formula matches trainer.py for one relevant movie."""
import math
import re


def ndcg_at_k(ranking, ground_truth, k=10):
    hits = [1.0 if item == ground_truth else 0.0 for item in list(ranking)[:k]]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg else 0.0


def hr_at_k(ranking, ground_truth, k=10):
    return float(ground_truth in list(ranking)[:k])


def parse_ranking(text, candidates):
    """Keep candidate ids in the order they appear, then append any missing ids."""
    allowed = set(candidates)
    found = []
    seen = set()
    for token in re.findall(r"\d+", text):
        movie_id = int(token)
        if movie_id in allowed and movie_id not in seen:
            found.append(movie_id)
            seen.add(movie_id)
    for movie_id in candidates:
        if movie_id not in seen:
            found.append(movie_id)
    return tuple(found)


def group_advantages(rewards):
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rewards in a group")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(reward - mean) / std for reward in rewards]
