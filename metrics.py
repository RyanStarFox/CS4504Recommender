"""Single-target HR/NDCG for the shared temporal next-item evaluation.

Course NCF keeps its existing multi-positive evaluation in trainer.py.
Pass ordered original movie IDs, not model scores, to these functions.
"""

import math
from numbers import Integral


def _positive_id(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("Movie IDs must be positive integers")
    return int(value)


def _target_rank(ranking, ground_truth, k):
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    target = _positive_id(ground_truth)
    items = [_positive_id(item) for item in ranking]
    if len(items) != len(set(items)):
        raise ValueError("A ranking must not contain duplicate movie IDs")
    try:
        return items[:k].index(target) + 1
    except ValueError:
        return None


def hr_at_k(ranking, ground_truth, k=10):
    """One if the single held-out movie appears in the first k positions."""
    return float(_target_rank(ranking, ground_truth, k) is not None)


def ndcg_at_k(ranking, ground_truth, k=10):
    """Single relevant movie: 1/log2(rank+1), or zero for a miss."""
    rank = _target_rank(ranking, ground_truth, k)
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def evaluate_rankings(rankings, ground_truths, k=10):
    """Macro-average over exactly the users in ground_truths, including misses.

    Both mappings must use the same user keys. Missing users raise an error,
    rather than silently increasing metrics by dropping difficult examples.
    """
    if not ground_truths:
        raise ValueError("Cannot evaluate an empty set of users")
    if set(rankings) != set(ground_truths):
        raise ValueError("Rankings and ground truths must cover exactly the same users")
    ranks = [
        _target_rank(rankings[user], target, k)
        for user, target in ground_truths.items()
    ]
    return {
        f"hr{k}": sum(rank is not None for rank in ranks) / len(ranks),
        f"ndcg{k}": sum(
            0.0 if rank is None else 1.0 / math.log2(rank + 1)
            for rank in ranks
        ) / len(ranks),
    }
