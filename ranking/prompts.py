"""Prompt and completion strings for Top-10 reranking.

The supervised target is a permutation of the candidate ids with the
ground-truth next movie first and the remaining ids in recall order.
DPO uses the same prompt. Its chosen completion puts the ground-truth
movie first; its rejected completion puts the other movie first.
"""


def user_prompt(example, titles):
    watched = "、".join(_movie(movie_id, titles) for movie_id in example.history)
    options = "、".join(_movie(movie_id, titles) for movie_id in example.candidates)
    return (
        f"看过：{watched}。"
        "请把下列候选电影编号按推荐优先级从高到低排成一行。"
        "只能使用这些编号，每个编号恰好出现一次，编号之间用空格分隔，不要输出其它内容。"
        f"候选：{options}。"
    )


def _movie(movie_id, titles):
    try:
        text = titles[movie_id]
    except KeyError as exc:
        raise KeyError(f"No title for movie {movie_id}") from exc
    return f"{text}（{movie_id}）"


def ranking_text(movie_ids):
    return " ".join(str(movie_id) for movie_id in movie_ids)


def place_first(candidates, first):
    if first not in candidates:
        raise ValueError(f"Movie {first} is not in the candidate list")
    return (first,) + tuple(movie_id for movie_id in candidates if movie_id != first)


def assert_non_thinking(prompt):
    """Qwen3.5 non-thinking mode closes an empty think block inside the prompt."""
    if "<think>" in prompt and "</think>" not in prompt:
        raise RuntimeError("Thinking mode is enabled. Ranking must keep enable_thinking off.")
    return prompt
