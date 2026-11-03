"""Interactive Agent demo: watched history -> filter -> recall -> rerank -> holdout check.

The repository does not track Qwen weights or LoRA adapters.  The default
command downloads the selected adapter from the repository documented in
README.md, loads it with ``ranking/lora.py``, and runs deterministic PyTorch
generation.  Therefore the command is useful in both online and offline states:

* online, it downloads the pinned Qwen base and the selected SFT/DPO/GRPO LoRA;
* offline without cached weights, it performs filtered SASRec recall and
  clearly reports that reranking was skipped;
* ``--reranker-backend llama`` remains available for a merged GGUF deployment.

Example::

    conda run -n CS4504 python agent_demo.py \
        --user-id 2 --request "我不喜欢恐怖片"

    conda run -n CS4504 python agent_demo.py \
        --user-id 2 --request "我不喜欢恐怖片" \
        --reranker-stage grpo --require-reranker
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from device import get_device
from ranking.batching import render_prompt
from ranking.prompts import user_prompt
from ranking.reward import parse_ranking
from ranking.sample import llama_completion
from run_sasrec import load_sequences, pad_histories
from sasrec import SASRec


ROOT = Path(__file__).resolve().parent
DEFAULT_GGUF = ROOT / "runs" / "llama" / "qwen_grpo" / "sft-f16.gguf"
DEFAULT_TOKENIZER = ROOT / "models" / "Qwen3.5-0.8B"
DEFAULT_ADAPTER_REPO = "RyanFoxW/cs4504-qwen35-movielens-rerank"
DEFAULT_ADAPTER_REVISION = "9fade15f08929a2bb5080fd0e21b9d5869c6c070"
DEFAULT_ADAPTER_ROOT = ROOT / "models" / "cs4504-qwen35-movielens-rerank"


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]


# MovieLens-1M uses these English labels.  Aliases are intentionally explicit:
# matching arbitrary words such as "怕" would make the demo silently filter the
# wrong category.
GENRE_ALIASES = {
    "Action": ("动作", "动作片", "action"),
    "Adventure": ("冒险", "冒险片", "adventure"),
    "Animation": ("动画", "动画片", "animation"),
    "Children's": ("儿童", "儿童片", "少儿", "children", "children's"),
    "Comedy": ("喜剧", "喜剧片", "comedy"),
    "Crime": ("犯罪", "犯罪片", "crime"),
    "Documentary": ("纪录", "纪录片", "documentary"),
    "Drama": ("剧情", "剧情片", "drama"),
    "Fantasy": ("奇幻", "奇幻片", "fantasy"),
    "Film-Noir": ("黑色电影", "film-noir", "film noir"),
    "Horror": ("恐怖", "恐怖片", "horror"),
    "Musical": ("音乐", "歌舞", "音乐片", "musical"),
    "Mystery": ("悬疑", "悬疑片", "mystery"),
    "Romance": ("爱情", "爱情片", "浪漫", "romance"),
    "Sci-Fi": ("科幻", "科幻片", "sci-fi", "science fiction"),
    "Thriller": ("惊悚", "惊悚片", "thriller"),
    "War": ("战争", "战争片", "war"),
    "Western": ("西部", "西部片", "western"),
}

NEGATIVE_MARKERS = (
    "不喜欢", "不想看", "不要", "不看", "讨厌", "排除", "拒绝", "避免",
    "dislike", "don't like", "do not like", "no ", "not ", "without",
)


def read_movies(path: str | Path) -> dict[int, Movie]:
    """Read MovieLens movies.dat while preserving original IDs."""
    movies: dict[int, Movie] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="latin-1").splitlines(), start=1
    ):
        parts = line.split("::")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number} is not movie_id::title::genres")
        movie_id = int(parts[0])
        if movie_id <= 0 or movie_id in movies:
            raise ValueError(f"{path}:{line_number} has an invalid or duplicate movie ID")
        genres = tuple(genre for genre in parts[2].split("|") if genre)
        if not genres:
            raise ValueError(f"{path}:{line_number} has no genre")
        movies[movie_id] = Movie(movie_id, parts[1], genres)
    if not movies:
        raise ValueError(f"No movies found in {path}")
    return movies


def normalize_genre(value: str) -> str:
    """Convert a canonical label or Chinese/English alias to MovieLens form."""
    normalized = value.strip().lower()
    for genre, aliases in GENRE_ALIASES.items():
        if normalized == genre.lower() or normalized in {alias.lower() for alias in aliases}:
            return genre
    choices = ", ".join(GENRE_ALIASES)
    raise ValueError(f"Unknown genre {value!r}; choose one of: {choices}")


def extract_excluded_genres(request: str) -> set[str]:
    """Extract only negatively expressed genres from a short user request."""
    excluded: set[str] = set()
    clauses = re.split(r"[，,。.!！;；但]+", request.lower())
    for clause in clauses:
        if not any(marker in clause for marker in NEGATIVE_MARKERS):
            continue
        for genre, aliases in GENRE_ALIASES.items():
            names = (genre, *aliases)
            if any(name.lower() in clause for name in names):
                excluded.add(genre)
    return excluded


def genre_filter(
    movie_ids: list[int] | tuple[int, ...],
    movies: dict[int, Movie],
    excluded_genres: set[str],
) -> list[int]:
    """Return catalog IDs that do not contain any excluded genre."""
    unknown = excluded_genres - set(GENRE_ALIASES)
    if unknown:
        raise ValueError(f"Unknown excluded genres: {sorted(unknown)}")
    return [
        movie_id for movie_id in movie_ids
        if movie_id in movies and excluded_genres.isdisjoint(movies[movie_id].genres)
    ]


def load_sasrec_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[SASRec, list[int], dict]:
    """Restore D's pure-ID checkpoint without loading arbitrary pickle objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "valid_movie_ids"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing {sorted(missing)}")
    model = SASRec(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    valid_movie_ids = [int(movie_id) for movie_id in checkpoint["valid_movie_ids"]]
    return model, valid_movie_ids, checkpoint


@torch.inference_mode()
def filtered_recall(
    model: SASRec,
    history: list[int],
    eligible_movie_ids: list[int],
    device: torch.device,
    topk: int = 10,
) -> list[int]:
    """Score the filtered, unseen catalog and return SASRec's ordered Top-K."""
    if topk <= 0:
        raise ValueError("topk must be positive")
    seen = set(history)
    candidates = [movie_id for movie_id in eligible_movie_ids if movie_id not in seen]
    if len(candidates) < topk:
        raise ValueError(
            f"Only {len(candidates)} unseen movies remain after filtering; need {topk}"
        )
    inputs = pad_histories([history], model.max_seq_len).to(device)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    scores = model.score_candidates(inputs, candidate_tensor)
    columns = scores[0].topk(topk).indices.cpu().tolist()
    return [candidates[column] for column in columns]


def _ranker_titles(movies: dict[int, Movie]) -> dict[int, str]:
    return {
        movie_id: f"Title: {movie.title}. Genres: {', '.join(movie.genres)}."
        for movie_id, movie in movies.items()
    }


def download_adapter(
    stage: str,
    repo_id: str = DEFAULT_ADAPTER_REPO,
    revision: str = DEFAULT_ADAPTER_REVISION,
    destination: str | Path = DEFAULT_ADAPTER_ROOT,
) -> Path:
    """Download one README-advertised adapter and validate its metadata."""
    if stage not in {"sft", "dpo", "grpo"}:
        raise ValueError("reranker stage must be sft, dpo or grpo")
    destination = Path(destination)
    adapter_dir = destination / stage
    files = (adapter_dir / "adapter.json", adapter_dir / "adapter.pt")
    if not all(path.is_file() for path in files):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
            cache_dir=ROOT / ".cache" / "huggingface" / "hub",
            token=False,
            allow_patterns=[f"{stage}/adapter.json", f"{stage}/adapter.pt"],
        )
    if not all(path.is_file() for path in files):
        raise FileNotFoundError(f"Adapter {stage} was not downloaded to {adapter_dir}")
    config = json.loads(files[0].read_text(encoding="utf-8"))
    if config.get("format_version") != 1 or config.get("stage") != stage:
        raise ValueError(f"Unexpected adapter metadata in {files[0]}")
    if config.get("model_id") != "Qwen/Qwen3.5-0.8B":
        raise ValueError(f"Adapter {stage} targets an unexpected base model")
    expected_revision = json.loads(
        (ROOT / "data" / "qwen_movie_features.json").read_text(encoding="utf-8")
    )["revision"]
    if config.get("revision") != expected_revision:
        raise ValueError(
            f"Adapter base revision {config.get('revision')} does not match "
            f"the project revision {expected_revision}"
        )
    return adapter_dir


def _ranking_prompt(tokenizer, candidates, history, movies, max_history):
    example = SimpleNamespace(
        history=tuple(history[-max_history:]), candidates=tuple(candidates)
    )
    return render_prompt(tokenizer, user_prompt(example, _ranker_titles(movies)))


@torch.inference_mode()
def rerank_with_hf(
    candidates: list[int],
    history: list[int],
    movies: dict[int, Movie],
    adapter_dir: str | Path,
    max_history: int = 10,
    max_new_tokens: int = 32,
) -> tuple[list[int], str, str]:
    """Load a README LoRA through ranking/lora.py and deterministically rerank."""
    from ranking.lora import load_adapter
    from ranking.model import load_ranker

    model, tokenizer, device, dtype = load_ranker()
    config = load_adapter(model, adapter_dir)
    model.eval()
    prompt = _ranking_prompt(tokenizer, candidates, history, movies, max_history)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = encoded["input_ids"].shape[1]
    completion = tokenizer.decode(
        generated[0, prompt_width:], skip_special_tokens=True
    ).strip()
    if not completion:
        raise RuntimeError("Qwen returned an empty completion")
    ranking = list(parse_ranking(completion, candidates))
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise RuntimeError("Qwen reranker did not produce a candidate permutation")
    detail = f"{config['stage']} LoRA on {device} ({dtype})"
    return ranking, completion, detail


def rerank_with_qwen(
    candidates: list[int],
    history: list[int],
    movies: dict[int, Movie],
    gguf: str | Path,
    tokenizer_path: str | Path,
    llama_bin: str,
    max_history: int = 10,
    max_new_tokens: int = 32,
    context: int = 2048,
) -> tuple[list[int], str]:
    """Run C's llama.cpp path and parse a complete candidate permutation."""
    gguf = Path(gguf)
    tokenizer_path = Path(tokenizer_path)
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF model is not at {gguf}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Qwen tokenizer is not at {tokenizer_path}")

    # Lazy import keeps the tracked-artifact-only fallback fast and offline.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt = _ranking_prompt(tokenizer, candidates, history, movies, max_history)
    completion = llama_completion(
        llama_bin, gguf, prompt, max_new_tokens=max_new_tokens,
        temperature=0, context=context,
    )
    ranking = list(parse_ranking(completion, candidates))
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise RuntimeError("Qwen reranker did not produce a candidate permutation")
    return ranking, completion


def run_demo(args: argparse.Namespace) -> dict:
    movies = read_movies(args.movies)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    model, valid_movie_ids, checkpoint = load_sasrec_checkpoint(args.checkpoint, device)
    sequences = load_sequences(args.sequences, set(movies))
    if args.user_id not in sequences:
        raise KeyError(f"User {args.user_id} is not in {args.sequences}")
    record = sequences[args.user_id]
    history = list(record.train)
    if args.history_split == "test":
        history.append(record.validation)
    target = record.test if args.history_split == "test" else record.validation

    excluded = extract_excluded_genres(args.request)
    excluded.update(normalize_genre(value) for value in args.exclude_genre)
    eligible = genre_filter(valid_movie_ids, movies, excluded)
    recalled = filtered_recall(model, history, eligible, device, args.top_k)

    final = recalled
    completion = None
    backend = args.reranker_backend
    if backend == "auto":
        backend = "llama" if args.gguf else "hf"
    reranker = {
        "used": False,
        "backend": backend,
        "stage": args.reranker_stage,
        "model": None,
        "detail": None,
        "reason": None,
    }
    try:
        if backend == "none":
            reranker["reason"] = "reranker disabled by --reranker-backend none"
        elif backend == "hf":
            adapter_dir = download_adapter(
                args.reranker_stage,
                repo_id=args.adapter_repo,
                revision=args.adapter_revision,
                destination=args.adapter_root,
            )
            final, completion, detail = rerank_with_hf(
                recalled, history, movies, adapter_dir,
                max_history=args.max_history,
                max_new_tokens=args.max_new_tokens,
            )
            reranker.update(
                used=True, model=str(adapter_dir), detail=detail, reason=None
            )
        elif backend == "llama":
            gguf = Path(args.gguf) if args.gguf else DEFAULT_GGUF
            final, completion = rerank_with_qwen(
                recalled, history, movies, gguf, args.tokenizer, args.llama_bin,
                max_history=args.max_history, max_new_tokens=args.max_new_tokens,
                context=args.context,
            )
            reranker.update(
                used=True, model=str(gguf),
                detail=f"{args.reranker_stage} merged GGUF via llama.cpp", reason=None,
            )
        else:
            raise ValueError(f"Unknown reranker backend {backend}")
    except Exception as error:
        if args.require_reranker:
            raise
        reranker["reason"] = f"{type(error).__name__}: {error}"

    def describe(movie_id: int, rank: int | None = None) -> dict:
        movie = movies[movie_id]
        description = {
            "movie_id": movie_id,
            "title": movie.title,
            "genres": list(movie.genres),
        }
        if rank is not None:
            description["rank"] = rank
        return description

    removed = len(valid_movie_ids) - len(eligible)
    history_limit = getattr(args, "history_limit", 10)
    shown_history = history[-history_limit:]
    first_position = len(history) - len(shown_history) + 1
    final_rank = final.index(target) + 1 if target in final else None
    return {
        "user_id": args.user_id,
        "request": args.request,
        "history_split": args.history_split,
        "history_length": len(history),
        "watched_history": {
            "total": len(history),
            "shown": len(shown_history),
            "movies": [
                {"position": position, **describe(movie_id)}
                for position, movie_id in enumerate(
                    shown_history, start=first_position
                )
            ],
        },
        "device": str(device),
        "filter": {
            "excluded_genres": sorted(excluded),
            "catalog_size": len(valid_movie_ids),
            "eligible_size": len(eligible),
            "removed": removed,
        },
        "sasrec": {
            "checkpoint": str(args.checkpoint),
            "validation_metrics": checkpoint.get("validation_metrics"),
            "test_metrics": checkpoint.get("test_metrics"),
            "recommendations": [
                describe(movie_id, rank)
                for rank, movie_id in enumerate(recalled, start=1)
            ],
        },
        "reranker": {**reranker, "raw_completion": completion},
        "final_recommendations": [
            describe(movie_id, rank)
            for rank, movie_id in enumerate(final, start=1)
        ],
        "held_out_evaluation": {
            "split": args.history_split,
            "target": describe(target),
            "hit_at_k": final_rank is not None,
            "rank": final_rank,
            "excluded_by_filter": target not in eligible,
        },
    }


def print_demo(result: dict) -> None:
    excluded = result["filter"]["excluded_genres"]
    print(f"用户 {result['user_id']}：{result['request']}")
    watched = result["watched_history"]
    print(
        f"已观看历史（按时间顺序，展示最近 {watched['shown']} / "
        f"{watched['total']} 部）："
    )
    for movie in watched["movies"]:
        genres = " / ".join(movie["genres"])
        print(
            f"   {movie['position']:>3}. [{movie['movie_id']}] "
            f"{movie['title']} — {genres}"
        )
    print(
        "1. Genre filter：排除 " + (", ".join(excluded) if excluded else "无")
        + f"；从 {result['filter']['catalog_size']} 部中移除 "
        + f"{result['filter']['removed']} 部。"
    )
    print(
        f"2. SASRec：使用 {result['history_length']} 条历史，在过滤后的目录中召回 "
        f"{len(result['sasrec']['recommendations'])} 部。"
    )
    if result["reranker"]["used"]:
        print(
            f"3. Qwen：已使用 {result['reranker']['detail']} 重排"
            f"（{result['reranker']['model']}）。"
        )
    else:
        print("3. Qwen：当前产物不可用，保留 SASRec 顺序。")
        print(f"   原因：{result['reranker']['reason']}")
    print("4. 最终推荐：")
    for movie in result["final_recommendations"]:
        genres = " / ".join(movie["genres"])
        print(f"   {movie['rank']:>2}. [{movie['movie_id']}] {movie['title']} — {genres}")
    evaluation = result["held_out_evaluation"]
    target = evaluation["target"]
    target_genres = " / ".join(target["genres"])
    if evaluation["hit_at_k"]:
        verdict = f"命中，最终排名第 {evaluation['rank']}"
    elif evaluation["excluded_by_filter"]:
        verdict = "未命中（被本次用户偏好过滤）"
    else:
        verdict = "未命中"
    print(
        f"5. 留出的真实下一部观看（{evaluation['split']}）："
        f"[{target['movie_id']}] {target['title']} — {target_genres}；Top-K {verdict}。"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, default=2)
    parser.add_argument("--request", default="我不喜欢恐怖片")
    parser.add_argument(
        "--exclude-genre", action="append", default=[],
        help="Explicit genre exclusion; repeat for multiple genres.",
    )
    parser.add_argument("--history-split", choices=("validation", "test"), default="test")
    parser.add_argument("--sequences", default="data/sequences.json")
    parser.add_argument("--movies", default="ml-1m/movies.dat")
    parser.add_argument("--checkpoint", default="checkpoints/sasrec_id.pt")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--reranker-backend", choices=("auto", "hf", "llama", "none"), default="auto",
        help="auto uses the README LoRA with PyTorch, unless --gguf selects llama.cpp.",
    )
    parser.add_argument(
        "--reranker-stage", choices=("sft", "dpo", "grpo"), default="grpo",
        help="README adapter to use; GRPO has the best reported test NDCG.",
    )
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--adapter-revision", default=DEFAULT_ADAPTER_REVISION)
    parser.add_argument("--adapter-root", default=str(DEFAULT_ADAPTER_ROOT))
    parser.add_argument("--gguf", help=f"Merged Qwen GGUF; default probe: {DEFAULT_GGUF}")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--llama-bin", default="llama-cli")
    parser.add_argument("--require-reranker", action="store_true")
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument(
        "--history-limit", type=int, default=10,
        help="Number of most recent watched movies to show in the demo.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    if args.user_id <= 0 or args.top_k <= 0:
        parser.error("--user-id and --top-k must be positive")
    if (
        args.max_history <= 0 or args.history_limit <= 0
        or args.max_new_tokens <= 0 or args.context <= 0
    ):
        parser.error("history, generation and context limits must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_demo(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_demo(result)


if __name__ == "__main__":
    main()
