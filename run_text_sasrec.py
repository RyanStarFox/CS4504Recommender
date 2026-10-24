"""B's 5a/5b experiments using D's unchanged SASRec training/evaluation loop."""

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

from device import get_device
import torch

from fusion import GatedMovieFusion
from run_sasrec import (
    SASRecTrainingDataset,
    load_sequences,
    read_movie_ids,
    train_model,
    upsert_result,
)
from sasrec import SASRec


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path, valid_movie_ids):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    features = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != metadata["tensor_sha256"]:
        raise ValueError("Feature tensor checksum mismatch")
    if features.shape != (max(valid_movie_ids) + 1, metadata["hidden_size"]):
        raise ValueError("Feature tensor shape does not match movie IDs/metadata")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError("Expected finite float32 feature cache")
    valid = torch.zeros(len(features), dtype=torch.bool)
    valid[valid_movie_ids] = True
    if features[~valid].count_nonzero() or not features[valid].ne(0).any(dim=-1).all():
        raise ValueError("Feature cache padding/missing/valid rows are incorrect")
    return features, metadata


def build_model(model_config, features, seed):
    # Keep ID/Transformer initialization and subsequent training RNG identical
    # across encoders, despite their different projection input dimensions.
    model = SASRec(**model_config, movie_features=features, user_features=None)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model.fusion_module = GatedMovieFusion(features.shape[1], model.rec_hidden_size)
    return model


def run(config, encoder):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config.get("threads", 4))
    device = get_device()
    movie_ids = read_movie_ids(config["movies"])
    sequences = load_sequences(config["sequences"], set(movie_ids))
    feature_path = Path(config["features"][encoder])
    features, feature_metadata = load_features(feature_path, movie_ids)
    if feature_metadata["encoder"] != encoder:
        raise ValueError("Encoder does not match feature metadata")
    if feature_metadata["data_sha256"] != file_sha256(config["movies"]):
        raise ValueError("Feature cache was generated from different movie metadata")
    model_config = dict(config["model"], item_num=max(movie_ids) + 1)
    model = build_model(model_config, features, config["seed"])
    data = SASRecTrainingDataset(
        sequences, movie_ids, model.max_seq_len, config["seed"]
    )
    model_name = f"sasrec_{encoder}_gated"
    print(
        f"{model_name}: {device}, {len(sequences)} users; text={features.shape[1]}; "
        "use_user_text=false, use_alignment=false",
        flush=True,
    )
    validation, test = train_model(
        model,
        data,
        sequences,
        movie_ids,
        device,
        seed=config["seed"],
        **config["training"],
    )
    checkpoint = Path(config["checkpoint_dir"]) / f"{model_name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model_name,
        "encoder": encoder,
        "experiment_config": config,
        "model_config": model_config,
        "feature_path": str(feature_path),
        "feature_sha256": file_sha256(feature_path),
        "sequences_sha256": file_sha256(config["sequences"]),
        "movies_sha256": file_sha256(config["movies"]),
        "validation_metrics": validation,
        "test_metrics": test,
        "selection": "validation NDCG@10; earliest best checkpoint",
        "use_user_text": False,
        "use_alignment": False,
        "history_and_scoring_share_fusion": True,
        "device": str(device),
        "dtype": "torch.float32",
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "checkpoint": str(checkpoint),
    }
    torch.save(dict(report, model_state=model.state_dict()), checkpoint)
    # Recreate from external frozen cache; checkpoints must contain all learned
    # fusion parameters, but must not contain the frozen feature tensors.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = build_model(model_config, features, config["seed"]).to(device)
    restored.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    restored.eval()
    history = torch.tensor(
        [sequences[min(sequences)].train[-model.max_seq_len :]],
        dtype=torch.long,
        device=device,
    )
    candidates = torch.tensor(movie_ids[:10], device=device)
    with torch.inference_mode():
        torch.testing.assert_close(
            model.score_candidates(history, candidates),
            restored.score_candidates(history, candidates),
            rtol=0,
            atol=0,
        )
    report["checkpoint_roundtrip_passed"] = True
    output = Path(config["report_dir"]) / f"{model_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    upsert_result(config["results"], model_name, test)
    results = Path(config["results"])
    results.write_text(
        results.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    print(f"Saved {checkpoint}; validation={validation}; test={test}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/text_sasrec.json")
    parser.add_argument("--encoder", choices=["gpt2", "qwen", "both"], default="both")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    for encoder in ["gpt2", "qwen"] if args.encoder == "both" else [args.encoder]:
        run(settings, encoder)
