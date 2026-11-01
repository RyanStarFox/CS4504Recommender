"""B's optional alignment update; D's dataset, BPR loss and evaluation stay shared."""

import copy
import math

import torch
from torch.utils.data import DataLoader

from fusion import alignment_loss
from run_sasrec import evaluate_split, sampled_bpr_loss


def batch_alignment_loss(model, inputs, positives, valid_items, generator, options):
    """Uniformly sample distinct train-history/positive movies with a private RNG."""
    ids = torch.unique(torch.cat((inputs.flatten(), positives.flatten())))
    ids = ids[ids.ne(0) & valid_items[ids]]
    if len(ids) > options["max_items"]:
        chosen = torch.randperm(len(ids), generator=generator)[: options["max_items"]]
        ids = ids[chosen.to(ids.device)]
    return alignment_loss(
        model.item_embedding(ids),
        model.fusion_module.project_text(model.movie_features[ids]),
        ids,
        valid_items,
        temperature=options["temperature"],
    )


def train_aligned(
    model,
    training_data,
    sequences,
    valid_movie_ids,
    device,
    epochs,
    batch_size,
    eval_batch_size,
    learning_rate,
    weight_decay,
    patience,
    seed,
    alignment,
):
    if (
        not math.isfinite(alignment["weight"])
        or alignment["weight"] < 0
        or not math.isfinite(alignment["temperature"])
        or alignment["temperature"] <= 0
        or alignment["max_items"] < 2
    ):
        raise ValueError("Invalid alignment weight, temperature or movie limit")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    valid_items = torch.zeros(model.item_num, dtype=torch.bool, device=device)
    valid_items[valid_movie_ids] = True
    best_state, best_metrics, best_epoch = None, None, None
    best_ndcg, stale_epochs = -math.inf, 0
    history = []
    for epoch in range(1, epochs + 1):
        training_data.resample_negatives(seed + epoch)
        loader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        # Sampling alignment pairs must not change recommendation/dropout RNG.
        alignment_rng = torch.Generator().manual_seed(seed + 100000 + epoch)
        model.train()
        rec_total, align_total, target_total, batch_total = 0.0, 0.0, 0, 0
        for users, inputs, positives, negatives in loader:
            users, inputs, positives, negatives = (
                x.to(device) for x in (users, inputs, positives, negatives)
            )
            pos_scores, neg_scores, valid = model.training_scores(
                inputs, positives, negatives, user_ids=users
            )
            rec_loss = sampled_bpr_loss(pos_scores, neg_scores, valid)
            aux_loss = batch_alignment_loss(
                model, inputs, positives, valid_items, alignment_rng, alignment
            )
            loss = rec_loss + alignment["weight"] * aux_loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = int(valid.sum())
            rec_total += float(rec_loss.detach()) * count
            align_total += float(aux_loss.detach())
            target_total += count
            batch_total += 1
        metrics, _ = evaluate_split(
            model, sequences, valid_movie_ids, "validation", device, eval_batch_size
        )
        record = {
            "epoch": epoch,
            "recommendation_loss": rec_total / target_total,
            "alignment_loss": align_total / batch_total,
            "validation": metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}: bpr={record['recommendation_loss']:.4f}, "
            f"align={record['alignment_loss']:.4f}, validation HR@10={metrics['hr10']:.4f}, "
            f"NDCG@10={metrics['ndcg10']:.4f}",
            flush=True,
        )
        if metrics["ndcg10"] > best_ndcg:
            best_ndcg, best_metrics, best_epoch = metrics["ndcg10"], metrics, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    if best_state is None:
        raise ValueError("Training must run at least one epoch")
    model.load_state_dict(best_state)
    test, _ = evaluate_split(
        model, sequences, valid_movie_ids, "test", device, eval_batch_size
    )
    return (
        best_metrics,
        test,
        {"best_epoch": best_epoch, "epochs_run": len(history), "history": history},
    )
