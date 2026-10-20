"""BPR matrix factorization on A's temporal MovieLens split."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metrics import evaluate_rankings


class BPRMF(nn.Module):
    def __init__(self, user_num, item_num, embed_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, embed_dim)
        self.item_embedding = nn.Embedding(item_num, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users, positive, negative):
        user = self.user_embedding(users)
        pos = self.item_embedding(positive)
        neg = self.item_embedding(negative)
        loss = -F.logsigmoid((user * (pos - neg)).sum(-1)).mean()
        regularizer = (user.square().sum(-1) + pos.square().sum(-1) + neg.square().sum(-1)).mean()
        return loss, regularizer


def train_epoch(model, optimizer, users, positives, histories, valid_items, batch_size, reg, rng, device):
    model.train()
    order = rng.permutation(len(users))
    total = 0.0
    for start in range(0, len(users), batch_size):
        index = order[start:start + batch_size]
        batch_users = users[index]
        negative = rng.choice(valid_items, size=len(index))
        invalid = np.fromiter((int(item) in histories[int(user)] for user, item in zip(batch_users, negative)), dtype=bool, count=len(index))
        while invalid.any():
            negative[invalid] = rng.choice(valid_items, size=int(invalid.sum()))
            invalid = np.fromiter((int(item) in histories[int(user)] for user, item in zip(batch_users, negative)), dtype=bool, count=len(index))
        user_tensor = torch.as_tensor(batch_users, dtype=torch.long, device=device)
        positive_tensor = torch.as_tensor(positives[index], dtype=torch.long, device=device)
        negative_tensor = torch.as_tensor(negative, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        bpr, regularizer = model(user_tensor, positive_tensor, negative_tensor)
        loss = bpr + reg * regularizer
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(index)
    return total / len(users)


@torch.no_grad()
def evaluate(model, sequences, valid_items, split, device, batch_size=256):
    model.eval()
    user_ids = list(sequences)
    valid_mask = torch.zeros(model.item_embedding.num_embeddings, dtype=torch.bool, device=device)
    valid_mask[torch.as_tensor(sorted(valid_items), dtype=torch.long, device=device)] = True
    item_vectors = model.item_embedding.weight
    rankings = {}
    for start in range(0, len(user_ids), batch_size):
        batch_ids = user_ids[start:start + batch_size]
        users = torch.as_tensor([int(user) for user in batch_ids], dtype=torch.long, device=device)
        scores = model.user_embedding(users) @ item_vectors.T
        scores[:, ~valid_mask] = -torch.inf
        for row, user in enumerate(batch_ids):
            history = sequences[user]["train"]
            if split == "test":
                history = history + [sequences[user]["validation"]]
            scores[row, history] = -torch.inf
        topk = scores.topk(10, dim=1).indices.cpu().tolist()
        rankings.update(zip(batch_ids, topk))
    targets = {user: row[split] for user, row in sequences.items()}
    return evaluate_rankings(rankings, targets)


def fit(sequences, valid_items, device, epochs=10, embed_dim=64, batch_size=8192,
        lr=0.005, reg=1e-5, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    user_num = max(map(int, sequences)) + 1
    item_num = max(valid_items) + 1
    model = BPRMF(user_num, item_num, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    users = np.fromiter((int(user) for user, row in sequences.items() for _ in row["train"]), dtype=np.int64)
    positives = np.fromiter((movie for row in sequences.values() for movie in row["train"]), dtype=np.int64)
    histories = {int(user): set(row["train"]) for user, row in sequences.items()}
    valid_items = np.asarray(sorted(valid_items), dtype=np.int64)
    best_score = -1.0
    best_validation = None
    best_state = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, users, positives, histories, valid_items, batch_size, reg, rng, device)
        validation = evaluate(model, sequences, valid_items, "validation", device)
        print(f"BPR-MF epoch {epoch}: loss={loss:.5f} validation={validation}", flush=True)
        if validation["ndcg10"] > best_score:
            best_score = validation["ndcg10"]
            best_validation = validation
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, sequences, valid_items, "test", device)
    return best_epoch, best_validation, test
