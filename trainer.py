"""NCF 训练与评测流程。

这一文件同时承担课堂上最重要的两段逻辑：
1. 用 BPR 损失更新模型参数；
2. 用全量排序计算 HR 和 NDCG。
"""

import math

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        eval_data,
        test_data=None,
        device="cuda",
        epochs=3,
        lr=1e-3,
        early_stop_patience=3,
    ):
        self.model = model.to(device)
        self.train_loader = train_data
        self.eval_loader = eval_data
        self.test_loader = test_data
        self.device = device
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience

        # Adam 根据损失函数的梯度更新用户 Embedding、电影 Embedding、
        # 文本投影层和 MLP 中的所有可训练参数。
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.best_valid_score = -np.inf
        self.best_valid_result = None

    def _train_epoch(self):
        """完成一轮训练，返回这一轮的平均 BPR 损失。"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            # batch 为 [3, B]：用户、正样本电影、负样本电影。
            pos_scores, neg_scores = self.model(batch)
            loss = self.model.calculate_loss(pos_scores, neg_scores)

            # 一次标准的 PyTorch 参数更新：清梯度、反向传播、更新参数。
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = batch.shape[1]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    @torch.no_grad()
    def _evaluate(self, eval_loader, topk=(10, 20)):
        """按用户做全量电影排序，并计算 HR@K 与 NDCG@K。

        这段评测代码在 teacher 版和学生版中完全相同。学生只需要补全
        训练与特征处理的核心部分，无需修改评测规则。
        """
        self.model.eval()
        values = {f"HR@{k}": [] for k in topk}
        values.update({f"NDCG@{k}": [] for k in topk})

        for users in eval_loader:
            user_ids = users.cpu().numpy()

            # scores 的形状是 [用户数, 电影数]，每一行是一个用户对所有
            # 电影的预测分数。训练和评测都经过 model.score()，计算口径一致。
            scores = self.model.full_sort_predict(users)
            train_items = eval_loader.get_user_train_pos_items(user_ids)
            positive_items = eval_loader.get_user_eval_pos_items(user_ids)

            # 已在训练集中出现的电影不应再次占据推荐榜，因此将其分数屏蔽。
            for row, seen in enumerate(train_items):
                if seen:
                    scores[row, list(seen)] = -torch.inf

            recommendations = scores.topk(max(topk), dim=1).indices.cpu().numpy()

            for row, positives in enumerate(positive_items):
                if not positives:
                    continue
                rank_list = recommendations[row]

                for k in topk:
                    hits = [item in positives for item in rank_list[:k]]

                    # HR@K：前 K 个推荐中只要命中一个测试集正样本就记为 1。
                    values[f"HR@{k}"].append(float(any(hits)))

                    # NDCG@K：命中位置越靠前，折损越小，得到的分数越高。
                    dcg = sum(
                        hit / math.log2(rank + 2)
                        for rank, hit in enumerate(hits)
                    )
                    ideal_count = min(k, len(positives))
                    idcg = sum(
                        1 / math.log2(rank + 2)
                        for rank in range(ideal_count)
                    )
                    values[f"NDCG@{k}"].append(dcg / idcg if idcg else 0.0)

        return {
            name: float(np.mean(result))
            for name, result in values.items()
        }

    def fit(self, save_model=False, model_path="checkpoint.pth"):
        """训练模型，并在结束后恢复验证集表现最好的参数。"""
        wait_epochs = 0
        best_state = None

        for epoch in tqdm(range(1, self.epochs + 1), desc="Training"):
            train_loss = self._train_epoch()
            valid_result = self._evaluate(self.eval_loader)
            print(
                f"Epoch {epoch}: loss={train_loss:.4f}, "
                f"HR@10={valid_result['HR@10']:.4f}, "
                f"NDCG@10={valid_result['NDCG@10']:.4f}"
            )

            # 用验证集 NDCG@10 选择模型。测试集不参与参数选择。
            if valid_result["NDCG@10"] > self.best_valid_score:
                self.best_valid_score = valid_result["NDCG@10"]
                self.best_valid_result = valid_result
                wait_epochs = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                if save_model:
                    torch.save(self.model.state_dict(), model_path)
            else:
                wait_epochs += 1
                if wait_epochs >= self.early_stop_patience:
                    print("Early stopping")
                    break

        # 后续测试和推荐都使用验证集上最好的参数。
        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_result = None
        if self.test_loader is not None:
            test_result = self._evaluate(self.test_loader)
        return self.best_valid_result, test_result
