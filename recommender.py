import torch
import torch.nn as nn


class HybridRecommender(nn.Module):
    """在基础 NCF 中加入 GPT-2 电影文本特征。"""

    def __init__(self, n_users, n_items, text_features, embed_dim=32):
        super().__init__()

        # 协同过滤部分：根据用户 ID 和电影 ID 学习可训练的稠密向量。
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # 内容特征部分：把 GPT-2 的 768 维特征压缩到与 ID Embedding 相同的维度。
        self.text_projector = nn.Linear(text_features.shape[1], embed_dim)

        # 三个 32 维向量拼接为 96 维，再由 MLP 输出一个推荐分数。
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

        # register_buffer 会让文本特征随模型移动到 GPU，但不会被优化器训练。
        self.register_buffer("text_features", text_features.float())

        # MovieLens 的电影编号有空缺；零向量所在位置不能进入推荐结果。
        self.register_buffer("valid_item_mask", text_features.abs().sum(dim=1).ne(0))
        self.n_items = len(text_features)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    @property
    def device(self):
        return next(self.parameters()).device

    def score(self, user_ids, item_ids):
        """计算一批用户—电影对的推荐分数。"""
        user_vec = self.user_embedding(user_ids)                    # [B, 32]
        item_vec = self.item_embedding(item_ids)                    # [B, 32]
        text_vec = self.text_projector(self.text_features[item_ids])  # [B, 32]

        # TODO 3：拼接三个向量，并通过 self.mlp 得到每个用户—电影对的分数。
        # 提示：沿最后一维拼接后形状为 [B, 96]；MLP 输出 [B, 1]，
        # 需要移除最后一个长度为 1 的维度。
        raise NotImplementedError("请完成特征融合与推荐分数计算")

    def forward(self, batch_data):
        """同一套 score 同时计算正样本和负样本，保证训练与推荐逻辑一致。"""
        user_ids, positive_item_ids, negative_item_ids = batch_data
        positive_scores = self.score(user_ids, positive_item_ids)
        negative_scores = self.score(user_ids, negative_item_ids)
        return positive_scores, negative_scores

    def calculate_loss(self, positive_scores, negative_scores):
        """BPR：鼓励正样本分数高于负样本分数。"""
        # TODO 4：实现 BPR 损失。
        # 提示：先计算 positive_scores - negative_scores，再依次使用
        # sigmoid、log、mean；最前面需要负号，可加入 1e-8 保持数值稳定。
        raise NotImplementedError("请完成 BPR 损失")

    @torch.no_grad()
    def full_sort_predict(self, user_ids):
        """为一批用户计算对全部电影的分数，供 HR/NDCG 评测使用。"""
        user_ids = user_ids.to(self.device)
        all_item_ids = torch.arange(self.n_items, device=self.device)
        batch_size = len(user_ids)

        users = user_ids[:, None].expand(batch_size, self.n_items).reshape(-1)
        items = all_item_ids[None, :].expand(batch_size, self.n_items).reshape(-1)
        scores = self.score(users, items).view(batch_size, self.n_items)
        scores[:, ~self.valid_item_mask] = -torch.inf
        return scores
