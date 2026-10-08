import math

import numpy as np
import torch


class TrainDataLoader:
    """把评分记录转换成 [用户, 正样本电影, 负样本电影] 训练批次。"""

    def __init__(self, dataset, batch_size=2048, shuffle=True, device="cuda"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.position = 0

        # 训练集中实际出现过的电影，作为负样本候选集合。
        self.all_items = dataset["movie_id"].unique()

        # history_items[user] 保存该用户训练集中交互过的全部电影。
        self.history_items = {
            user: set(items.values)
            for user, items in dataset.groupby("user_id")["movie_id"]
        }
        negative_items = self._sample_negative_items()

        # 三个一维张量长度相同，下标相同的位置构成一条训练样本。
        self.users = torch.tensor(dataset["user_id"].to_numpy(), dtype=torch.long, device=device)
        self.positive_items = torch.tensor(
            dataset["movie_id"].to_numpy(), dtype=torch.long, device=device
        )
        self.negative_items = torch.tensor(negative_items, dtype=torch.long, device=device)

    def _sample_negative_items(self):
        """每条正样本随机配一个该用户没有交互过的电影。"""
        negatives = np.zeros(len(self.dataset), dtype=np.int64)
        for index, row in enumerate(self.dataset.itertuples()):
            # TODO 2：从 all_items 随机选择一部当前用户未交互过的电影。
            # 提示：使用 while 循环反复采样 candidate，并检查它是否存在于
            # self.history_items[row.user_id]；找到后写入 negatives[index]。
            raise NotImplementedError("请完成训练负样本采样")
        return negatives

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        self.position = 0
        if self.shuffle:
            order = torch.randperm(len(self.dataset), device=self.device)
            self.users = self.users[order]
            self.positive_items = self.positive_items[order]
            self.negative_items = self.negative_items[order]
        return self

    def __next__(self):
        if self.position >= len(self.dataset):
            raise StopIteration
        end = self.position + self.batch_size
        current = slice(self.position, end)
        self.position = end

        # 返回形状 [3, B]：第 0/1/2 行分别是用户、正样本、负样本。
        return torch.stack(
            [self.users[current], self.positive_items[current], self.negative_items[current]]
        )


class EvalDataLoader:
    """按用户分批，为全量 Top-K 推荐评测提供历史电影和真实答案。"""

    def __init__(self, eval_dataset, train_dataset, batch_size=64, device="cuda"):
        self.dataset = eval_dataset
        self.batch_size = batch_size
        self.device = device
        self.position = 0
        self.eval_users = eval_dataset["user_id"].unique()
        self.train_positive_items = {
            user: set(items.values)
            for user, items in train_dataset.groupby("user_id")["movie_id"]
        }
        self.eval_positive_items = {
            user: set(items.values)
            for user, items in eval_dataset.groupby("user_id")["movie_id"]
        }

    def __len__(self):
        return math.ceil(len(self.eval_users) / self.batch_size)

    def __iter__(self):
        self.position = 0
        return self

    def __next__(self):
        if self.position >= len(self.eval_users):
            raise StopIteration
        users = self.eval_users[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        return torch.tensor(users, dtype=torch.long, device=self.device)

    def get_user_train_pos_items(self, users):
        return [self.train_positive_items.get(user, set()) for user in users]

    def get_user_eval_pos_items(self, users):
        return [self.eval_positive_items.get(user, set()) for user in users]
