"""基础版 GPT-2 + NCF 推荐系统的统一运行入口。"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from dataloader import EvalDataLoader, TrainDataLoader
from dataset import ML1MDataset
from recommender import HybridRecommender
from trainer import Trainer


def main(args):
    # 固定随机种子，方便课堂演示和不同同学之间比较结果。
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 有 NVIDIA GPU 时自动使用 CUDA；没有 GPU 时仍可在 CPU 上运行。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.features).is_file():
        raise FileNotFoundError(
            f"找不到 {args.features}，请先运行 python gpt2.py"
        )

    # 1. 读取 ML-1M，并取得固定的训练、验证、测试划分。
    dataset = ML1MDataset(args.data_path)
    train_data = dataset.get_split_data("train")
    valid_data = dataset.get_split_data("validation")
    test_data = dataset.get_split_data("test")

    # 快速模式只保留前 500 位训练用户，适合讲课时快速确认流程。
    if args.quick:
        selected_users = train_data["user_id"].drop_duplicates().head(500)
        train_data = train_data[train_data["user_id"].isin(selected_users)]
        valid_data = valid_data[valid_data["user_id"].isin(selected_users)]
        test_data = test_data[test_data["user_id"].isin(selected_users)]

    print(f"Device: {device}")
    print(
        "Train/validation/test: "
        f"{len(train_data)}/{len(valid_data)}/{len(test_data)}"
    )

    # 2. 载入 gpt2.py 预先生成的电影文本特征。
    text_features = torch.load(args.features, map_location="cpu")

    # 3. 构造混合推荐模型以及训练、验证、测试数据加载器。
    model = HybridRecommender(
        dataset.get_user_num(),
        dataset.get_item_num(),
        text_features,
        embed_dim=args.embed_dim,
    )
    train_loader = TrainDataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )
    valid_loader = EvalDataLoader(
        valid_data,
        train_data,
        batch_size=args.eval_batch_size,
        device=device,
    )
    test_loader = EvalDataLoader(
        test_data,
        train_data,
        batch_size=args.eval_batch_size,
        device=device,
    )

    # 4. 训练模型；训练结束后在测试集上进行一次最终评测。
    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    valid_result, test_result = trainer.fit(save_model=True)
    print("Best validation:", valid_result)
    print("Test:", test_result)

    # 5. 展示一个用户的实际 Top-10 推荐结果。
    user_id = int(train_data["user_id"].iloc[0])
    seen = set(
        train_data.loc[train_data["user_id"] == user_id, "movie_id"]
    )
    with torch.no_grad():
        scores = model.full_sort_predict(
            torch.tensor([user_id], device=device)
        )[0]
        scores[list(seen)] = -torch.inf
        recommended = scores.topk(10).indices.cpu().tolist()

    print(f"Top-10 recommendations for user {user_id}:")
    for movie_id in recommended:
        title = dataset.items_df.loc[movie_id, "title"]
        print(f"  {movie_id}: {title}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="ml-1m")
    parser.add_argument("--features", default="movie_features.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args())
