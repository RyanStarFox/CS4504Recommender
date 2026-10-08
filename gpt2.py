import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2Model, GPT2Tokenizer


def choose_model_source(model_path=None):
    """优先使用随学生包提供的本地权重；教师机也可以使用 Hugging Face 缓存。"""
    if model_path:
        return str(Path(model_path)), True
    local_path = Path("gpt2_model")
    if local_path.is_dir():
        return str(local_path), True
    return "gpt2", False


def extract_movie_features(data_path, output_path, batch_size=32, model_path=None):
    """把每部电影的标题和类型转换为一个固定的 768 维 GPT-2 特征。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    movies = pd.read_csv(
        Path(data_path) / "movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    # 给 GPT-2 的输入只包含本项目现成的两类内容信息：标题和电影类型。
    texts = [
        f"Title: {row.title}. Genres: {row.genres.replace('|', ', ')}."
        for row in movies.itertuples()
    ]

    model_source, local_only = choose_model_source(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_source, local_files_only=local_only)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_source, local_files_only=local_only)
    model = model.to(device).eval()

    encoded_batches = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}

        # 特征提取阶段不训练 GPT-2，因此关闭梯度以节省显存和计算时间。
        with torch.no_grad():
            # hidden 的形状为 [B, L, 768]：批大小、词元数、隐藏维度。
            hidden = model(**inputs).last_hidden_state

            # TODO 1：对最后一层隐藏状态进行掩码平均池化，得到 [B, 768]。
            # 提示：attention_mask 中真实词元为 1，padding 为 0；先扩充
            # 最后一维，再计算加权和并除以真实词元数。
            raise NotImplementedError("请完成 GPT-2 掩码平均池化")
        encoded_batches.append(pooled.cpu())

    encoded = torch.cat(encoded_batches)  # [3883, 768]

    # 建立“电影原始编号 -> 文本特征”的直接索引表。
    # MovieLens 的电影编号存在空缺，因此行数使用 max(movie_id)+1。
    item_num = int(movies["movie_id"].max()) + 1
    features = torch.zeros(item_num, encoded.shape[1])
    features[movies["movie_id"].to_numpy()] = encoded

    output_path = Path(output_path)
    torch.save(features, output_path)
    print(f"Device: {device}")
    print(f"Saved {len(movies)} movie features to {output_path}")
    print(f"Feature tensor shape: {tuple(features.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 MovieLens 电影的 GPT-2 特征")
    parser.add_argument("--data_path", default="ml-1m")
    parser.add_argument("--output", default="movie_features.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()
    extract_movie_features(args.data_path, args.output, args.batch_size, args.model_path)
