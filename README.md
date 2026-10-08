# 工科创：基于 GPT-2 特征增强的基础电影推荐系统

## 1. 项目目标

本项目使用 MovieLens-1M 数据训练一个基础 Top-K 电影推荐系统。模型同时使用两类信息：

- 用户 ID、电影 ID 的可训练 Embedding；
- GPT-2 从电影标题和类型中提取的文本 Embedding。

完成代码后，程序会训练推荐模型，输出 HR、NDCG 指标，并为一名用户生成 Top-10 推荐电影。

## 2. 包内文件

| 文件或目录 | 内容 |
| --- | --- |
| `ml-1m/` | MovieLens-1M 数据集 |
| `gpt2_model/` | 本地 GPT-2 模型权重，无需联网下载 |
| `dataset.py` | 数据读取与 8:1:1 划分 |
| `dataloader.py` | 训练负采样与评测数据组织 |
| `gpt2.py` | 电影文本特征提取 |
| `recommender.py` | GPT-2 + NCF 混合推荐模型 |
| `trainer.py` | 已完成的训练与评测代码，请勿修改评测部分 |
| `main.py` | 项目运行入口 |

## 3. 需要完成的四处代码

请搜索代码中的 `TODO`，完成以下四项：

1. `gpt2.py`：对 GPT-2 最后一层隐藏状态做掩码平均池化；
2. `dataloader.py`：为每条正样本随机选择一部用户未交互过的负样本电影；
3. `recommender.py`：拼接用户、电影 ID、电影文本三个向量，并用 MLP 计算分数；
4. `recommender.py`：根据正、负样本分数实现 BPR 损失。

每处均提供相邻提示。`trainer.py` 中的评测逻辑已经给出，请保持不变。

## 4. 环境准备

建议使用 Python 3.10。已具备 PyTorch 环境时，可以安装其余依赖：

```powershell
pip install -r requirements.txt
```

PyTorch 的安装方式与电脑是否具有 NVIDIA GPU 有关，可参考课程统一环境或 PyTorch 官方安装命令。代码会自动选择 CUDA 或 CPU。

## 5. 运行步骤

在本目录打开终端。

第一步，完成 `gpt2.py` 的填空并生成电影文本特征：

```powershell
python gpt2.py
```

应生成 `movie_features.pt`，特征形状为：

```text
(3953, 768)
```

第二步，完成另外三处填空并进行快速训练：

```powershell
python main.py --quick --epochs 2
```

第三步，确认快速运行成功后，可使用全部数据训练：

```powershell
python main.py --epochs 3
```

## 6. 运行结果检查

完整流程应满足：

- 成功读取 1,000,209 条评分；
- GPT-2 特征张量形状为 `(3953, 768)`；
- 训练时输出 loss、HR@10 和 NDCG@10；
- loss 总体下降；
- 最后输出验证集、测试集指标及 10 部推荐电影。

训练涉及随机负采样，不同设备上的数值可能有轻微差异。

## 7. 基础概念

- **Embedding**：把离散 ID 映射成可训练的稠密向量。
- **负采样**：从用户未交互电影中选择训练对照项。
- **BPR 损失**：推动正样本分数高于负样本分数。
- **HR@K**：前 K 个推荐中是否命中真实交互电影。
- **NDCG@K**：同时考虑命中情况和命中位置，位置越靠前贡献越大。
