# A：Phase 2 基线结果

工作分支：`split-baselines`。输入使用仓库已有 `ml-1m/`、`data/sequences.json`、
`data/mock_candidates.json` 与 B 交付的 `data/gpt2_movie_features.pt`。
GPT-2 特征是冻结、float32、形状 `[3953, 768]`，3883 个有效电影行。

## 口径

| 模型 | 划分 | 训练及评测 |
| --- | --- | --- |
| popularity | temporal | 只统计 `train` 观看次数，验证历史为 `train`，测试历史为 `train + validation`；测试 6040 人均计入 |
| bpr_mf | temporal | 用户与电影原始 ID 向量内积；每轮从未看过的有效电影采负样本；验证 NDCG@10 选轮次；测试 6040 人均计入 |
| ncf_gpt2 | random | 课程代码的随机 8:1:1 与多正样本 `trainer.py` 评测；只用训练交互构造负样本，验证 NDCG@10 选轮次 |

NCF 的评测协议不同，不能将它的绝对数字与前两行作性能优劣比较。
时间切分未把验证或测试目标强塞进候选；没有用测试集选超参数。

## 测试结果

| model | hr10 | ndcg10 | split |
| --- | ---: | ---: | --- |
| popularity | 0.03692053 | 0.01803267 | temporal |
| bpr_mf | 0.06721854 | 0.03370051 | temporal |
| ncf_gpt2 | 0.52500839 | 0.12534397 | random |

BPR-MF 使用 `seed=42`、64 维、Adam 学习率 0.005、batch size 8192、
正则系数 1e-5、共 10 轮。验证 NDCG@10 最优为第 8 轮
（HR@10 0.06953642，NDCG@10 0.03459776）。

NCF 使用 `seed=42`、32 维、Adam 学习率 0.001、batch size 2048、
共 3 轮。第 3 轮验证 HR@10 0.53639267、NDCG@10 0.12931089。
沿用课程 `trainer.py` 的评价函数，无评测规则改动。

## 复现

本次使用已有 Python 3.10.20 环境（PyTorch 2.7.1+cu126、NumPy 2.2.6、
Pandas 2.3.3），训练进程按 `device.py` 选择 CUDA。无需另建虚拟环境。
仓库 `requirements.txt` 是组内统一版本，若在其他机器复现，按该文件配置环境。

```bash
python -B run_baselines.py --models all
python -B -m unittest tests.test_split tests.test_metrics tests.test_phase1_data tests.test_a_phase2 -q
```

`run_baselines.py` 会在 `data/results.csv` 中按 `model,split` 更新 A 的三行，
保留其他组员已有行。时间切分数据仍由 `split.py` 再生成。
