# B：第 6 行对齐消融与汇报交付

第 5b 行保持已有 Qwen 冻结电影特征与门控模型。第 6 行从同一种子重新训练，
仅增加电影级对齐项；不加载用户属性，不更新语言模型，不从 5b 最优检查点继续训练。

## 配置与复现

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u run_text_sasrec.py --config configs/text_sasrec_alignment.json --encoder qwen
.venv/bin/python build_b_materials.py
```

`configs/text_sasrec_alignment.json` 去掉 `alignment` 后，与已交付的 5b 配置
完全相同：同一 A 划分、seed 42、64 维 SASRec、历史 50、2 层/2 头、dropout 0.2，
同一负采样与批次顺序、AdamW 学习率 0.001、batch 128、最多 50 轮、patience 5。
只按验证 NDCG@10 选最早的最佳模型，恢复后对完整测试集评价一次。

固定超参数为权重 0.01、温度 0.07，每批最多 256 部不同电影。本次不扫参，
也没有基于测试集结果重新选择权重或温度。

## 训练接口与公平性

第 5a/5b 仍使用 D 的原 `train_model`；新增的 `text_training.train_aligned`
复用 D 的数据集、BPR 损失与评测函数，保持优化器、梯度裁剪、早停和恢复规则一致。
没有修改 `sasrec.py`、`run_sasrec.py` 或 A/C 的训练代码。

对齐电影仅来自当前 batch 的训练历史与训练正样本。先排除 0/空缺并去重，超过
256 部时，使用独立 CPU Generator 均匀抽样；种子为 `seed + 100000 + epoch`。
这个 Generator 不改变推荐模型的 dropout 随机数。向量分别取原始电影 ID
embedding 与投影文本表示，经 L2 归一化后计算对称 InfoNCE。

总损失为 `BPR + 0.01 * InfoNCE`。投影与 ID embedding 接收对齐梯度，门控仍
由推荐损失更新；冻结文本缓存不接收梯度。对齐权重为 0 的两轮回归测试逐位核对
新旧训练循环的全部模型参数、指标和训练后 RNG 状态，结果完全一致。

## 产物

- `data/results.csv`：新增 `sasrec_qwen_aligned`，保留其他成员结果。
- `data/text_sasrec/sasrec_qwen_aligned.json`：配置、输入校验值、验证与测试结果、
  逐轮训练记录、最佳轮次和实际训练轮次。
- `checkpoints/sasrec_qwen_aligned.pt`：本机最优检查点，按项目规则忽略入 Git。
- `docs/figures/b-encoders.*`、`docs/figures/b-alignment.*`：PNG 与 SVG 对照图。
- `docs/b-representation.pptx`：供 A 合成的第 5、6、10 页，含演讲者备注。
- `docs/b-speaker-notes.md`：同一份讲稿的可读文本。

`build_b_materials.py` 从三份运行报告读取数字并生成图、幻灯片和讲稿，避免手工
抄写结果。重建图与 PPT 所需依赖统一写在 `requirements.txt`。

5u 用户属性文本属于可选消融，本次未启用，不影响 B 的必做任务验收。

## 正式结果与分析

两组均在同一最大预算下于第 32 轮早停、选择第 27 轮；全部 6040 位用户参与评价。

| 模型 | 验证 HR@10 | 验证 NDCG@10 | 测试 HR@10 | 测试 NDCG@10 |
| --- | --- | --- | --- | --- |
| 5b Qwen + 门控 | 0.121192 | 0.063981 | 0.115397 | 0.059449 |
| 6 Qwen + 门控 + 对齐 | 0.127483 | 0.065993 | 0.116060 | 0.060065 |

测试 HR@10 增加 0.00066225，对应多命中 4 位用户；NDCG@10 增加 0.00061589，相对约 1.04%。
单种子、固定超参数下观察到小幅提升，不作显著性或普遍有效性结论。对齐的平均
InfoNCE 从约 6.01 降到 0.56，但这不等于推荐指标必然大幅提升；正式结论以
恢复验证最优模型后的测试指标为准。没有因测试结果调整权重、温度或重选训练。

共 26 项项目测试及 16 项排序接口测试通过；正式检查点 strict 恢复后的候选
分数逐位一致。结果表保留原有 9 行，仅新增第 6 行；正式候选文件不变。

B 的必做实验、配置、结果、图、三页 PPT 与讲稿已交付。项目整体的 Agent
演示和最终合成 PPT 仍分别由 D、A 负责，本交付不替代其他成员的验收。
