# B：电影文本门控与 SASRec 对照

第一阶段特征继续复用 `data/gpt2_movie_features.pt` 与 `data/qwen_movie_features.pt`。
本阶段实现 `fusion.py`，并通过 `run_text_sasrec.py` 接入 D 的 SASRec。A 的
划分文件与 D 的模型、训练循环、评测函数均直接复用，不修改其代码。

按 `docs/phase.md`，B 的 phase2 验收已完成：5a/5b 共用划分与训练配置，
融合参数有梯度、可保存恢复，用户文本关闭；第 6 行的损失函数与接口已准备。
第 6 行正式对齐训练、5b/6 消融及分析属于 phase3（11/1），不计入本阶段结果。

## 运行与比较口径

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u run_text_sasrec.py --config configs/text_sasrec.json --encoder both
```

`--encoder gpt2` 和 `--encoder qwen` 可分别运行。输入为 A 交付的完整
`data/sequences.json`，覆盖 6040 位用户。验证历史为 train，测试历史为
train + validation；候选范围和历史屏蔽与 D 的 `evaluate_split` 完全相同。
两组均不加载用户属性，不使用对齐损失。

配置统一记录在 `configs/text_sasrec.json`：种子 42、推荐维度 64、历史长度
50、2 层/2 头、dropout 0.2、AdamW 学习率 0.001、weight decay 0、batch 128、
评测 batch 256、最多 50 轮，验证 NDCG@10 连续 5 轮未改善则早停，指标相等时
保留更早的模型。测试集仅在恢复验证最优参数后评价，不用于选模型或超参数。
这是 D 入口的默认训练配置；仓库中既有 `sasrec_id` 结果没有随附训练配置与
检查点，因此本次只保证 5a/5b 两组严格使用同一配置，不推断既有第 4 行的实际预算。

每组在同一个 seed 下先初始化 SASRec，再在隔离的 CPU 随机数上下文中初始化
融合模块。这样不同文本输入维度不会改变两组的 ID/Transformer 初始参数或后续
训练随机数状态；测试对此作逐位核对。批次顺序和逐轮负采样种子也相同。

## 融合接口

`GatedMovieFusion(text_hidden_size, rec_hidden_size)` 注册到
`SASRec.fusion_module`，实现 D 已约定的 `fuse_items(id_vectors, text_vectors)`。
历史输入和候选打分共用这一方法，保持所有前导维度与输出推荐维度不变：

```text
z = LayerNorm(W_text · frozen_text)
g = sigmoid(W_gate · concat(id_vector, z) + b_gate)
fused = id_vector + g * z
```

投影不带 bias，LayerNorm 无可训练仿射参数；零文本行的残差显式置零。门控初始
权重为零、bias 为 -2，初始文本贡献约 0.119。GPT-2 和 Qwen 使用同一结构，
只有投影输入维度分别由真实缓存决定。SASRec 原有的 ID 向量缩放保持不变。
`use_movie_text=False` 时返回原 ID 向量。`project_text` 主动 detach 冻结缓存，
只有投影、门控及推荐模型参数收到梯度。

用户文本不在当前模块中实现；本阶段调用者始终传 `user_features=None`。可选
5u 的注入位置与开关另行实现，避免电影文本实验因文件存在而启用用户模块。

## 检查点与交付

- `checkpoints/sasrec_gpt2_gated.pt`、`checkpoints/sasrec_qwen_gated.pt`：
  保存最优模型全部可训练参数、实验配置、输入校验值和指标；按项目规则不进 Git。
- `data/text_sasrec/sasrec_*_gated.json`：可复现运行记录，包含特征与序列 SHA-256、
  推荐模型配置、设备、dtype、版本、选模规则及指标。
- `data/results.csv`：仅新增或更新本阶段两行，`split=temporal`；其他成员的结果
  保留。本入口不写 `data/candidates.json`，C 继续使用 D 的原始正式 Top-10。

正式运行结束后，从外部冻结缓存重新创建模型并 strict 恢复检查点，核对候选分数
逐位一致。新测试还覆盖随机与两套真实特征的梯度、零行、模块关闭、形状、
历史/打分共用表示，以及融合参数进入 state_dict、冻结缓存不进入 state_dict。
两套真实特征均已完成 CPU 三用户/一轮入口验证，调试产物仅写入临时目录。

## 本次正式结果

两组均在 CUDA/float32 上覆盖全部 6040 位用户，最多 50 轮、patience 5，
实际均在第 32 轮早停，验证最优均出现在第 27 轮。

| 模型 | 验证 HR@10 | 验证 NDCG@10 | 测试 HR@10 | 测试 NDCG@10 |
| --- | --- | --- | --- | --- |
| 5a GPT-2 + 门控 | 0.128146 | 0.065965 | 0.114238 | 0.059923 |
| 5b Qwen + 门控 | 0.121192 | 0.063981 | 0.115397 | 0.059449 |

测试 HR@10 的差异约 0.00116，Qwen 略高；测试 NDCG@10 的差异约 0.00047，
GPT-2 略高。这是一组固定种子的结果，尚不能作显著性结论，也不支持预设
Qwen 在两项指标上都会胜出。后续第 6 行应保持 5b 配置，只改变对齐开关及
明确记录的对齐超参数。

24 项现有及新增测试全部通过。两份正式模型均通过 strict 检查点恢复并核对
候选分数逐位一致；冻结缓存未进入模型检查点。D 的正式候选文件与同步时的
`origin/main` 内容一致，没有被本实验覆盖。

## 第 6 行的准备

`fusion.alignment_loss(id_vectors, projected_text, movie_ids, valid_item_mask,
temperature=0.07)` 提供对称 InfoNCE。调用者传入原始 `item_embedding` 向量和
`fusion.project_text(cache[ids])`，函数排除 0、空缺 ID，并对电影 ID 去重。
少于两部有效电影时返回可反向传播的零损失；温度必须有限且大于零。

对齐权重由后续训练入口显式加入推荐损失。本次 5a/5b 不调用该函数；第 6 行
需要下一阶段在同一 5b 配置上只打开对齐，并记录温度和权重。测试已经验证
重复/无效 ID 排除、有限损失与梯度，尚不将这些测试当作正式对齐消融结果。
