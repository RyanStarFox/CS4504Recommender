# D：SASRec（第 1 阶段）

入口为 `run_sasrec.py`，模型为 `sasrec.py`。当前阶段实现纯电影 ID 的
SASRec、验证/测试全量排序、HR@10 / NDCG@10、检查点及正式 Top-10 导出。
不修改 A 的划分和指标文件，也不生成临时的正式候选。

## 输入与时间口径

程序严格读取 A 后续交付的 `data/sequences.json`，并保留 MovieLens 原始用户
和电影编号。电影候选从 `ml-1m/movies.dat` 读取，因此 ID 0 和原始编号中的
空缺不会进入推荐结果。

- 训练：只使用每位用户的 `train` 序列，最近 `max_seq_len + 1` 次交互组成
  next-item 训练对；负样本只排除该用户的训练交互，不查看验证或测试目标。
- 验证：以 `train` 为历史，预测 `validation`。
- 测试：以 `train + validation` 为历史，预测 `test`。
- 排序：屏蔽当前历史中看过的电影，再对全部有效电影取有序 Top-10。
- 选模：只按验证集 NDCG@10 early stop；测试集只在恢复最佳参数后评价。

上面的验证/测试历史和屏蔽规则是 D 当前固定的明确口径。A 的划分文件到位后，
如果全组最终采用不同口径，应在跑正式结果前统一调整。

## 运行

按项目统一环境建立 `.venv` 后执行：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_sasrec.py --device auto
```

默认产物为：

- `checkpoints/sasrec_id.pt`：验证集最优模型、配置、有效电影 ID 和指标；
- `data/candidates.json`：验证与测试各用户的有序 Top-10；
- `data/results.csv`：以 `model=sasrec_id, split=temporal` 写入测试指标，重复运行
  会更新该行，不会重复追加。

## 与 B 的接口

`SASRec` 的 `movie_features` 和 `user_features` 默认均为 `None`，此时是第 4 行
纯 ID 模型。冻结特征作为非持久 buffer 随模型移动设备，但不进入优化器或检查点。

第二阶段传入电影特征时，B 的可训练模块注册到 `fusion_module`，并提供：

```python
fuse_items(id_vectors, text_vectors) -> fused_item_vectors
fuse_users(sequence_states, user_vectors) -> fused_sequence_states  # 仅可选 5u
```

两种方法必须保持输入形状。`fuse_items` 同时用于历史序列和候选打分；投影和门控
参数因此会进入优化器及 `state_dict`。第 6 行的对齐损失仍由 B 在 `fusion.py`
实现，再由 D 的训练循环接入，当前阶段未提前实现。

## 第 1 阶段验收

合成数据测试覆盖因果掩码、左侧 padding、负采样、历史屏蔽、候选 JSON、冻结
特征及融合梯度，并实际完成一轮训练。B 已交付的 GPT-2 `[3953, 768]` 与 Qwen
`[3953, 1024]` 特征也已分别通过融合前向与梯度检查。正式训练仍需等待 A 的
`data/sequences.json`。
