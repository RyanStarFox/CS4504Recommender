# A：时间切分、指标与假候选（Phase 1）

在仓库根目录执行 `python split.py`，生成以下文件。此阶段使用 Python 标准库，
无需加载语言模型或安装 PyTorch；项目正式训练环境仍按 `requirements.txt` 配置。

| 文件 | 内容 |
| --- | --- |
| `split.py` | 数据读取、时间切分、流行度假候选及生成配置 |
| `metrics.py` | 时间切分的单目标 HR@K / NDCG@K 与全用户平均 |
| `data/sequences.json` | 全部 6040 名用户的 train / validation / test |
| `data/mock_candidates.json` | 验证、测试各用户有序的流行度 Top-10，供 C 调试 |
| `data/split_manifest.json` | 划分口径、数量、校验值与假候选调试指标 |

## 固定的数据口径

1. 所有评分视为隐式交互，不按星级筛选，不重新编号，不丢弃用户。
2. 每位用户按 timestamp 升序排列；相同 timestamp 保持 `ratings.dat` 中的
   原始行顺序。相同时间戳无法推断更细粒度的真实先后，这里采用固定规则复现。
3. 最后一次为 test，倒数第二次为 validation，其余为 train。发现交互过少、
   同一用户重复电影或未知编号时直接报错。
4. 验证历史为 train；测试历史为 train + validation，与 D 当前入口一致。
   屏蔽全部历史电影，屏蔽规则不因模型截断历史长度而变化。
5. 流行度只统计 train 内的交互次数，按次数降序、电影原始 ID 升序排序。
   对每位用户屏蔽相应历史后选 10 部。保留全部用户，不强行插入真实目标，
   不生成或覆盖 D 的正式 `data/candidates.json`。

## 指标接口

```python
from metrics import hr_at_k, ndcg_at_k, evaluate_rankings

hr_at_k([8, 4, 2], ground_truth=4, k=10)       # 1.0
ndcg_at_k([8, 4, 2], ground_truth=4, k=10)     # 1 / log2(3)
evaluate_rankings({"1": [8, 4], "2": [2, 3]}, {"1": 4, "2": 9})
# {"hr10": 0.5, "ndcg10": 0.31546487678572877}
```

每名用户只有一个真实下一部电影。按全部评测用户平均，目标未进候选的用户记
0 分；若预测和目标覆盖的用户不同则报错。输入须是有序、无重复、有效的正整数
电影 ID。返回值键为 `hr10` / `ndcg10`（传入其他 K 时相应变化）。课程 NCF
仍使用 `trainer.py` 的原始随机切分、多正样本评测。

固定 Top-10 内仅改变顺序时，HR@10 不变，NDCG@10 可变化。

## 本次交付与复现

| 项目 | 数量 |
| --- | --- |
| 用户 | 6040 |
| 有效电影 | 3883 |
| 训练交互 | 988129 |
| 验证交互 | 6040 |
| 测试交互 | 6040 |
| 合计 | 1000209 |
| 每个 split 的候选用户 | 6040 |

验证假候选命中 258 名用户，测试假候选命中 223 名用户。这里只用于接口调试，
指标记录在 manifest；第 1–3 行正式基线结果留到 Phase 2 写入 results.csv。

```bash
python split.py
python -m unittest tests.test_split tests.test_metrics tests.test_phase1_data -v
```

10 项测试覆盖合成样例、相同时间戳、负面输入、指标排名折扣、全部用户平均、
生成复现、源文件换行差异，以及全部原始交互和 12080 份候选列表的核对。
测试也实际调用 C 的 `ranking.data.load_bundle`；当前读入 233 个训练用户、
25 个选参用户、223 个测试用户，这是 C 当前读取器按目标是否入榜筛选后的
结果，交付 JSON 仍覆盖全体 6040 名用户。

输入文本 SHA-256 统一先将 CRLF 转为 LF；输出文件 SHA-256 校验实际字节。
`.gitattributes` 固定这三份生成 JSON 使用 LF，确保跨系统签出与复现一致。
源数据、两个输出 JSON 的校验值及调试指标可在 manifest 中核对。

## 与 C / D 的对接

- D 可以直接运行 `python run_sasrec.py --sequences data/sequences.json`。
  文件满足现有 `load_sequences` 的字段、ID 和历史长度要求。当前环境没有
  PyTorch，因此本次只执行 A 的标准库测试和 C 的数据读取，未执行 SASRec 训练。
- C 调试时指定 `--sequences data/sequences.json --candidates data/mock_candidates.json`。
  真实标签不在假候选中的情况会自然出现。
- C 当前读取器会过滤未命中的测试用户，并且测试提示历史仍只有 train。
  正式报告评测需覆盖全部用户，测试历史需与 D 对齐为 train + validation。
- C 当前使用 validation 目标构造训练样本；正式训练数据仍需按分工文件约定，
  从 train 序列内部构造历史前缀和下一次交互，保留验证目标用于选参。
  这些训练协议由 C 后续接入，不影响本次共享文件格式。
