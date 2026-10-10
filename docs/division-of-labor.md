# 分工

题目：语义对齐的序列推荐与语言模型重排。数据集用现成的 MovieLens-1M。文本只用标题、类型，以及用户的性别、年龄、职业。

四个人各守一块目录，第一天同时开工。对接只通过下面的文件格式，不通过口头约定。

## 用哪个模型

语言模型有两个。排序模型定为 `Qwen/Qwen3.5-0.8B`。

| 模型 | 权重 | 用在哪 |
| --- | --- | --- |
| GPT-2 | 本机的 `models/gpt2/`（整个 `models/` 忽略入 Git），隐藏维度 768 | 第 3 行、第 5a 行。只做冻结特征，不微调 |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B`，隐藏维度 1024 | 第 5b、6 行的冻结文本特征，第 7–9 行的 LoRA，以及最后的演示 |

不把 GPT-2 全部换掉。第 3 行是课程给定的 NCF + 冻结 GPT-2，划分仍用随机 8:1:1，只和原作业对齐，不和时间切分上的结果比高低。第 5a 行和第 5b 行是同一套 SASRec、同一套时间切分，只换编码器。第 6–9 行和演示都用 Qwen3.5-0.8B，不再用 GPT-2 重训排序。

Qwen3.5-0.8B 默认是非思考模式，排序时不要打开思考。只使用文本分支，不加载视觉编码器。语言模型特征维度从实际加载模型的文本配置动态读取（例如 `config.text_config.hidden_size` 或文本模型的 `config.hidden_size`），并核对输出张量；推荐模型维度单独配置为 `rec_hidden_size`，由投影层连接两者。课程原始 `transformers==4.40.2` 加载不了它，全组按 `requirements.txt` 统一到 Python 3.10、PyTorch 2.6.0、Transformers 5.8.0；CUDA wheel 可带平台后缀。

基础流程分别批量编码约 4000 部电影，存成 `.pt` 后按生成配置复用，不再重复编码；约 6000 个用户的属性文本仅在可选消融启用时生成。费时间的是第 7–9 行，只训 Qwen3.5 的 LoRA。DPO 和 GRPO 要同时放当前模型和一份冻结的参考模型。

## 打底运行方式

微调用 PyTorch + Transformers。推理用 llama.cpp。vLLM 不进这个作业。

| 步骤 | 用什么 | 说明 |
| --- | --- | --- |
| 电影、用户特征 | PyTorch + Transformers 前向 | 取出文本最后一层 hidden state。这是向量，不是生成，所以不走 llama.cpp |
| SASRec、MF、NCF | PyTorch | 与语言模型的部署方式无关 |
| SFT、DPO、GRPO 的参数更新 | PyTorch + Transformers + LoRA | 这三步都要梯度。环境变量设 `PYTORCH_ENABLE_MPS_FALLBACK=1`，MPS 上没有的算子回退到 CPU |
| GRPO 的一组排序 | llama.cpp 采样一次 | 从合并后的冻结 SFT 模型采样，存成文件。PyTorch 只读这些样本算 NDCG 和损失，训练中不再调用模型生成 |
| 第 7–9 行的测试指标、Agent 演示 | llama.cpp | 报告里的生成结果都从这里出。LoRA 先合并进基座，再转成 GGUF。需要新版 llama.cpp，旧版没有 Qwen3.5 的算子 |

`PYTORCH_ENABLE_MPS_FALLBACK=1` 在 CUDA 上没有效果，各台机器都照样设置。设备优先级仍是 `cuda > mps > cpu`。回退不能保证所有算子兼容，须先做小样本验证；DeltaNet 若大量落到 CPU，Mac 上的微调会变慢；全量 SFT、DPO、GRPO 优先放在 CUDA 上。

## 设备

优先级是 CUDA GPU，然后 Apple MPS，最后 CPU。四个人都调用同一个 `device.py`，由 B 在第一天写好。

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32
```

语言模型用这个 dtype 加载，再 `.to(device)`；MF、NCF、SASRec 及融合模块默认使用 float32，混合精度训练另行显式配置。不用 `device_map="auto"`，也不按机器改模型结构。存到磁盘上的 `movie_features.pt` 和 `user_features.pt` 一律是 `float32`，后面的 SASRec 不跟着语言模型的半精度走。batch size 可以按设备调小，数据格式和指标不变。

选择顺序固定为 `cuda > mps > cpu`。有 NVIDIA GPU 时用 CUDA；没有 CUDA、但是 Apple Silicon 时用 MPS；两者都没有时用 CPU。三套后端跑同一套模型和数据，只允许 batch size 和 dtype 不同。

微调进程启动前设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`。抽特征同样走 PyTorch。SFT、DPO、GRPO 的更新以 CUDA 和 MPS 为正式训练设备。CPU 必须能把同一条代码跑通，用来核对流程，不作为全量 GRPO 的训练机器。写进报告的重排结果用 llama.cpp 生成。

## 四个人各做什么

| 成员 | 负责 | 写这些文件 | 不改这些文件 |
| --- | --- | --- | --- |
| A 数据、经典基线、报告 | 时间切分；流行度、BPR-MF、课程 NCF；HR@10 / NDCG@10；用流行度先交一份假候选。后半段把四人的幻灯片收成一份并主讲 | `split.py`、`metrics.py`、`mf.py`、`run_baselines.py`，以及课程填空所需的 `dataloader.py`、`recommender.py`；最终 `report.pptx` | `sasrec.py`、`features.py`、`ranking/`、`agent_demo.py` |
| B 表示 | 冻结 GPT-2 与 Qwen3.5 的电影特征；门控注入；Qwen3.5 对比对齐；用户属性文本作为独立可选消融；`device.py` | `features.py`、`fusion.py`、`device.py`、课程 `gpt2.py` | `split.py`、`sasrec.py`、`ranking/`、`agent_demo.py` |
| C 排序 | SFT、DPO、GRPO-based ranking optimization；合并 LoRA 并交给 llama.cpp 出第 7–9 行的测试结果 | `ranking/` | `split.py`、`sasrec.py`、`features.py`、`agent_demo.py` |
| D 序列召回与演示 | SASRec；导出正式 Top-10；报告最后的 Agent 演示 | `sasrec.py`、`run_sasrec.py`、`agent_demo.py` | `features.py`、`ranking/`、`dataloader.py`、`recommender.py` |

课程代码里的 4 处 TODO 可复用本地 `master` 已完成的对应实现：`dataloader.py` 和 `recommender.py` 由 A 移植，`gpt2.py` 的池化由 B 移植；只按文件职责恢复，不合并整个旧分支。这些课程填空服务于第 3 行基线，SASRec 的门控和对齐另外实现。`trainer.py` 的评测函数保持原样，只给课程 NCF 那一行使用。四个人从 `device.py` 取设备，语言模型另取加载 dtype；推荐模型默认 float32。

## 共享文件格式

电影编号、用户编号都用 MovieLens 原始编号，直接作为张量下标。张量长度是 `max(id) + 1`，空缺编号处为 0。

### `data/sequences.json`（A 交）

每位用户按 `timestamp` 从早到晚排列。最后一次是测试，倒数第二次是验证，更早的是训练。MovieLens-1M 的用户都不少于 20 次评分，不用再丢用户。

```json
{
  "1": {
    "train": [1193, 661],
    "validation": 914,
    "test": 3408
  }
}
```

### `data/movie_features.pt` 与 `data/user_features.pt`（B 交）

`torch.float32`，维度用各自模型的 `hidden_size`，不写死。电影句子用标题和类型，用户句子用性别、年龄、职业。文件分成两套：

- `data/gpt2_movie_features.pt`：GPT-2，`[电影数, 768]`，给第 3 行和第 5a 行
- `data/qwen_movie_features.pt`：Qwen3.5-0.8B，给第 5b、6 行
- `data/qwen_user_features.pt`：独立可选的用户属性特征，仅在 `use_user_text=true` 的消融中使用；默认不生成、不加载。年龄、职业代码须映射为可读文字，不能把编号视为偏好语义。

GPT-2 不编码用户画像。第 3 行的课程模型只使用电影特征。

### `data/candidates.json`（D 交正式版，A 先交流行度假候选）

SASRec 对验证、测试各给 10 部电影。列表顺序就是召回顺序。

```json
{
  "validation": { "1": [914, 1193, 661] },
  "test": { "1": [3408, 1193, 661] }
}
```

### `data/results.csv`（谁跑完谁追加）

列固定为 `model,hr10,ndcg10,split`。`split` 只有 `temporal` 和 `random`。随机 8:1:1 只给课程 NCF 用，其余模型都写 `temporal`。

## 每一步做什么

前 6 行的语言模型都冻结，只训推荐模型自己的参数。第 7–9 行才训练 Qwen3.5 的 LoRA。

| 行 | 具体做什么 | 用的模型 | 谁跑 |
| --- | --- | --- | --- |
| 1 | 按训练集里的观看次数给电影排序，作为最低基线 | 无 | A |
| 2 | BPR-MF：用户 ID 和电影 ID 做内积，BPR 训练 | 无 | A |
| 3 | 补全课程的 4 处 TODO。NCF 拼接用户 ID、电影 ID、GPT-2 电影特征，随机 8:1:1 | 冻结 GPT-2，本地权重 | A 训练，B 提供 `gpt2_movie_features.pt` |
| 4 | 按时间切分，SASRec 只用电影 ID 预测下一部，导出验证和测试的 Top-10 | 无 | D |
| 5a | 同一套 SASRec，把冻结 GPT-2 的电影向量用门控加进 ID 向量 | 冻结 GPT-2 | D 的接口 + B 的特征 |
| 5b | 和 5a 保持相同输入与融合结构，只换成 Qwen3.5 电影向量，不加入用户文本 | 冻结 Qwen3.5-0.8B | B 的特征，D 的 SASRec |
| 5u（可选） | 在 5b 上单独打开用户属性文本，其他配置保持一致；与 5b 比较用户文本收益 | 冻结 Qwen3.5-0.8B | B 与 D |
| 6 | 在 5b（无用户文本）上加对比损失，让同一部电影的 ID 向量和 Qwen3.5 向量接近；只改变对齐开关。Qwen3.5 仍然冻结 | 冻结 Qwen3.5-0.8B | B |
| 7 | 用「看过 A、B、C，给 Top-10 编号排序」做监督微调，重排第 4 行的 Top-10 | Qwen3.5-0.8B + LoRA | C |
| 8 | 同一提示上做 DPO。chosen 是 ground-truth next movie，rejected 是同一个 Top-10 里的另一部 | 第 7 行的 LoRA 继续训，另留一份冻结参考模型 | C |
| 9 | GRPO-based ranking optimization。从冻结的 SFT 模型采样一组排序，奖励是相对 ground-truth 的 NDCG | 同一套 Qwen3.5-0.8B + LoRA | C |

第 4 行到第 5a 行看文本有没有用，第 5a 行到第 5b 行看换编码器有没有用，两边都只改一个因素。第 7 到第 9 行回答的问题是：Does policy optimization further improve LLM-based ranking? 三行都要写进表里。结果可以是 SFT < DPO < GRPO，也可以是 SFT 更好、后两行接近。表上不预设谁赢。

Agent 不进这张表。

## 偏好对怎么造

只用时间切分的下一部真实观看，不用人工标注，也不用模型自己的分数。

只保留 ground-truth 已经在 SASRec Top-10 里的用户：

- chosen = ground-truth next movie
- rejected = 同一个 Top-10 里的另一部电影

DPO 学的是对 ground-truth recommendation 的相对偏好。验证集用来选参数，测试集的下一部不参与训练。

GRPO 使用同一批固定的 Top-10。一组排序由 llama.cpp 从合并后的冻结 SFT 模型采样一次，存成文件；PyTorch 读取这些样本，奖励用相对 ground-truth 的 NDCG，再做组内归一化。训练时不再重新采样，也不和用户交互。报告里的名字用 GRPO-based ranking optimization。测试集上的重排同样用 llama.cpp。

测试时重排原始 Top-10。ground-truth 可以不在这 10 部里。

## 怎么并行

依赖只有三条：D 的 SASRec 要接 A 的序列，B 的融合要接 D 的 SASRec，C 的重排要接 D 的候选。Agent 等重排模型，由 D 最后接上。其余从第一天同时做。

```text
第 1 天（同时）
  A  写 split.py，交出 data/sequences.json 和 metrics.py
  A  另写一份 data/mock_candidates.json（流行度 Top-10），格式与正式候选相同
  B  复用课程池化，写 device.py；先小批量验证，再用 movies.dat 抽 GPT-2 和 Qwen3.5 电影特征，不需要划分结果。users.dat 特征留到可选消融
  D  用 A 的序列写 SASRec；序列未到之前，先把 sasrec.py 的接口和训练循环写好
  C  用 mock 候选把 SFT、DPO、GRPO 的训练循环跑通

第 2 段（同时）
  A  跑完第 1–3 行
  D  训 SASRec，把 data/candidates.json 换成正式 Top-10
  B  用随机向量把 fusion.py 测通；特征文件完成后接到 D 留出的接口
  C  正式候选一到，只换输入文件，重跑第 7–9 行

最后
  D  写 agent_demo.py，调用已有的 genre filter、SASRec 和 C 的 llama.cpp 重排。只演示，不进主表
  B、C、D  各交自己那几页幻灯片和讲稿要点
  A  收成一份 report.pptx，并负责 pre
```

D 在 `sasrec.py` 里留两个可选参数，默认 `None`：

```python
movie_features: torch.Tensor | None = None  # [item_num, hidden]
user_features: torch.Tensor | None = None   # [user_num, hidden]
```

为 `None` 时就是纯 ID 的第 4 行，D 不用等 B。B 完成后把张量传进去，得到第 5、6 行。第 6 行的对比损失写在 B 的 `fusion.py` 里，由 D 的训练循环调用。

### B 与 D 的最小接口约定

- `use_user_text=false`、`use_alignment=false` 为默认值；5a、5b 都只用电影文本，6 只增加对齐，5u 只增加用户文本。不因特征文件存在就自动启用模块。
- `features.py` 负责冻结编码与缓存，记录模型标识、文本模板、池化方式、截断长度和真实维度；缓存统一 float32，原始 ID 对齐，0 和空缺行保持零值。
- `fusion.py` 提供可训练电影投影与门控，D 将其注册为 SASRec 子模块并纳入优化器、检查点。双方明确历史输入与候选打分是否使用同一融合表示，以及可选用户向量的注入位置。
- 对齐损失接收同一组去重有效电影 ID 对应的原始 ID 表示与投影文本表示；排除 padding、空缺 ID，温度与损失权重显式配置。语言模型和缓存不更新，投影、门控及 ID 表示按目标反向传播。
- 先以随机特征验证形状、有效 ID、有限损失、梯度和开关行为，再接真实特征；正式对比固定划分、种子、训练预算和选模规则，仅改变被比较因素。

### A/C/D 仍需确认的事项（不阻塞 B 的冻结特征工作）

- 时间切分需固定相同 timestamp 的排序规则、验证/测试使用的历史及历史屏蔽规则；所有方法共用同一份划分。
- Top-10 内重排且仍输出 10 项，HR@10 不变；若希望改善命中率，需另行统一更大的候选数和候选召回率评价。
- 重排训练样本应来自训练序列内部的历史前缀与下一次交互，验证/测试目标不参与训练；现有候选格式还需另定义训练输入。
- SFT 完整排序与 DPO 单电影回答的输出任务需统一；固定 SFT 样本的奖励优化须明确采样概率与更新方式，不直接等同于标准在线 GRPO。

C 的数据读取只认 `sequences.json` 和 `candidates.json` 的路径。`mock_candidates.json` 与正式文件字段相同，所以调试阶段不用改训练代码。

## Agent 演示

放在报告最后一页，按这个顺序讲：

1. 用户说「我不喜欢恐怖片」
2. Agent 调用 genre filter
3. SASRec 在过滤后的电影里召回
4. 用第 7–9 行里验证集更好的那个 Qwen3.5 重排模型重排
5. 给出最终推荐

这一段用来演示系统，不报告 HR 和 NDCG。材料由 D 做成幻灯片，A 收到终稿里。

## 报告和 PPT

课程要交的报告就是这一份 PPT，10 页左右。不另写长文。每人只写自己负责的页，并附几句讲稿要点。A 的编码集中在前半段，后半段负责收成一份 `report.pptx` 并主讲。B、C、D 不改别人的技术结论，A 合成时也不改数字，只统一版式、页序和讲法。

| 页 | 内容 | 谁写 |
| --- | --- | --- |
| 1 | 问题：序列推荐、文本表示、重排 | A |
| 2 | 数据与时间切分 | A |
| 3 | 流行度、BPR-MF、课程 NCF | A |
| 4 | SASRec 与正式 Top-10 | D |
| 5 | GPT-2 与 Qwen3.5 特征、门控 | B |
| 6 | 对比对齐 | B |
| 7 | SFT 重排 | C |
| 8 | DPO 与 GRPO-based ranking optimization | C |
| 9 | 主结果表。第 4 行到第 5a 行、第 5a 行到第 5b 行各只改一个因素；第 7–9 行不预设谁更好 | A 排版，数字来自 `data/results.csv` |
| 10 | Agent 演示，以及四人各用两三句话写的贡献 | D 写演示，四人各写贡献，A 收成这一页 |

主讲是 A。讲到第 4–8 页和第 10 页的演示时，按对应作者写的讲稿要点讲，不临时改实验结论。

## 分支

从 `main` 拉四个分支：`split-baselines`、`text-features`、`sasrec`、`llm-rank`。各自只提交上表里自己的文件。下面这些数据文件直接进 Git，谁产生谁合并进 `main`，其他人再拉取：

- `data/sequences.json`、`data/candidates.json`、`data/mock_candidates.json`
- `data/gpt2_movie_features.pt`
- `data/qwen_movie_features.pt`、`data/qwen_user_features.pt`

GitHub 单文件上限是 100MB。这几份特征都是 float32：GPT-2 电影特征约 12MB，Qwen3.5 电影特征约 16MB，用户特征约 24MB。JSON 更小。都不用压缩，也不用 Git LFS。float32 特征几乎压不小，打成 zip 仍然要占仓库空间。

Git LFS 的免费额度大约是 1GB 存储和每月 1GB 流量，每次克隆都扣流量。这些小文件不值得占用额度。真正超限的是 `models/gpt2/model.safetensors`（约 523MB），它继续留在本机，不上传。
