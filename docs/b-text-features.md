# B：冻结电影特征（第 1 阶段）

入口为 `features.py`，共用 `device.py`。`gpt2.py` 保留课程函数和命令行参数，委托给同一提取实现。设备顺序为 CUDA > MPS > CPU；语言模型按设备选择 dtype，推荐模型不在这里转换精度。导入 `device.py` 会在导入 torch 前设置 MPS fallback 与确定性 CUDA 工作区。

## 环境与复现

全组使用 Python 3.10 和唯一依赖入口 `requirements.txt`，项目内虚拟环境统一命名 `.venv`：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python features.py --encoder gpt2 --batch-size 16
.venv/bin/python features.py --encoder qwen --batch-size 8
```

其他机器建立独立的 Python 3.10 虚拟环境并执行 `python -m pip install -r requirements.txt`。CUDA/MPS 可使用相同入口，只调整 batch size 和自动选择的 dtype。跨设备不保证逐位相同。

整个 `models/` 目录忽略入 Git，模型权重、配置和分词文件均在本机准备。GPT-2 从课程包放入 `models/gpt2/`，至少需 `config.json`、`vocab.json`、`merges.txt` 和 safetensors 权重；电影特征文件继续进入 Git。Qwen 权重在 `models/Qwen3.5-0.8B/`，Hub 缓存在项目 `.cache/huggingface/`，均忽略入 Git。首次下载可执行：

```bash
python features.py --encoder qwen --download-qwen --batch-size 8
```

下载先解析仓库 commit，再下载该 commit；本地 `revision.json` 与特征 sidecar 保存真实 revision。复现时将 `--revision` 指定为 `data/qwen_movie_features.json` 的 `revision`。已经下载完成的权重完全离线加载。

完整 revision 仅保存在机器可读配置中，用来锁定实际下载的模型版本；日常运行不需要手输 commit，也不需要在文档中重复整串哈希。

## 特征约定

仅编码 `movies.dat` 的标题、类型，模板为 `Title: {title}. Genres: {genres}.`，类型分隔符 `|` 转为 `, `。输入为原始文本，无聊天模板或生成步骤。两模型都取最后一层 hidden state，在有效 token 上以 float32 掩码平均池化，默认截断到 32 token、右侧 padding。Qwen 仅实例化 `Qwen3_5TextModel`，从多模态 safetensors 筛选文本分支参数，strict 加载；不实例化视觉模型和语言模型输出头。

GPT-2 从 `models/gpt2/vocab.json` 和 `merges.txt` 构建 byte BPE。`models/gpt2/tokenizer.json` 现已替换为校验正常的文件，标准 `from_pretrained` 也能成功加载。权重、配置、词表和合并规则保持原 SHA-256；对全部 3883 部电影，新分词文件与当前提取方式得到的 token ID 完全相同，因此既有电影特征无需重新生成。代码默认仍使用 `models/gpt2/`。

GPT-2 缓存指纹只记录实际使用的配置、词表、合并规则和权重，不包含未使用的 `tokenizer.json`、`tokenizer_config.json`。核验过两条样本文本，4.40.2 与 5.8.0 从词表和合并规则构建的分词结果一致。

交付 `data/gpt2_movie_features.pt`、`data/qwen_movie_features.pt`，都是单一 float32 张量，可用 `torch.load(path, map_location="cpu", weights_only=True)` 读取。行数为 `max(movie_id)+1`，按原始电影 ID 直接取行，0 与空缺电影编号保持零值。维度从实际文本配置解析，并与前向输出核对。用户文本默认不生成、不加载。

每个 `.pt` 有同名 `.json`，记录模型标识、真实 revision、模型及分词文件 SHA-256、数据 SHA-256、模板、池化、截断、设备、dtype、种子、线程数、依赖版本、维度、电影数、结果张量 SHA-256 与小批量验证结果。每次全量提取先检查冻结状态、有限值、重复前向一致性及变长 batch 的 padding 一致性。

重复相同 batch 的结果要求逐位一致。不同 batch 形状会改变半精度 GEMM 的舍入，padding 检查使用相对向量 L2 误差：float32 阈值 `1e-5`，float16 `0.01`，bfloat16 `0.03`；实测误差与阈值都写入 sidecar。GPT-2 的 CUDA/bfloat16 实测约 `0.00307`。该检查用于排除 padding 参与池化，并不承诺不同 batch size 的向量逐位一致。

同参数重复运行会核对配置和张量校验值并直接复用缓存。配置不同、缺少 sidecar 或结果损坏会报错；使用另一个 `--output` 或明确传 `--force` 重新生成。需要重新计算来核对复现时也使用 `--force`。

课程 NCF 可通过 `python main.py --features data/gpt2_movie_features.pt` 读取新特征；根目录旧 `movie_features.pt` 未作为本次交付。

## 本次验收结果

正式特征在 RTX 4060（8 GB）上使用 CUDA/bfloat16 编码，以 float32 保存。Python 3.10.20、PyTorch 2.6.0+cu124、Transformers 5.8.0。两模型均通过 CPU/float32 与 CUDA/bfloat16 小批量前向；MPS 未在本机验证。

| 文件 | 电影数 | 张量形状 | batch size | 保存大小 |
| --- | --- | --- | --- | --- |
| `data/gpt2_movie_features.pt` | 3883 | `[3953, 768]` | 16 | 约 12.1 MB |
| `data/qwen_movie_features.pt` | 3883 | `[3953, 1024]` | 8 | 约 16.2 MB |

两份文件的 0 和空缺 ID 共 70 行均为零；全部有效电影向量非零且有限。张量、输入数据、模型文件与 sidecar 的 SHA-256 已核对。3 个单元测试覆盖设备与 dtype 优先级、掩码池化的半精度溢出、原始 ID 与可读电影文本。GPT-2 缓存复用与截断参数变化时拒绝复用也已验证。

Qwen 使用 Transformers 的纯 PyTorch DeltaNet 实现，未安装额外融合内核。首次 CUDA 全量运行遇到一次 `CUDA unknown error`，未写出文件；同配置重试完成全量提取。小批量相同输入重复前向在两模型上都逐位一致。Qwen 变长 batch 的相对 L2 误差为 `0.00879792`（阈值 `0.03`）。

## 与 D 的后续对接

按 `docs/phase.md` 的 10/18 验收，本次已完成 GPT-2 池化、设备/dtype 策略、真实维度解析、两模型小批量验证、电影特征与生成配置。仍待 B/D 共同确认融合/损失接口；当前下面的约定是 B 的交付说明，不代表已取得 D 的确认。`fusion.py` 的实现、梯度和开关验证属于下一阶段。

D 按原始电影 ID 读取张量，并从 `features.shape[1]` 配置投影输入维度；推荐维度仍独立使用 `rec_hidden_size`。5a/5b 均只打开电影文本，`use_user_text=false`、`use_alignment=false`。第 6 行单独打开对齐。冻结缓存不进入优化器。

第二阶段的 `fusion.py` 尚未在本次实现：需与 D 固定历史输入与候选打分是否共用融合表示、用户向量注入位置，再注册投影/门控为 SASRec 子模块。对齐仅接收去重有效电影 ID 的原始 ID 表示与投影文本表示，排除 padding 和空缺 ID，温度与损失权重显式配置。

模型配置及文本类依据：[官方 Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。
