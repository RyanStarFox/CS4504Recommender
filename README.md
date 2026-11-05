# 基于大语言模型增强的推荐系统

基于 MovieLens-1M，实现经典推荐基线、SASRec 序列召回，以及两类语言模型增强实验：

- **召回侧**：冻结 GPT-2 / Qwen3.5-0.8B 的电影文本特征，与电影 ID 门控融合；在 Qwen 配置上增加对比对齐。
- **重排侧**：固定纯 ID SASRec 的 Top-10，用 Qwen 的 SFT、DPO、GRPO-based ranking optimization 调整顺序。DPO 和 GRPO-based 分别从同一份 SFT 出发。
- **案例演示**：先执行类型过滤，再调用 SASRec 和 Qwen；当前明确要求优先于历史偏好。

课程原有填空及各模块实现已完成。实验结果保存在 [data/results.csv](data/results.csv)。文本增强分支与重排分支分别评价，文本对齐模型不替换 C 使用的正式候选。

## 1. 成员分工

| 姓名 | 分工 | 具体工作 |
| --- | --- | --- |
| 周子涵 | A | 时间切分、共享指标、流行度/BPR-MF/课程 NCF、mock 候选、报告与汇报整合 |
| 聂显 | B | 冻结文本特征、设备策略、门控融合、对比对齐及相关实验 |
| 邵言 | C | Qwen LoRA 的 SFT/DPO/GRPO-based、导出与 llama.cpp 重排评估 |
| 何华宇 | D | SASRec、正式 Top-10 导出、融合接口、类型过滤与 Agent 演示 |

## 2. 目录结构

```text
.
├── README.md / requirements.txt
├── split.py / metrics.py               # 时间切分、单目标 HR/NDCG
├── mf.py / run_baselines.py            # 流行度、BPR-MF、课程 NCF
├── dataset.py / dataloader.py
├── recommender.py / trainer.py / main.py # 课程 NCF + GPT-2
├── device.py / features.py / gpt2.py    # 设备、冻结文本编码
├── sasrec.py / run_sasrec.py            # 序列召回及候选导出
├── fusion.py / text_training.py
├── run_text_sasrec.py / configs/        # 门控及对齐实验配置
├── ranking/                            # SFT、DPO、GRPO-based、生成和评估
├── agent_demo.py
├── tests/                              # 单元测试
├── docs/                               # 各成员的实现与实验说明
├── data/                               # 已报告结果、元数据及共享产物
├── ml-1m/                              # 原始数据；源码提交包不包含，另行下载
├── models/                             # 预训练权重/LoRA；另行下载
└── runs/reproduce/                     # 本文命令生成的新结果
```

GitHub 仓库中已有部分数据和特征缓存；精简源码 ZIP 不包含这些大型产物。下面的下载与生成步骤也适用于从源码 ZIP 开始运行。本文将新训练结果写入 `runs/reproduce/`，保留仓库中已有的结果表和正式候选。

## 3. 运行环境

推荐 **Python 3.10**。从仓库根目录执行；下列多行命令使用 Bash（Linux/macOS，Windows 可用 WSL）。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

主要依赖为 PyTorch 2.6.0、Transformers 5.8.0、safetensors 0.7.0、huggingface-hub 1.27.0、tokenizers 0.22.2、NumPy 1.23.5、Pandas 2.2.2，完整版本以 [requirements.txt](requirements.txt) 为准。Qwen3.5 需要对应的新版本 Transformers，不能沿用课程早期的 4.40.2。

PyTorch 的 CPU/CUDA 安装包需与本机匹配，可先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v260) 选择 **2.6.0** 对应安装包，再安装其余依赖。统一模块使用 CUDA > MPS > CPU；完整语言模型训练优先使用 CUDA，CPU 适合数据检查和小规模流程验证。课程独立入口 `main.py` 保留 CUDA/CPU 选择。

已记录的实验环境并非完全一致：A 使用 PyTorch 2.7.1+cu126，B 使用 2.6.0+cu124；详情见 [A 基线记录](docs/a-phase2.md) 与 [B 实验记录](docs/b-phase2.md)。依赖清单是复现起点，不表示所有历史结果都在同一软件环境产生。

## 4. 数据、模型与外部资源

| 资源 | 下载来源 | 本地位置/用途 |
| --- | --- | --- |
| MovieLens-1M | [GroupLens 数据页](https://grouplens.org/datasets/movielens/1m/) · [官方 ZIP](https://files.grouplens.org/datasets/movielens/ml-1m.zip) | `ml-1m/{ratings,movies,users}.dat` |
| GPT-2 | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `models/gpt2/`，冻结电影文本特征 |
| Qwen3.5-0.8B | [官方模型与固定版本](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | `models/Qwen3.5-0.8B/`，文本编码与重排基座 |
| 三份已训练 LoRA | [RyanFoxW/cs4504-qwen35-movielens-rerank](https://huggingface.co/RyanFoxW/cs4504-qwen35-movielens-rerank) | 固定修订 `9fade15f08929a2bb5080fd0e21b9d5869c6c070`；使用本项目 `ranking.lora` 加载 |
| llama.cpp | [官方仓库](https://github.com/ggml-org/llama.cpp) | 正式重排评估；转换器与二进制使用相同提交 `620d2225cc436ab0b001e3200e590e3b286ef9ab` |

### 4.1 下载数据

已有完整 `ml-1m/` 时跳过此步。仅下载并写入指定的四个文件：

```bash
python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

root = Path("ml-1m")
root.mkdir(exist_ok=True)
with urlopen("https://files.grouplens.org/datasets/movielens/ml-1m.zip") as response:
    archive = response.read()
with ZipFile(BytesIO(archive)) as z:
    for name in ("ratings.dat", "movies.dat", "users.dat", "README"):
        (root / name).write_bytes(z.read("ml-1m/" + name))
print("MovieLens-1M ready:", root.resolve())
PY
```

### 4.2 需要文本模型时再下载

最小运行示例不需要 GPT-2 或 Qwen；执行文本实验或重排时准备：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from features import download_qwen

snapshot_download(
    "openai-community/gpt2",
    local_dir="models/gpt2",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)
download_qwen(revision="2fc06364715b967f1860aea9cf38778875588b17")
PY
```

Qwen 基座修订也记录在 `data/qwen_movie_features.json`；精简提交包应保留该小型元数据文件。基座与 LoRA 必须来自兼容的修订。特征、权重和 GGUF 总体较大，不放入源码 ZIP。

## 5. 最小运行：不下载语言模型

安装依赖并准备 MovieLens 后，在根目录执行：

```bash
python split.py --data-path ml-1m \
  --output runs/reproduce/data/sequences.json \
  --mock-output runs/reproduce/data/mock_candidates.json \
  --manifest-output runs/reproduce/data/split_manifest.json

python run_baselines.py --models popularity \
  --sequences runs/reproduce/data/sequences.json \
  --mock-candidates runs/reproduce/data/mock_candidates.json \
  --results runs/reproduce/results.csv

python -B -m unittest discover -s tests -p 'test_split.py' -q
python -B -m unittest discover -s tests -p 'test_metrics.py' -q
```

预期数据规模为 **6,040 位用户、3,883 部有效电影、1,000,209 条交互**；训练/验证/测试交互数为 **988,129 / 6,040 / 6,040**。流行度测试约为 HR@10 = **0.03692053**、NDCG@10 = **0.01803267**。这一路径检验数据、候选格式和指标，不训练语言模型。

## 6. 训练与实验入口

以下命令继续使用第 5 节生成的时间序列。重新训练的参数、设备和随机性可能使结果有所变化，新结果不应直接覆盖已报告主表。

### 6.1 经典基线与冻结特征

```bash
python run_baselines.py --models mf \
  --sequences runs/reproduce/data/sequences.json \
  --results runs/reproduce/results.csv

python features.py --encoder gpt2 --batch-size 8 \
  --output runs/reproduce/features/gpt2_movie_features.pt
python features.py --encoder qwen --batch-size 8 \
  --output runs/reproduce/features/qwen_movie_features.pt

python run_baselines.py --models ncf \
  --sequences runs/reproduce/data/sequences.json \
  --features runs/reproduce/features/gpt2_movie_features.pt \
  --results runs/reproduce/results.csv
```

两套电影特征形状为 `[3953,768]` 和 `[3953,1024]`，使用原始电影 ID 作下标，空缺行与第 0 行为零；每个 `.pt` 必须保留同名 `.json`。编码器始终冻结，仅使用标题与类型。

`features.py` 会检查本地模型权重和缓存元数据。设备、批量或模型配置与旧缓存不一致时可能拒绝复用，因此上述命令使用新的输出路径。不要为了绕过检查而只复制 `.pt` 或盲目覆盖旧缓存。

### 6.2 纯 ID SASRec

```bash
python run_sasrec.py \
  --sequences runs/reproduce/data/sequences.json \
  --candidates runs/reproduce/data/candidates.json \
  --checkpoint runs/reproduce/checkpoints/sasrec_id.pt \
  --results runs/reproduce/results.csv
```

默认结构为 64 维、最近 50 次交互、2 层/2 头，完整历史用于排除已看电影。验证和测试各导出每用户十部有序候选。若只做运行检查，可额外传 `--epochs 1`，但该结果不能替代完整实验。

已有主表纯 ID 结果缺少完整历史运行超参及检查点，当前默认配置不能保证逐位复现其数值。上述命令重新训练的检查点可以用于本地演示。

### 6.3 门控与对齐实验

`run_text_sasrec.py` 通过 JSON 配置设置输入和输出。先生成独立复现配置：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("runs/reproduce")
root.mkdir(parents=True, exist_ok=True)
for name in ("text_sasrec", "text_sasrec_alignment"):
    config = json.loads(Path("configs", name + ".json").read_text())
    config["sequences"] = str(root / "data/sequences.json")
    config["features"] = {
        encoder: str(root / "features" / (encoder + "_movie_features.pt"))
        for encoder in ("gpt2", "qwen")
    }
    config["checkpoint_dir"] = str(root / "checkpoints")
    config["report_dir"] = str(root / "text_sasrec")
    config["results"] = str(root / "results.csv")
    (root / (name + ".json")).write_text(json.dumps(config, indent=2))
PY

python run_text_sasrec.py --config runs/reproduce/text_sasrec.json --encoder both
python run_text_sasrec.py --config runs/reproduce/text_sasrec_alignment.json --encoder qwen
```

三组实验共享划分、初始化与训练预算；对齐组从相同种子重新训练。详细配置与记录见 [门控实验](docs/b-phase2.md)、[对齐实验](docs/b-phase3.md)。

### 6.4 Qwen 重排与演示

重排默认候选路径是 mock，且训练默认仅 **1 步**。正式输入必须显式指定 `--candidates`，训练时须显式设置 `--steps`。例如 SFT 与 DPO 的运行入口为：

```bash
python -m ranking.train sft \
  --sequences runs/reproduce/data/sequences.json \
  --candidates runs/reproduce/data/candidates.json \
  --steps 712 --output runs/reproduce/sft

python -m ranking.train dpo \
  --sequences runs/reproduce/data/sequences.json \
  --candidates runs/reproduce/data/candidates.json \
  --adapter runs/reproduce/sft --steps 712 --output runs/reproduce/dpo
```

712 步与公开交付记录相符，其余未显式指定的参数是当前代码默认值，不能视为已核实的历史实跑配置。重排训练标签来自 validation 命中用户；测试目标不参与训练。代码虽划出了 select 子集，目前未将其接入自动早停与选模。

GRPO-based 必须从同一份 SFT 出发：先用冻结 SFT 一次性采样排序，保存样本，再更新参数；它不接在 DPO 后面。正式采样与测试使用 llama.cpp，准备方式见下一节。完整参数可查看：

```bash
python -m ranking.train --help
python agent_demo.py --help
```

如果只需要查看组合推荐流程，在第 6.2 节已有 SASRec 检查点后运行：

```bash
python agent_demo.py --user-id 2 --request "我不喜欢恐怖片" \
  --sequences runs/reproduce/data/sequences.json \
  --checkpoint runs/reproduce/checkpoints/sasrec_id.pt \
  --reranker-stage grpo --require-reranker
```

演示会自动获取固定修订的已交付 LoRA 和兼容 Qwen 基座。默认使用 HF/PyTorch；指定 `--reranker-backend llama --gguf ...` 可使用 llama.cpp。`--require-reranker` 保证失败时报错，避免把保留 SASRec 原顺序的降级结果当成成功重排。该案例不替代全体用户评估。更多说明见 [D 的交付说明](docs/d-sasrec.md)。

## 7. llama.cpp 正式重排评估

此节需要 Git、CMake、C/C++ 编译器和额外磁盘空间。固定转换器与二进制的提交，避免不兼容的 GGUF：

```bash
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build
cmake --build third_party/llama.cpp/build --config Release -j
python3.10 -m venv .venv-gguf
.venv-gguf/bin/python -m pip install --upgrade pip
.venv-gguf/bin/python -m pip install -r third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

CUDA 加速构建可在 CMake 配置阶段增加 `-DGGML_CUDA=ON`；其他后端按 [官方构建说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) 配置。该版本转换器依赖 PyTorch 2.11.0、Transformers 4.57.6 和 NumPy 2.2.x，与项目训练环境不同，因此使用独立的 `.venv-gguf`，并通过其解释器路径执行，不切换当前训练环境。下面先在项目环境合并 LoRA，再由独立解释器运行转换器；不用项目内部默认的 macOS 转换器路径：

```bash
python - <<'PY'
from pathlib import Path
from ranking.export import merge_adapter
merge_adapter(
    Path("runs/reproduce/sft").resolve(),
    Path("models/Qwen3.5-0.8B").resolve(),
    Path("runs/reproduce/llama/sft/merged-hf").resolve(),
)
PY

.venv-gguf/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  runs/reproduce/llama/sft/merged-hf \
  --outfile runs/reproduce/llama/sft/sft-f16.gguf --outtype f16 --no-mtp

python -m ranking.train sample --backend llama \
  --sequences runs/reproduce/data/sequences.json \
  --candidates runs/reproduce/data/candidates.json \
  --adapter runs/reproduce/sft \
  --gguf runs/reproduce/llama/sft/sft-f16.gguf \
  --llama-bin third_party/llama.cpp/build/bin/llama-cli \
  --output runs/reproduce/grpo_samples.json

python -m ranking.train grpo \
  --sequences runs/reproduce/data/sequences.json \
  --candidates runs/reproduce/data/candidates.json \
  --adapter runs/reproduce/sft --samples runs/reproduce/grpo_samples.json \
  --steps 712 --output runs/reproduce/grpo

python -m ranking.train eval \
  --sequences runs/reproduce/data/sequences.json \
  --candidates runs/reproduce/data/candidates.json \
  --adapter runs/reproduce/sft \
  --gguf runs/reproduce/llama/sft/sft-f16.gguf \
  --llama-bin third_party/llama.cpp/build/bin/llama-cli \
  --model-name reproduce_sft_test --results runs/reproduce/results.csv
```

对 DPO/GRPO 评估时，分别用相应 adapter 执行上述合并与转换，使用独立的输出目录和 GGUF 文件，再同步替换 `--adapter`、`--gguf` 和 `--model-name`。`merge_adapter` 会重建指定的合并输出目录，因此该目录应仅存放本次生成的中间模型。

`eval` 的生成缓存保存在 `runs/eval/{model-name}.jsonl`。更换候选、模型或评估划分时应使用新的 model-name，避免错误复用旧输出。若存在 `runs/sample_backend`，它会覆盖采样后端，运行前应检查其内容。正式评估只重排同一候选集合；输出解析会去重、过滤非法编号，并按原召回顺序补齐。

### 使用已交付结果核验输入

要评价公开 LoRA 在原实验候选上的表现，先从固定代码版本取得原正式候选，避免把新训练候选当成原候选：

```bash
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download

root = Path("runs/reported")
root.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/RyanStarFox/CS4504Recommender/f6fa138ef9cdeb93416829998298bd58b0fc2159/"
for name in ("sequences.json", "candidates.json"):
    urlretrieve(base + "data/" + name, root / name)
snapshot_download(
    "RyanFoxW/cs4504-qwen35-movielens-rerank",
    revision="9fade15f08929a2bb5080fd0e21b9d5869c6c070",
    local_dir=str(root / "adapters"),
    allow_patterns=["README.md", "sft/*", "dpo/*", "grpo/*"],
)
PY
```

然后使用 `runs/reported/adapters/sft`（或 dpo/grpo）导出 GGUF，并将评估输入切换为 `runs/reported/` 下两个 JSON。这些 adapter 是本项目的 `adapter.json + adapter.pt` 格式，使用 `ranking.lora`；不能直接当作标准 PEFT adapter 加载。

## 8. 指标、结果与限制

时间划分：每人最后一次测试、倒数第二次验证，其余训练；验证历史为 train，测试历史为 train + validation。所有评分作为隐式交互，完整屏蔽已看电影，评估全部 6,040 人。

| 模型 | HR@10 | NDCG@10 | 划分 |
| --- | ---: | ---: | --- |
| 流行度 | 0.036921 | 0.018033 | 时间 |
| BPR-MF | 0.067219 | 0.033701 | 时间 |
| SASRec · 纯 ID | 0.124669 | 0.064670 | 时间 |
| SASRec + GPT-2 门控 | 0.114238 | 0.059923 | 时间 |
| SASRec + Qwen 门控 | 0.115397 | 0.059449 | 时间 |
| SASRec + Qwen 门控 + 对齐 | 0.116060 | 0.060065 | 时间 |
| Qwen SFT 重排 | 0.124669 | 0.063627 | 时间 |
| Qwen DPO 重排 | 0.124669 | 0.064576 | 时间 |
| Qwen GRPO-based 重排 | 0.124669 | 0.065063 | 时间 |
| NCF + 冻结 GPT-2 | 0.525008 | 0.125344 | 随机，独立对照 |

- **HR@10**：前十是否包含真实目标；**NDCG@10**：同时考虑命中位置。
- NCF 是随机 8:1:1、多正样本评估，测试有 5,958 位用户；不可与时间划分的单目标数值直接比较。
- 重排集合固定，因此 HR 保持相同；只改变 NDCG。GRPO-based 相对原始 SASRec 的 NDCG 约提高 **0.61%**，幅度较小。
- 对齐相对 Qwen 门控的 NDCG 约提高 **1.04%**，仅完成单种子验证，尚无显著性结论。
- GRPO-based 是固定样本、组内奖励归一化的优化，不是标准在线 GRPO；纯 ID 完整历史训练预算及部分重排实跑超参缺失，不能保证精确复现原表。

## 9. 课程提交与源码打包

以小组为单位在 Canvas **分别上传**实验报告 PDF 和源码 ZIP；PDF 是主要评分材料，**不要放入 ZIP**。两份文件均包含四位成员姓名：

- `基于大语言模型增强的推荐系统_周子涵_邵言_何华宇_聂显_实验报告.pdf`
- `基于大语言模型增强的推荐系统_周子涵_邵言_何华宇_聂显_源代码.zip`

报告中补齐四位成员学号。截止时间按 Canvas 页面确认（课程通知为 11 月 15 日）。

源码 ZIP 应包含完整实现、测试、配置、README 及必要的小型实验记录；排除原始 MovieLens 数据、特征张量、模型/LoRA/GGUF、检查点、虚拟环境、缓存、运行输出、Git 历史和 PDF/PPT。下列脚本在**仓库根目录**运行，按白名单打包，并检查大小不超过 100 MB：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path.cwd()
required = ["README.md", "requirements.txt", "split.py", "ranking/train.py", "data/qwen_movie_features.json"]
for name in required:
    if not (root / name).is_file():
        raise SystemExit("请在完整项目根目录运行，缺少: " + name)

files = set(root.glob("*.py"))
files.update(root / name for name in ("README.md", "requirements.txt", ".gitignore", ".gitattributes")
             if (root / name).is_file())
for folder, suffixes in (
    ("ranking", {".py"}), ("tests", {".py"}), ("configs", {".json"}),
    ("docs", {".md", ".png", ".svg"}),
):
    files.update(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in suffixes)
for name in ("results.csv", "split_manifest.json", "gpt2_movie_features.json", "qwen_movie_features.json"):
    path = root / "data" / name
    if path.is_file():
        files.add(path)
files.update((root / "data/text_sasrec").glob("*.json"))

outdir = root / "submission"
outdir.mkdir(exist_ok=True)
output = outdir / "基于大语言模型增强的推荐系统_周子涵_邵言_何华宇_聂显_源代码.zip"
if output.exists():
    raise SystemExit("目标压缩包已存在，请先改名保留旧版: " + str(output))
with ZipFile(output, "w", ZIP_DEFLATED) as z:
    for path in sorted(files):
        z.write(path, "CS4504Recommender/" + path.relative_to(root).as_posix())
size = output.stat().st_size
with ZipFile(output) as z:
    assert z.testzip() is None
    assert not any(name.lower().endswith((".pdf", ".pptx", ".pt", ".gguf")) for name in z.namelist())
if size > 100_000_000:
    raise SystemExit("压缩包超过100 MB，请检查文件清单: " + str(size))
print(output)
print(f"{len(files)} files, {size / 1_000_000:.2f} MB")
PY
```

该脚本保留全部推荐实现和运行配置，不需要将报告 PDF 或模型权重塞入压缩包。收件人解压后，可按第 3–6 节安装环境、恢复数据和执行基本运行。提交前检查报告中的姓名/学号、ZIP 可解压，以及 README 资源链接可访问。
