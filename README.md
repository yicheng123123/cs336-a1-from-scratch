# CS336 A1 · LLM From Scratch

基于 Stanford CS336 Assignment 1 的个人学习项目：从 Byte-level BPE Tokenizer、基础神经网络层和 AdamW 出发，构建 Decoder-only Transformer，并完成 TinyStories 数据处理、GPU 训练、Checkpoint 恢复、固定窗口评估和文本生成。

项目范围为 **CS336 A1 Core Implementation + Selected Experiments**。它是对课程方法的工程实现与实验练习，没有提出新的 Transformer 架构，也不代表完成了全部书面题、OpenWebText 实验或 Architecture Ablation。

> 发布状态（2026-10-12）：下述实验已在本地核验。本次按限定范围仅发布文档、轻量指标和图片；51M 训练/评估/生成、LR Sweep 脚本及训练循环的新改动仍留在本地，未随本次提交发布。因此新克隆仓库目前可以检查核心组件，但尚不能独立复现全部训练实验。详见 [项目状态](docs/PROJECT_STATUS.md) 与 [复现说明](#reproduction)。

## Key Results

| 项目 | 已核实结果 |
| --- | --- |
| 模型参数 | **51,147,264（约 51M）** |
| 数据与词表 | TinyStoriesV2-GPT4；BPE vocab 10,000 |
| 训练终点 | **50,000 steps**，Checkpoint iteration **49,999** |
| 训练量 | **102.4M token prediction positions**，按最终训练路径计算 |
| 固定窗口 Validation Loss | **1.669270**（Cross-Entropy，nats） |
| 固定窗口 Perplexity | **5.3083**，计算方式为 exp(loss) |
| 评估设置 | Seed 2026；100 batches × 8 × 256 = **204,800 positions** |
| 核心参考测试 | **47 passed, 1 xfailed, 0 failed, 0 skipped** |

最终指标来自 [固定验证结果](docs/results/fixed_validation_final.csv)，对应本地 `artifacts/checkpoints/tinystories_51m.pt`。本次成功加载该 Checkpoint 并在 CUDA 上生成故事。评估是从 validation 文件随机抽取的固定窗口，可有重叠，并非完整验证集或 204,800 个互不重复的位置。50k steps 的计数包含 iteration 0（LR=0）；恢复时丢弃、重算的工作不额外计入 102.4M，训练量也不等于遍历了完整训练语料。

本次核验设备为 **NVIDIA RTX 4060 Laptop GPU（8188 MiB）**。没有找到峰值训练显存的持久化记录，因此不报告未经核实的训练显存、GPU latency 或吞吐 benchmark。

## Architecture

```mermaid
flowchart LR
    A["Token IDs"] --> B["Embedding"]
    B --> C["Pre-norm Transformer Block × 12"]
    C --> D["Final RMSNorm"]
    D --> E["Linear LM Head"]
    E --> F["Next-token logits"]
```

每个 Block 依次执行 RMSNorm → Causal Multi-Head Self-Attention + RoPE → Residual Add，再执行 RMSNorm → SwiGLU → Residual Add。Embedding 与 LM Head 的权重未共享。

| 模型配置 | 数值 |
| --- | ---: |
| Vocabulary / Context Length | 10,000 / 256 |
| d_model / Layers / Attention Heads | 512 / 12 / 8 |
| d_ff / RoPE theta | 1536 / 10,000 |
| Batch Size / Training Steps | 8 / 50,000 |
| Peak LR / Min LR / Warmup | 1e-3 / 1e-4 / 1,000 steps |
| Optimizer / Weight Decay | 自实现 AdamW / 0.01 |
| AdamW betas / epsilon | (0.9, 0.999) / 1e-8 |
| Max Gradient Norm | 1.0 |

配置核对本地 `scripts/train_tinystories_51m.py`、`cs336_basics/train.py` 与 `optimizer.py`；Checkpoint 权重已由同配置模型成功加载。参数量由该架构的参数形状核算，详见 [实验记录](docs/EXPERIMENTS.md)。

## Implementation

基于 PyTorch 张量运算与 autograd 实现以下组件，使用 PyTorch 模块容器、初始化工具及部分基础算子；不调用现成 Transformer 模型。

- **Tokenizer**：Byte-level BPE training、GPT-2 style Unicode pre-tokenization、encode/decode、special token、iterable encoding。
- **BPE 工程优化**：按完整故事分段读取语料；维护 pair 频率和受影响 pre-token 索引，在合并后增量更新统计。最高频 pair 的选择仍扫描当前 pair 表。
- **Transformer**：Linear、Embedding、RMSNorm、RoPE、Multi-Head Causal Self-Attention、SwiGLU、Transformer Block 与 LM Head。
- **训练组件**：数值稳定的 Cross-Entropy、自实现 AdamW、Warmup + Cosine LR Schedule、Global L2 Gradient Clipping、next-token batch。
- **本地实验链路**：memmap 数据读取、周期 validation、model/optimizer/iteration 保存与恢复、分段 CSV 日志、固定窗口评估与 Perplexity。
- **本地生成**：prompt encoding、Temperature、Top-p、max_new_tokens、EOS stopping；采用滑动 context window，无 KV Cache。

源文件位于 [cs336_basics](cs336_basics/)，参考测试位于 [tests](tests/)。新实验入口的发布状态见 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)。

## Dataset & Preprocessing

数据来源为 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的 `TinyStoriesV2-GPT4-train.txt` 与 `TinyStoriesV2-GPT4-valid.txt`。

```text
Raw TinyStories text
  → GPT-2 style pre-tokenization
  → Byte-level BPE（256 bytes + 1 special token + 9,743 merges）
  → vocab 10,000 + ordered merges
  → story-boundary streaming encoding
  → little-endian uint16 train.bin / valid.bin
  → memmap → next-token batches → Transformer LM
```

| Split | 原始文件（bytes） | Token IDs | 编码文件（bytes） |
| --- | ---: | ---: | ---: |
| Train | 2,227,753,162 | **541,229,347** | **1,082,458,694** |
| Validation | 22,502,601 | **2,381,256** | **4,762,512** |

最终 `train.bin` 和 `valid.bin` 均存在，本次不是 `.part` 状态。Token 数由文件大小 / 2 验证；分别在文件首部、1/4、1/2、3/4 和尾部抽样 163,840 个 IDs，均落在 [0, 10,000)；这不是重做全文件范围扫描。

全语料 BPE 的 stats 记录 **629.7227 s**、峰值进程 RSS **99.55 MiB**，实际 vocab 与 merges JSON 数量也已核对。流式 BPE 当前限定单个 `<|endoftext|>`；该 token ID 为 256。内存数字是 tokenizer 进程 RSS，不能解释成模型 GPU 显存。[BPE 规模实验与证据](docs/EXPERIMENTS.md#bpe-scalability)

原始数据、Tokenizer 产物、编码文件和 Checkpoint 均保留在本地并由 `.gitignore` 排除。此仓库不分发完整数据集或训练模型权重。

## Training & Experiments

### 小模型 Learning Rate Sweep

小模型为 **1,362,240 参数**：d_model=64、2 layers、4 heads、d_ff=128、context=128、batch=4，每组 1,000 steps。六组日志均结束于 iteration 999，最终随机 validation loss 与 summary CSV 一致。

| Peak LR | Final Validation Loss |
| --- | ---: |
| 1e-4 | 6.0545 |
| 1e-3 | 3.9514 |
| 3e-3 | 3.5294 |
| 5e-3 | 3.4998 |
| 1e-2 | 3.5499 |
| 3e-2 | 4.2422 |

![小模型 LR Sweep 的 validation loss](docs/figures/lr_sweep_loss_vs_steps.png)

图直接复制自本地已有绘图产物；原始脚本横轴为 iteration + 1。[轻量结果与数据来源](docs/results/lr_sweep_summary.csv)

进一步在 Seed 2026 的相同窗口上比较（102,400 evaluation positions）：

- LR 3e-3：Loss **3.492081**，PPL **32.8542**。
- LR 5e-3：Loss **3.472701**，PPL **32.2237**。

这仅说明 5e-3 在此次小模型、有限训练预算与固定窗口上更好。没有多 seed 的误差估计，**不能将其描述为 51M 模型的最佳 LR**。51M 实验使用 1e-3。详见 [固定窗口比较](docs/results/fixed_validation_lr_comparison.csv)。

### 51M 正式训练与 Checkpoint 恢复

![51M 模型的 Training Loss、Validation Loss 与 Learning Rate](docs/figures/tinystories_51m_training.png)

曲线来自 `artifacts/experiments/tinystories_51m/segment_*.csv`：78 + 24 条日志在 iteration 38,500 重叠，保留后一段记录，得到 **101 个唯一采样点**，覆盖 iteration 0–49,999。每 500 iterations 记录一次并额外记录终点，不存在每步 loss 的完整日志；图中连线未作平滑。

训练曲线的 train loss 是当步单个 batch，validation loss 是 20 个随机 batch 的均值。最后一条随机 validation loss 为 **1.698833**；Key Results 的 **1.669270** 来自独立固定窗口评估，二者不可混用。

Checkpoint 恢复 model、optimizer 和 iteration，LR 按全局 iteration 接续；目前不保存 RNG state，因此 resume 不保证与未中断训练逐步相同。各 Segment 的 elapsed time 从零重新开始，图仅使用 steps，未将局部计时拼成全局 wall-clock time。去重后的 [曲线 CSV](docs/results/tinystories_51m_curve.csv)、[SVG 原图](docs/figures/tinystories_51m_training.svg) 与 [合并规则](docs/EXPERIMENTS.md#51m-training-curve) 可供核查。

## Text Generation

本次通过现有本地生成脚本重新读取最终 Checkpoint，实际设置为 Seed=42、Temperature=0.8、Top-p=0.9、max_new_tokens=150、context=256，设备为 CUDA。

Prompt：

> Once upon a time, there was a little girl named Lily.

真实输出节选：

> Once upon a time, there was a little girl named Lily. She had a big heart and loved to help her mom in the kitchen. One day, her mom said, "Lily, I have a surprise for you." Lily was very happy.
>
> Lily asked her mom, "Can you help me make a cake?" Her mom smiled and said, "Yes, Lily! Let's make a cake together." They mixed the cake batter and put it in a pan. They mixed everything up.

同一样例后面出现了：

> They put it on a rack and the cake. Then, they added the cake to the cake.

[完整原始输出](docs/results/generation_51m.txt)保留了这些错误。模型能够生成基本连贯的短故事，但存在语义重复、语法缺陷、局部逻辑错误和人物角色不一致；它不是通用对话模型，这个样例也不是生成能力的全面评测。

## Reproduction

### 核心组件与已验证环境

从项目根目录运行。当前验证环境为 WSL2 / Ubuntu 24.04.1、Python **3.12.14**、PyTorch **2.5.1+cu124**、NumPy **2.4.4**；本机 CUDA 可用。[核验记录](docs/results/verification.md)

PowerShell 下使用现有 WSL 环境：

```powershell
wsl --cd /mnt/e/projects --exec /mnt/e/projects/.venv-wsl/bin/python -B -m pytest -p no:cacheprovider
```

结果：**47 passed, 1 xfailed in 50.37 s**（2026-10-12）。xfail 为 `test_encode_memory_usage` 的预期行为；另有 10 条旧式 `torch.load` FutureWarning。测试覆盖 BPE、Tokenizer、模型层、数值工具、优化器、get_batch 和 Checkpoint serialization，不等于完成全部课程实验。

新环境可依据 `pyproject.toml` / `uv.lock` 安装核心项目：

```sh
uv sync
uv run pytest -p no:cacheprovider
```

**依赖差异**：当前声明要求 `torch~=2.11.0`，并非上述已验证的 2.5.1 环境；这次没有修改依赖、重新安装环境或在全新环境复测。不应把 `uv sync` 的环境称为本次结果的精确复现环境，CUDA build 与驱动兼容性也需单独确认。

### 数据准备

已发布的 [BPE 脚本](scripts/train_tinystories_bpe.py) 和 [训练集编码脚本](scripts/encode_tinystories.py) 使用固定路径：

```sh
.venv-wsl/bin/python -m scripts.train_tinystories_bpe
.venv-wsl/bin/python -m scripts.encode_tinystories
```

输入置于 `data/`，vocab/merges 位于 `artifacts/tokenizer/full_tinystories_vocab10000/`，编码输出位于 `artifacts/encoded/`。以上是重新构建产物的命令，本次未执行；现有脚本会写入、替换对应文件，已有实验应先保留独立副本。编码脚本当前只提供 train 入口，没有独立的 validation 编码 CLI。

### 训练、评估与生成：当前仅本地入口

以下模块存在于此次检查的本地工作区，**尚未包含在远端代码快照中**。需先另行发布现有实验代码，并准备同格式数据/Tokenizer/Checkpoint，才能在新克隆中执行：

```sh
.venv-wsl/bin/python -m scripts.train_tinystories_51m
.venv-wsl/bin/python -m scripts.eval_tinystories_51m
.venv-wsl/bin/python -m scripts.generate_tinystories_51m
.venv-wsl/bin/python -m scripts.compare_lr_checkpoints
.venv-wsl/bin/python -m scripts.lr_sweep
.venv-wsl/bin/python -m scripts.plot_lr_sweep
```

训练脚本检测已有 Checkpoint 并从 iteration + 1 恢复；已达到 50k 时返回。没有“另建新实验”的 CLI，不能在原路径盲目启动新训练。评估脚本拒绝覆盖已有结果，但该检查位于计算之后；生成脚本只读取产物。本次只运行了短生成、参考测试及训练脚本的“已完成”检查分支，没有进行新的训练或完整验证评估。

`lr_sweep.py` 当前只列出 5e-3、1e-2、3e-2 三组；较低的三组保留了 CSV 与 Checkpoint，但未留存独立脚本版本。复现实验需要补齐版本化配置。

## Limitations & Future Work

- 优先补齐实验代码发布与一致的依赖锁定，让 clone 后的训练、评估、生成命令可直接使用；本次未提交任何用户代码改动。
- 保存 model config、RNG state、Tokenizer/数据版本和训练硬件信息，提升 Checkpoint 与结果的可追溯性。
- 固定窗口 PPL 只代表当前评估协议；需要全验证集、多 seed 和更多 prompt 的评测。Top-p 实现存在不代表 EOS 等采样行为已有独立测试覆盖。
- 尚未完成正式 Batch Size Sweep、Architecture Ablation、OpenWebText、全面 GPU profiling，以及完整官方书面作业/writeup。

[项目状态](docs/PROJECT_STATUS.md) · [实验记录](docs/EXPERIMENTS.md) · [下一阶段路线图](docs/ROADMAP.md)

## Acknowledgements

课程、作业规格、starter repository 和参考测试来自 **Stanford CS336: Language Modeling from Scratch**；保留 [MIT License](LICENSE) 中 Stanford University 的版权与上游归属。本项目不将上游测试或脚手架表述为个人原创。

- [Stanford CS336](https://cs336.stanford.edu/)
- [Official Assignment 1 starter repository](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment handout](cs336_assignment1_basics.pdf)
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
