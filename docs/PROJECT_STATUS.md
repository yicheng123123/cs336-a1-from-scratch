# 项目状态

审查日期：2026-10-12；真实根目录：`E:\projects`。

范围：**CS336 A1 Core Implementation + Selected Experiments**。以下区分本地已经实现/运行的工作，以及本次 GitHub 文档提交实际包含的内容。旧的“训练集仍编码中、无真实 LM 训练、无生成”状态已失效。

## 本地已完成并有证据的阶段

- Byte-level BPE、GPT-2 style pre-tokenization、incremental pair 更新、streaming TinyStories BPE、encode/decode、special token、iterable encoding。
- Linear / Embedding / RMSNorm / SwiGLU / RoPE / causal Attention / Transformer Block / Transformer LM。
- Cross-Entropy / Gradient Clipping / AdamW / cosine LR schedule / get_batch / checkpoint save-load。
- 全量 TinyStories BPE：vocab 10,000、merges 9,743；629.7227 s，99.55 MiB peak process RSS。
- 最终 uint16 train.bin / valid.bin 已存在：541,229,347 / 2,381,256 tokens。
- 真实 memmap 训练/validation；周期 Checkpoint、resume、分 Segment 日志。
- 1.36M 小模型六组 LR 实验；两组固定窗口比较。
- 51,147,264 参数模型完成 50,000 steps；最终 checkpoint iteration 49,999。model 与 optimizer 在完成检测路径成功恢复，训练直接返回。
- 最终固定 validation CSV：Loss 1.6692702758312226、PPL 5.308292793964665、Seed 2026、204,800 evaluation positions。
- 本次 CUDA 短生成复现 Lily 与妈妈做蛋糕的故事；Temperature、Top-p、max_new_tokens 与 EOS 分支在源码存在。
- 完整参考测试：47 passed、1 xfailed、0 failed、0 skipped，50.37 s；10 条 torch.load FutureWarning。

来源包括 [实验记录](EXPERIMENTS.md)、[轻量结果](results/)、[核验输出](results/verification.md)及[来源哈希](results/source_manifest.json)。只有最终评估 CSV 已保存的指标被当作已有评估结果；没有重跑长训练或整个固定评估。

## 数据 Pipeline

```text
TinyStories raw train / valid
  → 已完成 streaming BPE
  → vocab 10,000 + merges 9,743
  → 已完成 train.bin / valid.bin（little-endian uint16）
  → 真实 next-token LM training + validation
  → final checkpoint（iteration 49,999）
  → 固定窗口评估 + 文本生成
```

原始 train / valid 字节数为 2,227,753,162 / 22,502,601；编码文件为 1,082,458,694 / 4,762,512 bytes。本次在每个 binary 文件的五个位置各读 65,536 bytes，分别抽样 163,840 IDs，没有越出 vocab range。未重做全语料范围扫描或逐字节编码一致性验证。

当前 `encode_tinystories.py` 只提供 train 默认入口，validation 编码入口与历史验证日志未正式归档。数据、Tokenizer JSON、checkpoint 保持在本地。

## GitHub 代码发布状态

审查开始的分支为 `main`，基准 commit `939ef3e409393b07adc74f5633350dafe15a8808`，远端 `cs336-github` 指向用户仓库；`origin` 仍指向 Stanford starter。此前已压缩的历史本次不再改写。

已有用户工作：

- 6 个已跟踪修改：`cs336_basics/model.py`、`nn_utils.py`、`optimizer.py`、`train.py`、`utils.py`、`tests/adapters.py`。其中实际内容差异集中在 train.py，其余主要为文件 mode 差异。
- 8 个未跟踪脚本：`compare_lr_checkpoints.py`、`eval_tinystories_51m.py`、`generate_tinystories.py`、`generate_tinystories_51m.py`、`lr_sweep.py`、`plot_lr_sweep.py`、`smoke_experiment_logging.py`、`train_tinystories_51m.py`。

**按照本次“只提交 README、必要图片和文档”的要求，这些源码没有暂存、改写或提交。** 当前 GitHub 新文档因此展示本地实验证据，但新克隆仍缺少 51M/LR/生成等入口和新版训练循环。README 已明确该限制，未提供误导性的“一键复现全部训练”承诺。

## 未完成与证据限制

- 未完成全部官方书面题/writeup、OpenWebText、Architecture Ablation、正式 Batch Size Sweep。
- 还没有稳健的 GPU throughput/latency benchmark；未找到训练峰值显存的持久化记录，背景中的约 2.129 GiB 未计入已核实结果。
- 未验证多 seed 的指标方差；固定-window PPL 不代表完整 validation 集。
- Checkpoint 未保存 model config、RNG states、Tokenizer/数据版本，resume 不保证与未中断训练的逐步等价。
- 历史低 LR 三组没有独立脚本版本；当前 sweep 脚本只定义高 LR 三组。
- 未提供独立的完整 validation encoding CLI，也缺少生成边界行为测试、KV Cache 和系统性质量评估。
- 当前 `pyproject.toml` 的 torch~=2.11.0 与实际通过测试的 PyTorch 2.5.1+cu124 不一致。

无核心函数的 `NotImplementedError`、TODO 或仅 pass 的实现残留；发现的 pass 是 `tests/conftest.py` 的 DEFAULT sentinel 类，以及 `cs336_basics/__init__.py` 的可选包版本查询异常处理，不是待实现算法。没有将现有组件测试通过等同于完成全作业。

## 阶段性进度

本地的核心实现、TinyStories 数据准备、真实 GPU 训练、恢复、选定 LR 实验、固定评估与生成链路已经完成。尚未完成的课程要求与实验在上文列明；本次不使用缺乏任务权重依据的“全部完成”或总进度百分比。

对外可复现程度仍受“实验代码未发布”和“依赖版本不一致”限制；它们是下一阶段优先任务，而不是继续延长现有 51M 训练。

## 下一阶段

1. 单独审查并发布现有实验代码，统一依赖/版本化配置，确保 fresh clone 的入口可用。
2. 补全验证集编码入口、Checkpoint/RNG/config provenance 及短程 resume/generation 行为检查。
3. 固定实验协议后开展 batch-size/architecture 对照、多 seed、全验证集评估；再按课程范围考虑 OpenWebText 与完整 writeup。

详细拆分见 [ROADMAP.md](ROADMAP.md)。
