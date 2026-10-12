# 项目状态

审计日期：2026-10-11
项目在 Windows 与 WSL 环境中检查；数据与模型运行产物保留在本地。

## 已完成

### 核心实现

- `cs336_basics/tokenizer.py`：GPT-2 风格正则 pre-tokenization、Byte-level BPE、支持 streaming BPE 训练、增量 pair 统计更新、encode/decode、special token 与 iterable encode。
- `cs336_basics/model.py`：Linear、Embedding、RMSNorm、SwiGLU、RoPE、causal attention、多头 self-attention、Transformer Block、Transformer LM。
- `cs336_basics/nn_utils.py`：softmax、cross entropy、gradient clipping。
- `cs336_basics/optimizer.py`：AdamW、带 warmup 的 cosine learning-rate schedule。
- `cs336_basics/utils.py`：随机 next-token batch、checkpoint 保存与加载。
- `cs336_basics/train.py`：训练循环、validation loss 估计和周期保存 checkpoint。

### BPE 与 TinyStories 数据

- 原始训练文本存在：`data/TinyStoriesV2-GPT4-train.txt`，2,227,753,162 bytes。
- 原始 validation 文本存在：`data/TinyStoriesV2-GPT4-valid.txt`，22,502,601 bytes。
- 全语料 vocab、merges、stats 已写入 `artifacts/tokenizer/full_tinystories_vocab10000/`；实际 stats 为 vocab 10,000、merges 9,743、629.7227 s、峰值进程 RSS 99.55 MiB。
- `scripts/train_tinystories_bpe.py` 实现完整 BPE 训练、vocab/merges JSON 持久化和重载检查。
- `scripts/encode_tinystories.py` 实现故事边界分段、uint16 token 写入、memmap 范围和 batch 检查，并在检查通过后将 `.part` 改为正式文件名。
- `valid.bin` 已存在，大小 4,762,512 bytes，即 2,381,256 个 uint16 token IDs；此前及本次检查确认其数据范围和 batch 对齐有效。

## 当前进行中

`artifacts/encoded/train.bin.part` 正由 WSL 进程 `python -m scripts.encode_tinystories` 编码；审计期间文件持续增长。不要启动第二个编码进程，也不要删除、重命名或覆盖临时文件。

因为编码未结束，以下信息尚无最终证据：`train.bin` 是否生成、token 总数、完整 ID 范围与 memmap 验证。正式训练入口目前不会自动读取该文件。

## 未完成或当前限制

- **真实 TinyStories LM training**：训练函数可以接受数组和 validation 数据，但 `if __name__ == "__main__"` 使用 `np.arange(...) % vocab_size` 构造的周期合成序列；没有从 `.bin` 加载训练集或 validation 集。
- **resume**：`utils.py` 中 `load_checkpoint` 已实现并由 serialization 测试覆盖；训练入口不调用它，也没有恢复训练状态的配置流程。
- **checkpoint provenance**：根目录的 `checkpoint.pt` 是本地文件；未发现其由完整 TinyStories 训练产生的证据，因此不能视作真实语料 checkpoint。该文件被 Git 忽略。
- **generation**：没有 prompt generation、temperature、top-p、`max_new_tokens` 或 EOS stopping。
- **实验**：没有已完成的 LM learning-rate、batch-size、architecture ablation、GPU benchmark 或 OpenWebText 结果。
- **writeup/report**：未发现正式作业报告。
- **数据状态**：完整训练文本和 validation 文本在本地；validation token IDs 完成；train token IDs 仍处于 `.part` 编码阶段。

## 测试与检查

运行命令：

```sh
.venv-wsl/bin/python -m pytest -p no:cacheprovider
```

本次结果：**47 passed, 1 xfailed, 0 failed, 0 skipped in 48.67 s**。xfail 是测试预期行为：普通 `Tokenizer.encode` 被检查为超过 1 MB 内存限制。完整套件覆盖 tokenizer、BPE、模型、优化器、训练数据 batch、checkpoint 和数值工具。核心实现与两个数据脚本均可导入。

运行时有 CUDA warning：本机 NVIDIA driver 低于安装的 PyTorch CUDA runtime 要求。测试结果不验证 GPU 训练能力；GPU 是否可用需单独解决驱动问题后确认。

## Git 与上传安全

- 当前分支为 `main`，原 remote `origin` 仍是 Stanford starter repository；本次不会替换它。
- 准备发布的 `main` 可达历史中未发现大于 50 MB 的 blob；其中也没有 `data/`、`artifacts/` 或根目录 `checkpoint.pt`。
- 本地另有 Codex 检查点引用可达旧提交及 2,227,753,162-byte TinyStories blob。它不属于 `main`，发布时只推送 `main` 分支，不推送该引用。
- `.gitignore` 排除原始数据、实验产物、checkpoint、虚拟环境、临时编码文件和本地密钥配置；数据不会上传。
- 对已跟踪文件名和代码内容执行了凭据/私钥模式检查，未发现命中。
- 本次测试与审计没有修改核心算法。README、状态/实验/路线图文档和 `.gitignore` 属于本次整理范围。

## 当前进度估计

约 **55%**，是对 Assignment 1 工作项的粗略估计，不是测试通过率。主要核心实现、完整 BPE 和 validation 数据准备已完成；train token 编码仍在进行，真实 LM training、生成、性能/消融实验、OpenWebText 和 writeup 尚未完成。
