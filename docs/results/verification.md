# 2026-10-12 核验记录

本次审查使用现有本地文件；未下载数据、重训 BPE、执行新的 LM 更新或重跑完整固定窗口评估。文档整理后的 Git 操作限定正常提交与推送。

## 完整参考测试

```powershell
wsl --cd /mnt/e/projects --exec /mnt/e/projects/.venv-wsl/bin/python -B -m pytest -p no:cacheprovider
```

实际摘要：

```text
collected 48 items
47 passed, 1 xfailed, 10 warnings in 50.37s
```

0 failed、0 skipped。xfail 为 `test_encode_memory_usage`；10 warnings 是参考测试及 checkpoint loader 中旧式 `torch.load(weights_only=False)` 的 FutureWarning。它们未造成测试失败。本次无此前的 CUDA 驱动不兼容 warning。

测试覆盖的组件为 Tokenizer/BPE、Linear/Embedding、RMSNorm/SwiGLU、RoPE/Attention、Transformer Block/LM、Cross-Entropy、Gradient Clipping、AdamW、LR scheduler、get_batch、Checkpoint serialization；不覆盖整个 51M 长训练或全部生成边界行为。

## 最终 Checkpoint 加载

已有生成脚本实际输出 `Device: cuda` 与 `Checkpoint iteration: 49999`，并生成 [完整故事](generation_51m.txt)。该脚本只读取数据，不修改模型。

检查过训练脚本的完成判断后，仅执行其“已完成”返回路径，输出：

```text
Model parameters: 51147264
Resuming from iteration: 50000
Training already completed
```

这同时验证同架构 model 与 optimizer state 能加载；没有执行新的训练 iteration。Checkpoint 检查前后均为 613,878,304 bytes，修改时间未变；SHA-256 见 [manifest](source_manifest.json)。

## 环境

`python -B -m torch.utils.collect_env` 与 Windows `nvidia-smi` 核实：

- Ubuntu 24.04.1 LTS / WSL2；Python 3.12.14。
- PyTorch 2.5.1+cu124；NumPy 2.4.4；CUDA build 12.4；CUDA available=True。
- NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，driver 555.97。
- 当前 `pyproject.toml` 要求 PyTorch 2.11，和此环境不一致；本次未更改依赖，也未重新创建环境。

以上是审查时环境，不是历史各实验自动保存的完整环境锁文件。未找到峰值训练显存、历史 GPU profiler 或精确 GPU latency 的记录。

## 数据轻量检查

| 文件 | Bytes | uint16 token 数 | 抽样 IDs | 抽样范围 |
| --- | ---: | ---: | ---: | --- |
| train.bin | 1,082,458,694 | 541,229,347 | 163,840 | 10–9985 |
| valid.bin | 4,762,512 | 2,381,256 | 163,840 | 10–9977 |

以 little-endian uint16 读取首部、1/4、1/2、3/4 和尾部各 65,536 bytes；文件大小均为偶数，抽样 ID 小于 vocab size。token 数源于最终文件大小，而非重编码后的比较。此抽样不能替代全文件 ID 范围和原始语料逐字节一致性验证。

实际 vocab JSON 有 10,000 entries，merges JSON 有 9,743 rules；`<|endoftext|>` 对应 ID 256。

## 实验与发布边界

最终评估 CSV 的数值及 exp(loss) 关系已核对；本次没有再次评估全部 100 batches。两段 51M 日志均已读取，合并曲线保留 101 个唯一 iterations。六组 LR Sweep 的终点分别与 summary 相符。

记录了当前代码与日志 SHA-256，便于识别今后工作区变动，但不能据此证明训练发生时的代码与当前工作区逐字节相同。新的实验脚本与 `train.py` 改动保留为用户未提交工作；本次 GitHub 更新仅包含文档与轻量图片/指标。
