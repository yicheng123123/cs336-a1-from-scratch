# CS336 Assignment 1 路线图

按当前依赖关系和优先级排列。第一项对应正在运行的训练集编码；继续观察现有进程，不另起编码任务。

## 1. 完成 `train.bin` 和最终验证

- 等现有 `scripts.encode_tinystories` 进程正常结束。
- 确认临时文件被改名为 `artifacts/encoded/train.bin`；记录实际字节数、token 数、dtype 和 ID 范围。
- 使用 memmap 验证大小与 token 数一致、ID 小于 vocab size，并确认 next-token batch 可读取。
- 独立确认 `valid.bin` 与 train/validation split 都是 little-endian uint16。

## 2. 接通真实 LM training

- 让训练入口使用 train/valid token IDs（适合大文件的内存映射读取）。
- 核实 batch 起点和标签偏移不会越过文件边界。
- 记录模型配置、随机种子、训练/验证 loss 和运行设备。
- 将 `load_checkpoint` 接入训练入口，恢复 model、optimizer、iteration 和必要的训练状态。

## 3. TinyStories smoke test

- 用很小的模型、短迭代数和已编码的真实语料确认一次端到端 forward/backward、validation 与 checkpoint 保存。
- 检查日志有限且 loss 为有限值；确认恢复后 iteration 和 optimizer state 接续。
- smoke test 通过后再计划长时间训练。

## 4. GPU training 与 checkpoint resume

- 先解决当前 PyTorch 提示的 NVIDIA driver/CUDA runtime 不兼容，确认设备可用。
- 记录硬件、软件版本、batch/context/model config、吞吐与 validation loss。
- 验证中断后可从 checkpoint 恢复；不把 CPU 单元测试结果写成 GPU benchmark。

## 5. Generation

- 增加 prompt 编码、模型逐 token 生成与 decode。
- 支持 `temperature`、top-p、`max_new_tokens` 和 EOS stopping。
- 用固定 prompt 和 seed 做小型行为检查，记录采样参数。

## 6. Experiments / ablations

- 按作业要求设计 architecture ablation、learning-rate 和 batch-size 实验。
- 固定其他变量，记录配置、随机种子、训练/验证指标和运行成本。
- 只报告可追溯的测量；不从 tokenizer benchmark 推断 LM 性能。

## 7. OpenWebText

- 获取课程要求的 OpenWebText 子集，确认授权/下载来源、磁盘空间和数据忽略规则。
- 使用 tokenizer 编码并对 split、dtype、vocab 范围和 batch 读取做验证。
- 资源允许时运行作业要求的训练/比较实验。

## 8. Writeup

- 对照 handout 汇总方法、真实训练设置、实验设计、结果、限制和失败案例。
- 确保每个表格与图都能追溯到日志或 stats 文件，并准确区分已完成与计划工作。
