# 下一阶段路线图

基于 2026-10-12 的实际文件、测试和产物。全量 BPE、train.bin / valid.bin、51M 50k 训练和基本生成已完成，不再把它们列为尚待启动的工作。

## P0：把已有成果变成可独立复现的仓库

1. **发布现有实验代码**：单独审查本地 train.py 改动与 8 个未跟踪脚本；本次文档任务没有提交这些代码。保留现有数据与 checkpoint，避免触发重新训练或覆盖输出。
2. **统一环境记录**：对齐当前声明的 PyTorch 2.11 与实际已验证的 2.5.1+cu124；记录可重建的依赖锁与 GPU/驱动兼容要求。
3. **版本化实验配置**：保留 51M 模型/训练/评估/生成设置、全部六组 LR 设置，以及原始 CSV 来源，消除路径硬编码与“脚本后来改过”的复现缺口。

验收目标：全新 clone 能运行核心测试，并在显式提供本地数据/产物后找到可用的训练、评估、生成入口；不把恢复现有实验与新建实验混为同一操作。

## P1：数据与 Checkpoint 可追溯性

- 为 validation 编码补齐独立入口和完成记录；归档 train/valid split、dtype、Tokenizer vocab/merges 与输入来源。
- 保存 model config、RNG states、数据/Tokenizer 版本和软件环境；用短程对照检查恢复后 iteration、optimizer 与采样序列。
- 增加生成行为检查：空 prompt、采样参数范围、EOS、长度限制、context 截断。
- 保留当前 SHA-256 manifest 与轻量结果，并在下一次实验中保存实际执行版本，而非只记审查时的源码版本。

## P2：评估和实验扩展

- 对最终模型做全验证集或更充分的固定评估，增加多 seed 和多个 prompt，明确采样/重复窗口协议。
- 正式比较 batch size、模型规模与 architecture choices，固定其他变量、训练预算和评估样本；不要直接把小模型 5e-3 最优观测推广到 51M。
- 设计独立 GPU profiling，记录实际峰值显存、tokens/s、latency、warmup、同步和重复次数；不从训练 CSV 的 elapsed time 倒推可靠 GPU benchmark。
- 确认资源和课程要求后规划 OpenWebText，不自动下载或训练。

## P3：完整 Assignment Writeup

对照 handout 逐题列出实现、推导、实验、限制与未完成项。当前 README 和 EXPERIMENTS 是工程总结，不替代官方全部书面作业。

前三个最优先任务：**现有代码发布、依赖/配置统一、数据与恢复行为验证**。
