# BPE Tokenizer 实验记录

本表从本地 `artifacts/tokenizer/**/tokenizer_stats.json` 读取。输入字节数、目标词表大小、merge 数、训练时间和峰值 RSS 均为 stats 文件中的实际记录；没有用 README 或计划文档补数。MB 使用十进制单位（1 MB = 1,000,000 bytes）。

| 实验目录 | Corpus (bytes) | Vocab size | Merge rules | Training time (s) | Peak process RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tokenizer` | 100,807 | 300 | 43 | 0.06 | 22.24 |
| `streaming_smoke` | 100,807 | 300 | 43 | 0.05 | 22.08 |
| `scale_1mb` | 1,007,405 | 300 | 43 | 0.39 | 26.69 |
| `scale_10mb` | 10,074,023 | 1,000 | 743 | 3.48 | 66.60 |
| `scale_10mb_vocab3000` | 10,074,023 | 3,000 | 2,743 | 4.04 | 66.99 |
| `scale_10mb_vocab10000` | 10,074,023 | 10,000 | 9,743 | 4.86 | 66.75 |
| `scale_100mb_vocab10000` | 100,741,548 | 10,000 | 9,743 | 29.25 | 54.55 |
| `full_tinystories_vocab10000` | 2,227,753,162 | 10,000 | 9,743 | 629.7227 | 99.55 |

## 结果来源

- 统计文件位置为每行对应实验目录中的 `tokenizer_stats.json`。
- 完整语料产物：`artifacts/tokenizer/full_tinystories_vocab10000/`，其中还包含 `tinystories_vocab.json` 与 `tinystories_merges.json`。
- 输入 corpus 是本地 TinyStories 的相应规模文件；数据本身不纳入仓库。
- 训练耗时与峰值 RSS 是单次运行结果。运行环境、并发负载和数据规模不同会影响测量，不能据此推断模型训练速度或 GPU 性能。

## 观察

- 在相同的 10,000 vocab 配置下，10 MB、100 MB、完整 2.23 GB 语料分别记录 4.86 s、29.25 s、629.7227 s；这是本机观测，不代表线性扩展规律。
- 小语料实验的 vocab target 不全相同：100 KB 与 1 MB 使用 300，10 MB 另外测试了 1,000、3,000 和 10,000。因此这些行不能直接当成只改变 corpus size 的控制变量实验。
- 峰值 RSS 没有随着语料规模单调增加，表中数据反映单次进程峰值，并不构成内存复杂度证明。
- 完整语料结果记录了 BPE tokenizer 训练。尚无真实 Transformer LM training loss、validation loss、PPL 或 GPU latency。
