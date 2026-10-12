
from pathlib import Path
import time
import numpy as np

from cs336_basics.tokenizer import Tokenizer
from scripts.train_tinystories_bpe import load_vocab, load_merges


# ==========================================
# 1. 配置
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_DIR = (
    ROOT
    / "artifacts"
    / "tokenizer"
    / "full_tinystories_vocab10000"
)

INPUT_PATH = (
    ROOT
    / "data"
    / "TinyStoriesV2-GPT4-train.txt"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "encoded"
    / "train.bin"
)

TEMP_PATH = OUTPUT_PATH.with_suffix(".bin.part")

SPECIAL_TOKEN = "<|endoftext|>"
CHUNK_SIZE = 1_000_000

CONTEXT_LENGTH = 128
BATCH_SIZE = 4


# ==========================================
# 2. 加载 Tokenizer
# ==========================================

def load_tokenizer():

    vocab = load_vocab(
        TOKENIZER_DIR / "tinystories_vocab.json"
    )

    merges = load_merges(
        TOKENIZER_DIR / "tinystories_merges.json"
    )

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=[SPECIAL_TOKEN],
    )

    assert len(vocab) <= 65536
    assert len(tokenizer.encode(SPECIAL_TOKEN)) == 1

    print("Tokenizer loaded.")
    print("Vocab size:", len(vocab))
    print("Merge rules:", len(merges))

    return tokenizer


# ==========================================
# 3. Streaming 编码
# ==========================================

def encode_corpus(tokenizer):

    total_tokens = 0
    total_segments = 0

    start = time.perf_counter()

    TEMP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(INPUT_PATH, "r", encoding="utf-8") as src:
        with open(TEMP_PATH, "wb") as dst:

            remainder = ""

            while True:

                chunk = src.read(CHUNK_SIZE)

                if not chunk:
                    break

                text = remainder + chunk

                # 按完整故事分隔符切分
                parts = text.split(SPECIAL_TOKEN)

                # 最后一段可能不完整
                for part in parts[:-1]:

                    segment = part + SPECIAL_TOKEN

                    ids = tokenizer.encode(segment)

                    # 最初三个故事验证可逆性
                    if total_segments < 3:
                        assert tokenizer.decode(ids) == segment

                    # Token IDs -> uint16 -> binary
                    token_array = np.asarray(
                        ids,
                        dtype="<u2",
                    )

                    token_array.tofile(dst)

                    total_tokens += len(ids)
                    total_segments += 1

                    # 输出进度
                    if total_segments % 10000 == 0:

                        elapsed = time.perf_counter() - start

                        print(
                            f"Encoded {total_segments:,} segments | "
                            f"{total_tokens:,} tokens | "
                            f"{elapsed:.1f}s",
                            flush=True,
                        )

                remainder = parts[-1]

            # 处理文件最后一个不完整分隔段
            if remainder:

                ids = tokenizer.encode(remainder)

                assert tokenizer.decode(ids) == remainder

                np.asarray(
                    ids,
                    dtype="<u2",
                ).tofile(dst)

                total_tokens += len(ids)

    elapsed = time.perf_counter() - start

    print("\nEncoding finished!")
    print("Segments:", total_segments)
    print("Total tokens:", total_tokens)
    print("Encoding time:", round(elapsed, 2), "seconds")

    return total_tokens


# ==========================================
# 4. 验证 binary 文件
# ==========================================

def verify_binary(total_tokens, vocab_size):

    print("\nVerifying binary file...")

    output_bytes = TEMP_PATH.stat().st_size

    print("Output bytes:", output_bytes)

    assert output_bytes == total_tokens * 2
    assert total_tokens > CONTEXT_LENGTH * BATCH_SIZE

    print("Binary size verification: PASS")

    # 以内存映射方式读取
    data = np.memmap(
        TEMP_PATH,
        dtype="<u2",
        mode="r",
    )

    print("Total tokens:", len(data))
    print("Data shape:", data.shape)

    assert len(data) == total_tokens

    # 验证 Token ID 范围
    min_id = int(data.min())
    max_id = int(data.max())

    print("Min token ID:", min_id)
    print("Max token ID:", max_id)

    assert 0 <= min_id
    assert max_id < vocab_size

    print("Token ID range: PASS")

    # ======================================
    # 构建一个训练 Batch
    # ======================================

    starts = np.arange(BATCH_SIZE) * CONTEXT_LENGTH

    x = np.stack([
        data[i:i + CONTEXT_LENGTH]
        for i in starts
    ])

    y = np.stack([
        data[i + 1:i + CONTEXT_LENGTH + 1]
        for i in starts
    ])

    print("\nX shape:", x.shape)
    print("Y shape:", y.shape)

    print("X first 10:", x[0, :10].tolist())
    print("Y first 10:", y[0, :10].tolist())

    assert x.shape == (BATCH_SIZE, CONTEXT_LENGTH)
    assert y.shape == (BATCH_SIZE, CONTEXT_LENGTH)

    # Next-token prediction 对齐验证
    assert np.array_equal(
        x[:, 1:],
        y[:, :-1],
    )

    print("Batch verification: PASS")

    del data


# ==========================================
# 5. 主函数
# ==========================================

def main():

    print("Starting TinyStories encoding...")
    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)

    tokenizer = load_tokenizer()

    total_tokens = encode_corpus(tokenizer)

    verify_binary(
        total_tokens,
        vocab_size=len(tokenizer.vocab),
    )

    # 所有检查通过后再改名
    TEMP_PATH.replace(OUTPUT_PATH)

    print("\nAll checks passed.")
    print("Saved:", OUTPUT_PATH)
    print("Total tokens:", total_tokens)


if __name__ == "__main__":
    main()
