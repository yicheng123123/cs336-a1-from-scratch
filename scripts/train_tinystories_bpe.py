from pathlib import Path
import time
import json
import resource
from cs336_basics.tokenizer import train_bpe, Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
)

VOCAB_SIZE = 10000

SPECIAL_TOKENS = ["<|endoftext|>"]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "full_tinystories_vocab10000"
)
def save_vocab(vocab, path):
    data = {
        str(token_id): token.hex()
        for token_id, token in vocab.items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = {
        int(token_id): bytes.fromhex(token_hex)
        for token_id, token_hex in data.items()
    }

    return vocab


def save_merges(merges, path):
    data = [
        [left.hex(), right.hex()]
        for left, right in merges
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
def load_merges(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merges = [
        (bytes.fromhex(left), bytes.fromhex(right))
        for left, right in data
    ]

    return merges
def main():
    print("Starting BPE training...")

    start = time.perf_counter()

    vocab, merges = train_bpe(
        INPUT_PATH,
        VOCAB_SIZE,
        SPECIAL_TOKENS,
        streaming=True,
    )

    elapsed = time.perf_counter() - start
    peak_memory_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    peak_memory_mib = peak_memory_kib / 1024





    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = OUTPUT_DIR / "tinystories_vocab.json"
    merges_path = OUTPUT_DIR / "tinystories_merges.json"

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)






    stats = {
        "input_path": str(INPUT_PATH),
        "corpus_size_bytes": INPUT_PATH.stat().st_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": SPECIAL_TOKENS,
        "training_time_seconds": elapsed,
        "peak_process_memory_mib": peak_memory_mib,
    }

    stats_path = OUTPUT_DIR / "tokenizer_stats.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved stats:", stats_path)
    print(f"Peak process memory: {peak_memory_mib:.2f} MiB")



    loaded_vocab = load_vocab(vocab_path)
    loaded_merges = load_merges(merges_path)

    assert vocab == loaded_vocab
    assert merges == loaded_merges
    tokenizer = Tokenizer(
        vocab=loaded_vocab,
        merges=loaded_merges,
        special_tokens=SPECIAL_TOKENS,
    )
    text = "Once upon a time, there was a little girl."

    ids = tokenizer.encode(text)
    recovered = tokenizer.decode(ids)

    print("Original text:", text)
    print("Token IDs:", ids)
    print("Recovered text:", recovered)

    assert recovered == text
    special = "<|endoftext|>"

    special_ids = tokenizer.encode(special)

    print("Special token IDs:", special_ids)

    assert len(special_ids) == 1
    assert tokenizer.decode(special_ids) == special

    print("Special token: PASS")
    print("Encode/decode: PASS")
    print("Vocab reload: PASS")
    print("Merges reload: PASS")
    print("Saved vocab:", vocab_path)
    print("Saved merges:", merges_path)

    print("Training finished!")
    print("Vocab size:", len(vocab))
    print("Number of merges:", len(merges))
    print("Training time:", elapsed, "seconds")


if __name__ == "__main__":
    main()