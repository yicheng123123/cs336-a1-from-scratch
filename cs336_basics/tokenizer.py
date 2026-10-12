import regex as re


# GPT-2 style pre-tokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    streaming=False,
):
    # --------------------------------------------------
    # 1. 初始化 byte vocabulary
    # --------------------------------------------------

    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    # --------------------------------------------------
    # 2. 加入 special tokens
    # --------------------------------------------------

    if special_tokens:
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

    # --------------------------------------------------
    # 3. 读取 corpus
    # --------------------------------------------------

    if streaming:
        if special_tokens != ["<|endoftext|>"]:
            raise ValueError(
                "当前 streaming 只支持 TinyStories 单特殊 token"
            )

        pretoken_counts = get_pretoken_counts_streaming(
            input_path
        )
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        pretoken_counts = get_pretoken_counts(
            text,
            special_tokens,
        )

    # --------------------------------------------------
    # 5. 一次性建立 pair statistics
    # --------------------------------------------------

    pair_counts, pair_to_pretokens = build_pair_stats(
        pretoken_counts
    )

    merges = []

    # --------------------------------------------------
    # 6. BPE training loop
    # --------------------------------------------------

    while len(vocab) < vocab_size:

        # 找出现次数最多的 pair
        best_pair = get_best_pair(pair_counts)

        if best_pair is None:
            break

        # 只更新受 best_pair 影响的 pre-token
        merge_and_update(
            pretoken_counts,
            pair_counts,
            pair_to_pretokens,
            best_pair,
        )

        # 新 token = 两个 token 的 bytes 拼接
        new_token = best_pair[0] + best_pair[1]

        # 加入 vocab
        vocab[len(vocab)] = new_token

        # 保存 merge rule
        merges.append(best_pair)

    return vocab, merges


# ======================================================
# Pretokenization
# ======================================================


def pretokenize(text):
    pretokens = []

    for match in re.finditer(PAT, text):
        pretokens.append(match.group())

    return pretokens


def get_pretoken_counts(text, special_tokens):
    """
    把文本：

        "hello world hello"

    转成：

        {
            (b'h', b'e', b'l', b'l', b'o'): 1,
            (b' ', b'w', b'o', b'r', b'l', b'd'): 1,
            (b' ', b'h', b'e', b'l', b'l', b'o'): 1,
        }

    同时保证 special token 不参与 BPE。
    """

    pretoken_counts = {}

    # --------------------------------------------------
    # 先用 special token 把文本切开
    # --------------------------------------------------

    if special_tokens:
        # 长的 special token 优先，避免 overlapping token 出问题
        sorted_special_tokens = sorted(
            special_tokens,
            key=len,
            reverse=True,
        )

        pattern = "|".join(
            re.escape(token)
            for token in sorted_special_tokens
        )

        text_parts = re.split(pattern, text)

    else:
        text_parts = [text]

    # --------------------------------------------------
    # 每个普通文本区域进行 GPT-2 pretokenization
    # --------------------------------------------------

    for part in text_parts:

        for match in re.finditer(PAT, part):

            pretoken = match.group()

            # "hello"
            #
            # ↓ UTF-8
            #
            # b"hello"
            #
            # ↓ 每个 byte 一个初始 token
            #
            # (b'h', b'e', b'l', b'l', b'o')

            token_sequence = tuple(
                bytes([b])
                for b in pretoken.encode("utf-8")
            )

            pretoken_counts[token_sequence] = (
                pretoken_counts.get(token_sequence, 0)
                + 1
            )

    return pretoken_counts
def get_pretoken_counts_streaming(
    input_path,
    separator="<|endoftext|>",
    chunk_size=1_000_000,
):
    counts = {}

    def process_part(part):
        for match in re.finditer(PAT, part):
            token_sequence = tuple(
                bytes([b])
                for b in match.group().encode("utf-8")
            )

            counts[token_sequence] = (
                counts.get(token_sequence, 0) + 1
            )

    with open(input_path, "r", encoding="utf-8") as f:
        remainder = ""

        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            text = remainder + chunk

            parts = text.split(separator)

            # 只有已经遇到分隔符的片段才完整
            for part in parts[:-1]:
                process_part(part)

            # 最后一段可能还没读取完整
            remainder = parts[-1]

            if len(remainder) > 10_000_000:
                raise ValueError(
                    "连续文本过长，未找到完整故事分隔符"
                )

        # 文件末尾剩余的文本
        process_part(remainder)

    return counts

# ======================================================
# Pair counting
# ======================================================


def local_pair_counts(tokens):
    """
    只统计一个 pre-token 内部的 pair。

    例如：

        (b'a', b'n', b'a', b'n')

    →

        {
            (b'a', b'n'): 2,
            (b'n', b'a'): 1
        }
    """

    counts = {}

    for i in range(len(tokens) - 1):

        pair = (
            tokens[i],
            tokens[i + 1],
        )

        counts[pair] = counts.get(pair, 0) + 1

    return counts


def build_pair_stats(pretoken_counts):
    """
    一开始完整扫描一次所有 pre-token。

    返回：

    pair_counts:
        pair 在整个 corpus 出现多少次

    pair_to_pretokens:
        哪些 pre-token 包含这个 pair
    """

    pair_counts = {}
    pair_to_pretokens = {}

    for tokens, freq in pretoken_counts.items():

        local_counts = local_pair_counts(tokens)

        for pair, occurrences in local_counts.items():

            # 一个 pre-token 内出现次数
            # ×
            # 这个 pre-token 在 corpus 中出现次数

            contribution = occurrences * freq

            pair_counts[pair] = (
                pair_counts.get(pair, 0)
                + contribution
            )

            if pair not in pair_to_pretokens:
                pair_to_pretokens[pair] = set()

            pair_to_pretokens[pair].add(tokens)

    return pair_counts, pair_to_pretokens


# ======================================================
# Choose best pair
# ======================================================


def get_best_pair(pair_counts):
    """
    选择：
        1. frequency 最大
        2. frequency 相同时，pair lexicographically 最大
    """

    best_pair = None
    best_count = -1

    for pair, count in pair_counts.items():

        if count > best_count:
            best_count = count
            best_pair = pair

        elif count == best_count and pair > best_pair:
            best_pair = pair

    return best_pair


# ======================================================
# Merge
# ======================================================


def merge_pair(tokens, pair):
    """
    例如：

        tokens =
        (b'b', b'a', b'n', b'a', b'n', b'a')

        pair =
        (b'a', b'n')

    →

        [
            b'b',
            b'an',
            b'an',
            b'a'
        ]
    """

    merged_tokens = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and (tokens[i], tokens[i + 1]) == pair
        ):

            merged_tokens.append(
                tokens[i] + tokens[i + 1]
            )

            i += 2

        else:

            merged_tokens.append(tokens[i])

            i += 1

    return merged_tokens


# ======================================================
# Incremental pair-statistics update
# ======================================================


def merge_and_update(
    pretoken_counts,
    pair_counts,
    pair_to_pretokens,
    best_pair,
):
    """
    不重新扫描整个 corpus。

    只处理包含 best_pair 的 pre-token：

        1. 删除旧 pair contribution
        2. merge
        3. 加入新 pair contribution
    """

    # --------------------------------------------------
    # best_pair 影响到哪些 pre-token
    # --------------------------------------------------

    affected_tokens = list(
        pair_to_pretokens.get(best_pair, set())
    )

    # --------------------------------------------------
    # 非常重要：
    #
    # 必须先保存所有旧 frequency。
    #
    # 因为：
    # old_sequence A merge 后，
    # 可能刚好变成另一个 old_sequence B。
    #
    # 如果边处理边修改，会导致同一轮重复 merge。
    # --------------------------------------------------

    affected_items = [
        (
            tokens,
            pretoken_counts[tokens],
        )
        for tokens in affected_tokens
    ]

    # ==================================================
    # Phase 1:
    # 删除所有受影响 old sequences 的贡献
    # ==================================================

    for old_tokens, freq in affected_items:

        old_local_counts = local_pair_counts(
            old_tokens
        )

        for pair, occurrences in old_local_counts.items():

            contribution = occurrences * freq

            pair_counts[pair] -= contribution

            pair_to_pretokens[pair].discard(
                old_tokens
            )

            # 如果 pair 已经彻底消失
            if pair_counts[pair] == 0:

                del pair_counts[pair]

                del pair_to_pretokens[pair]

        # 从 pre-token frequency table 删除
        del pretoken_counts[old_tokens]

    # ==================================================
    # Phase 2:
    # merge 所有 old sequences
    #
    # 先暂存在 merged_counts，
    # 避免不同 old sequence merge 成相同 sequence。
    # ==================================================

    merged_counts = {}

    for old_tokens, freq in affected_items:

        new_tokens = tuple(
            merge_pair(
                old_tokens,
                best_pair,
            )
        )

        merged_counts[new_tokens] = (
            merged_counts.get(new_tokens, 0)
            + freq
        )

    # ==================================================
    # Phase 3:
    # 把 merged sequences 加回 corpus
    # 并更新新的 pair statistics
    # ==================================================

    for new_tokens, added_freq in merged_counts.items():

        # 如果这个 new_tokens 本来就存在于 corpus：
        #
        # 保留已有 frequency，
        # 再加新产生的 frequency。

        pretoken_counts[new_tokens] = (
            pretoken_counts.get(new_tokens, 0)
            + added_freq
        )

        new_local_counts = local_pair_counts(
            new_tokens
        )

        for pair, occurrences in new_local_counts.items():

            contribution = (
                occurrences
                * added_freq
            )

            pair_counts[pair] = (
                pair_counts.get(pair, 0)
                + contribution
            )

            if pair not in pair_to_pretokens:
                pair_to_pretokens[pair] = set()

            pair_to_pretokens[pair].add(
                new_tokens
            )
class Tokenizer:

    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ):
        self.vocab = vocab

        self.merges = merges

        self.special_tokens = (
            special_tokens
            if special_tokens is not None
            else []
        )

        # bytes → token_id
        self.inverse_vocab = {
            token: token_id
            for token_id, token in vocab.items()
        }

        # merge pair → priority
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }
    def _apply_bpe(self,pretoken):
        tokens=[bytes([b])  for b in pretoken.encode("utf-8")]
        while len(tokens) > 1:

            best_pair = None
            best_rank = None

        # 看当前所有相邻 pair
            for i in range(len(tokens) - 1):

                pair = (
                    tokens[i],
                    tokens[i + 1]
                )

                # 这个 pair 有没有在训练阶段被学到？
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                # rank 越小，优先级越高
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 当前已经没有可以继续 merge 的 pair
            if best_pair is None:
                break

            # merge 当前优先级最高的 pair
            tokens = merge_pair(
                tokens,
                best_pair
            )

        return tokens
    def encode(self, text):
        ids = []

        # 没有 special tokens
        if not self.special_tokens:
            parts = [text]

        else:
            sorted_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            pattern = (
                "("
                + "|".join(
                    re.escape(token)
                    for token in sorted_special_tokens
                )
                + ")"
            )

            parts = re.split(pattern, text)

        for part in parts:

            if part == "":
                continue

            # -----------------------------
            # special token
            # -----------------------------
            if part in self.special_tokens:

                token_bytes = part.encode("utf-8")

                token_id = self.inverse_vocab[
                    token_bytes
                ]

                ids.append(token_id)

            # -----------------------------
            # normal text
            # -----------------------------
            else:

                pretokens = pretokenize(part)

                for pretoken in pretokens:

                    bpe_tokens = self._apply_bpe(
                        pretoken
                    )

                    for token in bpe_tokens:

                        token_id = (
                            self.inverse_vocab[token]
                        )

                        ids.append(token_id)

        return ids

    def decode(self, ids):
        byte_data = b""

        for token_id in ids:
            byte_data += self.vocab[token_id]

        text = byte_data.decode(
            "utf-8",
            errors="replace"
        )

        return text
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
