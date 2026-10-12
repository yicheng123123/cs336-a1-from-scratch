from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def run_linear(
    d_in,
    d_out,
    weights,
    in_features,
):
    from cs336_basics.model import Linear

    linear = Linear(
        d_in=d_in,
        d_out=d_out,
    )

    linear.weight.data.copy_(weights)

    return linear(in_features)


def run_embedding(
    vocab_size,
    d_model,
    weights,
    token_ids,
):
    from cs336_basics.model import Embedding

    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
    )

    embedding.weight.data.copy_(weights)

    return embedding(token_ids)
def run_silu(in_features):
    import torch.nn.functional as F
    return F.silu(in_features)
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight,
    w2_weight,
    w3_weight,
    in_features,
):
    from cs336_basics.model import SwiGLU

    swiglu = SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
    )

    swiglu.w1.weight.data.copy_(w1_weight)
    swiglu.w2.weight.data.copy_(w2_weight)
    swiglu.w3.weight.data.copy_(w3_weight)

    return swiglu(in_features)

def run_scaled_dot_product_attention(
    Q,
    K,
    V,
    mask=None,
):
    from cs336_basics.model import scaled_dot_product_attention

    return scaled_dot_product_attention(
        Q,
        K,
        V,
        mask,
    )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    in_features,
):
    from cs336_basics.model import MultiHeadSelfAttention

    mha = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    mha.q_proj.weight.data.copy_(q_proj_weight)
    mha.k_proj.weight.data.copy_(k_proj_weight)
    mha.v_proj.weight.data.copy_(v_proj_weight)
    mha.output_proj.weight.data.copy_(o_proj_weight)

    return mha(in_features)

def run_multihead_self_attention_with_rope(
    d_model,
    num_heads,
    max_seq_len,
    theta,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    in_features,
    token_positions=None,
):
    from cs336_basics.model import MultiHeadSelfAttention

    mha = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
    )

    mha.q_proj.weight.data.copy_(q_proj_weight)
    mha.k_proj.weight.data.copy_(k_proj_weight)
    mha.v_proj.weight.data.copy_(v_proj_weight)
    mha.output_proj.weight.data.copy_(o_proj_weight)

    return mha(
        in_features,
        token_positions=token_positions,
    )


def run_rope(
    d_k,
    theta,
    max_seq_len,
    in_query_or_key,
    token_positions,
):
    from cs336_basics.model import RotaryPositionalEmbedding

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device,
    )

    return rope(
        in_query_or_key,
        token_positions,
    )

def run_transformer_block(
    d_model,
    num_heads,
    d_ff,
    max_seq_len,
    theta,
    weights,
    in_features,
):
    from cs336_basics.model import TransformerBlock

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        max_seq_len=max_seq_len,
    )

    block.ln1.weight.data.copy_(
        weights["ln1.weight"]
    )

    block.attn.q_proj.weight.data.copy_(
        weights["attn.q_proj.weight"]
    )

    block.attn.k_proj.weight.data.copy_(
        weights["attn.k_proj.weight"]
    )

    block.attn.v_proj.weight.data.copy_(
        weights["attn.v_proj.weight"]
    )

    block.attn.output_proj.weight.data.copy_(
        weights["attn.output_proj.weight"]
    )

    block.ln2.weight.data.copy_(
        weights["ln2.weight"]
    )

    block.ffn.w1.weight.data.copy_(
        weights["ffn.w1.weight"]
    )

    block.ffn.w2.weight.data.copy_(
        weights["ffn.w2.weight"]
    )

    block.ffn.w3.weight.data.copy_(
        weights["ffn.w3.weight"]
    )

    return block(in_features)
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices,
):
    from cs336_basics.model import TransformerLM

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    model.load_state_dict(weights)

    return model(in_indices)

def run_rmsnorm(
    d_model,
    eps,
    weights,
    in_features,
):
    from cs336_basics.model import RMSNorm

    rmsnorm = RMSNorm(
        d_model=d_model,
        eps=eps,
    )

    rmsnorm.weight.data.copy_(weights)

    return rmsnorm(in_features)


def run_get_batch(
    dataset,
    batch_size,
    context_length,
    device,
):
    from cs336_basics.utils import get_batch

    return get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

def run_softmax(
    in_features,
    dim,
):
    from cs336_basics.nn_utils import softmax

    return softmax(
        in_features,
        dim,
    )

def run_cross_entropy(
    inputs,
    targets,
):
    from cs336_basics.nn_utils import cross_entropy

    return cross_entropy(inputs, targets)

def run_gradient_clipping(
    parameters,
    max_l2_norm,):
    from cs336_basics.nn_utils import gradient_clipping

    return gradient_clipping(
        parameters,
        max_l2_norm,
    )

def get_adamw_cls():
    from cs336_basics.optimizer import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    from cs336_basics.optimizer import get_lr_cosine_schedule

    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model,
    optimizer,
    iteration,
    out,
):
    from cs336_basics.utils import save_checkpoint

    return save_checkpoint(
        model,
        optimizer,
        iteration,
        out,
    )

def run_load_checkpoint(
    src,
    model,
    optimizer,
):
    from cs336_basics.utils import load_checkpoint

    return load_checkpoint(
        src,
        model,
        optimizer,
    )


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer import Tokenizer

    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """

    from cs336_basics.tokenizer import train_bpe

    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )