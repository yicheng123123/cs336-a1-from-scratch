import numpy as np
import torch


def get_batch(
    dataset,
    batch_size,
    context_length,
    device,
):
    

    # 起点最大不能让 y 越界。
    #
    # x:
    # dataset[start : start + context_length]
    #
    # y:
    # dataset[start + 1 : start + context_length + 1]
    #
    # 所以必须满足：
    # start + context_length < len(dataset)
    max_start = len(dataset) - context_length

    # 随机抽 batch_size 个起点。
    #
    # np.random.randint 的 high 不包含，
    # 所以合法起点范围是：
    #
    # 0 ... max_start - 1
    starts = np.random.randint(
        0,
        max_start,
        size=batch_size,
    )

    # 为每个随机起点构造 x。
    x = np.stack([
        dataset[
            start : start + context_length
        ]
        for start in starts
    ])

    # y 比 x 整体向右移动一个 token。
    y = np.stack([
        dataset[
            start + 1 : start + context_length + 1
        ]
        for start in starts
    ])

    # numpy -> torch
    #
    # token ids 必须是整数，
    # Embedding 接收的 index 通常应该是 torch.long。
    x = torch.tensor(
        x,
        dtype=torch.long,
        device=device,
    )

    y = torch.tensor(
        y,
        dtype=torch.long,
        device=device,
    )

    return x, y
def save_checkpoint(
    model,
    optimizer,
    iteration,
    out,
):
    """
    Save model state, optimizer state, and current iteration.
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(
        checkpoint,
        out,
    )
def load_checkpoint(
    src,
    model,
    optimizer,
):
    """
    Restore model, optimizer, and iteration.
    """

    checkpoint = torch.load(
        src,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    iteration = checkpoint["iteration"]

    return iteration