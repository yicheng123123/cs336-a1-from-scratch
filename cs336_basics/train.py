import torch
import numpy as np

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.utils import get_batch, save_checkpoint


@torch.no_grad()
def estimate_loss(
    model,
    data,
    batch_size,
    context_length,
    device,
    num_batches=20,
):
    """
    在 validation data 上计算平均 loss。

    这里只做 forward，不做 backward，
    因此使用 @torch.no_grad()。
    """

    # 切换到 evaluation mode
    model.eval()

    losses = []

    for _ in range(num_batches):

        # 随机抽一个 validation batch
        x, y = get_batch(
            data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Forward
        logits = model(x)

        # Compute validation loss
        loss = cross_entropy(
            logits,
            y,
        )

        losses.append(loss.item())

    # 验证结束，切回 training mode
    model.train()

    return sum(losses) / len(losses)


def train(
    model,
    train_data,
    val_data,
    optimizer,
    num_iters,
    batch_size,
    context_length,
    max_lr,
    min_lr,
    warmup_iters,
    max_grad_norm,
    device,
    eval_interval=100,
    checkpoint_interval=1000,
    checkpoint_path="checkpoint.pt",
):
    """
    Main Transformer training loop.
    """

    model.train()

    for iteration in range(num_iters):

        # ==================================================
        # 1. Sample training batch
        #
        # x: [B, T]
        # y: [B, T]
        #
        # y 是 x 向右移动一个 token 的结果
        # ==================================================
        x, y = get_batch(
            train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # ==================================================
        # 2. Clear gradients from previous iteration
        #
        # PyTorch 默认会累积 gradient，
        # 所以每一步训练都需要清空。
        # ==================================================
        optimizer.zero_grad(
            set_to_none=True
        )

        # ==================================================
        # 3. Forward
        #
        # x:
        # [B, T]
        #
        # logits:
        # [B, T, vocab_size]
        # ==================================================
        logits = model(x)

        # ==================================================
        # 4. Cross entropy
        #
        # logits: [B,T,V]
        # y:      [B,T]
        #
        # output:
        # scalar loss
        # ==================================================
        loss = cross_entropy(
            logits,
            y,
        )

        # ==================================================
        # 5. Backpropagation
        #
        # 计算所有 parameter.grad
        # ==================================================
        loss.backward()

        # ==================================================
        # 6. Gradient clipping
        #
        # 如果 global gradient L2 norm 太大，
        # 按比例缩小所有 gradients。
        # ==================================================
        gradient_clipping(
            model.parameters(),
            max_l2_norm=max_grad_norm,
        )

        # ==================================================
        # 7. Learning-rate schedule
        #
        # warmup:
        # 0 -> max_lr
        #
        # cosine:
        # max_lr -> min_lr
        # ==================================================
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=num_iters,
        )

        # AdamW 从 param_groups 中读取当前 lr
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ==================================================
        # 8. Update parameters
        # ==================================================
        optimizer.step()

        # ==================================================
        # 9. Validation
        # ==================================================
        if iteration % eval_interval == 0:

            val_loss = estimate_loss(
                model=model,
                data=val_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            print(
                f"iter={iteration} "
                f"train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"lr={lr:.6e}"
            )

        # ==================================================
        # 10. Save checkpoint
        # ==================================================
        if (
            iteration > 0
            and iteration % checkpoint_interval == 0
        ):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                out=checkpoint_path,
            )
if __name__ == "__main__":

    # =========================
    # Device
    # =========================
    device = "cpu"

    # =========================
    # Tiny model config
    # =========================
    vocab_size = 100
    context_length = 16

    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128

    rope_theta = 10000.0

    # =========================
    # Training config
    # =========================
    batch_size = 8
    num_iters = 200

    max_lr = 1e-3
    min_lr = 1e-4

    warmup_iters = 20

    max_grad_norm = 1.0
    weight_decay = 0.01
    train_data = np.arange(
        10000,
        dtype=np.int64,
    ) % vocab_size

    val_data = np.arange(
        2000,
        dtype=np.int64,
    ) % vocab_size
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    model = model.to(device)


    # 4. 创建 optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
    )


    # 5. 开始训练
    train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        num_iters=num_iters,
        batch_size=batch_size,
        context_length=context_length,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        max_grad_norm=max_grad_norm,
        device=device,
        eval_interval=20,
        checkpoint_interval=100,
        checkpoint_path="checkpoint.pt",
    )