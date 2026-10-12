import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # m_t
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # decoupled weight decay
                if weight_decay != 0:
                    p.mul_(
                        1 - lr * weight_decay
                    )

                # Adam update
                p.addcdiv_(
                    m_hat,
                    torch.sqrt(v_hat) + eps,
                    value=-lr,
                )
import math
def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it>cosine_cycle_iters:
        return min_learning_rate
     # -------------------------------------------------
    # Stage 2: Cosine decay
    #
    # 先把当前 iteration 转换成 [0, 1] 的进度。
    #
    # it = warmup_iters
    # -> progress = 0
    #
    # it = cosine_cycle_iters
    # -> progress = 1
    
    # 再把 cosine coefficient 从 [1, 0]
    # 映射到学习率 [max_lr, min_lr]
    #
    # cosine = 1 -> max_learning_rate
    # cosine = 0 -> min_learning_rate
    # -------------------------------------------------
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay