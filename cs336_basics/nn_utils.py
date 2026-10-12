import torch


def softmax(x, dim):
    max_value = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    shifted = x - max_value

    numerator = torch.exp(shifted)

    denominator = torch.sum(numerator,dim=dim,keepdim=True, )

    return numerator / denominator


def cross_entropy(inputs, targets):
    max_value=torch.max(inputs,dim=-1,keepdim=True).values
    shift=inputs-max_value
    log_sum_ex=(torch.log(torch.sum(torch.exp(shift),
                dim=-1,)) + max_value.squeeze(-1))
    target_logits=torch.gather(inputs,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_ex - target_logits

    return loss.mean()
def gradient_clipping(
        parameters,
        max_l2_norm,
    ):
        """
        Clip gradients using the global L2 norm.

        Args:
            parameters:
                Iterable of model parameters.

            max_l2_norm:
                Maximum allowed global L2 norm of all gradients.
        """

        # parameters 通常是 generator，例如 model.parameters()
        # 转成 list，避免后面遍历一次之后就耗尽。
        parameters = list(parameters)

        # -------------------------------------------------
        # 1. 计算所有 gradient 合在一起后的 global L2 norm
        #
        # global_norm =
        # sqrt(sum(g_1^2) + sum(g_2^2) + ...)
        # -------------------------------------------------
        total_squared_norm = 0.0

        for p in parameters:
            # 有些参数当前 step 可能没有梯度
            if p.grad is None:
                continue

            total_squared_norm += torch.sum(
                p.grad ** 2
            )

        total_norm = torch.sqrt(
            total_squared_norm
        )

        # -------------------------------------------------
        # 2. 只有 gradient norm 超过阈值时才进行 clipping
        #
        # scale = max_norm / (current_norm + eps)
        # -------------------------------------------------
        if total_norm > max_l2_norm:
            scale = (
                max_l2_norm
                / (total_norm + 1e-6)
            )

            # 所有参数必须使用相同 scale
            for p in parameters:
                if p.grad is None:
                    continue

                p.grad.mul_(scale)
