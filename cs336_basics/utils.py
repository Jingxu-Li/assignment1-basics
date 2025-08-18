from math import cos, pi
from typing import Iterable
from jaxtyping import Float, Int
from torch import Tensor
import torch


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> torch.Tensor:
    """
    Compute the softmax of a tensor along the specified dimension.
    """
    # 数值稳定性：减去最大值
    max_value = in_features.max(dim=dim, keepdim=True).values
    exp_features = torch.exp(in_features - max_value)

    # 归一化
    return exp_features / exp_features.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> torch.Tensor:
    """
    Compute the average cross-entropy loss between logits and target indices.

    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size)
        targets: Target indices of shape (batch_size,)

    Returns:
        Average cross-entropy loss
    """
    # 使用PyTorch的log_softmax，它内部已经优化了log和exp的抵消
    log_softmax = torch.log_softmax(inputs, dim=-1)

    # 获取目标类别的log概率
    batch_size = inputs.size(0)
    target_log_probs = log_softmax[torch.arange(batch_size), targets]

    # 计算平均交叉熵损失（注意交叉熵是负对数似然）
    loss = -target_log_probs.mean()

    return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return (
            min_learning_rate
            + 0.5 * (max_learning_rate - min_learning_rate)
            * (
                1
                + cos(
                    pi * (it - warmup_iters)
                    / (cosine_cycle_iters - warmup_iters)
                )
            )
        )
    else:
        return min_learning_rate


def get_gradient_clipping_fn(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by the global L2 norm across all parameters that have gradients.

    Compute a single scaling factor using the combined norm of all grads,
    and scale each grad in-place. Parameters with p.grad is None are ignored.
    Matches the behavior of torch.nn.utils.clip_grad.clip_grad_norm_.
    """
    grads = [p.grad for p in parameters if getattr(
        p, "grad", None) is not None]
    if not grads:
        return

    device = grads[0].device
    total_sq = torch.zeros((), device=device)
    for g in grads:
        total_sq = total_sq + g.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)

    eps = 1e-6
    clip_coef = (max_l2_norm / (total_norm + eps)).item()
    if clip_coef >= 1.0:
        return

    for p in parameters:
        if p.grad is None:
            continue
        p.grad.mul_(clip_coef)
