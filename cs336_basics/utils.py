from math import cos, pi
from typing import Iterable
from jaxtyping import Float, Int
from torch import Tensor
import torch
import os
from typing import BinaryIO, IO
import numpy as np
import numpy.typing as npt


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
    # 确保targets是long类型用于索引
    targets = targets.long()
    target_log_probs = log_softmax[torch.arange(batch_size, device=inputs.device), targets]

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

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data from the dataset.
    
    Args:
        dataset: 输入数据集，形状为 (seq_len,)
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备类型 ('cpu' 或 'cuda')
        
    Returns:
        tuple: (x, y) 其中：
            - x: 输入序列，形状为 (batch_size, context_length)
            - y: 目标序列，形状为 (batch_size, context_length)
    """
    # 确保数据集长度足够
    if len(dataset) < context_length + 1:
        raise ValueError(f"数据集长度 {len(dataset)} 小于 context_length + 1 = {context_length + 1}")
    
    # 随机选择起始位置
    # 确保每个样本都有足够的后续token作为目标
    max_start_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # 创建批次数据
    x = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    y = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    
    for i, start_idx in enumerate(start_indices):
        # 输入序列：从 start_idx 开始的 context_length 个token
        x[i] = dataset[start_idx:start_idx + context_length]
        # 目标序列：从 start_idx + 1 开始的 context_length 个token（向右偏移1）
        y[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    # 转换为PyTorch张量并移动到指定设备
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    
    return x_tensor, y_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    Save the model and optimizer state to a file.
    """
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load the model and optimizer state from a file.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']