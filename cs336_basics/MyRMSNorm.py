import torch
import torch.nn as nn


class MyRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(MyRMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        RMS Norm前向传播

        RMS Norm公式: y = x / sqrt(mean(x^2) + eps) * weight

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., d_model)

        返回:
            torch.Tensor: 归一化后的张量，形状与输入相同
        """
        output = rms_norm_functional(x, self.weight, self.eps)
        return output


def rms_norm_functional(x, weight, eps=1e-5):
    """
    函数式RMS Norm实现

    参数:
        x (torch.Tensor): 输入张量
        weight (torch.Tensor): 权重参数
        eps (float): 数值稳定性常数

    返回:
        torch.Tensor: 归一化后的张量
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def rms_norm_einsum(x, weight, eps=1e-5):
    """
    使用einsum的RMS Norm实现

    参数:
        x (torch.Tensor): 输入张量
        weight (torch.Tensor): 权重参数
        eps (float): 数值稳定性常数

    返回:
        torch.Tensor: 归一化后的张量
    """
    from einops import einsum

    # 计算RMS: sqrt(mean(x^2))
    x_squared = einsum(x, x, "... d, ... d -> ...")
    rms = torch.sqrt(torch.mean(x_squared, dim=-1, keepdim=True) + eps)

    # 归一化并应用权重
    normalized = einsum(x, rms, "... d, ... -> ... d")
    output = einsum(normalized, weight, "... d, d -> ... d")

    return output
