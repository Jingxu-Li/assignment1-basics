import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import einsum


class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff, in_features, d_type=None, device=None):
        super(SwiGLU, self).__init__()
        self.w1_weight = nn.Parameter(torch.empty(
            d_ff, d_model, device=device, dtype=d_type))
        self.w2_weight = nn.Parameter(torch.empty(
            d_model, d_ff, device=device, dtype=d_type))
        self.w3_weight = nn.Parameter(torch.empty(
            d_ff, d_model, device=device, dtype=d_type))
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x):
        # 输入x的形状: (..., d_model)
        # 计算SwiSH门控值: w1_weight @ x
        # w1_weight: (d_ff, d_model), x: (..., d_model)
        # 结果: (..., d_ff)
        gate = einsum(x, self.w1_weight,
                      "... d_model, d_ff d_model -> ... d_ff")

        # 应用SwiSH激活函数
        silu = gate * F.sigmoid(gate)  # (..., d_ff)

        # 计算值投影: w3_weight @ x
        # w3_weight: (d_ff, d_model), x: (..., d_model)
        # 结果: (..., d_ff)
        value = einsum(x, self.w3_weight,
                       "... d_model, d_ff d_model -> ... d_ff")

        # 逐个元素相乘：silu ⊗ value
        # 方法1：使用einsum
        # silu_w3x = einsum(silu, value, "...")

        # 方法2：直接使用*操作符（更简单）
        silu_w3x = silu * value

        # 输出投影: w2_weight @ silu_w3x
        # w2_weight: (d_model, d_ff), silu_w3x: (..., d_ff)
        # 结果: (..., d_model)
        # 等价于 silu_w3x @ w2_weight.T
        output = einsum(self.w2_weight, silu_w3x,
                        "d_model d_ff, ... d_ff -> ... d_model")

        return output
