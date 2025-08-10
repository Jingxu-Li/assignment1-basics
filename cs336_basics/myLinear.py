import math
import torch
import torch.nn as nn
from einops import einsum


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(MyLinear, self).__init__()
        input_std = math.sqrt((2 / (in_features+out_features)))
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype))
        self.weight = nn.init.trunc_normal_(
            self.weight, mean=0, std=input_std, a=-3*input_std, b=3*input_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
