from jaxtyping import Float
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
