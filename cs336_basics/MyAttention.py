import torch
from cs336_basics.utils import softmax


def run_scaled_dot_product_attention(
    Q: torch.Tensor,  # Float[Tensor, " ... queries d_k"]
    K: torch.Tensor,  # Float[Tensor, " ... keys d_k"]
    V: torch.Tensor,  # Float[Tensor, " ... values d_v"]
    # Float[Tensor, " ... queries keys"] | None
    mask: torch.Tensor | None = None,
) -> torch.Tensor:  # Float[Tensor, " ... queries d_v"]
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # Calculate the scale factor based on the key dimension
    d_k = K.size(-1)
    scale = d_k ** 0.5

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # K.transpose(-2, -1): (..., d_k, keys)
    # Result: (..., queries, keys)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    # Apply mask if provided
    if mask is not None:
        # mask: (..., queries, keys)
        # Set masked positions to -inf so they become 0 after softmax
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get attention weights
    attn_weights = softmax(attn_scores, dim=-1)

    # Apply attention weights to values
    # attn_weights: (..., queries, keys), V: (..., keys, d_v)
    # Result: (..., queries, d_v)
    output = torch.matmul(attn_weights, V)

    return output
