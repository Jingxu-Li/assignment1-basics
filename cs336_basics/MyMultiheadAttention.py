import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.MyAttention import run_scaled_dot_product_attention


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    实现多头自注意力机制，基于 run_scaled_dot_product_attention

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q投影权重，包含所有头的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K投影权重，包含所有头的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V投影权重，包含所有头的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 输入特征

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: 多头自注意力的输出
    """
    # 计算每个头的维度
    d_k = d_model // num_heads
    d_v = d_model // num_heads

    # 获取输入的形状
    batch_shape = in_features.shape[:-2]  # 除了最后两个维度 (sequence_length, d_model)
    seq_len = in_features.shape[-2]

    # 1. 线性投影得到 Q, K, V
    # 重塑输入以便进行批量矩阵乘法
    # (batch_size * seq_len, d_model)
    in_features_reshaped = in_features.view(-1, d_model)

    # 计算 Q, K, V - 权重矩阵乘以输入向量
    # (batch_size * seq_len, d_model)
    Q = torch.matmul(q_proj_weight, in_features_reshaped.T).T
    # (batch_size * seq_len, d_model)
    K = torch.matmul(k_proj_weight, in_features_reshaped.T).T
    # (batch_size * seq_len, d_model)
    V = torch.matmul(v_proj_weight, in_features_reshaped.T).T

    # 2. 重塑为多头格式
    # 首先重塑为 (batch_size, seq_len, d_model)
    Q = Q.view(*batch_shape, seq_len, d_model)
    K = K.view(*batch_shape, seq_len, d_model)
    V = V.view(*batch_shape, seq_len, d_model)

    # 然后重塑为 (batch_size, num_heads, seq_len, d_k/d_v)
    # 使用 transpose 来正确交换维度
    Q = Q.view(*batch_shape, seq_len, num_heads,
               d_k).transpose(-3, -2).contiguous()  # (batch, heads, seq, d_k)
    K = K.view(*batch_shape, seq_len, num_heads,
               d_k).transpose(-3, -2).contiguous()  # (batch, heads, seq, d_k)
    V = V.view(*batch_shape, seq_len, num_heads,
               d_v).transpose(-3, -2).contiguous()  # (batch, heads, seq, d_v)

    # 3. 构造因果mask (causal mask)
    # 使用 torch.triu 创建上三角矩阵，然后反转得到下三角mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = ~causal_mask  # 反转，True表示可以attend，False表示不能attend

    # 扩展到多头格式: (seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
    attention_mask = causal_mask.unsqueeze(
        0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    attention_mask = attention_mask.expand(
        *batch_shape, num_heads, seq_len, seq_len)

    # 转换为浮点类型，其中1.0表示可以attend，0.0表示不能attend
    attention_mask = attention_mask.float()

    # 4. 应用缩放点积注意力
    # 使用 run_scaled_dot_product_attention 处理每个头
    # 输入形状: (batch_size, num_heads, seq_len, d_k/d_v)
    # 输出形状: (batch_size, num_heads, seq_len, d_v)
    attn_output = run_scaled_dot_product_attention(Q, K, V, attention_mask)

    # 5. 重塑回原始格式
    # 从 (batch_size, num_heads, seq_len, d_v) 到 (batch_size, seq_len, d_model)
    # 使用 transpose 和 contiguous 来确保内存布局正确
    attn_output = attn_output.transpose(-3, -
                                        2).contiguous().view(*batch_shape, seq_len, d_model)

    # 6. 应用输出投影
    # 重塑为 (batch_size * seq_len, d_model) 以便进行矩阵乘法
    attn_output_reshaped = attn_output.view(-1, d_model)

    # 应用输出投影
    # (batch_size * seq_len, d_model)
    output = torch.matmul(o_proj_weight, attn_output_reshaped.T).T

    # 重塑回原始形状
    output = output.view(*batch_shape, seq_len, d_model)

    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    rope_theta: float = 10000.0,
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    实现带RoPE的多头自注意力机制，基于 run_scaled_dot_product_attention

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q投影权重，包含所有头的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K投影权重，包含所有头的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V投影权重，包含所有头的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 输入特征
        rope_theta (float): RoPE参数
        token_positions (Int[Tensor, "... sequence_length"] | None): token位置信息

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: 多头自注意力的输出
    """
    # 计算每个头的维度
    d_k = d_model // num_heads
    d_v = d_model // num_heads

    # 获取输入的形状
    batch_shape = in_features.shape[:-2]  # 除了最后两个维度 (sequence_length, d_model)
    seq_len = in_features.shape[-2]

    # 1. 线性投影得到 Q, K, V
    # 重塑输入以便进行批量矩阵乘法
    # (batch_size * seq_len, d_model)
    in_features_reshaped = in_features.view(-1, d_model)

    # 计算 Q, K, V - 权重矩阵乘以输入向量
    # (batch_size * seq_len, d_model)
    Q = torch.matmul(q_proj_weight, in_features_reshaped.T).T
    # (batch_size * seq_len, d_model)
    K = torch.matmul(k_proj_weight, in_features_reshaped.T).T
    # (batch_size * seq_len, d_model)
    V = torch.matmul(v_proj_weight, in_features_reshaped.T).T

    # 2. 重塑为多头格式
    # 首先重塑为 (batch_size, seq_len, d_model)
    Q = Q.view(*batch_shape, seq_len, d_model)
    K = K.view(*batch_shape, seq_len, d_model)
    V = V.view(*batch_shape, seq_len, d_model)

    # 然后重塑为 (batch_size, num_heads, seq_len, d_k/d_v)
    # 使用 transpose 来正确交换维度
    Q = Q.view(*batch_shape, seq_len, num_heads,
               d_k).transpose(-3, -2).contiguous()  # (batch, heads, seq, d_k)
    K = K.view(*batch_shape, seq_len, num_heads,
               d_k).transpose(-3, -2).contiguous()  # (batch, heads, seq, d_k)
    V = V.view(*batch_shape, seq_len, num_heads,
               d_v).transpose(-3, -2).contiguous()  # (batch, heads, seq, d_v)

    # 3. 应用RoPE到Q和K（但不应用到V）
    # 将头维度作为batch维度处理
    # (batch, heads, seq, d_k) -> (batch * heads, seq, d_k)
    Q_reshaped = Q.view(-1, seq_len, d_k)
    K_reshaped = K.view(-1, seq_len, d_k)

    # 如果token_positions为None，使用默认的连续位置
    if token_positions is None:
        # 创建默认的位置编码：从0到seq_len-1
        token_positions = torch.arange(
            seq_len, device=in_features.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(*batch_shape, -1)

    # 扩展token_positions以匹配头维度
    # 确保token_positions的形状是 (batch, seq)
    if token_positions.dim() == 1:
        # 如果是 (seq,)，扩展为 (1, seq)
        token_positions = token_positions.unsqueeze(0).expand(*batch_shape, -1)
    elif token_positions.dim() == 2:
        # 如果已经是 (batch, seq)，确保批次维度匹配
        if token_positions.shape[0] != batch_shape[0]:
            token_positions = token_positions.expand(*batch_shape, -1)

    # 扩展为 (batch, heads, seq) 然后重塑为 (batch * heads, seq)
    token_positions_expanded = token_positions.unsqueeze(
        1).expand(-1, num_heads, -1)
    token_positions_reshaped = token_positions_expanded.view(-1, seq_len)

    # 确保token_positions_reshaped的形状与Q_reshaped的批次维度匹配
    # Q_reshaped: (batch * heads, seq, d_k)
    # token_positions_reshaped: (batch * heads, seq)
    assert token_positions_reshaped.shape[0] == Q_reshaped.shape[0], \
        f"token_positions_reshaped batch size {token_positions_reshaped.shape[0]} " \
        f"does not match Q_reshaped batch size {Q_reshaped.shape[0]}"

    # 应用RoPE
    from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(rope_theta, d_k, seq_len)
    Q_rope = rope.forward(Q_reshaped, token_positions_reshaped)
    K_rope = rope.forward(K_reshaped, token_positions_reshaped)

    # 重塑回多头格式
    Q = Q_rope.view(*batch_shape, num_heads, seq_len, d_k)
    K = K_rope.view(*batch_shape, num_heads, seq_len, d_k)

    # 4. 构造因果mask (causal mask)
    # 使用 torch.triu 创建上三角矩阵，然后反转得到下三角mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = ~causal_mask  # 反转，True表示可以attend，False表示不能attend

    # 扩展到多头格式: (seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
    attention_mask = causal_mask.unsqueeze(
        0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    attention_mask = attention_mask.expand(
        *batch_shape, num_heads, seq_len, seq_len)

    # 转换为浮点类型，其中1.0表示可以attend，0.0表示不能attend
    attention_mask = attention_mask.float()

    # 5. 应用缩放点积注意力
    # 使用 run_scaled_dot_product_attention 处理每个头
    # 输入形状: (batch_size, num_heads, seq_len, d_k/d_v)
    # 输出形状: (batch_size, num_heads, seq_len, d_v)
    attn_output = run_scaled_dot_product_attention(Q, K, V, attention_mask)

    # 6. 重塑回原始格式
    # 从 (batch_size, num_heads, seq_len, d_v) 到 (batch_size, seq_len, d_model)
    # 使用 transpose 和 contiguous 来确保内存布局正确
    attn_output = attn_output.transpose(-3, -
                                        2).contiguous().view(*batch_shape, seq_len, d_model)

    # 7. 应用输出投影
    # 重塑为 (batch_size * seq_len, d_model) 以便进行矩阵乘法
    attn_output_reshaped = attn_output.view(-1, d_model)

    # 应用输出投影
    # (batch_size * seq_len, d_model)
    output = torch.matmul(o_proj_weight, attn_output_reshaped.T).T

    # 重塑回原始形状
    output = output.view(*batch_shape, seq_len, d_model)

    return output
