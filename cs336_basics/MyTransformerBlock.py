import torch
import torch.nn as nn
from einops import einsum


class MyTransformerBlock(nn.Module):
    """
    Transformer Block实现

    包含:
    1. 第一个RMSNorm
    2. 多头自注意力 (带RoPE)
    3. 残差连接
    4. 第二个RMSNorm  
    5. SwiGLU前馈网络
    6. 残差连接
    """

    def __init__(self, d_model, num_heads, d_ff, max_seq_len=2048, theta=10000.0):
        super(MyTransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 第一个RMSNorm (注意力前的归一化)
        from cs336_basics.MyRMSNorm import MyRMSNorm
        self.ln1 = MyRMSNorm(d_model)

        # 注意力权重
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj_weight = nn.Parameter(torch.empty(d_model, d_model))

        # 第二个RMSNorm (前馈网络前的归一化)
        self.ln2 = MyRMSNorm(d_model)

        # SwiGLU前馈网络权重
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 使用xavier_uniform初始化所有权重
        for param in [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
                      self.o_proj_weight, self.w1_weight, self.w2_weight, self.w3_weight]:
            nn.init.xavier_uniform_(param)

    def load_weights(self, weights):
        """加载预训练权重"""
        self.ln1.weight.data = weights['ln1.weight']
        self.q_proj_weight.data = weights['attn.q_proj.weight']
        self.k_proj_weight.data = weights['attn.k_proj.weight']
        self.v_proj_weight.data = weights['attn.v_proj.weight']
        self.o_proj_weight.data = weights['attn.output_proj.weight']
        self.ln2.weight.data = weights['ln2.weight']
        self.w1_weight.data = weights['ffn.w1.weight']
        self.w2_weight.data = weights['ffn.w2.weight']
        self.w3_weight.data = weights['ffn.w3.weight']

    def _attention(self, x):
        """多头自注意力机制"""
        from cs336_basics.MyMultiheadAttention import run_multihead_self_attention_with_rope

        return run_multihead_self_attention_with_rope(
            d_model=self.d_model,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            o_proj_weight=self.o_proj_weight,
            in_features=x,
            rope_theta=self.theta
        )

    def _swiglu_ffn(self, x):
        """SwiGLU前馈网络"""
        # 计算SwiSH门控值: w1_weight @ x
        gate = einsum(x, self.w1_weight,
                      "... d_model, d_ff d_model -> ... d_ff")

        # 应用SwiSH激活函数
        silu = gate * torch.sigmoid(gate)

        # 计算值投影: w3_weight @ x
        value = einsum(x, self.w3_weight,
                       "... d_model, d_ff d_model -> ... d_ff")

        # 逐个元素相乘：silu ⊗ value
        silu_w3x = silu * value

        # 输出投影: w2_weight @ silu_w3x
        output = einsum(self.w2_weight, silu_w3x,
                        "d_model d_ff, ... d_ff -> ... d_model")

        return output

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        # 第一个子层：注意力 + 残差连接
        # 1. RMSNorm
        normed_x = self.ln1(x)

        # 2. 多头自注意力
        attn_output = self._attention(normed_x)

        # 3. 残差连接
        x = x + attn_output

        # 第二个子层：前馈网络 + 残差连接
        # 4. RMSNorm
        normed_x = self.ln2(x)

        # 5. SwiGLU前馈网络
        ffn_output = self._swiglu_ffn(normed_x)

        # 6. 残差连接
        output = x + ffn_output

        return output


def run_transformer_block(
    d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features
):
    """
    函数式Transformer Block实现

    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络维度
        max_seq_len (int): 最大序列长度
        theta (float): RoPE参数
        weights (dict): 权重字典
        in_features (torch.Tensor): 输入特征

    Returns:
        torch.Tensor: 输出特征
    """
    # 创建Transformer Block实例
    transformer_block = MyTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta
    )

    # 加载权重
    transformer_block.load_weights(weights)

    # 前向传播
    return transformer_block(in_features)
