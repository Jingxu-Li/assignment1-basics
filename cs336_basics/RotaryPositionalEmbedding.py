import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 计算逆频率
        inv_freq = 1.0 / \
            (self.theta ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 初始化缓存
        self.max_seq_len_cached = 0
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_k) 或 (batch_size * num_heads, seq_len, d_k)
        # token_positions: (batch_size, seq_len) 或 (batch_size * num_heads, seq_len)

        # 确保token_positions在正确的设备上
        if token_positions.device != self.cos_cached.device:
            token_positions = token_positions.to(self.cos_cached.device)

        # 保证缓存长度足够
        max_pos = int(token_positions.max().item()) + 1
        if max_pos > self.max_seq_len_cached:
            self._update_cos_sin_cache(max_pos)

        # 获取每个 token 位置对应的 cos/sin 编码
        # cos_cached 和 sin_cached 的形状是 (seq_len, d_k)
        # token_positions 的形状是 (batch_size, seq_len) 或 (batch_size * num_heads, seq_len)
        # 结果形状是 (batch_size, seq_len, d_k) 或 (batch_size * num_heads, seq_len, d_k)
        cos, sin = self.cos_cached[token_positions], self.sin_cached[token_positions]
        cos1, _ = cos.chunk(2, dim=-1)
        sin1, _ = sin.chunk(2, dim=-1)

        # 确保cos1和sin1的形状与x的批次维度匹配
        if cos1.shape[0] != x.shape[0]:
            # 如果批次维度不匹配，需要广播
            cos1 = cos1.expand(x.shape[0], -1, -1)
            sin1 = sin1.expand(x.shape[0], -1, -1)

        # 偶数位和奇数位分组
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        # 旋转编码
        rot_even = x_even * cos1 - x_odd * sin1
        rot_odd = x_even * sin1 + x_odd * cos1

        # 交错合并回原始顺序
        rotated_x = torch.empty_like(x)
        rotated_x[..., ::2] = rot_even
        rotated_x[..., 1::2] = rot_odd

        return rotated_x


if __name__ == "__main__":
    x = torch.randn(1, 10, 128)
    token_positions = torch.arange(10, device=x.device)
    rop = RotaryPositionalEmbedding(10000, 128, 1024, device=x.device)
    print(rop(x, token_positions))
