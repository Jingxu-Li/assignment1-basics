import torch.nn as nn

from cs336_basics.MyTransformerBlock import MyTransformerBlock
from cs336_basics.myEmbedding import MyEmbedding
from cs336_basics.utils import softmax
from cs336_basics.myLinear import MyLinear
from cs336_basics.MyRMSNorm import MyRMSNorm


class MyLMBlock(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, in_indices, max_seq_len=2048, rope_theta=10000.0, weights=None):
        super(MyLMBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.transformer_blocks = nn.ModuleList([
            MyTransformerBlock(
                d_model, num_heads, d_ff, max_seq_len, rope_theta
            ) for _ in range(num_layers)
        ])
        self.token_embeddings = MyEmbedding(vocab_size, d_model)
        self.ln_final = MyRMSNorm(d_model)
        self.output_linear = MyLinear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        # init token embeddings
        nn.init.xavier_uniform_(self.token_embeddings.weight)

        # 初始化每个 transformer block 的参数
        for transformer_block in self.transformer_blocks:
            for param in [
                transformer_block.q_proj_weight,
                transformer_block.k_proj_weight,
                transformer_block.v_proj_weight,
                transformer_block.o_proj_weight,
                transformer_block.w1_weight,
                transformer_block.w2_weight,
                transformer_block.w3_weight
            ]:
                nn.init.xavier_uniform_(param)

        # init linear (lm_head)
        nn.init.xavier_uniform_(self.output_linear.weight)
        # lm_head 通常没有 bias，所以不初始化 bias

    def load_weights(self, weights):
        # 加载 token embeddings
        self.token_embeddings.weight.data = weights['token_embeddings.weight']

        # 加载输出层权重 (lm_head)
        self.output_linear.weight.data = weights['lm_head.weight']

        self.ln_final.weight.data = weights['ln_final.weight']

        # 加载每个 transformer block 的权重
        for i, transformer_block in enumerate(self.transformer_blocks):
            # 为每个 layer 创建子字典
            layer_weights = {
                'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
                'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
                'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
                'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
                'ln1.weight': weights[f'layers.{i}.ln1.weight'],
                'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
                'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight'],
                'ffn.w3.weight': weights[f'layers.{i}.ffn.w3.weight'],
                'ln2.weight': weights[f'layers.{i}.ln2.weight']
            }
            transformer_block.load_weights(layer_weights)

    def forward(self, in_indices):
        # 1. Token Embeddings
        in_indices = self.token_embeddings(in_indices)
        # 2. Transformer Blocks
        for transformer_block in self.transformer_blocks:
            in_indices = transformer_block(in_indices)
        # 3. Final Layer Norm
        in_indices = self.ln_final(in_indices)
        # 4. Linear (lm_head) - 返回未归一化的logits
        in_indices = self.output_linear(in_indices)
        # 注意：不应用softmax，因为函数文档要求返回未归一化的预测分布
        return in_indices
