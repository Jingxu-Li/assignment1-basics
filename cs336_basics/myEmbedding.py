import torch
import torch.nn as nn


class MyEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings is vocab size
        # embedding_dim is d_model
        super(MyEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.weight = nn.init.trunc_normal_(
            self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # embedding metrix is vocabsize*d_model
        # for each token, select embedding in embedding metrix
        # token_ids is batch*sequence
        return self.weight[token_ids]
