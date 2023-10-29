import math
from typing import Optional, List

import torch
from torch import nn
from labml import tracker


class PrepareForMultiHeadAttention(nn.Module):
    """
    This module does a linear transformation and split s the vector into given number of heads
    for muli head Attention. This is used to transform key ,query and value vector
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # input has shape seq_len, batch_size, d_model or batch_size, d_model
        head_shape = x.shape[:-1]
        x = self.linear(x)
        # split
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True
    ) -> None:
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(
            d_model=d_model, heads=self.heads, d_k=self.d_k, bias=bias
        )
        self.key = PrepareForMultiHeadAttention(
            d_model=d_model, heads=self.heads, d_k=self.d_k, bias=bias
        )
        self.value = PrepareForMultiHeadAttention(
            d_model=d_model, heads=self.heads, d_k=self.d_k, bias=True
        )

        # soft max for attention along the time dimesion of key
        self.softmax = nn.Softmax(dim=1)

        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)

        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

    def get_scores(slef, query: torch.Tensor, key: torch.Tensor):
        # 爱因斯坦求和约定计算矩阵乘法 Q*K^(T)
        return torch.einsum("ibhd,jbhd->ijbh", query, key)

    def prepare_mask(
        self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]
    ):
        # mask must has shape seq_len_q seq_len_k batch_size
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(
                mask=mask, query_shape=query.shape, key_shape=key.shape
            )
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query=query, key=key)

        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)

        tracker.debug("attn", attn)

        attn = self.dropout(attn)

        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)
