import math
import torch
import torch.nn as nn

from labml_nn.utils import clone_module_list
from .feed_forward import FeedForward
from .mha import MultiHeadAttention
from .positional_encoding import get_positional_encodeing


class EmbeddingsWithPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000) -> None:
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer(
            "positional_encodings", get_positional_encodeing(d_model, max_len)
        )

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[: x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[: x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe


class EmbeddingsWithLearnedEncoding(nn.Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000) -> None:
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[: x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe

class TransformerLayer(nn.Module)