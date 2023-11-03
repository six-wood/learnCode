import math
import torch
from torch import nn

from labml_helpers.module import Module


class FeedForward(Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_prob: float = 0.1,
        activation=nn.ReLU(),
        is_grated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)

        self.dropout = nn.Dropout(dropout_prob)

        self.activation = activation

        self.is_grated = is_grated
        if is_grated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_grated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)
