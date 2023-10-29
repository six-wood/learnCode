import math

import numpy as np
import torch
from torch import nn


def get_positional_encodeing(d_model: int, max_len: int = 5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)

    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings


class PositionalEncodeing(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "positional_encoding",
            get_positional_encodeing(d_model=d_model, max_len=max_len),
            False,
        )

    def forward(self, x: torch.Tensor):
        """
        一、tensor.detach()
        返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
        这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，
        到该调用detach()的tensor就会停止，不能再继续向前进行传播
        注意：
        使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变。
        """
        pe = self.positional_encoding[: x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        return x


def _test_positional_encoding():
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    pe = get_positional_encodeing(20, 100)
    plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional Encoding")
    plt.show()


if __name__ == "__main__":
    _test_positional_encoding()
