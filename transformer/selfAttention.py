import math
import torch
import torch.nn

# self Attention

"""

softMax(x*x')*x,自注意力原理
Attention = softMax( q*k'/sqrt(d) )*v,q k v都是x的线性变换,用于增强泛化能力
    增加了参数量，增加模型的表达能力。
    加入了不同的线性变换相当于对做了不同的投影,将向量投影到不同空间,增加模型的泛化能力,不要那么hard。
    允许某个token对其他位置token的注意力大于对自己的注意力,才能更好的捕捉全局位置的注意力。
sqrt(d)是为了防止梯度爆炸
    归一化方差,平滑分布
    大量实验已经验证如果不scale,模型预训练很难收敛

"""


class SelfAttention(torch.nn.Module):
    # input : batch_size * seq_len * input_dim
    # q: query
    # k: key
    # v: value
    def __init__(self, input_dim, dim_k, dim_v) -> None:
        super(SelfAttention, self).__init__()
        self.q = torch.nn.Linear(input_dim, dim_k)
        self.k = torch.nn.Linear(input_dim, dim_k)
        self.v = torch.nn.Linear(input_dim, dim_v)
        self.__normal_factor = 1 / math.sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Atten = torch.nn.Softmax(dim=-1)(
            torch.bmm(Q, K.transpose(1, 2)) * self.__normal_factor
        )  # Q * K.T() # batch_size * seq_len * seq_len
        out = torch.bmm(Atten, V)
        return out


def testAtten():
    x = torch.randn(10, 20, 30)
    selfAttention = SelfAttention(30, 40, 50)
    out = selfAttention(x)


"""
在《Attention Is All You Need》这篇原论文原文中解释了多头的作用:
    将隐状态向量分成多个头，形成多个子语义空间，可以让模型去关注不同维度语义空间的信息
    (或者说让模型去关注不同方面的信息)

multi-head-attention中大部分头没有捕捉到语法/句法信息
    
多头的核心思想就是ensemble,如随机森林一样,将特征切分,每个head就像是一个弱分类器,
让最后得到的embedding关注多方面信息,不要过拟合到某一种pattern上,这一点上面的实验图像可以很清晰的看出来。
    通过观察大量样本的attention矩阵我们发现,
    其实几乎每一个token在全句中的注意力都是稀疏的,即每个token只关注非常有限个其他token,
    其余注意力基本可以看成是0(softmax无法严格为0),大量稀疏就意味着我们可以对其进行低秩分解
"""

"""
self Attention都是全局Attention,即每个token都关注其他所有token
self Attention得到的attention矩阵是稀疏的,即每个token只关注非常有限个其他token,
可以分为多个头,即multi-head-attention
解释:
    1. 通过多头attention可以让模型去关注不同维度语义空间的信息(或者说让模型去关注不同方面的信息)
    比如随机森林一样,有的Attention矩阵基本只关注语法/句法信息,有的Attention矩阵基本只关注语义信息
    2.既然是稀疏的,那么就可以对其进行低秩分解,减少参数量
    可以看成是对特征进行切分,每个head就像是一个弱分类器,让最后得到的embedding关注多方面信息,不要过拟合到某一种pattern上
"""


class selfAttentionMultiHead(torch.nn.Module):
    # input : batch_size * seq_len * input_dim
    # q: query
    # k: key
    # v: value

    def __init__(self, input_dim, dim_k, dim_v, num_dead) -> None:
        super(selfAttentionMultiHead, self).__init__()
        self.q = torch.nn.Linear(input_dim, dim_k)
        self.k = torch.nn.Linear(input_dim, dim_k)
        self.v = torch.nn.Linear(input_dim, dim_v)

        self.num_head = num_dead
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.__normal_factor = 1 / math.sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q_ = Q.view(-1, Q.shape[1], self.num_head, self.dim_k // self.num_head)
        K_ = K.view(-1, K.shape[1], self.num_head, self.dim_k // self.num_head)
        V_ = V.view(-1, V.shape[1], self.num_head, self.dim_v // self.num_head)

        atten = torch.nn.Softmax(dim=-1)(
            torch.matmul(Q_, K_.permute(0, 1, 3, 2))
        )  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.matmul(atten, V_).reshape(
            x.shape[0], x.shape[1], -1
        )  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output


def testMultiHeadAtten():
    x = torch.randn(10, 20, 30)
    selfAttention = selfAttentionMultiHead(30, 40, 50, 10)
    out = selfAttention(x)


if __name__ == "__main__":
    testAtten()
    testMultiHeadAtten()
