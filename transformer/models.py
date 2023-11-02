import math
import torch
import torch.nn as nn

from labml_nn.utils import clone_module_list
from labml_nn.transformers import Encoder, MultiHeadAttention
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


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        """
        Some implementations, including the paper seem to have differences in where the layer-normalization is done. 
        Here we do a layer normalization before attention and feed-forward networks, 
        and add the original residual vectors. Alternative is to do a layer normalization after adding the residuals.
        But we found this to be less stable when training. 
        We found a detailed discussion about this in the paper On Layer Normalization in the Transformer Architecture
        """
        """
        在transformer中一般采用LayerNorm，LayerNorm也是归一化的一种方法，
        与BatchNorm不同的是它是对每单个batch进行的归一化，
        而batchnorm是对所有batch一起进行归一化的
        因此train()和eval()对LayerNorm没有影响
        
        """
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm(d_model)
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.is_save_ff_input = False

    def forward(
        self,
        *,
        x: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        # normalize vector before atten
        z = self.norm_self_attn(x)
        # run self attention
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # add the self attention result
        x = x + self.dropout(self_attn)
        if src is not None:
            z = self.norm_src_attn(x)
            src_attn = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(src_attn)

        z = self.norm_ff(x)

        if self.is_save_ff_input:
            self.ff_input = z.clone()

        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layer: int) -> None:
        super().__init__()
        self.layers = clone_module_list(layer, n_layer)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layer: int) -> None:
        super().__init__()
        self.layers = clone_module_list(layer, n_layer)
        self.norm = nn.LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, n_vocab: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x: torch.Tensor):
        return self.projection(x)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        for p in self.parameters():
            if p.dim > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        enc = self.encoder(src, src_mask)
        return self.decoder(enc, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask=src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
