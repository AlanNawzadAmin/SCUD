import numpy as np
from sequence_models.layers import PositionFeedForward
from sequence_models.convolutional import MaskedConv1d, ByteNetBlock
from torch import nn
import torch
import math
import torch.nn.functional as F
from .dit_text import modulate_fused, TimestepEmbedder

class ByteNetLMTimeNew(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, simple_embed=True, n_tokens=31, d_aa_emb=8, d_embedding=128, d_model=1024, n_layer=16,
                 kernel_size=5, r=128, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.1, slim=True, activation='gelu',
                 schedule_conditioning=True, **kwargs):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        """
        super().__init__()
        self.simple_embed = simple_embed
        self.schedule_conditioning = schedule_conditioning
        if not schedule_conditioning:
            self.time_embed_input = TimestepEmbedderNew(2 * d_model) 
            self.time_embed_block = TimestepEmbedderNew(d_embedding) 
            self.time_embed_input.mlp[2].weight.data.zero_()
            self.time_embed_input.mlp[2].bias.data.zero_()
        if schedule_conditioning:
            self.s_embed_input = TimestepEmbedderNew(2 * d_model)
            self.s_embed_block = TimestepEmbedderNew(d_embedding)
            self.s_embed_input.mlp[2].weight.data.zero_()
            self.s_embed_input.mlp[2].bias.data.zero_()
        if not simple_embed:
            self.embedder = nn.Embedding(n_tokens, d_aa_emb,
                                         padding_idx=padding_idx)
            self.up_embedder = nn.Linear(d_aa_emb, d_model)
        else:
            self.embedder = nn.Embedding(n_tokens, d_model, padding_idx=padding_idx)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layer)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        self.layers = nn.ModuleList([
            ByteNetBlock_wmod(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation, dropout=dropout)
            for d in dilations
        ])
        self.c_mod_layers = nn.ModuleList([nn.Linear(d_embedding, 2*d_h) for d in dilations])
        for layer in self.c_mod_layers:
            layer.weight.data.zero_()
            layer.bias.data.zero_()
        self.dropout = dropout
        self.decoder = PositionFeedForward(d_model, n_tokens)
        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x, t, input_mask=None, S=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        x = self.embedder(x)
        if not self.simple_embed:
            x = self.up_embedder(x)

        if self.schedule_conditioning:
            S_out = F.silu(self.s_embed_input(S.reshape(-1))).reshape(S.shape+(-1,))
            x = modulate_fused(x,*S_out.chunk(2, dim=-1))
            c = F.silu(self.s_embed_block(S.reshape(-1))).reshape(S.shape+(-1,))
        else:
            t_out = F.silu(self.time_embed_input(t))[:, None, :]
            x = modulate_fused(x,*t_out.chunk(2, dim=-1))
            c = F.silu(self.time_embed_block(t))[:, None, :]

        for layer, c_layer in zip(self.layers, self.c_mod_layers):
            c_mod = c_layer(c)
            x = layer(x, c_mod, input_mask=input_mask.unsqueeze(-1))
        return self.decoder(self.last_norm(x))


class ByteNetBlock_wmod(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).
         
         Shape:
            Input: (N, L, d_in)
            input_mask: (N, L, 1), optional
            Output: (N, L, d_out)

    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1,
                 causal=False, activation='gelu', rank=None, dropout=0.0):
        super().__init__()
        self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        act = nn.GELU
        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h, rank=rank),
        ]
        layers_mod = [
            nn.LayerNorm(d_h),
            act()
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out, rank=rank),
        ]
        self.dropout = dropout
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence_mod = nn.Sequential(*layers_mod)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, c_mod, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        skip_x = x
        x = self.sequence1(x)
        x = modulate_fused(x,*c_mod.chunk(2, dim=-1))
        x = self.sequence_mod(x)
        x = F.dropout(x, self.dropout)
        x = self.conv(x, input_mask=input_mask)
        x = self.sequence2(x)
        return skip_x + x


class TimestepEmbedderNew(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=1280):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size, bias=True),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = 2 * 3.14159 * torch.exp(
      - math.log(max_period)
      * (torch.arange(start=0, end=half, dtype=torch.float32) - half/3)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class ByteNetLMTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, simple_embed=True, n_tokens=31, d_aa_emb=8, d_embedding=128, d_model=1024, n_layer=16,
                 kernel_size=5, r=128, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.1, slim=True, activation='gelu',
                 schedule_conditioning=True, **kwargs):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        """
        super().__init__()
        self.simple_embed = simple_embed
        self.time_encoding = TimestepEmbedderOld(d_embedding) # Timestep encoding
        self.time_mod_layer = nn.Linear(d_embedding, d_model)
        self.schedule_conditioning = schedule_conditioning
        if schedule_conditioning:
            self.s_embed_input = TimestepEmbedderOld(d_model)
            self.s_embed_block = TimestepEmbedderOld(d_embedding)
        if not simple_embed:
            self.embedder = nn.Embedding(n_tokens, d_aa_emb, padding_idx=padding_idx)
            self.up_embedder = nn.Linear(d_aa_emb, d_model)
        else:
            self.embedder = nn.Embedding(n_tokens, d_model, padding_idx=padding_idx)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layer)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        self.layers = nn.ModuleList([
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ])
        self.c_mod_layers = nn.ModuleList([nn.Linear(d_embedding, 2*d_model) for d in dilations])
        for layer in self.c_mod_layers:
            layer.weight.data.zero_()
            layer.bias.data.zero_()
        self.dropout = dropout
        self.decoder = PositionFeedForward(d_model, n_tokens)
        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x, t, input_mask=None, S=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        x = self.embedder(x)
        if not self.simple_embed:
            x = self.up_embedder(x)
        c = F.silu(self.time_encoding(t))[:, None, :]

        x = x + self.time_mod_layer(c)
        if S is not None:
            bs, seq_len = S.shape[0], S.shape[1]
            S_out = F.silu(self.s_embed_input(S.reshape(-1))).reshape(bs, seq_len, -1)
            x = x + S_out
            S_out = F.silu(self.s_embed_block(S.reshape(-1))).reshape(bs, seq_len, -1)
            c = c + S_out

        for layer, c_layer in zip(self.layers, self.c_mod_layers):
            x = layer(x, input_mask=input_mask.unsqueeze(-1))
            c_mod = c_layer(c)
            x = modulate_fused(x, *c_mod.chunk(2, dim=-1))
            if self.dropout > 0.0:
                x = F.dropout(x, self.dropout)
        return self.decoder(self.last_norm(x))
        
class TimestepEmbedderOld(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=1280):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * (torch.arange(start=0, end=half, dtype=torch.float32))
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb
