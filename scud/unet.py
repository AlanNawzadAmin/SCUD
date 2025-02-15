# Code is adapted from https://github.com/google-research/google-research/tree/master/d3pm and https://github.com/google-research/vdm

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

class NormalizationLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        num_groups = 32
        num_groups = min(num_channels, num_groups)
        assert num_channels % num_groups == 0
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)

def pad_image(x, target_size):
    """Preprocess image to target size with padding."""
    _, _, h, w = x.shape
    # target_size = math.ceil(max([h, w]) // 2 ** sec_power) * 2 ** sec_power
    if h == target_size and w == target_size:
        return x
    
    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    padding = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
    return F.pad(x, padding, mode='constant', value=0)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, dropout, semb_dim=0, cond=False, film=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = NormalizationLayer(in_channels)
        self.norm2 = NormalizationLayer(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = emb_dim
        self.semb_dim = semb_dim
        self.film = film
        if emb_dim>0:
            self.temb_proj = nn.Linear(emb_dim, out_channels)
            if self.film:
                self.temb_proj_mult = nn.Linear(emb_dim, out_channels)
        if semb_dim>0:
            self.semb_proj = nn.Linear(semb_dim, out_channels)
            if self.film:
                self.semb_proj_mult = nn.Linear(semb_dim, out_channels)
        if cond:
            self.y_proj = nn.Linear(emb_dim, out_channels)
            if self.film:
                self.y_proj_mult = nn.Linear(emb_dim, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb, y, semb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add in timestep embedding
        if self.emb_dim > 0:
            if self.film:
                gam = 1 + self.temb_proj_mult(F.silu(temb))[:, :, None, None]
            else:
                gam = 1
            bet = self.temb_proj(F.silu(temb))[:, :, None, None]
            h = gam * h + bet

        if self.semb_dim > 0:
            if self.film:
                gam = 1 + self.semb_proj_mult(F.silu(semb.transpose(-1, -3))).transpose(-1, -3)
            else:
                gam = 1
            bet = self.semb_proj(F.silu(semb.transpose(-1, -3))).transpose(-1, -3)
            h = gam * h + bet
        
        # Add in class embedding
        if y is not None:
            if self.film:
                gam = 1 + self.y_proj_mult(y)[:, :, None, None]
            else:
                gam = 1
            bet = self.y_proj(y)[:, :, None, None]
            h = gam * h + bet
        
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, channels, width, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.height = width
        self.width = width
        self.head_dim = channels // self.num_heads
        self.norm = NormalizationLayer(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B = x.shape[0]
        h = self.norm(x).view(B, self.channels, self.height*self.width).transpose(1, 2)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, -1)
        q = q.view(B, self.height*self.width, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.height*self.width, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.height*self.width, self.num_heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(B, self.height, self.width, self.channels)
        h = self.proj_out(h)
        return x + h.transpose(2, 3).transpose(1, 2)


class KingmaUNet(nn.Module):
    def __init__(self,
                 n_channel=3,
                 N=256,
                 s_lengthscale=50,
                 time_lengthscale=1,
                 schedule_conditioning=False,
                 s_dim=16,
                 ch=128,
                 time_embed_dim=128,
                 s_embed_dim=128,
                 num_classes=1,
                 n_layers=32,
                 inc_attn=False,
                 dropout=0.1,
                 num_heads=1,
                 n_transformers=1,
                 width=32,
                 not_logistic_pars=True,
                 semb_style="learn_embed", # "learn_nn", "u_inject"
                 first_mult=False,
                 input_logits=False,
                 film=False,
                 **kwargs
                ):
        super().__init__()

        self.first_mult = first_mult
        if schedule_conditioning:
            in_channels = ch * n_channel + n_channel * s_dim

            emb_dim = s_dim//2
            semb_sin = 10000**(-torch.arange(emb_dim)/(emb_dim-1))
            self.register_buffer("semb_sin", semb_sin)
            if semb_style != "learn_embed":
                self.S_embed_sinusoid = lambda s: torch.cat([
                    torch.sin(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale),
                    torch.cos(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale)], dim=-1)
                in_channels = ch * n_channel + s_embed_dim
                self.S_embed_nn = nn.Sequential(
                    nn.Linear(n_channel * s_dim, s_embed_dim),
                    nn.SiLU(),
                    nn.Linear(s_embed_dim, s_embed_dim),
                )
                if self.first_mult:
                    self.S_mult_nn = nn.Sequential(
                        nn.Linear(n_channel * s_dim, s_embed_dim),
                        nn.SiLU(),
                        nn.Linear(s_embed_dim, ch * n_channel),
                    )
            else:
                s = torch.arange(10000).reshape(-1, 1) * 1000 / s_lengthscale
                semb = torch.cat([torch.sin(s * semb_sin), torch.cos(s * semb_sin)], dim=1)
                self.S_embed_sinusoid = nn.Embedding(10000, s_dim)
                self.S_embed_sinusoid.weight.data = semb
                s_embed_dim = 0
                self.S_embed_nn = nn.Identity()
            if semb_style != "u_inject":
                s_embed_dim = 0
        else:
            s_embed_dim = 0
            in_channels = ch * n_channel
        self.N = N
        self.n_channel = n_channel
        out_channels = n_channel * N 
        self.ch = ch
        self.n_layers = n_layers
        self.inc_attn = inc_attn
        self.num_classes = num_classes
        self.time_lengthscale = time_lengthscale
        self.width = width
        self.not_logistic_pars = not_logistic_pars

        self.input_logits = input_logits
        if not self.input_logits:
            self.x_embed = nn.Embedding(N, ch)
        else:
            self.x_embed = nn.Sequential(
                nn.Linear(n_channel * N, ch * n_channel),
                nn.SiLU(),
                nn.Linear(ch * n_channel, ch * n_channel),
            )
        # Time embedding
        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            if self.first_mult:
                self.time_embed_mult_nn = nn.Sequential(
                    nn.Linear(ch, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, ch * n_channel),
                )

        # Class embedding
        self.cond = num_classes > 1 
        if self.cond:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i_level in range(self.n_layers):
            block = nn.ModuleList()
            block.append(ResnetBlock(ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film)
        self.mid_attn = nn.Sequential(*[AttnBlock(ch, width, num_heads)
                                        for i in range(n_transformers)])
        self.mid_block2 = ResnetBlock(ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i_level in range(self.n_layers+1):
            block = nn.ModuleList()
            block.append(ResnetBlock(2 * ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.up_blocks.append(block)

        self.norm_out = NormalizationLayer(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    @torch.compile()
    def flat_unet(self, x, temb, yemb, semb):
        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down_blocks:
            h = blocks[0](h, temb, yemb, semb)
            h = blocks[1](h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, temb, yemb, semb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, yemb, semb)

        # Upsampling
        for i, blocks in enumerate(self.up_blocks):
            h = blocks[0](torch.cat([h, hs[self.n_layers-(i+1)]], dim=1), temb, yemb, semb)
            h = blocks[1](h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h
    
    def forward(self, x, t, y=None, S=None):
        B, C, H, W, *_ = x.shape
        if not self.input_logits:
            x_onehot = F.one_hot(x.long(), num_classes=self.N).float()
            x = self.x_embed(x.permute(0,2,3,1))
            x = x.reshape(*x.shape[:-2], -1).permute(0,3,1,2)
        else:
            x_onehot = 0
            x = (x - x.mean(-1)[..., None]).permute(0,2,3,1,4)
            x = self.x_embed(x.reshape(*x.shape[:-2], -1)).permute(0,3,1,2)

        # Time embedding        
        if self.time_embed_dim > 0:
            t = t.float().reshape(-1, 1) * 1000 / self.time_lengthscale
            emb_dim = self.ch//2
            temb_sin = 10000**(-torch.arange(emb_dim, device=t.device)/(emb_dim-1))
            temb_sin = torch.cat([torch.sin(t * temb_sin), torch.cos(t * temb_sin)], dim=1)
            temb = self.time_embed(temb_sin)
            if self.first_mult:
                t_mult = self.time_embed_mult_nn(temb_sin)
                x = x * t_mult[:, :, None, None]
        else:
            temb = None

        # S embedding
        if S is not None:
            semb_sin = self.S_embed_sinusoid(S.permute(0,2,3,1))
            semb = self.S_embed_nn(semb_sin.reshape(*semb_sin.shape[:-2], -1)).permute(0,3,1,2)

            if self.first_mult:
                s_mult = self.S_mult_nn(semb_sin.reshape(*semb_sin.shape[:-2], -1)).permute(0,3,1,2)
                x = x * s_mult
            x = torch.cat([x, semb], dim=1)
        else:
            semb = None

        # Class embedding
        if y is not None and self.num_classes > 1:
            yemb = self.class_embed(y)
        else:
            yemb = None
        
        # Reshape output
        h = self.flat_unet(x, temb, yemb, semb)
        h = h[:, :, :H, :W].reshape(B, C, self.N, H, W).permute((0, 1, 3, 4, 2))
        return h + self.not_logistic_pars * x_onehot

