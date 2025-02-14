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

class UNet(nn.Module):
    def __init__(self,
                 n_channel=3,
                 N=256,
                 s_lengthscale=50,
                 time_lengthscale=1,
                 schedule_conditioning=False,
                 s_dim=16,
                 ch=128,
                 time_embed_dim=512,
                 s_embed_dim=512,
                 num_classes=1,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 attn_resolutions=(1,),
                 dropout=0.1,
                 num_heads=1,
                 width=32,
                 not_logistic_pars=True,
                 semb_style="learn_embed", # "learn_nn"
                 film=False,
                 **kwargs
                ):
        super().__init__()

        if schedule_conditioning:
            in_channels = n_channel + n_channel * s_dim

            emb_dim = s_dim//2
            semb_sin = 10000**(-torch.arange(emb_dim)/(emb_dim-1))
            self.register_buffer("semb_sin", semb_sin)
            if semb_style != "learn_embed":
                self.S_embed_sinusoid = lambda s: torch.cat([
                    torch.sin(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale),
                    torch.cos(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale)], dim=-1)
                in_channels = n_channel + s_embed_dim
                self.S_embed_nn = nn.Sequential(
                    nn.Linear(n_channel * s_dim, s_embed_dim),
                    nn.SiLU(),
                    nn.Linear(s_embed_dim, s_embed_dim),
                )
            else:
                s = torch.arange(10000).reshape(-1, 1) * 1000 / s_lengthscale
                semb = torch.cat([torch.sin(s * semb_sin), torch.cos(s * semb_sin)], dim=1)
                self.S_embed_sinusoid = nn.Embedding(10000, s_dim)
                self.S_embed_sinusoid.weight.data = semb
                self.S_embed_nn = nn.Identity()
        else:
            in_channels= n_channel
        self.N = N
        self.n_channel = n_channel
        out_channels = n_channel * N 
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_classes = num_classes
        self.time_lengthscale = time_lengthscale
        self.width = width
        self.not_logistic_pars = not_logistic_pars

        # Time embedding
        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
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
        in_ch = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch, time_embed_dim, dropout, cond=self.cond, film=film))
                in_ch = out_ch
                if i_level in attn_resolutions:
                    block.append(AttnBlock(in_ch, self.width//(2**i_level), num_heads))
                else:
                    block.append(nn.Identity())
            if i_level != self.num_resolutions - 1:
                block.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
            else:
                block.append(nn.Identity())
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(in_ch, in_ch, time_embed_dim, dropout, cond=self.cond, film=film)
        self.mid_attn = AttnBlock(in_ch, self.width//(2**i_level), num_heads)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, time_embed_dim, dropout, cond=self.cond, film=film)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        prev_out = ch * ch_mult[-1]
        for i_level in list(reversed(range(self.num_resolutions))):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for j in range(self.num_res_blocks + 1):
                in_ch = ch * ((1,)+ch_mult)[i_level + 1 - (j==self.num_res_blocks)]
                block.append(ResnetBlock(in_ch + prev_out, out_ch, time_embed_dim, dropout, cond=self.cond, film=film))
                prev_out = out_ch
                if i_level in attn_resolutions:
                    block.append(AttnBlock(prev_out, self.width//(2**i_level), num_heads))
                else:
                    block.append(nn.Identity())
            if i_level != 0:
                block.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                           nn.Conv2d(prev_out, prev_out, 3, padding=1)))
            else:
                block.append(nn.Identity())
            self.up_blocks.append(block)

        self.norm_out = NormalizationLayer(prev_out)
        self.conv_out = nn.Conv2d(prev_out, out_channels, 3, padding=1)

    # @torch.compile(fullgraph=True, dynamic=False)
    def unet_main(self, x, temb, yemb):
        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down_blocks:
            for i in range(self.num_res_blocks):
                h = blocks[2 * i](h, temb, yemb)
                h = blocks[2 * i + 1](h)
                hs.append(h)
            h = blocks[-1](h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, temb, yemb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, yemb)

        # Upsampling
        len_hs = (self.num_res_blocks + 1) * self.num_resolutions # skip last addition
        for j, blocks in enumerate(self.up_blocks):
            for i in range(self.num_res_blocks+1):
                index = (self.num_res_blocks+1) * j + i
                old_h = hs[len_hs-(index+1)]
                h = blocks[2 * i](torch.cat([h, old_h], dim=1), temb, yemb)
                h = blocks[2 * i + 1](h)
            h = blocks[-1](h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h
    
    def forward(self, x, t, y=None, S=None):
        x_onehot = F.one_hot(x.long(), num_classes=self.N).float()
        x = pad_image(x, self.width)
        x = x.float()
        x[..., :self.n_channel, :, :] = (2 * x[..., :self.n_channel, :, :] / self.N) - 1.0

        # S embedding
        if S is not None:
            semb = self.S_embed_sinusoid(S.permute(0,2,3,1))
            semb = self.S_embed_nn(semb.reshape(*semb.shape[:-2], -1)).permute(0,3,1,2)

            x = torch.cat([x, semb], dim=1)
        else:
            semb = None

        # Time embedding   
        if self.time_embed_dim > 0:
            t = t.float().reshape(-1, 1) * 1000 / self.time_lengthscale
            emb_dim = self.ch//2
            temb = 10000**(-torch.arange(emb_dim, device=t.device)/(emb_dim-1))
            temb = torch.cat([torch.sin(t * temb), torch.cos(t * temb)], dim=1)
            temb = self.time_embed(temb)
        else:
            temb = None

        # Class embedding
        if y is not None and self.num_classes > 1:
            yemb = self.class_embed(y)
        else:
            yemb = None
        
        # Reshape output and add to x_onehot
        B, C, H, W, _ = x_onehot.shape
        h = self.unet_main(x, temb, yemb)
        h = h[:, :, :H, :W].reshape(B, C, self.N, H, W).permute((0, 1, 3, 4, 2))
        return h + self.not_logistic_pars * x_onehot

########################


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


########################

class GigaUNet(nn.Module):
    def __init__(self,
                 n_channel=3,
                 N=256,
                 s_lengthscale=50,
                 time_lengthscale=1,
                 schedule_conditioning=False,
                 s_dim=16,
                 ch=128,
                 time_embed_dim=128,
                 num_classes=1,
                 n_layers=32,
                 inc_attn=False,
                 dropout=0.1,
                 num_heads=1,
                 n_transformers=1,
                 width=32,
                 logistic_pars=True,
                 semb_style="u_inject", # "learn_nn", "u_inject"
                 film=False,
                 **kwargs
                ):
        super().__init__()

        s_embed_dim = ch
        if schedule_conditioning:
            in_channels = ch * n_channel + s_embed_dim

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
        assert s_embed_dim != 0
        self.N = N
        self.n_channel = n_channel
        out_channels = n_channel * N 
        self.ch = ch
        self.n_layers = n_layers
        self.inc_attn = inc_attn
        self.num_classes = num_classes
        self.time_lengthscale = time_lengthscale
        self.width = width
        self.logistic_pars = logistic_pars

        self.x_embed = nn.Embedding(N, ch)
        # Time embedding
        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Class embedding
        self.cond = num_classes > 1 
        if self.cond:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.conv_in_S = nn.Conv2d(s_embed_dim, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i_level in range(self.n_layers):
            block = nn.ModuleList()
            block.append(ResnetBlock(ch, ch, time_embed_dim, dropout, ch, self.cond, film))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.down_blocks.append(block)
        self.down_blocks_S = nn.ModuleList()
        for i_level in range(self.n_layers):
            block = nn.ModuleList()
            block.append(ResnetBlock(ch, ch, time_embed_dim, dropout, 0, self.cond, film))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.down_blocks_S.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(ch, ch, time_embed_dim, dropout, s_embed_dim, self.cond, film)
        self.mid_attn = nn.Sequential(*[AttnBlock(ch, width, num_heads)
                                        for i in range(n_transformers)])
        self.mid_block2 = ResnetBlock(ch, ch, time_embed_dim, dropout, s_embed_dim, self.cond, film)

        # Middle S
        self.mid_block1_S = ResnetBlock(ch, ch, time_embed_dim, dropout, 0, self.cond, film)
        self.mid_attn_S = nn.Sequential(*[AttnBlock(ch, width, num_heads)
                                        for i in range(n_transformers)])
        self.mid_block2_S = ResnetBlock(ch, ch, time_embed_dim, dropout, 0, self.cond, film)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i_level in range(self.n_layers+1):
            block = nn.ModuleList()
            block.append(ResnetBlock(2 * ch, ch, time_embed_dim, dropout, s_embed_dim, self.cond, film))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.up_blocks.append(block)

        # Upsampling
        self.up_blocks_S = nn.ModuleList()
        for i_level in range(self.n_layers+1):
            block = nn.ModuleList()
            block.append(ResnetBlock(2 * ch, ch, time_embed_dim, dropout, 0, self.cond, film))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.up_blocks_S.append(block)

        self.norm_out = NormalizationLayer(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    @torch.compile()
    def flat_unet(self, x, temb, yemb, semb):
        # Downsampling
        h = self.conv_in(x)
        h_S = self.conv_in_S(semb)
        hs = [h]
        hs_S = [h_S]
        for blocks_S, blocks in zip(self.down_blocks_S, self.down_blocks):
            h_S = blocks_S[0](h_S, temb, yemb, None)
            h_S = blocks_S[1](h_S)
            hs_S.append(h_S)
            h = blocks[0](h, temb, yemb, h_S)
            h = blocks[1](h)
            hs.append(h)

        # Middle
        h_S = self.mid_block1_S(h_S, temb, yemb, None)
        h_S = self.mid_attn_S(h_S)
        h = self.mid_block1(h, temb, yemb, h_S)
        h = self.mid_attn(h)
        h_S = self.mid_block2_S(h_S, temb, yemb, None)
        h = self.mid_block2(h, temb, yemb, h_S)

        # Upsampling
        for i, (blocks_S, blocks) in enumerate(zip(self.up_blocks_S, self.up_blocks)):
            h_S = blocks_S[0](torch.cat([h_S, hs_S[self.n_layers-(i+1)]], dim=1), temb, yemb, None)
            h_S = blocks_S[1](h_S)
            h = blocks[0](torch.cat([h, hs[self.n_layers-(i+1)]], dim=1), temb, yemb, h_S)
            h = blocks[1](h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h
    
    def forward(self, x, t, y=None, S=None):
        x_onehot = F.one_hot(x.long(), num_classes=self.N).float()
        x = self.x_embed(x.permute(0,2,3,1))
        x = x.reshape(*x.shape[:-2], -1).permute(0,3,1,2)

        # S embedding
        if S is not None:
            semb = self.S_embed_sinusoid(S.permute(0,2,3,1))
            semb = self.S_embed_nn(semb.reshape(*semb.shape[:-2], -1)).permute(0,3,1,2)

            x = torch.cat([x, semb], dim=1)
        else:
            semb = None

        # Time embedding        
        if self.time_embed_dim > 0:
            t = t.float().reshape(-1, 1) * 1000 / self.time_lengthscale
            emb_dim = self.ch//2
            temb = 10000**(-torch.arange(emb_dim, device=t.device)/(emb_dim-1))
            temb = torch.cat([torch.sin(t * temb), torch.cos(t * temb)], dim=1)
            temb = self.time_embed(temb)
        else:
            temb = None
        
        # Class embedding
        if y is not None and self.num_classes > 1:
            yemb = self.class_embed(y)
        else:
            yemb = None
        
        # Reshape output and add to x_onehot
        B, C, H, W, _ = x_onehot.shape
        h = self.flat_unet(x, temb, yemb, semb)
        h = h[:, :, :H, :W].reshape(B, C, self.N, H, W).permute((0, 1, 3, 4, 2))
        return h + self.logistic_pars * x_onehot

########################

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm(self.conv2(x))
        return F.relu(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channel, N, n_T, schedule_conditioning=False, s_dim=16):
        super().__init__()

        if schedule_conditioning:
            in_channel = n_channel + n_channel * s_dim

            self.S_embed = nn.Embedding(n_T, s_dim)
            self.S_embed.weight.data = torch.stack(
                [torch.sin(torch.arange(n_T) * 3.1415 * 2**i) for i in range(s_dim // 2)] + 
                [torch.cos(torch.arange(n_T) * 3.1415 * 2**i) for i in range(s_dim // 2)]
            ).T
        else:
            in_channel = n_channel

        self.N = N
        self.n_channel = n_channel
        out_channel = n_channel * N 
        
        # Encoder (downsampling)
        self.enc1 = SimpleConvBlock(in_channel, 64)
        self.enc2 = SimpleConvBlock(64, 128)
        self.enc3 = SimpleConvBlock(128, 256)
        
        # Decoder (upsampling)
        self.dec3 = SimpleConvBlock(256 + 128, 128)  # 256 from enc3, 128 from enc2
        self.dec2 = SimpleConvBlock(128 + 64, 64)    # 128 from dec3, 64 from enc1
        self.dec1 = SimpleConvBlock(64, 32)
        
        self.final = nn.Conv2d(32, out_channel, 1)

    def forward(self, x, t, cond, S=None):
        x = x.float()
        x[..., :self.n_channel, :, :] = (2 * x[..., :self.n_channel, :, :] / self.N) - 1.0
    
        if S is not None:
            s_embed = self.S_embed(S.permute(0,2,3,1))
            s_embed = s_embed.reshape(*s_embed.shape[:-2], -1).permute(0,3,1,2)

            x = torch.cat([x, s_embed], dim=1)
        
        # Encoder
        e1 = self.enc1(x.float())
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(e3, scale_factor=2, mode='nearest'), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), e1], dim=1))
        d1 = self.dec1(d2)

        d0 = self.final(d1)
        
        return d0.reshape(d0.shape[0], -1, self.N, *x.shape[2:]).transpose(2, -1)

