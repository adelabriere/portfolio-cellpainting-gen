import torch.nn as nn
from einops import rearrange
from torch import einsum


def get_activation_function(name,dic_args):
    vname = name.lower()
    if vname == "leaky_relu":
        return nn.LeakyReLU(**dic_args)
    elif vname == "relu":
        return nn.ReLU(**dic_args)
    elif vname == "silu":
        return nn.SiLU(**dic_args)
    
class Conv2dBlockUnetDiff(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ConvNextBlock(nn.Module):
    # Taken from https://arxiv.org/pdf/2201.03545"
    def __init__(self, dim, time_dim=None, dim_out=None, mult=3):
        super().__init__()
        self.dim_out = dim_out
        self.time_dim = time_dim
        self.mult = mult
        self.initial_conv = nn.Conv2d(dim, dim, kernel_size = 7, padding = "same", groups = dim)
        final_dim = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            nn.GroupNorm(1,dim),
            nn.Conv2d(dim, dim * self.mult, kernel_size = 3, padding="same"),
            nn.GELU(),
            nn.GroupNorm(1,dim * self.mult),
            nn.Conv2d(dim * self.mult, final_dim, kernel_size = 3, padding="same")
        )

        self.time_mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_dim, dim)
        ) if self.time_dim is not None else None

        self.residual_layer = nn.Conv2d(dim, dim_out, kernel_size = 1) if dim_out is not None else nn.Identity()

    def forward(self, x, time_emb=None):
        xres = self.residual_layer(x)
        x = self.initial_conv(x)

        if self.time_dim is not None:
            t = self.time_mlp(time_emb)
            t = t[:, :, None, None]
            x = x +t 

        x = self.net(x)
        return x + xres


class Conv2dStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, activation="leaky_relu", activation_args=None):
        super(Conv2dStack, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation_args is None:
            # FOr compatibility purpose
            activation_args = {"negative_slope":0.1, "inplace": True}
        self.activation_layer = get_activation_function(activation, activation_args)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_layer(x)
        return x

class Conv2dTransposeStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, output_padding=0, activation="leaky_relu", activation_args=None, bias=False):
        super(Conv2dTransposeStack, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                       stride=stride, output_padding=output_padding, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_args is None:
            # FOr compatibility purpose
            activation_args = {"negative_slope":0.1, "inplace": True}
        self.activation_layer = get_activation_function(activation, activation_args)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_layer(x)
        return x

class UpsampleConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_dim, mode="nearest",activation="leaky_relu", activation_args=None):
        super(UpsampleConvStack, self).__init__()
        self.upsample = nn.Upsample(size=output_dim, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_args is None:
            # FOr compatibility purpose
            activation_args = {"negative_slope":0.1, "inplace": True}
        self.activation_layer = get_activation_function(activation, activation_args)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_layer(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """See https://haileyschoelkopf.github.io/blog/2024/linear-attn/"""
    def __init__(self, dim, nhead=4, dim_head=32, add_residuals=False):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.add_residuals = add_residuals

        self.to_qkv = nn.Conv2d(dim, dim_head * nhead * 3, 1, bias=False)
        self.out = nn.Conv2d(dim_head * nhead, dim, 1, bias=False)
        
    def forward(self, x):
        B, C, X, Y = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape q,k,v
        exp_einops = 'b (h c) x y -> b h c (x y)'
        q = rearrange(q, exp_einops,h=self.nhead)
        k = rearrange(k, exp_einops,h=self.nhead)
        v = rearrange(v, exp_einops,h=self.nhead)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        phi_kv = einsum("b h d n, b h e n -> b h d e", k , v)
        att = einsum("b h d e, b h d n -> b h e n", phi_kv, q)
        att = rearrange(att, 'b h c (x y) -> b (h c) x y', x=X)
        out = self.out(att)
        if self.add_residuals:
            out = out + x
        return out