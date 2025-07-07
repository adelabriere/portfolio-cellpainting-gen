import torch.nn as nn



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
    def __init__(self, dim, time_dim, dim_out=None):
        self.dim_out = dim_out
        self.initial_conv = nn.Conv2d(dim, dim, kernel_size = 7, padding = "same", groups = dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1,dim),
            nn.Conv2d(dim, dim * 3, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim if dim_out is None else dim, kernel_size = 1)
        )

        self.time_mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_dim, dim)
        )

        self.residual_layer = nn.Conv2d(dim, dim_out, kernel_size = 1) if dim_out is not None else nn.Identity()

    def forward(self, x, time_emb):
        xres = self.residual_layer(x)
        t = self.time_mlp(time_emb)
        x = self.initial_conv(x)
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