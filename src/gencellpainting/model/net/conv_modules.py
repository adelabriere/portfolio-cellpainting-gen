import torch.nn as nn



def get_activation_function(name,dic_args):
    if name == "leaky_relu":
        return nn.LeakyReLU(**dic_args)
    elif name == "relu":
        return nn.ReLU(**dic_args)
    

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