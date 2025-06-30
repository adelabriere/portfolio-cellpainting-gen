from .conv_modules import Conv2dStack,Conv2dTransposeStack
import torch
import math
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class DoubleConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=None,kernel_size=3):
        super(DoubleConvBn, self).__init__()
        if middle_channels is None:
            middle_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding="same")
        self.b1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.b2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.b2(x)
        x = self.activation(x)
        return x


class UnetContractionStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UnetContractionStack, self).__init__()
        self.dconv = DoubleConvBn(in_channels, out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(stride = 2, kernel_size = 2)

    def forward(self, x):
        xi = self.dconv(x)
        return self.pool(xi), xi
    
class UnetExpansionStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UnetExpansionStack, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Reducing the channel
        self.dconv = DoubleConvBn(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x, cx):
        x = self.up(x)
        x = torch.cat([x, cx] , dim=1)
        return self.dconv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, network_capacity=64, nlayers=4):
        super(UNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.network_capacity = network_capacity
        self.nlayers = nlayers

        network_channels = [network_capacity*(2**i) for i in range(nlayers + 1)]
        self.network_channels = network_channels
    
        self.contracting_path = nn.ModuleList()
        self.extending_path = nn.ModuleList()

        for i in range(nlayers):
            cin_channels = network_channels[i - 1] if i !=  0 else in_channels
            cout_channels = network_channels[i]
            self.contracting_path.append(UnetContractionStack(cin_channels, cout_channels))

        self.middle_layer = DoubleConvBn(network_channels[-2], network_channels[-2], middle_channels=network_channels[-1])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # COntractring path
        for i in range(nlayers):


            cin_channels = network_channels[nlayers - i]
            cout_channels = network_channels[nlayers - i - 2] if i != (nlayers-1) else network_channels[nlayers - i - 1]
            self.extending_path.append(UnetExpansionStack(cin_channels, cout_channels))

        self.final_layer = nn.Conv2d(network_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        contractions = []
        for i, down in enumerate(self.contracting_path):
            x, xi = down(x)
            contractions.append(xi)

        x = self.middle_layer(x)

        for i, (up, xi) in enumerate(zip(self.extending_path, contractions[::-1])):
            x = up(x , xi)
        
        return self.final_layer(x)


