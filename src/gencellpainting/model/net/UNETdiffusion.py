from .conv_modules import Conv2dStack,Conv2dTransposeStack, ConvNextBlock
import torch
import math
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, middle_channels=None,kernel_size=3):
        super(ResNetBlock, self).__init__()
        if middle_channels is None:
            middle_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding="same")
        self.b1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.b2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.time_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_dim, middle_channels)
        )
        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.activation(h)
        # print("TIME_EMB {}".format(time_emb.device))
        time_proj = self.time_projection(time_emb)
        time_proj = time_proj[:,:,None,None]
        h = h + time_proj

        h = self.conv2(h)
        h = self.b2(h)
        h = self.activation(h)
        return h + self.residual_layer(x)

class UnetContractionDiffusionStack(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, kernel_size=3, nblocks=1):
        super(UnetContractionDiffusionStack, self).__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(nblocks):
            cin_channels = in_channels if i==0 else out_channels
            self.res_blocks.append(ResNetBlock(cin_channels, out_channels, time_channels, kernel_size=kernel_size))

        self.pool = nn.MaxPool2d(stride = 2, kernel_size = 2)

    def forward(self, x, time_emb):
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        return self.pool(x), x
    
class UnetExpansionDiffusionStack(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, kernel_size=3, nblocks=1):
        super(UnetExpansionDiffusionStack, self).__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(nblocks):
            cin_channels = in_channels if i==0 else out_channels
            self.res_blocks.append(ResNetBlock(cin_channels, out_channels, time_channels, kernel_size=kernel_size))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, time_emb):
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        return self.up(x)


class UNetDiffusion(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels,\
                  network_capacity=64, nlayers=4, activation="none"):
        super(UNetDiffusion,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.network_capacity = network_capacity
        self.nlayers = nlayers
        self.activation = activation

        network_channels = [network_capacity*(2**i) for i in range(nlayers + 1)]
        self.network_channels = network_channels
    
        self.contracting_path = nn.ModuleList()
        self.extending_path = nn.ModuleList()

        # Contracting path
        for i in range(nlayers):
            cin_channels = network_channels[i - 1] if i !=  0 else in_channels
            cout_channels = network_channels[i]
            self.contracting_path.append(UnetContractionDiffusionStack(cin_channels, cout_channels, time_channels))

        self.middle_layer = UnetExpansionDiffusionStack(network_channels[nlayers-1], network_channels[nlayers-1], time_channels)

        # Expanding path
        for i in range(nlayers-1):
            cin_channels = network_channels[nlayers - i]
            cout_channels = network_channels[nlayers - i - 2] if i != (nlayers-1) else network_channels[nlayers - i - 1]
            self.extending_path.append(UnetExpansionDiffusionStack(cin_channels, cout_channels, time_channels))

        final_layers = [nn.Conv2d(network_channels[0], out_channels, kernel_size=1)]

        if activation == "tanh":
            final_layers.append(nn.Tanh())
        elif activation =="sigmoid":
            final_layers.append(nn.Sigmoid())
        self.final_layer = nn.Sequential(*final_layers)
    
    def forward(self, x, time_emb):
        contractions = []
        for i, down in enumerate(self.contracting_path):
            x, xres = down(x, time_emb)
            contractions.append(xres)


        x = self.middle_layer(x, time_emb)

        for i, (up, xi) in enumerate(zip(self.extending_path, contractions[::-1])):
            x = torch.cat([x, xi], axis = 1)
            x = up(x , time_emb)
        return self.final_layer(x)
