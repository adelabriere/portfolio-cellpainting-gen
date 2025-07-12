from .conv_modules import Conv2dStack,Conv2dTransposeStack, ConvNextBlock, LinearAttention, Attention
import torch
import math
import torch.nn as nn
from einops import rearrange


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
        # print("tproj init ",time_proj.shape)
        time_proj = time_proj[:,:,None,None]
        # print("tproj, h ",time_proj.shape, h.shape)
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



class UNetDiffusionV2(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels,\
                  network_capacity=64, nlayers=4, init_dim= None):#, activation="none"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.network_capacity = network_capacity
        self.nlayers = nlayers
        # self.activation = activation
        if init_dim is None:
            init_dim = network_capacity // 3 * 2

        network_channels = [init_dim]+[network_capacity*(2**i) for i in range(nlayers)]
        self.network_channels = network_channels

        # Sequence of in and out dimensions
        in_out_channels = list(zip(network_channels[:-1],network_channels[1:]))


        self.init_conv = nn.Conv2d(in_channels, init_dim, kernel_size=7, padding="same")

        self.contracting_path = nn.ModuleList()
        self.extending_path = nn.ModuleList()

        # Contracting path
        for i, (cin_channels, cout_channels) in enumerate(in_out_channels):
            downsample = None
            if i == nlayers-1:
                downsample = nn.Identity()
            else:
                downsample = nn.Conv2d(cout_channels, cout_channels, kernel_size=4, padding=1, stride=2)
            
            # print("down {} cin {} cout {}".format(i,cin_channels,cout_channels))
            current_module_list = nn.ModuleList([
                ConvNextBlock(cin_channels, time_channels, cout_channels),
                ConvNextBlock(cout_channels, time_channels, cout_channels),
                nn.GroupNorm(1, cout_channels),
                LinearAttention(cout_channels, add_residuals=False),
                downsample
            ])
            self.contracting_path.append(current_module_list)

        _,mid_dim = in_out_channels[-1]
        self.middle_block = nn.ModuleList([
            ConvNextBlock(mid_dim, time_channels, mid_dim),
            nn.GroupNorm(1, mid_dim),
            Attention(mid_dim),
            ConvNextBlock(mid_dim, time_channels, mid_dim)
        ])


        # Expanding path
        for i, (cin_channels, cout_channels) in enumerate(reversed(in_out_channels)):#[1:])):
            upsample = None
            if i == nlayers-1:
                upsample = nn.Identity()
            else:
                upsample = nn.ConvTranspose2d(cin_channels, cin_channels, kernel_size=4, padding=1, stride=2)
            # print("up {} cin {} cout {}".format(i,cout_channels,cin_channels))
            current_module_list = nn.ModuleList([
                ConvNextBlock(2 * cout_channels, time_channels, cin_channels),
                ConvNextBlock(cin_channels, time_channels, cin_channels),
                nn.GroupNorm(1, cin_channels),
                LinearAttention(cin_channels, add_residuals=False),
                upsample
            ])
            self.extending_path.append(current_module_list)

        self.final_layers = nn.Sequential(
            ConvNextBlock(init_dim,None, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        # if activation == "tanh":
        #     final_layers.append(nn.Tanh())
        # elif activation =="sigmoid":
        #     final_layers.append(nn.Sigmoid())
        # self.final_layer = nn.Sequential(*final_layers)
    
    def forward(self, x, time_emb):
        # print("init x {}".format(x.shape))
        x = self.init_conv(x)
        # print("initial_conv x {}".format(x.shape))
        contractions = []
        for ilayer,(c1, c2, norm, att, down) in enumerate(self.contracting_path):

            # print("down {} x {}".format(ilayer, x.shape))
            # print("c1dimout:{} c1residual:{} c1net:{}".format(c1.dim_out,c1.residual_layer,c1.net))
            x = c1(x, time_emb)
            # print("c2dimout:{} c2residual:{} c2net:{}".format(c2.dim_out,c2.residual_layer,c2.net))
            gx = c2(x, time_emb)
            x = norm(gx)
            x = att(x) + gx
            contractions.append(x)
            x = down(x)
        c1mid, normmid, attmid, c2mid = self.middle_block

        gx = c1mid(x, time_emb)
        x = normmid(gx)
        x = attmid(x) + gx
        x = c2mid(x, time_emb)

        for ilayer,(c1, c2, norm, att, up) in enumerate(self.extending_path):
            # print("up {} x {}".format(ilayer, x.shape))
            x = torch.cat([x, contractions.pop()], dim = 1)
            x = c1(x, time_emb)
            gx = c2(x, time_emb)
            x = norm(gx)
            x = att(x) + gx
            x = up(x)
        return self.final_layers(x)
