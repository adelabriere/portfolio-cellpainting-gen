from .conv_modules import Conv2dStack,Conv2dTransposeStack
import torch
import math
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal



    # def __init__(self, latent_dim, out_channels, network_capacity = 32, kernel_size=4,\
    #              image_size=128, nlayers=None):
    #     super(Decoder, self).__init__()
    #     self.latent_dim = latent_dim
    #     self.out_channels = out_channels
    #     self.network_capacity = network_capacity
    #     self.image_size=image_size
    #     self.kernel_size = kernel_size
    #     self.bias = True
    #     self.activation = "relu"


    #     # We count the number of layers
    #     if nlayers is None:
    #         nlayers = int(math.log2(self.image_size)-1)

    #     networks_channels = [network_capacity*2**i for i in range(1,nlayers)]
    #     self.networks_channels = networks_channels[::-1]

    #     layers_list = []
    #     # Initial reshaping layer
    #     layers_list.append(
    #         Conv2dTransposeStack(self.latent_dim, out_channels=self.networks_channels[0],\
    #             kernel_size=4, stride=1, padding=0, output_padding=0,activation_args={},
    #                   activation=self.activation, bias=self.bias),#(B, latent_dim, 4, 4)
            
    #     )

    #     for ilayer in range(0, nlayers-2):
    #         cin = self.networks_channels[ilayer]
    #         cout = self.networks_channels[ilayer+1]
    #         cstride = 2
    #         layers_list.append(
    #             Conv2dTransposeStack(cin, out_channels=cout, kernel_size=self.kernel_size, stride=cstride,\
    #                             padding=1, output_padding=0, activation_args={},\
    #                                 activation=self.activation, bias=self.bias) #(B, cout, 4*2**(ilayer+1), 4*2**(ilayer+1))
    #         )

    #     self.net = nn.Sequential(*layers_list)
        
    #     self.output_layers = nn.Sequential(
    #         nn.ConvTranspose2d(self.networks_channels[-1], out_channels=out_channels, kernel_size=4,\
    #                                 stride=2, padding=1, bias=self.bias), # Output: (B, out_channels, 128, 128)
    #         nn.Sigmoid()
    #     )


class DoubleConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConvBn, self).__init__()
        vpad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=vpad)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=vpad)
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
        x = self.dconv(x)
        return self.pool(x)
    
class UnetExpansionStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        vpad = kernel_size // 2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Reducing the channel
        self.dconv = DoubleConvBn(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x, cx):
        # We first upscale X 
        x = self.up(x)
        x = torch.cat([x, cx] , dim=1)
        return self.dconv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, network_capacity=64, kernel_size=4, nlayers=4):
        super(UNET,self).__init__()
        self.out_channels = out_channels
        self.network_capacity = network_capacity
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.bias = True
        self.activation = "relu"

        network_channels = [network_capacity*(2**i) for i in range(nlayers + 1)]
        self.network_channels = network_channels
    
        self.contracting_path = nn.ModuleList()
        self.expansive_path = nn.ModuleList()

        for i in range(nlayers-1):
            in_channels = network_channels[i]
            out_channels = network_channels[i + 1]
            self.contracting_path.append(UnetContractionStack(in_channels, out_channels))

        # COntractring path
        for i in range(nlayers-1):
            in_channel = 





    def forward(self, x):
        contractions = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            contractions.append(x)

        x = self.middle_layer(x)

        for i, up in enumerate(self.expansive_path):
            x = self.upsample(x)
            x = torch.cat([x, contractions[-(i-1])], dim=1


