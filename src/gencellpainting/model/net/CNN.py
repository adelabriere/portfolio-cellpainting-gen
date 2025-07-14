from .conv_modules import Conv2dStack,Conv2dTransposeStack
import torch
import math
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class EncoderWithPooling(nn.Module):
    def __init__(self, in_channels, latent_dim, network_capacity=32, image_size=128, nlayers=None):
        super(EncoderWithPooling, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.network_capacity = network_capacity
        self.image_size = image_size
        self.nlayers = nlayers
        if nlayers is None:
            nlayers = int(math.log2(image_size))
        networks_channels = [in_channels]+[network_capacity*2**i for i in range(nlayers)]
        final_image_size = image_size//(2**nlayers)

        # image size is expected to be a power of 2
        seq_layers = []

        for i in range(nlayers):
            imsize = int(image_size/(2**i))
            cin_channels = networks_channels[i]
            cout_channels = networks_channels[i+1]
            seq_layers.extend([
                Conv2dStack(cin_channels, out_channels=cout_channels, kernel_size=min(3,imsize), stride=1, padding="same"),
                nn.MaxPool2d(kernel_size=2, stride = 2)  
            ])
        seq_layers.append(nn.Flatten())
        self.model = nn.Sequential(*seq_layers)

        # We create the last layer that will output the latent space
        self.latent_layer = nn.Linear(networks_channels[-1] * final_image_size * final_image_size, 2 * latent_dim)
        self.softplus = nn.Softplus()

    def forward(self, x, eps = 1e-8):
        x = self.model(x)
        l = self.latent_layer(x)
        mu, logvar = torch.chunk(l, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = MultivariateNormal(loc = mu, scale_tril=scale_tril)
        return dist


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, network_capacity = 32, kernel_size=3,\
                 image_size=128, nlayers=None):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.network_capacity = network_capacity
        self.image_size=image_size
        self.kernel_size = kernel_size
        self.bias = True
        self.activation = "relu"


        # We count the number of layers
        if nlayers is None:
            nlayers = int(math.log2(self.image_size)-1)

        networks_channels = [network_capacity*2**i for i in range(1,nlayers+1)]
        self.networks_channels = networks_channels[::-1]

        layers_list = []
        # Initial reshaping layer
        layers_list.append(
            Conv2dTransposeStack(self.latent_dim, out_channels=self.networks_channels[0],\
                kernel_size=4, stride=1, padding=0, output_padding=0,activation_args={},
                      activation=self.activation, bias=self.bias),#(B, latent_dim, 4, 4)
        )

        for ilayer in range(0, nlayers-1):
            cin = self.networks_channels[ilayer]
            cout = self.networks_channels[ilayer+1]
            cstride = 2
            layers_list.append(
                Conv2dTransposeStack(cin, out_channels=cout, kernel_size=self.kernel_size, stride=cstride,\
                                padding=1, output_padding=1, activation_args={},\
                                    activation=self.activation, bias=self.bias) #(B, cout, 4*2**(ilayer+1), 4*2**(ilayer+1))
            )

        self.net = nn.Sequential(*layers_list)
        
        self.output_layers = nn.Sequential(
            nn.ConvTranspose2d(self.networks_channels[-1], out_channels=out_channels, kernel_size=self.kernel_size,\
                                    stride=1, bias=self.bias, padding=1), # Output: (B, out_channels, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, z):
        # x = self.fc(z)
        x = z.view(-1, self.latent_dim, 1, 1)
        x = self.net(x)
        x = self.output_layers(x)
        return x
