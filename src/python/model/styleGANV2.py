
from conv_modules import UpsampleConvStack, Conv2dTransposeStack, Conv2dStack
import torch.nn as nn
import torch.nn.functional as F
import torch
from VAE import Decoder, Encoder
import lightning as L
from abc_model import UnsupervisedImageGenerator




class SG2ModConvNorm(nn.Module):
    # Weight modified convolution
    def __init__(self, in_channel, out_channel, kernel_size, epsilon = 1e-7, demod = True):
        super(SG2ModConvNorm, self).__init__()
        # Weights of the ocncolution
        self.weights = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.demod = demod
        self.epsilon = epsilon

    def forward(self, X, S):
        # Input:
        #   X: Images, B x C x H x W
        #   S: Style vector, B x C
        # Output:
        #      Images B x Co x H x W

        B, Ci, H, W = X.shape
        K = self.kernel_size
        Co = self.out_channel

        # B x 1 x Ci x 1 x 1
        Se = S[:,None,:,None,None]

        # 1 x Co x Ci x K x K
        We = self.weights[None,:,:,:,:]

        # Instance normalization using broadcasting
        # B x Co x Ci x K x K
        w1 = We * Se

        if self.demod:
            # B x Co x 1 x 1 x 1
            norm_factor = torch.rsqrt( ((w1**2).sum(dim=(2,3,4), keepdim=True) + self.epsilon) )

            # Demodulating
            # B x Co x Ci x K x K
            weights = w1 *  norm_factor

        # 1 x (B x Ci) x H x W
        X = X.reshape(1, -1, H, W)

        # (B x Co) x Ci x K x K
        weights = weights.reshape(B * Co, Ci, K, K)

        # 1 x (B x Co) x H x W
        X = F.conv2d(X, weights, padding="same", groups=B)
        X = X.reshape(B, Co, H, W)
        return X


class SG2OutputBlock(nn.Module):
    """Output block for sytle GAN 2. Convert high diemnsionnal image  and stile to image with output channel im"""
    def __init__(self, latent_dim, in_channel, out_channel, upsample):
        super(SG2OutputBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.to_style = nn.Linear(latent_dim, in_channel)
        self.conv = SG2ModConvNorm(in_channel, out_channel, 1, demod=False)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)#,
            # Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, s):
        s = self.to_style(s)
        x = self.conv(x, s)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SG2GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_channel, out_channel, final_channel, kernel_size=3, upsample=True, epsilon=1e-7):
        super(SG2GeneratorBlock, self).__init__()
        
        self.latent_dim = latent_dim
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.epsilon = epsilon

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        
        # Block 1
        self.to_style1 = nn.Linear(latent_dim, in_channel)
        self.to_noise1 = nn.Linear(1,out_channel)        
        self.conv1 = SG2ModConvNorm(in_channel, out_channel, kernel_size, epsilon=epsilon)

        # Block 2
        self.to_style2 = nn.Linear(latent_dim, out_channel)
        self.to_noise2 = nn.Linear(1,out_channel)        
        self.conv2 = SG2ModConvNorm(out_channel, out_channel, kernel_size, epsilon=epsilon)


        self.activation = nn.LeakyReLU(negative_slope = 0.1)
        self.output_block = SG2OutputBlock(latent_dim, in_channel, final_channel, upsample)

    def forward(self, x, s, noise, prev_output):
        # Input:
        #   x: B x Ci x H x W
        #   s: 1 x B x L
        #   noise: B x H x W x 1
        # Output
        #   x: B x Co x Hu x Wu (u depending of upsample or not)
        #   output: 

        # Upsampling first if necessary
        if self.upsample is not None:
            x = self.upsample(x)

        B,C,H,W = x.shape

        # seems unncessary
        rnoise = noise[:,:H,:W ,:]

        # B x H x W x Co
        noise1 = self.to_noise1(rnoise)
        noise2 = self.to_noise2(rnoise)

        # Reordering
        # B x Co x H x W
        noise1 = noise1.permute(0,3,1,2)
        noise2 = noise2.permute(0,3,1,2)

        # 1 x B x Co
        s1 = self.to_style1(s)

        # 1 x B x Co
        s2 = self.to_style2(s)

        x = self.conv1(x, s1)
        x = self.activation(x + noise1)
        x = self.conv2(x, s2)
        x = self.activation(x + noise2)

        # Progressive output, added to build the image
        output = self.output_block(x, prev_output, s)
        return x, output

