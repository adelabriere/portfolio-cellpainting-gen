
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from .abc_model import UnsupervisedImageGenerator
import lightning as L



# Utility funciton
def generate_noise(batch_size, image_size, device):
    # 1 isa ont th elast diemsion to be passed through a linear layer
    return torch.randn(batch_size, image_size, image_size, 1).to(device)

def generate_style(nlayers, batch_size, latent_dim, device):
    return torch.randn(nlayers, batch_size, latent_dim).to(device)


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

        # B x Ci
        # B x 1 x Ci x 1 x 1
        Se = S[:,None,:,None,None]

        # 1 x Co x Ci x K x K
        We = self.weights[None,:,:,:,:]

        # Instance normalization using broadcasting
        # B x Co x Ci x K x K
        w1 = We * Se

        weights = w1
        if self.demod:
            # B x Co x 1 x 1 x 1
            norm_factor = torch.rsqrt( ((w1**2).sum(dim=(2,3,4), keepdim=True) + self.epsilon) )

            # Demodulating
            # B x Co x Ci x K x K
            weights = weights *  norm_factor

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
        self.latent_dim = latent_dim

        self.to_style = nn.Linear(latent_dim, in_channel)
        self.conv = SG2ModConvNorm(in_channel, out_channel, 1, demod=False)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)#,
            # Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, s):
        # print("shapes x {} s {}".format(x.shape, s.shape))
        s = self.to_style(s)
        x = self.conv(x, s)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SG2GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_channel, out_channel, final_channel, kernel_size=3, upsample=True, upsample_output=True, epsilon=1e-7):
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
        self.output_block = SG2OutputBlock(latent_dim, out_channel, final_channel, upsample_output)

    def forward(self, x, s, noise, prev_output):
        # Input:
        #   x: B x Ci x H x W
        #   s: 1 x B x L
        #   noise: B x H x W x 1
        # Output
        #   x: B x Co x Hu x Wu (u depending of upsample or not)

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


# Discriminator section


class SG2DiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super(SG2DiscriminatorBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.downsample = downsample

        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.reshape_residual = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=(2 if downsample else 1))

        self.downsample = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride = 2, padding=1) if downsample else None

    def forward(self, x):
        # Input:
        #   x: B x Ci x H x W
        # Output
        #   x: B x Co x Hd x Wd (d depending of downsample or not)

        res = self.reshape_residual(x)
        x = self.layers(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


# D

class Generator(nn.Module):
    def __init__(self, in_channel, image_size, latent_dim, network_capacity=16):
        super(Generator, self).__init__()
        self.in_channel = in_channel
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.network_capacity = 16

        # Num layers is based on initial image size
        num_layers = int(math.log2(image_size) - 1)
        self.num_layers = num_layers

        # Learned initial tensor 
        self.initial_tensor = nn.Parameter(torch.randn(1, in_channel, 4, 4))


        # Input image size, output image size, number of channel 
        conv_channel = [network_capacity*4*(2**i) for i in range(num_layers)]
        conv_channel = conv_channel[::-1]

        input_image_channel = conv_channel[:-1]
        output_image_channel = conv_channel[1:]

        self.initial_conv = nn.Conv2d(in_channel, conv_channel[0], 3, padding=1)

        # BUIlding the layers
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            coutput_size = output_image_channel[i]
            cinput_size = input_image_channel[i]
            print("i {} {}".format(cinput_size,coutput_size))
            self.layers.append(SG2GeneratorBlock(latent_dim=self.latent_dim,in_channel=cinput_size, out_channel=coutput_size,\
                                                final_channel = in_channel, upsample = i!=0, upsample_output= (i!=(num_layers-1))))
        
    def forward(self, styles, noise):
        # Input:
        #  styles: W x B x C
        x = self.initial_tensor.expand(styles.shape[1], -1, -1, -1)
        x = self.initial_conv(x)
        layer_idx = 0
        o = None
        for s,layer in zip(styles,self.layers):
            x, o = layer(x, s, noise, o)
            layer_idx += 1
        return o

      
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channel, image_size, network_capacity=16):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.image_size = image_size
        self.layers = nn.ModuleList()

        # Num layers is based on initial image size
        num_layers = int(math.log2(image_size) - 1)
        conv_channel = [in_channel]+[network_capacity*4*(2**i) for i in range(num_layers)]

        input_image_channel = conv_channel[:-1]
        output_image_channel = conv_channel[1:]

        self.layers = nn.ModuleList()

# class SG2DiscriminatorBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, downsample=True):

        for i in range(num_layers):
            self.layers.append(SG2DiscriminatorBlock(in_channel=input_image_channel[i],\
                                                     out_channel=output_image_channel[i], downsample= True))#i!=(num_layers-1)))


        last_dim = conv_channel[-1] * 2 * 2

        self.flatten = nn.Flatten()
        self.to_logit = nn.Linear(last_dim, 1)

    def forward(self, x):
        ilayer = 0
        for layer in self.layers:
            x = layer(x)
            ilayer += 1
        x = self.flatten(x)
        x = self.to_logit(x)
        return x
    
# Style encoding values
class StyleEncoder(nn.Module):
    def __init__(self, latent_dim, depth):
        super(StyleEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth

        layers_list = []
        

        for i in range(self.depth):
            layers_list.append(nn.Linear(self.latent_dim, self.latent_dim))
            layers_list.append(nn.LeakyReLU(0.2))
        
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):

        x = F.normalize(x, dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class SG2SimpleGAN(UnsupervisedImageGenerator):
    def __init__(self, latent_dim, image_channel, image_size, style_depth=6, network_capacity=4,\
                 disc_training_interval=5, epoch_monitoring_interval=1, n_images_monitoring=6, learning_rate = 1e-4):
        super(SG2SimpleGAN, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)

        self.latent_dim = latent_dim
        self.image_channel = image_channel
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.style_depth = style_depth
        self.disc_training_interval = disc_training_interval

        self.G = Generator(image_channel, image_size, latent_dim, network_capacity)
        self.D = Discriminator(image_channel, image_size, network_capacity)
        self.Snet = StyleEncoder(latent_dim, depth = style_depth)
        
        self.gen_layers = self.G.num_layers

        self.loss = F.binary_cross_entropy
        self.learning_rate = learning_rate
        self.automatic_optimization = False


    # Utility function for training
    def disable_discriminator_training(self):
        for param in self.D.parameters():
            param.requires_grad = False

    def enable_discriminator_training(self):
        for param in self.D.parameters():
            param.requires_grad = True

    def disable_generator_training(self):
        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.Snet.parameters():
            param.requires_grad = False
    
    def enable_generator_training(self):
        for param in self.G.parameters():
            param.requires_grad = True
        for param in self.Snet.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.Snet.parameters()), lr=self.learning_rate)
        disc_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate)
        return gen_optimizer, disc_optimizer
    
    def generate_images(self, batch=None, n=None):
        device = batch.device
        noise = generate_noise(n, self.image_size, device=device)
        S = generate_style(self.gen_layers, n, self.latent_dim, device = device)
        W = self.Snet(S)
        return self.G(W, noise)
    
    def training_step(self, batch, batch_idx):
        real_imgs, real_imgs_dics = batch
        B = real_imgs.shape[0]
        Bd = real_imgs_dics.shape[0]
        device = real_imgs.device

        Gopt, Dopt = self.optimizers()

        # Training the discriminator
        if self.current_epoch % self.disc_training_interval == 0:
            self.enable_discriminator_training()
            self.disable_generator_training()

            Dopt.zero_grad()
            
            # Generate images
            fake_imgs = self.generate_images(real_imgs,n=Bd)

            # Discriminator loss
            p_fake = F.sigmoid(self.D(fake_imgs))
            p_real = F.sigmoid(self.D(real_imgs_dics))

            disc_loss = F.binary_cross_entropy(p_fake, torch.zeros_like(p_fake), reduction='mean') + \
                        F.binary_cross_entropy(p_real, torch.ones_like(p_real), reduction='mean')
            self.log('disc_loss', disc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            self.manual_backward(disc_loss)
            Dopt.step()
        
        # Training the discriminator
        self.enable_generator_training()
        self.disable_discriminator_training()
        Gopt.zero_grad()

        # Generate images
        fake_imgs = self.generate_images(real_imgs,n=B)

        # Generator loss
        p_fake = F.sigmoid(self.D(fake_imgs))
        gen_loss = F.binary_cross_entropy(p_fake, torch.ones_like(p_fake), reduction='mean')
        self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.manual_backward(gen_loss)
        Gopt.step()
            
        return super().training_step(batch[0], batch_idx)
