from .conv_modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .VAE import Decoder
import lightning as L
from .abc_model import UnsupervisedImageGenerator, AbstractGAN

   

class WGANCritic(nn.Module):
    def __init__(self, in_channels, image_size, nlayers=None, network_capacity = 32, kernel_size=4):
        super(WGANCritic, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.network_capacity = network_capacity 
        self.image_size=image_size
        self.kernel_size = kernel_size
        self.bias = True
        self.activation = "relu"


        # We count the number of layers
        if nlayers is None:
            nlayers = int(math.log2(self.image_size)-1)

        network_channels = [network_capacity*2**i for i in range(nlayers)]
        self.network_channels = network_channels
        # self.networks_channels = networks_channels[::-1]

        layers_list = []
        # Initial reshaping layer
        layers_list.extend([
            nn.Conv2d(self.in_channels, out_channels=network_channels[0],\
                                      kernel_size=self.kernel_size, stride=2, padding=1), # (B, _, 64, 64)
            nn.InstanceNorm2d(network_channels[0], affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        for ilayer in range(0, nlayers-2):
            cin = self.network_channels[ilayer]
            cout = self.network_channels[ilayer+1]
            cstride = 2
            layers_list.extend([
                nn.Conv2d(cin, out_channels=cout, kernel_size=4, stride=2, padding=2), # (B, _, 64, 64)
                nn.InstanceNorm2d(cout, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        self.net = nn.Sequential(*layers_list)
        self.output = nn.Conv2d(cout, out_channels= 1, kernel_size=4, stride=2, padding=0)

    def forward(self, x):
        x = self.net(x)
        return self.output(x)


class WGAN_GP(AbstractGAN):
    """Own implemntation of https://arxiv.org/pdf/1701.07875 the parameters are taken from the paper"""
    def __init__(self, in_channels, out_channels, noise_dim, n_critic = 5, vlambda=10,\
                 network_capacity=16,image_size=128,\
                 learning_rate = 1e-5,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(WGAN_GP, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        # self.generator = GeneratorV2(out_channels,latent_dim=noise_dim)
        self.generator  = Decoder(noise_dim, out_channels, image_size=image_size, network_capacity=network_capacity)
        self.discriminator = WGANCritic(in_channels, image_size)

        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.learning_rate = learning_rate
        self.vlambda = vlambda

        # Classification loss
        self.classification_loss = F.binary_cross_entropy

        self.automatic_optimization = False

    def sample_z(self, batch_size):
        # unirofrm between 0 and 1
        Z = torch.rand(batch_size, self.noise_dim)
        return Z
    
    def generate_images(self, batch, n):
        Z = self.sample_z(n).to(batch.device)
        return self.generator(Z)
    
    # Utility function for training
    def disable_discriminator_training(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def enable_discriminator_training(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def disable_generator_training(self):
        for param in self.generator.parameters():
            param.requires_grad = False
    
    def enable_generator_training(self):
        for param in self.generator.parameters():
            param.requires_grad = True

    def clamp_parameters(self, model, value):
        for p in model.parameters():
            p.data.clamp_(-value, value)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        B, C, H, W = real_samples.size()

        # We sample interpolation wieights
        eta = torch.rand(B, 1, 1 ,1)
        eta = eta.expand(-1, C, H, W).to(real_samples.device)

        interpolated_samples = eta * real_samples + ((1 - eta) * fake_samples)

        # We compute the gradient based on this
        interpolated_samples.requires_grad = True

        # We compute the discriminator output
        d_interpolated = self.discriminator(interpolated_samples)

        # We compute the gradient
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_samples,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(B,-1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty


    def training_step(self, batch, batch_idx):
        real_images, critic_images = batch

        opt_gen,opt_disc = self.optimizers()

        critic_images = critic_images.chunk(self.n_critic, dim=0)

        # Training discriminator
        self.disable_generator_training()
        self.enable_discriminator_training()

        total_grad_penalty = 0.0

        mones = torch.tensor(-1).to(real_images.device)
        ones = torch.tensor(1).to(real_images.device)

        for ic in range(self.n_critic):

            opt_disc.zero_grad()
            # Sample the real data
            R = critic_images[ic]

            # Sample the prior data
            d_R_disc = self.discriminator(R)
            d_R_disc = d_R_disc.mean()
            self.manual_backward(d_R_disc,mones)


            # PRevent gradient ocmputation on the first model
            Z = self.sample_z(R.size(0)).to(R.device)
            G_Z = self.generator(Z)
            d_G_disc = self.discriminator(G_Z)
            d_G_disc = d_G_disc.mean()
            self.manual_backward(d_G_disc,ones)






            grad_penalty = self.vlambda*self.compute_gradient_penalty(R, G_Z)
            self.manual_backward(grad_penalty)

            total_grad_penalty += grad_penalty

            # Loss used in the paper
            D_loss_disc = d_G_disc - d_R_disc + self.vlambda * grad_penalty

            # Updating parameters
            opt_disc.step()

        total_grad_penalty = total_grad_penalty / self.n_critic

        # Training generator
        self.disable_discriminator_training()
        self.enable_generator_training()

        opt_gen.zero_grad()

        Z = self.sample_z(real_images.size(0)).to(real_images.device)
        G_Z = self.generator(Z)
        d_G = self.discriminator(G_Z)
        d_G = d_G.mean()
        self.manual_backward(d_G,mones)
        opt_gen.step()

        # We can log the metrics
        self.log_dict({"G_loss": -d_G, "C_loss": D_loss_disc, "d_C_loss": d_R_disc, "d_G_loss": d_G_disc, "GradPen": total_grad_penalty},prog_bar=True)
    
        # Label for Precision and Recall
        super().training_step(batch[0], batch_idx, None, None)

    def configure_optimizers(self):
        opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)
        opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        return opt_gen, opt_disc



