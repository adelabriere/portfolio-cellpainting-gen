from conv_modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from GAN import GeneratorV2, GeneratorV2ShallowFC
import lightning as L
from .abc_model import UnsupervisedImageGenerator

   

class WGANCritic(nn.Module):
    def __init__(self, in_channels):
        super(WGANCritic, self).__init__()
        # Input size (B, in_channels, 128, 128)
        self.model = nn.Sequential(
            Conv2dStack(in_channels, out_channels=32, kernel_size=4, stride=2, padding=1), # (B, 32, 64, 64)
            Conv2dStack(32, out_channels=64, kernel_size=4, stride=2, padding=1), # (B, 64, 32, 32)
            Conv2dStack(64, out_channels=128, kernel_size=3, stride=2, padding=1), # (B, 128, 16, 16)
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1)
        )

    def forward(self, x):
        return self.model(x)
    

class WGAN(UnsupervisedImageGenerator):
    """Own implemntation of https://arxiv.org/pdf/1701.07875 the parameters are taken from the paper"""
    def __init__(self, in_channels, out_channels, noise_dim, n_critic = 5, clip_value = 0.01,\
                 learning_rate = 1e-5,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(WGAN, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        # self.generator = GeneratorV2(out_channels,latent_dim=noise_dim)
        self.generator  = GeneratorV2ShallowFC(out_channels,noise_dim)
        self.discriminator = WGANCritic(in_channels)

        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.learning_rate = learning_rate

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

    
    
    def training_step(self, batch, batch_idx):
        real_images, critic_images = batch

        opt_gen,opt_disc = self.optimizers()

        critic_images = critic_images.chunk(self.n_critic, dim=0)

        # Training discriminator
        self.disable_generator_training()
        self.enable_discriminator_training()

        for ic in range(self.n_critic):
            opt_disc.zero_grad()
            # Sample the real data
            R = critic_images[ic]

            # Sample the prior data
            Z = self.sample_z(R.size(0)).to(R.device)
            
            d_R_disc = self.discriminator(R)

            # PRevent gradient ocmputation on the first model
            G_Z = self.generator(Z)
            d_G_disc = self.discriminator(G_Z)

            d_R_disc = d_R_disc.mean()
            d_G_disc = d_G_disc.mean()

            # Loss used in the paper
            D_loss_disc = d_R_disc - d_G_disc

            # Computing the gradient
            self.manual_backward(D_loss_disc)
            # Updating parameters
            opt_disc.step()

            # We can now clip the parameters
            self.clamp_parameters(self.discriminator, self.clip_value)

        # Training generator
        self.disable_discriminator_training()
        self.enable_generator_training()

        opt_gen.zero_grad()
        Z = self.sample_z(real_images.size(0)).to(real_images.device)
        G_Z = self.generator(Z)
        d_G = self.discriminator(G_Z)
        d_G = d_G.mean()
        G_loss = d_G
        self.manual_backward(G_loss)
        opt_gen.step()

        # We can log the metrics
        self.log_dict({"G_loss": G_loss, "D_loss": D_loss_disc, "d_R": d_R_disc, "d_G": d_G_disc},prog_bar=True)

        super().training_step(batch[0], batch_idx)

    def configure_optimizers(self):
        opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)
        opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        return opt_gen, opt_disc



