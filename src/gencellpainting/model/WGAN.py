from .conv_modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from .GAN import GeneratorVAEDecoder
import lightning as L
from .abc_model import UnsupervisedImageGenerator, AbstractGAN

   

class WGANCritic(nn.Module):
    def __init__(self, in_channels, network_capacity = 32):
        super(WGANCritic, self).__init__()
        self.in_channels = in_channels
        self.network_capacity = network_capacity 
        networks_channels = [network_capacity*2**i for i in range(5)]
        # Input size (B, in_channels, 128, 128)
        self.model = nn.Sequential(
            # we dont use batch norm
            nn.Conv2d(in_channels, out_channels=networks_channels[0], kernel_size=4, stride=2, padding=1), # (B, _, 64, 64)
            nn.InstanceNorm2d(networks_channels[0], affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(networks_channels[0], out_channels=networks_channels[1], kernel_size=4, stride=2, padding=1), # (B, _, 32, 32)
            nn.InstanceNorm2d(networks_channels[1], affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(networks_channels[1], out_channels=networks_channels[2], kernel_size=4, stride=2, padding=1), # (B, _, 16, 16)
            nn.InstanceNorm2d(networks_channels[2], affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(networks_channels[2], out_channels=networks_channels[3], kernel_size=4, stride=2, padding=1), # (B, _, 8, 8)
            nn.InstanceNorm2d(networks_channels[3], affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(networks_channels[3], out_channels=networks_channels[4], kernel_size=4, stride=2, padding=1), # (B, _, 4, 4)
            nn.InstanceNorm2d(networks_channels[4], affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=networks_channels[-1], out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)

class WGAN(AbstractGAN):
    """Own implemntation of https://arxiv.org/pdf/1701.07875 the parameters are taken from the paper"""
    def __init__(self, in_channels, out_channels, noise_dim, n_critic = 5, network_capacity=32,\
                 clip_value = 0.01,learning_rate = 1e-4,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(WGAN, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        # self.generator = GeneratorV2(out_channels,latent_dim=noise_dim)
        self.generator  = GeneratorVAEDecoder(noise_dim, out_channels, network_capacity=network_capacity)
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
        Z = torch.randn(batch_size, self.noise_dim)
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
        # self.disable_generator_training()
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
        # self.enable_generator_training()

        opt_gen.zero_grad()
        Z = self.sample_z(real_images.size(0)).to(real_images.device)
        G_Z = self.generator(Z)
        d_G = self.discriminator(G_Z)
        d_G = d_G.mean()
        G_loss = -d_G
        self.manual_backward(G_loss)
        opt_gen.step()

        # We can log the metrics
        self.log_dict({"G_loss": G_loss, "D_loss": D_loss_disc, "d_R": d_R_disc, "d_G": d_G_disc},prog_bar=True)
    
        # Label for Precision and Recall
        super().training_step(batch[0], batch_idx, None, None)

    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.0,0.9))
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.0,0.9))
        return opt_gen, opt_disc




class WGAN_GP(AbstractGAN):
    """Own implemntation of https://arxiv.org/pdf/1701.07875 the parameters are taken from the paper"""
    def __init__(self, in_channels, out_channels, noise_dim, n_critic = 5, vlambda=10,network_capacity=16,\
                 learning_rate = 1e-5,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(WGAN_GP, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        # self.generator = GeneratorV2(out_channels,latent_dim=noise_dim)
        self.generator  = GeneratorVAEDecoder(noise_dim, out_channels, network_capacity=network_capacity)
        self.discriminator = WGANCritic(in_channels)

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

        for ic in range(self.n_critic):
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

            grad_penalty = self.compute_gradient_penalty(R, G_Z)

            total_grad_penalty += grad_penalty

            # Loss used in the paper
            D_loss_disc = d_G_disc - d_R_disc + self.vlambda * grad_penalty

            opt_disc.zero_grad()
            # Computing the gradient
            self.manual_backward(D_loss_disc)
            # Updating parameters
            opt_disc.step()

        total_grad_penalty = total_grad_penalty / self.n_critic

        # Training generator
        self.disable_discriminator_training()
        self.enable_generator_training()

        Z = self.sample_z(real_images.size(0)).to(real_images.device)
        G_Z = self.generator(Z)
        d_G = self.discriminator(G_Z)
        d_G = -d_G.mean()
        G_loss = d_G
        opt_gen.zero_grad()
        self.manual_backward(G_loss)
        opt_gen.step()

        # We can log the metrics
        self.log_dict({"G_loss": G_loss, "C_loss": D_loss_disc, "d_C_loss": d_R_disc, "d_G_loss": d_G_disc, "GradPen": total_grad_penalty},prog_bar=True)
    
        # Label for Precision and Recall
        super().training_step(batch[0], batch_idx, None, None)

    def configure_optimizers(self):
        opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)
        opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        return opt_gen, opt_disc



