from .conv_modules import UpsampleConvStack, Conv2dTransposeStack, Conv2dStack
import torch.nn as nn
import torch.nn.functional as F
import torch
from .VAE import Decoder, Encoder
import lightning as L
from .abc_model import UnsupervisedImageGenerator



class Generator(nn.Module):
    def __init__(self,out_channels,hidden_dim):
        super(Generator, self).__init__()
        self.decoder = Decoder(latent_dim=hidden_dim, out_channels=out_channels)

    def forward(self, z):
        return self.decoder(z)
    

class GeneratorV2(nn.Module):
    def __init__(self, out_channels, latent_dim):
        super(GeneratorV2, self).__init__()
        # We remove the uflly connected layer and try to use a convolutionnal layer of smaller size

        # Input after reshaping (B, latent_dim (noise), 1 , 1)
        self.model = nn.Sequential(
            # Input after reshaping (B, latent_dim (noise), 1 , 1)
            Conv2dTransposeStack(latent_dim, out_channels=512, kernel_size=4, stride=1, padding=0, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 512, 4, 4)
            Conv2dTransposeStack(512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 256, 8, 8)
            Conv2dTransposeStack(256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 128, 16, 16)
            Conv2dTransposeStack(128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 64, 32, 32)
            Conv2dTransposeStack(64, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 32, 64, 64)
            nn.ConvTranspose2d(32, out_channels=out_channels, kernel_size=4, stride=2, padding=1), # Output: (B, out_channels, 128, 128)
            nn.Sigmoid()
        )

        self.latent_dim = latent_dim
    
    def forward(self, z):
        # We reshape the input z
        x = z.view(z.size(0), self.latent_dim , 1, 1)
        x = self.model(x)
        return x
    

class GeneratorV2ShallowFC(nn.Module):
    """For testing purpose model version with shallower version"""
    def __init__(self, out_channels, latent_dim):
        super(GeneratorV2ShallowFC, self).__init__()
        # We remove the uflly connected layer and try to use a convolutionnal layer of smaller size
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        # Input after reshaping (B, latent_dim (noise), 1 , 1)
        self.model = nn.Sequential(
            Conv2dTransposeStack(256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 128, 16, 16)
            Conv2dTransposeStack(128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 64, 32, 32)
            Conv2dTransposeStack(64, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # Output: (B, 32, 64, 64)
            nn.ConvTranspose2d(32, out_channels=out_channels, kernel_size=4, stride=2, padding=1), # Output: (B, out_channels, 128, 128)
            nn.Sigmoid()
        )

        self.latent_dim = latent_dim
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 8, 8)
        x = self.model(x)
        return x
    
class GeneratorUpsampling(nn.Module):
    def __init__(self,  out_channels, latent_dim):
        super(GeneratorUpsampling, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Buld the first layer 
            
            # Input after reshaping (B, latent_dim (noise), 1 , 1)
            Conv2dTransposeStack(latent_dim, out_channels=512, kernel_size=4, stride=1, padding=0, output_padding=0, activation="relu", activation_args={"inplace": True}),
            # (B, 512, 4, 4)
            UpsampleConvStack(512, out_channels=256, kernel_size=4, output_dim=(8,8), padding="same", mode="nearest"),
            # (B, 256, 8, 8)
            UpsampleConvStack(256, out_channels=128, kernel_size=4, output_dim=(16,16), padding="same", mode="nearest"),
            # (B, 128, 16, 16)
            UpsampleConvStack(128, out_channels=64, kernel_size=4, output_dim=(32,32), padding="same", mode="nearest"),
            # (B, 64, 32, 32)
            UpsampleConvStack(64, out_channels=32, kernel_size=4, output_dim=(64,64), padding="same", mode="nearest"),
            # (B, 32, 64, 64)
            nn.ConvTranspose2d(32, out_channels=out_channels, kernel_size=4, stride=2, padding=1), # Output: (B, out_channels, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        # Reshaping input
        x = x.view(x.size(0), self.latent_dim, 1, 1)
        return self.model(x)
   

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # Input size (B, in_channels, 128, 128)
        self.model = nn.Sequential(
            Conv2dStack(in_channels, out_channels=32, kernel_size=4, stride=2, padding=1), # (B, 32, 64, 64)
            Conv2dStack(32, out_channels=64, kernel_size=4, stride=2, padding=1), # (B, 64, 32, 32)
            Conv2dStack(64, out_channels=128, kernel_size=3, stride=2, padding=1), # (B, 128, 16, 16)
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)
    

# class VAE(UnsupervisedImageGenerator):
#     def __init__(self, latent_dim, in_channels, out_channels, alpha=0.1, epoch_monitoring_interval=1, n_images_monitoring=6):
#         super(VAE, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
#         #self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
#         self.encoder = EncoderWithPooling(in_channels=in_channels, latent_dim=latent_dim)
#         self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels)
#         self.latent_dim = latent_dim
#         self.alpha = alpha

    
#     def training_step(self, batch, batch_idx):
#         X = batch
#         dist = self.encoder(X)

#         # We sample the object
#         z = dist.rsample()
#         X_hat = self.decoder(z)
#         loss = nn.functional.mse_loss(X_hat, X, reduction='mean')

#         # We add the Kullbach Lieber divergnce
        
#         # Defining the reference distribution
#         std_normal = torch.distributions.MultivariateNormal(
#             torch.zeros_like(z, device=z.device),
#             scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
#         )
#         loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

#         # Total loss
#         total_loss = loss + self.alpha * loss_kl

#         # Log metrics
#         self.log_dict({'train_mse': loss, 'train_kl':loss_kl,'total_loss':total_loss}, prog_bar=True, logger=True, on_epoch=True)
        
#         super().training_step(batch, batch_idx)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         X = batch
#         dist = self.encoder(X)
#         z = dist.rsample()
#         X_hat = self.decoder(z)
#         loss = nn.functional.mse_loss(X_hat, X, reduction='mean')
#         self.log_dict({'val_loss': loss}, prog_bar=True, logger=True, on_epoch=True)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
#         return optimizer
    
#     def generate_images(self, batch=None):
#         X = batch
#         dist = self.encoder(X)
#         z = dist.rsample()
#         X_hat = self.decoder(z)
#         return X_hat

class GAN(UnsupervisedImageGenerator):
    def __init__(self, generator, discriminator, in_channels, out_channels, noise_dim, epoch_monitoring_interval=1, n_images_monitoring=6):
        super(GAN, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)

        # Using V2 for testing purposes
        if generator is None:
            generator = GeneratorV2ShallowFC(out_channels,noise_dim)
        self.generator = generator
        # self.generator = GeneratorUpsampling(out_channels,noise_dim)
        if discriminator is None:
            discriminator = Discriminator(in_channels)
        self.discriminator = discriminator

        self.noise_dim = noise_dim

        # Classification loss
        self.classification_loss = F.binary_cross_entropy

        self.automatic_optimization = False

    def sample_z(self, batch_size):
        Z = torch.randn(batch_size, self.noise_dim)
        return Z
    
    def generate_images(self, batch, n):
        Z = self.sample_z(n).to(batch.device)
        return self.generator(Z)
    
    def training_step(self, batch, batch_idx):
        real_images = batch

        opt_gen,opt_disc = self.optimizers()

        opt_disc.zero_grad()

        # Writng the labels
        real_labels = torch.ones(real_images.size(0), 1, device=real_images.device)
        fake_labels = torch.zeros(real_images.size(0), 1, device=real_images.device)

        # Generating the noise
        Z = self.sample_z(real_images.size(0)).to(real_images.device)

        # Generating the fake images
        fake_images = self.generator(Z).to(real_images.device)

        # Training the discriminator
        real_loss = F.binary_cross_entropy(self.discriminator(real_images), real_labels)
        fake_loss = F.binary_cross_entropy(self.discriminator(fake_images.detach()), fake_labels)
        total_loss_disc = real_loss + fake_loss

        self.manual_backward(total_loss_disc)
        opt_disc.step()

        opt_gen.zero_grad()

        # Training the generator
        pred_fakes = self.discriminator(fake_images)
        gen_loss = F.binary_cross_entropy(pred_fakes, real_labels)


        self.manual_backward(gen_loss)
        opt_gen.step()

        # We log the different losses
        self.log_dict({
            "loss_d":total_loss_disc,
            "loss_g":gen_loss,
            "loss_d_real":real_loss,
            "loss_d_fake":fake_loss
        },prog_bar=True)

        super().training_step(batch, batch_idx)
    
    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return opt_gen, opt_disc



