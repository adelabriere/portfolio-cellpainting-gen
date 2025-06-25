from .conv_modules import UpsampleConvStack, Conv2dTransposeStack, Conv2dStack
import torch.nn as nn
import torch.nn.functional as F
import torch
from .VAE import Decoder, Encoder
import lightning as L
from .abc_model import AbstractGAN

    

# class GeneratorV2ShallowFC(nn.Module):
#     """For testing purpose model version with shallower version"""
#     def __init__(self, out_channels, latent_dim):
#         super(GeneratorV2ShallowFC, self).__init__()
#         # We remove the uflly connected layer and try to use a convolutionnal layer of smaller size
#         self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

#         # Input after reshaping (B, 256, 8 , 8)
#         self.model = nn.Sequential(
#             Conv2dTransposeStack(256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu", activation_args={"inplace": True}),
#             # Output: (B, 128, 16, 16)
#             Conv2dTransposeStack(128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu", activation_args={"inplace": True}),
#             # Output: (B, 64, 32, 32)
#             Conv2dTransposeStack(64, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu", activation_args={"inplace": True}),
#             # Output: (B, 32, 64, 64)
#             nn.ConvTranspose2d(32, out_channels=out_channels, kernel_size=2, stride=2, padding=0, output_padding=0), # Output: (B, out_channels, 128, 128)
#             nn.Conv2d(out_channels, out_channels=out_channels, stride=1, kernel_size=1,padding="same"),
#             nn.Sigmoid()
#         )

#         self.latent_dim = latent_dim
    
#     def forward(self, z):
#         x = self.fc(z)
#         x = x.view(x.size(0), 256, 8, 8)
#         x = self.model(x)
#         return x
    

class GeneratorVAEDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, network_capacity = 32):
        super(GeneratorVAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.network_capacity = network_capacity


        networks_channels = [network_capacity*2**i for i in range(5)]
        self.networks_channels = networks_channels[::-1]
        # self.fc = nn.Linear(latent_dim, self.networks_channels[0] * 4 * 4)
        self.model = nn.Sequential(
            Conv2dTransposeStack(self.latent_dim, out_channels=self.networks_channels[0],\
                                 kernel_size=4, stride=1, padding=0, output_padding=0, activation="leaky_relu", bias=False),
            Conv2dTransposeStack(self.networks_channels[0], out_channels=self.networks_channels[1],\
                                 kernel_size=2, stride=2, padding=0, output_padding=0, activation="leaky_relu", bias=False),#activation="relu", activation_args={"inplace": True}), # Output: (B, 256, 8, 8)
            Conv2dTransposeStack(self.networks_channels[1], out_channels=self.networks_channels[2],\
                                 kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu", bias=False),#, activation="relu", activation_args={"inplace": True}), # Output: (B, 128, 16, 16)
            Conv2dTransposeStack(self.networks_channels[2], out_channels=self.networks_channels[3],\
                                 kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu", bias=False),#, activation="relu", activation_args={"inplace": True}), # Output: (B, 64, 32, 32)
            Conv2dTransposeStack(self.networks_channels[3], out_channels=self.networks_channels[4],\
                                 kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu", bias=False),#, activation="relu", activation_args={"inplace": True}), # Output: (B, 32, 64, 64)
            nn.ConvTranspose2d(self.networks_channels[4], out_channels=out_channels, kernel_size=4,\
                                stride=2, padding=1), # Output: (B, out_channels, 128, 128)
            # We add convolution layer
            nn.Conv2d(out_channels, out_channels=out_channels, stride=1, kernel_size=1,padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # x = self.fc(z)
        x = z.view(-1, self.latent_dim, 1, 1)
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
            Conv2dStack(in_channels, out_channels=32, kernel_size=4, stride=1, padding="same"), # (B, 32, 128, 128)
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, 64, 64)
            Conv2dStack(32, out_channels=64, kernel_size=4, stride=1, padding="same"), # (B, 32, 64, 64)
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, 32, 32)
            Conv2dStack(64, out_channels=128, kernel_size=3, stride=1, padding="same"), # (B, 128, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 128, 16, 16)
            Conv2dStack(128, out_channels=256, kernel_size=2, stride=1, padding="same"), # (B, 256, 16, 16)
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 256, 8, 8)
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)
    

class EncoderWithPooling(nn.Module):
    def __init__(self, in_channels, latent_dim, network_capacity=32,):

        self.network_capacity = network_capacity

        networks_channels = [network_capacity*2**i for i in range(4)]

        super(EncoderWithPooling, self).__init__()
        self.model = nn.Sequential(
            Conv2dStack(in_channels, out_channels=networks_channels[0], kernel_size=4, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 16, 64, 64)
            Conv2dStack(networks_channels[0], out_channels=networks_channels[1], \
                        kernel_size=4, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, 32, 32)
            Conv2dStack(networks_channels[1], out_channels=networks_channels[2], \
                        kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, 16, 16)
            Conv2dStack(networks_channels[2], out_channels=networks_channels[3], \
                        kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, 8, 8)
            nn.Flatten()
        )

        self.softplus = nn.Softplus()
        self.latent_layer = nn.Linear(networks_channels[3] * 8 * 8, latent_dim * 2)


    def forward(self, x, eps = 1e-8):
        x = self.model(x)
        l = self.latent_layer(x)
        mu, logvar = torch.chunk(l, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = MultivariateNormal(loc = mu, scale_tril=scale_tril)
        return dist


class GAN(AbstractGAN):
    def __init__(self, generator, discriminator, in_channels, out_channels, noise_dim, epoch_monitoring_interval=1, n_images_monitoring=6, learning_rate=1e-4):
        super(GAN, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)

        # Using V2 for testing purposes
        if generator is None:
            generator = GeneratorVAEDecoder(noise_dim,out_channels)
        self.generator = generator
        # self.generator = GeneratorUpsampling(out_channels,noise_dim)
        if discriminator is None:
            discriminator = Discriminator(in_channels)
        self.discriminator = discriminator
        self.learning_rate = learning_rate
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


        # Writng the labels
        real_labels = torch.ones(real_images.size(0), 1, device=real_images.device)
        fake_labels = torch.zeros(real_images.size(0), 1, device=real_images.device)

        # Generating the noise
        Z = self.sample_z(real_images.size(0)).to(real_images.device)

        # Generating the fake images
        fake_images = self.generator(Z).to(real_images.device)

        # print(" fake_images {} real_images {}".format(fake_images.shape,real_images.shape))

        # Training the discriminator
        real_loss = F.binary_cross_entropy(self.discriminator(real_images), real_labels)
        fake_loss = F.binary_cross_entropy(self.discriminator(fake_images.detach()), fake_labels)
        total_loss_disc = real_loss + fake_loss

        opt_disc.zero_grad()
        self.manual_backward(total_loss_disc)
        opt_disc.step()



        # Training the generator
        pred_fakes = self.discriminator(fake_images)
        gen_loss = F.binary_cross_entropy(pred_fakes, real_labels)

        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()

        # We log the different losses
        self.log_dict({
            "loss_d":total_loss_disc,
            "loss_g":gen_loss,
            "loss_d_real":real_loss,
            "loss_d_fake":fake_loss
        },prog_bar=True)

        with torch.no_grad():
            pred_real = self.discriminator(real_images)
        
        pred_full = torch.cat([pred_fakes,pred_real],dim=0)
        targets_full = torch.cat([fake_labels,real_labels])
        super().training_step(batch, batch_idx, pred_full, targets_full)
    
    def configure_optimizers(self):
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        return opt_gen, opt_disc




