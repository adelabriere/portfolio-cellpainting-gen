from conv_modules import Conv2dStack,Conv2dTransposeStack
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import lightning as L
from abc_model import UnsupervisedImageGenerator

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            Conv2dStack(in_channels, out_channels=32, kernel_size=4, stride=2, padding=1), # (B, 32, 64, 64)
            Conv2dStack(32, out_channels=64, kernel_size=4, stride=2, padding=1), # (B, 64, 32, 32)
            Conv2dStack(64, out_channels=128, kernel_size=3, stride=2, padding=1), # (B, 128, 16, 16)
            nn.Flatten()
        )


        self.softplus = nn.Softplus()
        self.latent_layer = nn.Linear(128 * 16 * 16, latent_dim * 2)


    def forward(self, x, eps = 1e-8):
        x = self.model(x)
        l = self.latent_layer(x)
        mu, logvar = torch.chunk(l, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = MultivariateNormal(loc = mu, scale_tril=scale_tril)
        return dist
    

class EncoderWithPooling(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(EncoderWithPooling, self).__init__()
        self.model = nn.Sequential(
            Conv2dStack(in_channels, out_channels=16, kernel_size=4, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2), # (B, 16, 64, 64)
            Conv2dStack(16, out_channels=32, kernel_size=4, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2), # (B, 32, 32, 32)
            Conv2dStack(32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2), # (B, 64, 16, 16)
            Conv2dStack(64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2), # (B, 64, 8, 8)
            nn.Flatten()
        )

        self.softplus = nn.Softplus()
        self.latent_layer = nn.Linear(64 * 8 * 8, latent_dim * 2)


    def forward(self, x, eps = 1e-8):
        x = self.model(x)
        l = self.latent_layer(x)
        mu, logvar = torch.chunk(l, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = MultivariateNormal(loc = mu, scale_tril=scale_tril)
        return dist


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.model = nn.Sequential(
            Conv2dTransposeStack(512, out_channels=256, kernel_size=2, stride=2, padding=0, output_padding=0, activation="leaky_relu"),#activation="relu", activation_args={"inplace": True}), # Output: (B, 256, 8, 8)
            Conv2dTransposeStack(256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu"),#, activation="relu", activation_args={"inplace": True}), # Output: (B, 128, 16, 16)
            Conv2dTransposeStack(128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu"),#, activation="relu", activation_args={"inplace": True}), # Output: (B, 64, 32, 32)
            Conv2dTransposeStack(64, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=0, activation="leaky_relu"),#, activation="relu", activation_args={"inplace": True}), # Output: (B, 32, 64, 64)
            nn.ConvTranspose2d(32, out_channels=out_channels, kernel_size=4, stride=2, padding=1), # Output: (B, out_channels, 128, 128)
            # We add convolution layer
            nn.Conv2d(out_channels, out_channels=out_channels, stride=1, kernel_size=1,padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.model(x)
        return x



# class UnsupervisedImageGenerator(L.LightningModule):
#     """
#     Parent class for image generation, mainly including tensorboard image logging.
#     """
#     def __init__(self, epoch_monitoring_interval=None, n_images=6):
#     def training_step(self, batch, batch_idx):
#     def configure_optimizers(self):
#     def monitor_training(self, batch):
#     def generate_images(self, batch=None, n=6)


class VAE(UnsupervisedImageGenerator):
    def __init__(self, latent_dim, in_channels, out_channels, alpha=0.1, epoch_monitoring_interval=1, n_images_monitoring=6):
        super(VAE, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        #self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.encoder = EncoderWithPooling(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels)
        self.latent_dim = latent_dim
        self.alpha = alpha

    
    def training_step(self, batch, batch_idx):
        X = batch
        dist = self.encoder(X)

        # We sample the object
        z = dist.rsample()
        X_hat = self.decoder(z)
        loss = nn.functional.mse_loss(X_hat, X, reduction='mean')

        # We add the Kullbach Lieber divergnce
        
        # Defining the reference distribution
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        # Total loss
        total_loss = loss + self.alpha * loss_kl

        # Log metrics
        self.log_dict({'train_mse': loss, 'train_kl':loss_kl,'total_loss':total_loss}, prog_bar=True, logger=True, on_epoch=True)
        
        super().training_step(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X = batch
        dist = self.encoder(X)
        z = dist.rsample()
        X_hat = self.decoder(z)
        loss = nn.functional.mse_loss(X_hat, X, reduction='mean')
        self.log_dict({'val_loss': loss}, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
    
    def generate_images(self, batch=None, n=None):
        X = batch
        dist = self.encoder(X)
        z = dist.rsample()
        X_hat = self.decoder(z)
        return X_hat