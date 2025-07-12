import torch

import torch.nn as nn
from torch.distributions import MultivariateNormal

from gencellpainting.constants import MONITORED_LOSS

from .abc_model import UnsupervisedImageGenerator
from .net.CNN import EncoderWithPooling, Decoder

class VAE(UnsupervisedImageGenerator):
    """
    Variational Autoencoder (VAE) for unsupervised image generation.

    This model consists of an encoder that maps input images to a latent distribution
    and a decoder that reconstructs images from latent samples. The model is trained
    with a combination of reconstruction loss (MSE) and KL divergence to a standard normal.

    Args:
        latent_dim (int): Dimension of the latent space
        in_channels (int): Number of input image channels
        out_channels (int): Number of output image channels (Often the same)
        alpha (float, optional): Weight of the KL divergence term. Defaults to 0.1.
        network_capacity (int, optional): Base capacity of the network, number of intemediate channel is multiplied by this number. Defaults to 32.
        image_size (int, optional): Size of input/output images. Defaults to 128.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        epoch_monitoring_interval (int, optional): Interval for monitoring epochs. Defaults to 1.
        n_images_monitoring (int, optional): Number of images to monitor. Defaults to 6.
    """
    def __init__(self, latent_dim, in_channels, out_channels, alpha=0.1,\
                 network_capacity=32,image_size=128,decoder_nlayers=None, encoder_nlayers=None,
                 learning_rate=1e-3, epoch_monitoring_interval=1,n_images_monitoring=6):

        super(VAE, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        self.encoder = EncoderWithPooling(in_channels=in_channels, latent_dim=latent_dim,\
                                           network_capacity=network_capacity, image_size=image_size, nlayers=encoder_nlayers)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels,\
                                network_capacity=network_capacity, image_size=image_size, nlayers=decoder_nlayers)
        
        self.save_hyperparameters()

    
    def training_step(self, batch, batch_idx):
        X = batch
        dist = self.encoder(X)

        # We sample the object
        z = dist.rsample()
        X_hat = self.decoder(z)
        loss = nn.functional.mse_loss(X_hat, X, reduction='mean')

        # We add the Kullbach Lieber divergnce
        
        # Defining the reference distribution
        std_normal = MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = self.hparams.alpha * torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        # Total loss
        total_loss = loss + loss_kl

        # Log metrics
        self.log_dict({'MSE_train': loss, 'KL_train':loss_kl,"total_loss_train":total_loss}, prog_bar=True, logger=True, on_epoch=True)
        super().training_step(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        loss_kl = self.hparams.alpha * torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        # Total loss
        total_loss = loss + loss_kl

        # Log metrics
        self.log_dict({'MSE_val': loss, 'KL_val':loss_kl,MONITORED_LOSS:total_loss}, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]
    
    def generate_images(self, batch=None, n=None):
        X = batch
        dist = self.encoder(X)
        z = dist.rsample()
        X_hat = self.decoder(z)
        return X_hat