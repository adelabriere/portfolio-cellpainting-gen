from .net.conv_modules import Conv2dStack,Conv2dTransposeStack
from .net.CNN import EncoderWithPooling, Decoder
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from .abc_model import UnsupervisedImageGenerator

class VAE(UnsupervisedImageGenerator):
    def __init__(self, latent_dim, in_channels, out_channels, alpha=0.1,\
                 network_capacity=32,image_size=128,learning_rate=1e-3, \
                epoch_monitoring_interval=1,n_images_monitoring=6):
        super(VAE, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        #self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.encoder = EncoderWithPooling(in_channels=in_channels, latent_dim=latent_dim,\
                                           network_capacity=network_capacity)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels,\
                                network_capacity=network_capacity, image_size=image_size)
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.network_capacity = network_capacity
    
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
        # print("\nX {} X_shape {} z {}".format(X.shape,X_hat.shape,z.shape))
        loss = nn.functional.mse_loss(X_hat, X, reduction='mean')
        self.log_dict({'val_loss': loss}, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]
        return optimizer
    
    def generate_images(self, batch=None, n=None):
        X = batch
        dist = self.encoder(X)
        z = dist.rsample()
        X_hat = self.decoder(z)
        return X_hat