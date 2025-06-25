from .conv_modules import Conv2dStack,Conv2dTransposeStack
import torch
import math
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import lightning as L
from .abc_model import UnsupervisedImageGenerator

# class Encoder(nn.Module):
#     def __init__(self, in_channels, latent_dim, network_capacity=32):
#         super(Encoder, self).__init__()

#         self.in_channels = in_channels
#         self.latent_dim = latent_dim
#         self.network_capacity = network_capacity

#         networks_channels = [network_capacity*2**i for i in range(3)]

#         self.model = nn.Sequential(
#             Conv2dStack(in_channels, out_channels=networks_channels[0], kernel_size=4, stride=2, padding=1), # (B, 32, 64, 64)
#             Conv2dStack(32, out_channels=64, kernel_size=4, stride=2, padding=1), # (B, 64, 32, 32)
#             Conv2dStack(64, out_channels=128, kernel_size=3, stride=2, padding=1), # (B, 128, 16, 16)
#             nn.Flatten()
#         )


#         self.softplus = nn.Softplus()
#         self.latent_layer = nn.Linear(128 * 16 * 16, latent_dim * 2)


#     def forward(self, x, eps = 1e-8):
#         x = self.model(x)
#         l = self.latent_layer(x)
#         mu, logvar = torch.chunk(l, 2, dim=-1)
#         scale = self.softplus(logvar) + eps
#         scale_tril = torch.diag_embed(scale)
#         dist = MultivariateNormal(loc = mu, scale_tril=scale_tril)
#         return dist
    

class EncoderWithPooling(nn.Module):
    def __init__(self, in_channels, latent_dim, network_capacity=32,):

        self.network_capacity = network_capacity

        networks_channels = [network_capacity*2**i for i in range(4)]

        super(EncoderWithPooling, self).__init__()
        self.model = nn.Sequential(
            Conv2dStack(in_channels, out_channels=networks_channels[0], kernel_size=4, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride = 2), # (B, 16, 64, 64)
            Conv2dStack(networks_channels[0], out_channels=networks_channels[1], \
                        kernel_size=4, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride = 2), # (B, 32, 32, 32)
            Conv2dStack(networks_channels[1], out_channels=networks_channels[2], \
                        kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride = 2), # (B, 64, 16, 16)
            Conv2dStack(networks_channels[2], out_channels=networks_channels[3], \
                        kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride = 2), # (B, 64, 8, 8)
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


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, network_capacity = 32, kernel_size=4,\
                 image_size=128, nlayers=None):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.network_capacity = network_capacity
        self.image_size=image_size
        self.kernel_size = kernel_size
        self.bias = True
        self.activation = "relu"


        # We count the number of layers
        if nlayers is None:
            nlayers = int(math.log2(self.image_size)-1)

        networks_channels = [network_capacity*2**i for i in range(1,nlayers)]
        self.networks_channels = networks_channels[::-1]

        layers_list = []
        # Initial reshaping layer
        layers_list.append(
            Conv2dTransposeStack(self.latent_dim, out_channels=self.networks_channels[0],\
                kernel_size=4, stride=1, padding=0, output_padding=0,activation_args={},
                      activation=self.activation, bias=self.bias),#(B, latent_dim, 4, 4)
            
        )

        for ilayer in range(0, nlayers-2):
            cin = self.networks_channels[ilayer]
            cout = self.networks_channels[ilayer+1]
            cstride = 2
            layers_list.append(
                Conv2dTransposeStack(cin, out_channels=cout, kernel_size=self.kernel_size, stride=cstride,\
                                padding=1, output_padding=0, activation_args={},\
                                    activation=self.activation, bias=self.bias) #(B, cout, 4*2**(ilayer+1), 4*2**(ilayer+1))
            )

        self.net = nn.Sequential(*layers_list)
        
        self.output_layers = nn.Sequential(
            nn.ConvTranspose2d(self.networks_channels[-1], out_channels=out_channels, kernel_size=4,\
                                    stride=2, padding=1, bias=self.bias), # Output: (B, out_channels, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, z):
        # x = self.fc(z)
        x = z.view(-1, self.latent_dim, 1, 1)
        x = self.net(x)
        x = self.output_layers(x)
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