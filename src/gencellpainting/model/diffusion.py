from .conv_modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .VAE import Decoder
import lightning as L
from .abc_model import UnsupervisedImageGenerator, AbstractGAN

   

def cosine_scheduling(T, s=1e-6):
    f = torch.cos(((torch.linspace(0,1,T)+s)/(2+2*s))*math.pi)**2
    alphas = f/f[0]
    betas = torch.clamp(1-alphas[1:]/alphas[:-2], max=0.999)
    return alphas, betas


class DiffusionProcess(UnsupervisedImageGenerator):
    def __init__(self, in_channels, nsteps, network_capacity=32,image_size=128,\
                learning_rate = 1e-4,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(DiffusionProcess, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        
        self.nsteps = nsteps
        self.in_channels = in_channels
        self.alphas, self.betas = cosine_scheduling(self.nsteps)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.m_sqrt_alphas = torch.sqrt(1-self.alphas)
        self.image_size = image_size
        self.model = None # TODO add the model

    def gaussian_noise(self,images):
        epsilon = torch.randn(*images.size(),device=images.device)
        return epsilon

    def forward_process(self, images, epsilon, t):
        B,C,H,W = images.size()
        alphat = self.sqrt_alphas[t]
        malphat = self.m_sqrt_alphas[t]
        malphat = malphat.expand(-1,C,H,W)
        alphat = alphat.expand(-1,C,H,W)
        # Computing the noisy image
        noised_images = torch.sqrt(alphat) * images + (1 - alphat) * epsilon
        return noised_images

    def backward_process(self,batch):
        pass

    def training_step(self, batch, batch_idx):
        images = batch # B x C x H x W
        B,C,H,W = batch.size()

        # Sampling noise
        epsilon = self.gaussian_noise(images)

        # Sampling t
        t = torch.randint(low=1,high=self.nsteps)

        # forqward process
        noised_images = self.forward_process(images, epsilon,)

        # Predicting the values of espilon
        estimated_epsilon = self.model(noised_images, t)

        # Computing the loss
        loss = F.mse_loss(estimated_epsilon, epsilon)
        self.log("MSE",loss)
        super().training_step(batch, batch_idx)
        return loss
    
    def generate_images(self, batch=None, n=6):
        if batch is None:
            batch = torch.zeros(n,self.in_channels,self.image_size,self.image_size)
        x = self.gaussian_noise(batch)
        for t in range(self.nsteps,0,-1):
            




        return super().generate_images(batch, n)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)