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
    def __init__(self, in_channels, out_channels, nsteps, network_capacity=32,\
                learning_rate = 1e-4,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(DiffusionProcess, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        
        self.nsteps = nsteps
        self.alphas, self.betas = cosine_scheduling(self.nsteps)

    def gaussian_noise(self,images):
        

    def forward_process(self,images):

        

    def backward_process(self,batch):
        pass

    def training_step(self, batch, batch_idx):
        images = batch # B x C x H x W
        B,C,H,W = batch.size()

        # Forward step ie adding noise
        self.forward_process(images)

        # Sampling t
        t = torch.randint(low=1,high=self.nsteps)

        mt = (t[:,None,None,None]).expand(-1,C,H,W)
        
        alphat = self.alphas[t]

        loss = 


        return super().training_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return super().configure_optimizers()