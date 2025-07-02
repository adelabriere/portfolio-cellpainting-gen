from .net.conv_modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .VAE import Decoder
import lightning as L
from .abc_model import UnsupervisedImageGenerator, AbstractGAN

   

def cosine_beta_scheduling(nsteps, s=8e-3):
    Ts = nsteps+1
    x = torch.linspace(0,1,Ts)
    f = torch.cos(((x+s)/(1+2)) * torch.pi * 0.5 ) ** 2
    alphas = f/f[0]
    betas = torch.clamp(1-alphas[1:]/alphas[:-1],min=0.0001, max=0.999)
    return betas

class TransformerPositionalEmbedding(nn.Module):
    """
    Directly taken from 'https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py'
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000, device="cuda"):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        pe_matrix = torch.zeros(max_timesteps, dimension)
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(max_timesteps)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

        # Register the parameters as buffer
        self.register_buffer("pe_matrix",pe_matrix)

    def forward(self, timestep):
        # [bs, d_model]
        return self.pe_matrix[timestep]

class DiffusionProcess(UnsupervisedImageGenerator):
    def __init__(self, in_channels, nsteps, model, time_dim=128,image_size=128,\
                learning_rate = 1e-4,epoch_monitoring_interval=1, n_images_monitoring=6, device="cuda"):
        super(DiffusionProcess, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        
        self.nsteps = nsteps
        self.in_channels = in_channels
        cbetas = cosine_beta_scheduling(self.nsteps)
        calphas = 1 - cbetas
        csum_alphas = torch.cumprod(calphas, axis=0)
        csum_sqrt_alphas = torch.sqrt(csum_alphas)

        # Registering the buffers
        self.register_buffer("alphas",calphas)
        self.register_buffer("betas",cbetas)
        self.register_buffer("sqrt_alphas",torch.sqrt(calphas))
        self.register_buffer("inv_sqrt_alphas",1/torch.sqrt(calphas))
        self.register_buffer("cum_alphas",torch.sqrt(csum_alphas))
        self.register_buffer("sqrt_cum_alphas",torch.sqrt(csum_sqrt_alphas))
        self.register_buffer("m_sqrt_cum_alphas",torch.sqrt(1-csum_sqrt_alphas))
        self.image_size = image_size
        self.time_dim = time_dim
        self.model = model
        self.time_model = TransformerPositionalEmbedding(self.time_dim, max_timesteps=nsteps, device=device)

    def gaussian_noise(self,images):
        epsilon = torch.randn(*images.size(),device=images.device)
        return epsilon

    def forward_process(self, images, epsilon, t):
        B,C,H,W = images.size()
        alphat = self.sqrt_cum_alphas[t]
        malphat = self.m_sqrt_cum_alphas[t]
        alphat = alphat[:,None,None,None]
        malphat = malphat[:,None,None,None]
        malphat = malphat.expand(-1,C,H,W)
        alphat = alphat.expand(-1,C,H,W)
        # Computing the noisy image
        images = torch.sqrt(alphat) * images + malphat * epsilon
        return images

    def backward_process(self,batch):
        pass

    def training_step(self, batch, batch_idx):
        images = batch # B x C x H x W
        B,C,H,W = batch.size()

        # Sampling noise
        epsilon = self.gaussian_noise(images)

        # Sampling t
        t = torch.randint(low=1,high=self.nsteps, size = (B,)).to(batch.device)
        emb_t = self.time_model(t)

        # forqward process
        noised_images = self.forward_process(images, epsilon, t)

        # Predicting the values of espilon
        estimated_epsilon = self.model(noised_images, emb_t)

        # Computing the loss
        loss = F.mse_loss(estimated_epsilon, epsilon)
        self.log("MSE",loss)
        super().training_step(batch, batch_idx)
        return loss
    
    def generate_images(self, batch=None, n=6):

        if batch is None:
            batch = torch.zeros(n,self.in_channels,self.image_size,self.image_size)
            batch = batch.to(self.device)
        x = self.gaussian_noise(batch)
        timesteps = torch.arange(self.nsteps, 0, -1) - 1
        timesteps = timesteps[:,None]
        timesteps = timesteps.to(self.device)
        emb_timesteps = self.time_model(timesteps) 
        for t,emb_t in zip(range(self.nsteps,0,-1),emb_timesteps):
            z = self.gaussian_noise(batch) if t>1 else torch.zeros_like(batch)
            cinv_sqrt_alpha = self.inv_sqrt_alphas[t-1]
            csqrt_cum_alpha = self.m_sqrt_cum_alphas[t-1]
            malpha = 1 - self.alphas[t-1]
            csigma = self.betas[t-1]
            x = cinv_sqrt_alpha *(x - (malpha/csqrt_cum_alpha)*self.model(x,emb_t)) + z * csigma
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)