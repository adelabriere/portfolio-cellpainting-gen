from .net.conv_modules import *
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
    betas = torch.clamp(1-alphas[1:]/alphas[:-1], max=0.999)
    return alphas, betas

class TransformerPositionalEmbedding(nn.Module):
    """
    Directly taken from 'https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py'
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1000, device="cuda"):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)
        self.pe_matrix = self.pe_matrix.to(device)

    def forward(self, timestep):
        # [bs, d_model]
        return self.pe_matrix[timestep].to(timestep.device)

class DiffusionProcess(UnsupervisedImageGenerator):
    def __init__(self, in_channels, nsteps, model, time_dim=128,image_size=128,\
                learning_rate = 1e-4,epoch_monitoring_interval=1, n_images_monitoring=6, device="cuda"):
        super(DiffusionProcess, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        
        self.nsteps = nsteps
        self.in_channels = in_channels
        self.alphas, self.betas = cosine_scheduling(self.nsteps)
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        self.sqrt_alphas = torch.sqrt(self.alphas).to(device)
        self.m_sqrt_alphas = torch.sqrt(1-self.alphas).to(device)
        self.image_size = image_size
        self.time_dim = time_dim
        self.model = model
        self.time_model = TransformerPositionalEmbedding(self.time_dim, device=device)
        self.time_model = self.time_model.to(device)

    def gaussian_noise(self,images):
        epsilon = torch.randn(*images.size(),device=images.device)
        return epsilon

    def forward_process(self, images, epsilon, t):
        B,C,H,W = images.size()
        alphat = self.sqrt_alphas[t]
        malphat = self.m_sqrt_alphas[t]
        alphat = alphat[:,None,None,None]
        malphat = malphat[:,None,None,None]
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
        timesteps = torch.arange(self.nsteps, 0, -1)
        timesteps = timesteps[:,None]
        timesteps = timesteps.to(self.device)
        emb_timesteps = self.time_model(timesteps) 
        for t,emb_t in zip(range(self.nsteps,0,-1),emb_timesteps):
            z = self.gaussian_noise(batch) if t>1 else torch.zeros_like(batch)
            calpha = self.alphas[t-1]
            malpha = 1-calpha
            # print("calpha {} malpha {} embt_t {} x {}".format(calpha.device,malpha.device,emb_t.device,x.device))
            x = (1/calpha.sqrt())*(x - (malpha/calpha.sqrt())*self.model(x,emb_t)) + z
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)