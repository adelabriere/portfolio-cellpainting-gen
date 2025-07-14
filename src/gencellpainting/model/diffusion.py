import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from .abc_model import UnsupervisedImageGenerator
from .net.conv_modules import *
from gencellpainting.constants import MONITORED_LOSS

def cosine_beta_scheduling(nsteps, s=8e-3):
    Ts = nsteps+1
    x = torch.linspace(0,1,Ts)
    f = torch.cos(((x+s)/(1+s)) * torch.pi * 0.5 ) ** 2
    alphas = f/f[0]
    betas = torch.clamp(1-alphas[1:]/alphas[:-1],min=0.0001, max=0.9999)
    return betas

# Not used
def linear_beta_scheduling(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class TransformerPositionalEmbedding(nn.Module):
    """
    Directly taken from 'https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py'
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=10000, device="cuda"):
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
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, timesteps=200):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        vtime = torch.arange(timesteps)
        embeddings = vtime[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Register the parameters as buffer
        self.register_buffer("embeddings",embeddings)

    def forward(self, time):
        return self.embeddings[time,:]


class DiffusionProcess(UnsupervisedImageGenerator):
    def __init__(self, in_channels, nsteps, model, time_dim=128,alpha=0.1,image_size=128,include_time_emb=True,\
                learning_rate = 5e-4,epoch_monitoring_interval=1, n_images_monitoring=6):
        super(DiffusionProcess, self).__init__(epoch_monitoring_interval=epoch_monitoring_interval, n_images_monitoring=n_images_monitoring, add_original=True)
        
        self.nsteps = nsteps
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.include_time_emb = include_time_emb
        self.alpha = alpha
        cbetas = linear_beta_scheduling(self.nsteps)
        #cbetas = cosine_beta_scheduling(self.nsteps)
        calphas = 1 - cbetas
        cprod_alphas = torch.cumprod(calphas, axis=0)
        cprod_sqrt_alphas = torch.sqrt(cprod_alphas)

        self.loss = nn.SmoothL1Loss()


        # Registering the buffers
        self.register_buffer("alphas",calphas)
        self.register_buffer("betas",cbetas)
        self.register_buffer("sqrt_alphas",torch.sqrt(calphas))
        self.register_buffer("inv_sqrt_alphas",torch.sqrt(1. / calphas))
        self.register_buffer("cum_alphas",cprod_alphas)
        self.register_buffer("sqrt_cum_alphas",cprod_sqrt_alphas)
        self.register_buffer("m_sqrt_cum_alphas",torch.sqrt(1. - cprod_alphas))

        # POosterior variance
        csum_alphas_prev = F.pad(cprod_alphas[:-1], (1, 0), value=1.0)
        csigma = cbetas * (1. - csum_alphas_prev) / (1. - cprod_alphas)
        self.register_buffer("sigma",torch.sqrt(csigma))

        self.image_size = image_size
        self.time_dim = time_dim
        self.model = model
        if self.include_time_emb:
            self.time_model = nn.Sequential(
                SinusoidalPositionEmbeddings(32, timesteps=nsteps),
                nn.Linear(32, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        
        self.save_hyperparameters()

    def gaussian_noise(self,images):
        epsilon = torch.randn_like(images,device=images.device)
        return epsilon
    
    def offset_noise(self, images):
        epsilon =self.gaussian_noise(images)
        B,C,_,_ = images.shape
        offset = torch.randn(B, C, 1, 1, device=images.device)
        return epsilon + self.alpha*offset

    def q_sample(self, images, epsilon, t):
        B,C,H,W = images.size()
        sqrt_alphas_cumprod = self.sqrt_cum_alphas[t]
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[:,None,None,None]
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.expand(-1,C,H,W)

        m_sqrt_alphas_cumprod = self.m_sqrt_cum_alphas[t]
        m_sqrt_alphas_cumprod = m_sqrt_alphas_cumprod[:,None,None,None]
        m_sqrt_alphas_cumprod = m_sqrt_alphas_cumprod.expand(-1,C,H,W)
        # Computing the noisy image
        return sqrt_alphas_cumprod * images + m_sqrt_alphas_cumprod * epsilon

    def backward_process(self,batch):
        pass

    def training_step(self, batch, batch_idx):
        images = batch # B x C x H x W
        B,_,_,_ = batch.size()

        # Sampling noise
        epsilon = self.offset_noise(images)
        # epsilon = self.gaussian_noise(images)

        # Sampling t
        t = torch.randint(low=0,high=self.nsteps, size = (B,)).to(batch.device)
        if self.include_time_emb:
            emb_t = self.time_model(t)
        else:
            emb_t = t

        # forqward process
        noised_images = self.q_sample(images, epsilon, t)

        # Predicting the values of espilon
        estimated_epsilon = self.model(noised_images, emb_t)

        # Computing the loss
        vloss = self.loss(estimated_epsilon, epsilon)
        self.log("total_loss_train",vloss)
        super().training_step(batch, batch_idx)
        return vloss
    
    def validation_step(self, batch, batch_idx):
        images = batch # B x C x H x W
        B,_,_,_ = batch.size()

        # Sampling noise
        epsilon = self.gaussian_noise(images)

        # Sampling t
        t = torch.randint(low=0,high=self.nsteps, size = (B,)).to(batch.device)
        if self.include_time_emb:
            emb_t = self.time_model(t)
        else:
            emb_t = t

        # forqward process
        noised_images = self.q_sample(images, epsilon, t)

        # Predicting the values of espilon
        estimated_epsilon = self.model(noised_images, emb_t)

        # Computing the loss
        vloss = self.loss(estimated_epsilon, epsilon)
        self.log(MONITORED_LOSS,vloss)
        return vloss

    def step_sampling(self, x, t, t_emb):
            cinv_sqrt_alpha = self.inv_sqrt_alphas[t]
            cm_sqrt_cum_alpha = self.m_sqrt_cum_alphas[t]
            cbeta = self.betas[t]
            
            model_mean = cinv_sqrt_alpha * (
                x - cbeta * self.model(x,t_emb) / cm_sqrt_cum_alpha
            )

            if t==0:
                return model_mean
            else:
                noise = self.gaussian_noise(x)
                csigma = self.sigma[t]

                return model_mean + csigma * noise
        # Sampling the right constants
    @torch.no_grad()
    def generate_images(self, batch=None, n=6, return_intermediate=False, return_frequency=10):
        if batch is None:
            batch = torch.zeros(n,self.in_channels,self.image_size,self.image_size)
            batch = batch.to(self.device)
        x = self.gaussian_noise(batch)
        seq_timesteps = list(reversed(range(0,self.nsteps)))
        timesteps = torch.tensor(seq_timesteps).view(-1,1)
        timesteps = timesteps.to(self.device)
        if self.include_time_emb:
            emb_timesteps = self.time_model(timesteps)
        else:
            emb_timesteps = timesteps

        intermediates = []

        for t,t_emb in zip(seq_timesteps,emb_timesteps):
            # We expanding the mebedding
            B = batch.size(0)
            # t_emb_e = t_emb[None,:]
            # print("t_emb_e {}".format(t_emb_e.shape))
            t_emb_e = t_emb.expand(B, -1)
            x = self.step_sampling(x, t, t_emb_e)
            if t % return_frequency == 0:
                st = str(t)
            if return_intermediate and t % return_frequency == 0:
                intermediates.append(x.detach().cpu())
        clipped_image = torch.clamp(x, min=-1., max=1.)
        if not return_intermediate:
            return clipped_image
        return clipped_image, intermediates
        
    
    def configure_optimizers(self):
        to_optim = list(self.model.parameters())
        if self.include_time_emb:
            to_optim += list(self.time_model.parameters())
        return torch.optim.Adam(to_optim, lr=self.learning_rate)