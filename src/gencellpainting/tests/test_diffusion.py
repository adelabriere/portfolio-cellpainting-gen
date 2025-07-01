import pytest
import torch
import math
from gencellpainting.model.net.UNETdiffusion import UNetDiffusion
from gencellpainting.model.diffusion import DiffusionProcess

CIN = 6
NSTEPS = 32
IMG_SIZE = 128
BATCH_SIZE = 16
NLAYERS = 2
TIME_EMBEDDING_DIM = 32
DEVICE = "cuda"


@pytest.fixture
def fake_data():
    imgs = torch.rand((BATCH_SIZE, CIN, IMG_SIZE, IMG_SIZE))*255
    imgs =imgs.to(DEVICE)
    time_embs = torch.rand((BATCH_SIZE, TIME_EMBEDDING_DIM)).to(DEVICE)
    undiff = UNetDiffusion(in_channels = CIN, out_channels=CIN, network_capacity=32,\
                           nlayers=4, time_channels=TIME_EMBEDDING_DIM)
    undiff = undiff.to(DEVICE)
    diff_process = DiffusionProcess(CIN, NSTEPS, undiff, TIME_EMBEDDING_DIM, IMG_SIZE, device=DEVICE)
    diff_process = diff_process.to(DEVICE)
    ts = torch.randint(low=0,high=NSTEPS-1, size = (BATCH_SIZE,))
    ts =ts.to(DEVICE)
    return imgs, ts, time_embs, diff_process


# def test_diffusion_forward(fake_data):
#     imgs, ts, _, diff_process = fake_data
#     epsilon = diff_process.gaussian_noise(imgs)
#     noised_imgs = diff_process.forward_process(imgs, epsilon, ts)
#     vdiff = noised_imgs-imgs
#     abs_norm = torch.norm(vdiff,p=2)
#     assert noised_imgs.shape == imgs.shape
#     assert abs_norm > 0.02

def test_image_generation(fake_data):
    imgs, ts, time_embs, diff_process = fake_data
    NGEN = 3
    gimgs = diff_process.generate_images(n=NGEN)
    assert gimgs.shape == (NGEN, CIN, IMG_SIZE, IMG_SIZE)
