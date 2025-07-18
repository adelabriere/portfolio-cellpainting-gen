import pytest
import torch
import math
from gencellpainting.model.net.UNETdiffusion import UNetDiffusion, UNetDiffusionV2
CIN = 5
COUT = 5
N = 32
IMG_SIZE = 64
BATCH_SIZE = 16
NLAYERS = 3
TIME_EMBEDDING_DIM = 32


@pytest.fixture
def fake_data():
    imgs = torch.rand((BATCH_SIZE, CIN, IMG_SIZE, IMG_SIZE))*255
    time_embs = torch.rand((BATCH_SIZE, TIME_EMBEDDING_DIM))
    return imgs, time_embs


# def test_unet_dim(fake_data):
#     unet = UNet(in_channels = CIN, out_channels=COUT, network_capacity=64, nlayers=4)
#     imgs, _ = fake_data
#     x = unet(imgs)
#     assert x.shape == (BATCH_SIZE, COUT, IMG_SIZE, IMG_SIZE)


def test_unet_diff_dim(fake_data):
    undiff = UNetDiffusionV2(in_channels = CIN, out_channels=COUT, network_capacity=2,\
                           nlayers=3, time_channels=TIME_EMBEDDING_DIM)
    imgs, time_embs = fake_data
    x = undiff(imgs, time_embs)
    assert x.shape == (BATCH_SIZE, COUT, IMG_SIZE, IMG_SIZE)
