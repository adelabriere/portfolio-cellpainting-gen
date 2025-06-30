import pytest
import torch
import math
from gencellpainting.model.net.UNET import UNet
CIN = 6
COUT = 5
N = 32
IMG_SIZE = 256
BATCH_SIZE = 16
NLAYERS = 3


@pytest.fixture
def fake_data():
    imgs = torch.rand((BATCH_SIZE, CIN, IMG_SIZE, IMG_SIZE))*255
    return imgs




def test_unet_dim(fake_data):
    unet = UNet(in_channels = CIN, out_channels=COUT, network_capacity=64, nlayers=4)
    imgs = fake_data
    x = unet(imgs)
    assert x.shape == (BATCH_SIZE, COUT, IMG_SIZE, IMG_SIZE)

