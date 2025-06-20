import pytest
import torch
import math
from gencellpainting.model.styleGANV2 import Generator,Discriminator, generate_noise


@pytest.fixture
def fake_data():

    C,L,IMG_SIZE,BATCH_SIZE = (6,32,64,16)
    nlayers = math.log2(IMG_SIZE)-1

    imgs = torch.rand(BATCH_SIZE, C, IMG_SIZE, IMG_SIZE)*255
    styles = torch.rand(nlayers,BATCH_SIZE, L)
    noise = generate_noise(BATCH_SIZE, IMG_SIZE)
    {"images":imgs,"styles":styles,"noise":noise}


def test_stylegan2(fake_data):
    C = 6
    L = 32
    IMG_SIZE = 64
    G = Generator(C, IMG_SIZE, L, network_capacity=4)
    gen_images = G(fake_data["styles"], fake_data["noise"], None)

    print(gen_images.shape)
    assert gen_images.shape == (16, 6, 64, 64)


