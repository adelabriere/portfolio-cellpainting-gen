import pytest
import torch
import math
from gencellpainting.model.styleGANV2 import Generator,Discriminator, generate_noise

C = 6
L = 32
IMG_SIZE = 64
BATCH_SIZE = 16


@pytest.fixture
def fake_data():
    nlayers = int(math.log2(IMG_SIZE)-1)

    imgs = torch.rand((BATCH_SIZE, C, IMG_SIZE, IMG_SIZE))*255
    styles = torch.rand((nlayers,BATCH_SIZE, L))
    noise = generate_noise(BATCH_SIZE, IMG_SIZE)
    return {"images":imgs,"styles":styles,"noise":noise}


def test_stylegan2_generator(fake_data):

    G = Generator(C, IMG_SIZE, L, network_capacity=4)
    gen_images = G(fake_data["styles"], fake_data["noise"])

    print(gen_images.shape)
    assert gen_images.shape == (BATCH_SIZE, C, IMG_SIZE, IMG_SIZE)


def test_stylegan2_discriminator(fake_data):

    D = Discriminator(C, IMG_SIZE, network_capacity=4)

    dprob = D(fake_data["images"])

    assert dprob.shape == (BATCH_SIZE, 1)

# def test_stylagan2(fake_data):

