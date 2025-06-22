import pytest
import torch
import math
from gencellpainting.model.styleGANV2 import Generator,Discriminator, SG2SimpleGAN, generate_noise, generate_style

C = 6
L = 32
IMG_SIZE = 64
BATCH_SIZE = 16


@pytest.fixture
def fake_data():
    nlayers = int(math.log2(IMG_SIZE)-1)

    imgs = torch.rand((BATCH_SIZE, C, IMG_SIZE, IMG_SIZE))*255
    styles = generate_style(nlayers,BATCH_SIZE, L)
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


def test_stylegan2_lighning(fake_data):
    disc_images = fake_data["images"]

    sg2 = SG2SimpleGAN(L, C, IMG_SIZE, network_capacity=4)

    new_images = sg2.generate_images(n=4)

    assert new_images.shape == (4, C, IMG_SIZE, IMG_SIZE)
