from gencellpainting.model.net.conv_modules import LinearAttention, ConvNextBlock
import torch
import pytest


CIN = 6
COUT = 5
N = 32
IMG_SIZE = 128
BATCH_SIZE = 16
NLAYERS = 3
TIME_EMBEDDING_DIM = 128


@pytest.fixture
def fake_data():
    imgs = torch.rand((BATCH_SIZE, CIN, IMG_SIZE, IMG_SIZE))*255
    time_embs = torch.rand((BATCH_SIZE, TIME_EMBEDDING_DIM))
    return imgs, time_embs


def test_linear_attention(fake_data):
    imgs, _ = fake_data
    lin_att = LinearAttention(CIN)
    out = lin_att(imgs)
    assert out.shape == (BATCH_SIZE, CIN, IMG_SIZE, IMG_SIZE)

def test_convnext_block(fake_data):
    imgs, time_embs = fake_data
    convnext = ConvNextBlock(CIN, TIME_EMBEDDING_DIM)
    out = convnext(imgs, time_embs)
    assert out.shape == (BATCH_SIZE, CIN, IMG_SIZE, IMG_SIZE)