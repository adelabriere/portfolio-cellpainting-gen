import pytest
import torch
import math
from gencellpainting.evaluation.clip_fih import FrechetCLIPDistance
C = 6
L = 32
IMG_SIZE = 64
BATCH_SIZE = 16


@pytest.fixture
def fake_data():
    nlayers = int(math.log2(IMG_SIZE)-1)

    imgs = torch.rand((BATCH_SIZE, C, IMG_SIZE, IMG_SIZE))*255
    fake_imgs = torch.rand((BATCH_SIZE, C, IMG_SIZE, IMG_SIZE))*255
    return [imgs,fake_imgs]


def test_clipfih_metrics(fake_data):
    imgs,fake_imgs  = fake_data

    fid = FrechetCLIPDistance().to("cuda")

    fid.update(imgs.to("cuda"), is_real=True)
    fid.update(fake_imgs.to("cuda"), is_real=False)

    # Compute the metrics
    fid_score = fid.compute()
    assert ~fid_score.isnan()

    fid2 = FrechetCLIPDistance().to("cuda")

    fid2.update(imgs.to("cuda"), is_real=True)
    fid2.update(imgs.to("cuda"), is_real=False)
    # Compute the metrics

    fid2_score =fid2.compute()
    assert ~fid2_score.isnan()
    
    assert float(fid_score)>float(fid2_score)