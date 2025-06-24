

import torch
import torchvision
import numpy as np


def multi_channel_tensor_to_flat_matrix(image,nrow=6):
    # Split along the the channel dimension
    imgs = list(torch.split(image,1,dim=0))

    # Build the grid
    grid = torchvision.utils.make_grid(imgs,nrow = nrow)

    return grid
