import torch
import torch.nn as nn

with torch.no_grad():
    a = torch.rand(32,256,8,8)
    clayer1 = nn.ConvTranspose2d(256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
    a1 = clayer1(a)
    print("a1 {}".format(a1.shape))
    clayer2 = nn.ConvTranspose2d(128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
    a2 = clayer2(a1)
    print("a2 {}".format(a2.shape))
    clayer3 = nn.ConvTranspose2d(64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
    a3 = clayer3(a2)
    print("a3 {}".format(a3.shape))
    clayero = nn.ConvTranspose2d(32, out_channels=5, kernel_size=2, stride=2, padding=0, output_padding=0) # Output: (B, out_channels, 128, 128)
    oo = clayero(a3)
    print("oo {}".format(oo.shape))
