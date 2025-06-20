import torch
import torch.nn as nn

cn_conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=2, padding=0, stride=1)

cn_conv.weight.shape


# Let s try to perform the addition we will have to do with the style vector

X = torch.tensor([
    [
        [
            [1,2],[3,4]
        ]
    ],[
        [
            [11,12],[13,14]
        ]
    ],[
        [
            [21,22],[23,24]
        ]
    ]
])

X.shape
K = X.shape[2]

S = torch.tensor([1,2,3])

exS = (S[Nonw,:,None,None]).expand(B,-1,K,K)

exS.shape
X.shape

exS * X

X.reshape(1,-1,2,2).shape


#X (1 , B x C , H , W)


# B x Co x Ci x K x K

#W (B * Co , Ci , K, K)


print([x[0].shape for x in zip(X)])
imgs = torch.randn(4,3,128,128)


conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=0)
conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=1, padding=0)

conv(imgs).shape
conv1(imgs)[1,1,:,:]
imgs[1,1,:,:]



conv1.weight.shape

# CHeck idf we can rebuild the tensor

imgs[1,1,:,:,]