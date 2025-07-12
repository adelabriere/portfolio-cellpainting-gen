# Cell Painting Data Generation

This notebook aims to explore various algorithms of image generations for Cell Painting Assay images. This is mainly a fun projects.

## Models considered

### Variationnal convolutionnal $\beta$-VAE (VAE)
The first and simplest model is a $\beta$-VAE [(Higgins, 2016)](https://arxiv.org/abs/1606.05579). It is composed of 2 models, one to encode the image into a latent space (Encoder) and one to decode the latent space into an image (Decoder). The model is trained with a combination of reconstruction loss (MSE) and KL divergence to a standard normal. An image is actually encoded into a distribution, ensuring some variability.

* __Encoder__: The encoder is a convolutionnal neural network using 2D max pooling for downsampling.
* __Decoder__: The decoder is a convolutionnal neural network using transposed convolution for upsampling

### Wasserstein Generative Adversarial network (WGAN-GP)
The Wasserstein GAN [(Arjovsky, 2017)](https://arxiv.org/abs/1701.07875) with gradient penalty [(Gulrajani, 2017)](https://arxiv.org/abs/1704.00028) use 2 models, one to generate an image from a noise patch (Generator) and one evaluating if the image can be discriminated from the other (Discriminator). As GAN training is notoriously unstable, this implementation is a Wasserstein GAN with gradient penalty (WGAN-GP), see () which is easier to train and more stable.

* __Generator__: The generator is a simple convolutionnal neural network using transposed convolution, it has exactly the same structure as the decoder of the VAE.
* __Discriminator__:

### Denoising diffusion probabilistic model with UNET (DDPM-UNET)
The third implementation is an implementation of the [(Ho, 2020)](https://arxiv.org/pdf/2006.11239). In this case, the generating model is a UNET [(Ronneberger, 2015)](https://arxiv.org/abs/1505.04597) with a more moder architecture, notably including ConvNext block [(Lio, 2022)](https://arxiv.org/abs/2201.03545). In this case 

```
| Name       | Type            | Params | Mode 
-------------------------------------------------------
0 | loss       | SmoothL1Loss    | 0      | train
1 | model      | UNetDiffusionV2 | 7.2 M  | train
2 | time_model | Sequential      | 6.0 K  | train
-------------------------------------------------------
7.3 M     Trainable params
0         Non-trainable params
7.3 M     Total params
29.000    Total estimated model params size (MB)
230       Modules in train mode
0         Modules in eval mode
```

### ResNet50 backbone baseline
In order to have a simple baseline


## Parameters optimizations

Given that the experiments were ran on a personnal laptop with very limited VRAM, the parameters optimization was limited in scale, as such a simple grid search was used to look for size of the latent space and the inner convolution layers channels for both the VAE and the WGAN-GP. Given the longer training time of the DDPM-UNET model, its parameters were not optimized. The parameters optimization script is given in ....

## Training the models

Evaluating the model

## Evaluating the model

### Visual inspection
Her is osme examples of images

### Frecht Inception Distance
Frechet inception distance ocn

