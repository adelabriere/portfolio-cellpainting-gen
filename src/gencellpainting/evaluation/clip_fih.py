
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from importlib.util import find_spec
from typing import Any, Optional, TypeVar, Union

import torch
from torch import nn, Tensor
from torchmetrics import Metric
from transformers import AutoProcessor, AutoModel

if find_spec("torchvision") is not None:
    from torchvision import models

    _TORCHVISION_AVAILABLE = True
else:
    _TORCHVISION_AVAILABLE = False


NAME_MODEL = "openai/clip-vit-base-patch32"

TFrechetInceptionDistance = TypeVar("TFrechetInceptionDistance")

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


def _validate_torchvision_available() -> None:
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "You must have torchvision installed to use FID, please install torcheval[image]"
        )




def batch_tensor_to_clip_inputs(tens, average_channel=True):
    # Consider each of the cellpainting channel as 1 separate image then replicate the channel to convert to RGB expected by CLIP.
    if average_channel:
        tens =  tens.repeat_interleave(3,dim=1)
        tens = tens.view(tens.shape[0]*tens.shape[1],tens.shape[2],tens.shape[3])
    return tens.split(3,dim=0)


def compute_clip_embeddings(tens,processor,device):
    # Input tensor: B x C x H x W
    B, C, _, _ = tens.shape
    img_list = batch_tensor_to_clip_inputs(tens)
    inputs = processor(images=img_list, return_tensors="pt")
    inputs = inputs.to(device)
    features = model.get_image_features(**inputs)
    features = features.view(B ,C, -1)
    return features.mean(axis=1)


class FIDCLIP(nn.Module):
    def __init__(
        self, average_channel = True
    ) -> None:
        """
        This class wraps the CLIP model to compute FID.

        Args:
            average_channel [Boolean]: Sould the channel be duplicated and averaged, considering each of them as a single image.
        """
        super().__init__()
        # pyre-ignore
        self.model = AutoModel.from_pretrained(NAME_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",device_map="auto",)
        self.processor = AutoProcessor.from_pretrained(NAME_MODEL,use_fast=True)
        self.average_channel = average_channel

    def _prepare_tensor(self, tensor):
        if self.average_channel:
            tensor =  tensor.repeat_interleave(3,dim=1)
            ts = tensor.shape
            tensor = tensor.view(ts[0]*ts[1],*ts[2:])
        return tensor.split(3,dim=0)


    def forward(self, x: Tensor) -> Tensor:
    # Input tensor: B x C x H x W
        B, C, _, _ = x.shape
        img_list = self._prepare_tensor(x)
        inputs = self.processor(images=img_list, return_tensors="pt")
        inputs = inputs.to(x.device)
        features = self.model.get_image_features(**inputs)
        return features.view(B ,C, -1).mean(axis=1)


class FrechetCLIPDistance(Metric):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        feature_dim: int = 512,
    ) -> None:
        """
        Computes the Frechet Inception Distance (FID) between two distributions of images (real and generated).

        The original paper: https://arxiv.org/pdf/1706.08500.pdf

        Args:
            model (nn.Module): Module used to compute feature activations.
                If None, a default InceptionV3 model will be used.
            feature_dim (int): The number of features in the model's output,
                the default number is 2048 for default InceptionV3.
            device (torch.device): The device where the computations will be performed.
                If None, the default device will be used.
        """
        _validate_torchvision_available()

        super().__init__()

        self._FID_parameter_check(model=model, feature_dim=feature_dim)

        if model is None:
            model = FIDCLIP(average_channel=True)

        # Set the model and put it in evaluation mode
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        # Initialize state variables used to compute FID
        self.add_state("real_sum", torch.zeros(feature_dim))
        self.add_state(
            "real_cov_sum", torch.zeros((feature_dim, feature_dim))
        )
        self.add_state("fake_sum", torch.zeros(feature_dim))
        self.add_state(
            "fake_cov_sum", torch.zeros((feature_dim, feature_dim))
        )
        self.add_state("num_real_images", torch.tensor(0).int())
        self.add_state("num_fake_images", torch.tensor(0).int())


    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self, images: Tensor, is_real: bool
    ):
        """
        Update the states with a batch of real and fake images.

        Args:
            images (Tensor): A batch of images.
            is_real (Boolean): Denotes if images are real or not.
        """

        self._FID_update_input_check(images=images, is_real=is_real)

        images = images.to(self.device)

        # Compute activations for images using the given model
        activations = self.model(images)

        batch_size = images.shape[0]

        # Update the state variables used to compute FID
        if is_real:
            self.num_real_images += batch_size
            self.real_sum += torch.sum(activations, dim=0)
            self.real_cov_sum += torch.matmul(activations.T, activations)
        else:
            self.num_fake_images += batch_size
            self.fake_sum += torch.sum(activations, dim=0)
            self.fake_cov_sum += torch.matmul(activations.T, activations)

        return self

    @torch.inference_mode()
    def merge_state(
        self, metrics
    ):
        """
        Merge the state of another FID instance into this instance.

        Args:
            metrics (Iterable[FID]): The other FID instance(s) whose state will be merged into this instance.
        """
        for metric in metrics:
            self.real_sum += metric.real_sum.to(self.device)
            self.real_cov_sum += metric.real_cov_sum.to(self.device)
            self.fake_sum += metric.fake_sum.to(self.device)
            self.fake_cov_sum += metric.fake_cov_sum.to(self.device)
            self.num_real_images += metric.num_real_images.to(self.device)
            self.num_fake_images += metric.num_fake_images.to(self.device)

        return self

    @torch.inference_mode()
    def compute(self) -> Tensor:
        """
        Compute the FID.

        Returns:
            tensor: The FID.
        """

        # If the user has not already updated with at lease one
        # image from each distribution, then we raise an Error.
        if (self.num_real_images < 2) or (self.num_fake_images < 2):
            warnings.warn(
                "Computing FID requires at least 2 real images and 2 fake images,"
                f"but currently running with {self.num_real_images} real images and {self.num_fake_images} fake images."
                "Returning 0.0",
                RuntimeWarning,
                stacklevel=2,
            )

            return torch.tensor(0.0)

        # Compute the mean activations for each distribution
        real_mean = (self.real_sum / self.num_real_images).unsqueeze(0)
        fake_mean = (self.fake_sum / self.num_fake_images).unsqueeze(0)

        # Compute the covariance matrices for each distribution
        real_cov_num = self.real_cov_sum - self.num_real_images * torch.matmul(
            real_mean.T, real_mean
        )
        real_cov = real_cov_num / (self.num_real_images - 1)
        fake_cov_num = self.fake_cov_sum - self.num_fake_images * torch.matmul(
            fake_mean.T, fake_mean
        )
        fake_cov = fake_cov_num / (self.num_fake_images - 1)

        # Compute the Frechet Distance between the distributions
        fid = self._calculate_frechet_distance(
            real_mean.squeeze(), real_cov, fake_mean.squeeze(), fake_cov
        )
        return fid

    def _calculate_frechet_distance(
        self,
        mu1: Tensor,
        sigma1: Tensor,
        mu2: Tensor,
        sigma2: Tensor,
    ):
        """
        Calculate the Frechet Distance between two multivariate Gaussian distributions.

        Args:
            mu1 (Tensor): The mean of the first distribution.
            sigma1 (Tensor): The covariance matrix of the first distribution.
            mu2 (Tensor): The mean of the second distribution.
            sigma2 (Tensor): The covariance matrix of the second distribution.

        Returns:
            tensor: The Frechet Distance between the two distributions.
        """

        # Compute the squared distance between the means
        mean_diff = mu1 - mu2
        mean_diff_squared = mean_diff.square().sum(dim=-1)
    
        # Addition stickign to def
        sigma_mm = torch.matmul(sigma1, sigma2)
        # Calculate the sum of the traces of both covariance matrices
        trace_sum = sigma1.trace() + sigma2.trace()

        # Compute the eigenvalues of the matrix product of the real and fake covariance matrices
        # sigma_mm = torch.matmul(sigma1, sigma2)
        eigenvals = torch.linalg.eigvals(sigma_mm)

        # Take the square root of each eigenvalue and take its sum
        sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)

        # Calculate the FID using the squared distance between the means,
        # the sum of the traces of the covariance matrices, and the sum of the square roots of the eigenvalues
        fid = mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum

        return fid

    def _FID_parameter_check(
        self,
        model: Optional[nn.Module],
        feature_dim: int,
    ) -> None:
        # Whatever the model, the feature_dim needs to be set
        if feature_dim is None or feature_dim <= 0:
            raise RuntimeError("feature_dim has to be a positive integer")

        if model is None and feature_dim != 512:
            raise RuntimeError(
                "When the default Inception v3 model is used, feature_dim needs to be set to 2048"
            )

    def _FID_update_input_check(
        self, images: torch.Tensor, is_real: bool
    ) -> None:
        if not torch.is_tensor(images):
            raise ValueError(f"Expected tensor as input, but got {type(images)}.")


    def to(
        self,
        device: Union[str, torch.device],
        *args: Any,
        **kwargs: Any,
    ):
        super().to(device=device)
        self.model.to(device)
        return self