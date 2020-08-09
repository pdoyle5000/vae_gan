from dataclasses import dataclass
from typing import Optional
from encoder import Encoder
from enum import Enum, unique
from decoder import Decoder
from discriminator import Discriminator
import torch.nn as nn
import torch
import torct.nn.functional as F
from torch.autograd import Variable
import numpy as np


class VaeGan(nn.Module):
    def __init__(self, size: int = 128, reconstruct: int = 3, is_training: bool = True):
        super(VaeGan, self).__init__()
        self.size = size
        self.encoder = Encoder(size)
        self.decoder = Decoder(size)
        self.discriminator = Discriminator(reconstruct, 3)
        self.initialize_networks()
        self.is_training = is_training

    def initialize_networks(self):
        for module in self.modules():
            is_layer = isinstance(module, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
            if is_layer and _requires_grad(module):
                scalar = 1.0 / np.sqrt(np.prod(module.weight.shape[1:])) / np.sqrt(3)
                nn.init.uniform(module.weight, -scalar, scalar)
            if _requires_bias(module):
                nn.init.constant(module.bias, 0.0)

    def forward(self, input_tensor: Optional[torch.Tensor]):
        if self.is_training:
            orig_tensor = input_tensor
            vector, log_variances = self.encoder(input_tensor)
            variances = torch.exp(log_variances * 0.5)
            rand_tensor = Variable(
                torch.randn(len(input_tensor), self.size).cuda(), requires_grad=True
            )
            input_tensor = rand_tensor * variances + vector

            input_tensor = self.decoder(input_tensor)
            tensor_layer = self.discriminator(input_tensor, orig_tensor, DiscMode.rec)
            rand_tensor = Variable(
                torch.randn(len(input_tensor), self.size).cuda(), requires_grad=True
            )
            input_tensor = self.decoder(rand_tensor)
            pred = self.discriminator(orig_tensor, input_tensor, DiscMode.gan)
            return input_tensor, pred, tensor_layer, vector, log_variances
        else:
            return self.sample_decode(input_tensor)

    def sample_decode(self, input_tensor: Optional[torch.Tensor]):
        if input_tensor is None:
            input_tensor = Variable(
                torch.randn(10, self.size).cuda(), requires_grad=False
            )
            input_tensor = self.decoder(input_tensor)
        else:
            vector, log_variances = self.encoder(input_tensor)
            variances = torch.exp(log_variances * 0.5)
            rand = Variable(
                torch.randn(len(input_tensor), self.size).cuda(), requires_grad=False
            )
            input_tensor = rand * variances + vector
            input_tensor = self.decoder(input_tensor)
        return input_tensor

    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)


@dataclass
class VaeganLoss:
    orig_input: torch.Tensor
    predicted: torch.Tensor
    orig_latent_discrim: torch.Tensor
    pred_latent_discrim: torch.Tensor
    orig_labels: torch.Tensor
    pred_labels: torch.Tensor
    sampled_labels: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor

def vaegan_loss(inputs: VaeganLoss) -> VaeganLoss:



# this has to go in the discriminator
@unique
class DiscMode(Enum):
    gan = 0
    rec = 1


def _requires_grad(module: nn.Module) -> bool:
    return all(
        [
            hasattr(module, "weight"),
            module.weight is not None,
            module.weight.requires_grad,
        ]
    )


def _requires_bias(module: nn.Module) -> bool:
    return all(
        [hasattr(module, "bias"), module.bias is not None, module.bias.requires_grad]
    )
