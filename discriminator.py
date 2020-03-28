import torch.nn as nn
from typing import Tuple

# https://arxiv.org/pdf/1512.09300.pdf
# 5x5 32 chan, BNorm ReLU
# 5x5 128 chan BNorm ReLU
# 5x5 256 chan BNorm ReLU
# 5x5 256 chan BNorm ReLU
# 512 FC BNorm ReLU
# 1 FC sigmoid


def _conv_2d_block(in_chan: int, out_chan: int, kernel: Tuple[int, int]):
    return nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel),
        nn.BatchNorm2d(out_chan, eps=1e-5, momentum=0.9),
        nn.ReLU(inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = _conv_2d_block(3, 32, (5, 5))
        self.conv2 = _conv_2d_block(32, 128, (5, 5))
        self.conv3 = _conv_2d_block(128, 256, (5, 5))
        self.conv4 = _conv_2d_block(256, 256, (5, 5))
        self.dense1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        return self.dense2(out)
