import torch.nn as nn
from typing import Tuple

# https://arxiv.org/pdf/1512.09300.pdf
# 5x5 64 chan, BNorm ReLU
# 5x5 128 chan BNorm ReLU
# 5x5 256 chan BNorm ReLU
# 2048 FC BNorm ReLU


def _conv_2d_block(in_chan: int, out_chan: int, kernel: Tuple[int, int]):
    return nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel),
        nn.BatchNorm2d(out_chan, eps=1e-5, momentum=0.9),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = _conv_2d_block(3, 64, (5, 5))
        self.conv2 = _conv_2d_block(64, 128, (5, 5))
        self.conv3 = _conv_2d_block(128, 256, (5, 5))
        self.dense = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm2d(2048, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        return self.dense(out)
