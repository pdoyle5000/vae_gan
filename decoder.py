import torch.nn as nn
from typing import Tuple

# https://arxiv.org/pdf/1512.09300.pdf
# 8*8*256 FC BNorm ReLU
# 5x5 256 chan BNorm ReLU
# 5x5 128 chan BNorm ReLU
# 5x5 32 chan BNorm ReLU
# 5x5 3 chan tanh


def _conv_transpose_block(in_chan: int, out_chan: int, kernel: Tuple[int, int]):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chan, out_chan, kernel),
        nn.BatchNorm2d(out_chan, eps=1e-5, momentum=0.9),
        nn.ReLU(inplace=True),
    )


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(2048, 8 * 8 * 256),
            nn.BatchNorm2d(8 * 8 * 256, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        # the input might be 1 channel.
        self.conv1 = _conv_transpose_block(8 * 8 * 256, 256, (5, 5))
        self.conv2 = _conv_transpose_block(256, 128, (5, 5))
        self.conv3 = _conv_transpose_block(128, 32, (5, 5))
        self.conv4 = nn.Sequential(
            _conv_transpose_block(128, 3, (5, 5)),
            nn.BatchNorm2d(3, eps=1e-5, momentum=0.9),
            nn.Tanh(inplace=True),
        )

    def forward(self, x):
        out = self.dense(x)
        # probably need a reshape here
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.conv4(out)
