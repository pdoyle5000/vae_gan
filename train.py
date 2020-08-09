import torch
from torch.optim import Adam
from torch.nn import nn
from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from dataset import VaeganDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def _maybe_distribute(net):
    if torch.cuda.device_count > 1:
        return nn.DataParallel(net)


class VaeganTrainer:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.dataset = VaeganDataset()
        self.encoder = _maybe_distribute(Encoder().to(self.device))
        self.decoder = _maybe_distribute(Decoder().to(self.device))
        self.discriminator = _maybe_distribute(Discriminator().to(self.device))
        self.encoder_opt = Adam(self.encoder.params(), lr=1e-4)
        self.decoder_opt = Adam(self.decoder.params(), lr=1e-4)
        self.discriminator_opt = Adam(self.discriminator.params(), lr=1e-4)
        self.criterion = None  # ph
        self.scheduler = None  # ph
        self.writer = SummaryWriter()

    def train(self, output_file: str = "output.pth", epochs: int = 10, batch_size: int = 4):

        dataloader = DataLoader(self.dataset, batch_size=batch_size)
        for epoch in epochs:
            for i, inputs in enumerate(dataloader):
                # forward pass
                # calc loss
                # backward pass
                # opt step
                # metrics
        # save models
