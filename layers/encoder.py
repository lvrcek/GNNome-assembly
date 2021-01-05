import torch.nn as nn


class EncoderNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):
        super(EncoderNetwork, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.leaky_relu(x)
        return x
