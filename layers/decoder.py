import torch.nn as nn


class DecoderNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):
        super(DecoderNetwork, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.linear(x)
