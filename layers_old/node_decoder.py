import torch
import torch.nn as nn


class NodeDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, g, h):
        g.ndata['h'] = h       
        p = self.linear(h)
        return p
