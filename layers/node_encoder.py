import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """
    Module that encodes the node features into a high-dimensional
    vector.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer used to encode the edge attributes
    """

    def __init__(self, in_channels, out_channels, bias=True):
        """
        Parameters:
        in_channels : int
            Dimension of the input vectors
        out_channels : int
            Dimension of the output (encoded) vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        """Return the encoded edge attributes."""
        x = x.unsqueeze(-1)
        return self.linear(x)
