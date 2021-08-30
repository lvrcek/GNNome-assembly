import torch
import torch.nn as nn


class EdgeEncoder(nn.Module):
    """
    Module that normalizes and encodes edge attributes
    (overlap length and similarity) into vectors.

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

    def normalize(self, tensor):
        """Normalize the tensor to mean = 0, std = 1."""
        tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
        return tensor

    def forward(self, overlap_similarity, overlap_length):
        """Return the encoded edge attributes."""
        overlap_similarity = overlap_similarity.unsqueeze(-1)  # Can't normalize if all are 1.0!
        overlap_length = self.normalize(overlap_length.float()).unsqueeze(-1)
        e = torch.cat((overlap_similarity, overlap_length), dim=1)
        return self.linear(e)
