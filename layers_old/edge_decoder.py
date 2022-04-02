import torch
import torch.nn as nn


class EdgeDecoder(nn.Module):
    """
    Module that decodes the node and edge information and returns
    the conditional probabilities for traversing each edge.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layers used to decode node and edge attributes
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        """
        Parameters:
        in_channels : int
            Dimension of the input vectors (latent dimension)
        out_channels : int
            Dimension of the output vectors (usually 1)
        """
        self.linear = nn.Linear(3 * in_channels, out_channels, bias=bias)

    def concatenate(self, edges):
        """Concatenate vectors of two adjacent nodes and their edge."""
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        e_f = edges.data['e_f']
        # e_b = edges.data['e_b']
        # p = torch.cat((h_src, h_dst, e_f, e_b), dim=1)
        p = torch.cat((h_src, h_dst, e_f), dim=1)
        return {'p': p}

    def forward(self, g, h, e_f, e_b):
        """Return the conditional probability for each edge."""
        g.ndata['h'] = h
        g.edata['e_f'] = e_f
        # g.edata['e_b'] = e_b
        g.apply_edges(self.concatenate)
        p = self.linear(g.edata['p'])
        # p = torch.sigmoid(p)  # I think it's better to return logits
        return p
