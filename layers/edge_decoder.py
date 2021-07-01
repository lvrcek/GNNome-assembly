import torch
import torch.nn as nn


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(3 * in_channels, out_channels, bias=bias)

    def concatenate(self, edges):
        h_k = edges.src['h']
        h_i = edges.dst['h']
        e_ki = edges.data['e']
        p = torch.cat((h_k, h_i, e_ki), dim=1)
        return {'p': p}

    def forward(self, g, h, e):
        g.ndata['h'] = h
        g.edata['e'] = e
        g.apply_edges(self.concatenate)
        p = self.linear(g.edata['p'])
        # p = torch.sigmoid(p)  # I think it's better to return logits
        return p
