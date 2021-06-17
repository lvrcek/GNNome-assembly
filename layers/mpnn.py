import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from hyperparameters import get_hyperparameters

class MPNN(nn.Module):
    def __init__(self, in_channels, out_channels, edge_features, bias=False):
        super(MPNN, self).__init__()
        self.device = get_hyperparameters()['device']
        self.out_channels = out_channels
        self.M = nn.Sequential(nn.Linear(2 * in_channels + edge_features, out_channels, bias=bias),
                               nn.LeakyReLU())
        self.U = nn.Sequential(nn.Linear(2 * in_channels, out_channels, bias=bias),
                               nn.LeakyReLU())

    def message_func(self, edges):
        return {'m': self.M(torch.cat((edges.src['h'], edges.dst['h'], edges.data['e']), dim=1)) }

    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['e'] = e
            g.update_all(message_func=self.message_func, reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat((h, h_N), dim=1)
            return self.U(h_total).detach()
