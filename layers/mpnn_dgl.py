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

    def u_mul_e_udf(edges):
        return {'m': edges.src['h'] * edges.data['w']}

    def sum_udf(nodes):
        return {'h': nodes  }

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat((h, h_N), dim=1)
            print(h_N.shape)
            print(h_total.shape)
            print(self.U)
            # return h_total
            return self.U(h_total).detach()
