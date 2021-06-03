import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import add_self_loops

from hyperparameters import get_hyperparameters

class MPNN(nng.MessagePassing):

    def __init__(self, in_channels, out_channels, edge_features, aggr='mean', bias=False, flow='source_to_target'):
        super(MPNN, self).__init__(aggr=aggr, flow=flow)
        self.device = get_hyperparameters()['device']
        self.out_channels = out_channels
        self.M = nn.Sequential(nn.Linear(2 * in_channels + edge_features, out_channels, bias=bias),
                               nn.LeakyReLU())
        self.U = nn.Sequential(nn.Linear(2 * in_channels, out_channels, bias=bias),
                               nn.LeakyReLU())
        self.gru = nn.GRUCell(out_channels, out_channels, bias=bias).to(self.device)

    def zero_hidden(self, num_nodes):
        self.hidden = torch.zeros((num_nodes, self.out_channels)).to(self.device).detach()
        return self.hidden

    def forward(self, x, edge_attr, edge_index):
        # print(edge_index.shape)
        # print(edge_attr.shape)
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr).to(self.device)

    def message(self, x_i, x_j, edge_attr):
        return self.M(torch.cat((x_i, x_j, edge_attr), dim=1)).to(self.device)

    def update(self, aggr_out, x):
        self.hidden = self.U(torch.cat((x, aggr_out), dim=1)).to(self.device).detach()
        return self.hidden
