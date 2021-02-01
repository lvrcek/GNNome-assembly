import torch
import torch.nn as nn
import torch_geometric.nn as nng


class MPNN(nng.MessagePassing):

    def __init__(self, in_channels, out_channels, edge_features, aggr='max', bias=False, flow='source_to_target'):
        super(MPNN, self).__init__(aggr=aggr, flow=flow)
        # ---testing GPU ----
        self.device = 'cuda:5'
        # -------------------
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
        return self.propagate(edge_index, x=x, edge_attr=edge_attr).to(self.device)

    def message(self, x_i, x_j, edge_attr):
        return self.M(torch.cat((x_i, x_j, edge_attr), dim=1)).to(self.device)

    def update(self, aggr_out, x):
        # TODO: Doesn't detach() defeat the purpose of GRU?
        self.hidden = self.gru(self.U(torch.cat((x, aggr_out), dim=1)), self.hidden).to(self.device).detach()
        return self.hidden
