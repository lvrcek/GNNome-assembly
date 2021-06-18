import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class Net(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.A_1 = nn.Linear(in_channels, out_channels)
        self.A_2 = nn.Linear(in_channels, out_channels)
        self.A_3 = nn.Linear(in_channels, out_channels)
        self.B_1 = nn.Linear(in_channels, out_channels)
        self.B_2 = nn.Linear(in_channels, out_channels)
        self.B_3 = nn.Linear(in_channels, out_channels)

        self.bn_h = nn.BatchNorm1d(out_channels)
        self.bn_e = nn.BatchNorm1d(out_channels)

        self.batch_norm = True
        self.residual = True
        self.dropout = 0.5

    def message_forward(self, edges):
        A2h_j = edges.src['A2h']
        e_ji = edges.src['B1h'] + edges.dst['B2h'] + edges.data['B3e']  # e_ji = B_1*h_j + B_2*h_i + B_3*e_ji
        # Add relu, batchnorm, residual connection
        return {'A2h_j': A2h_j, 'e_ji': e_ji}

    def reduce_forward(self, nodes):
        A2h_j = nodes.mailbox['A2h_j']
        e_ji = nodes.mailbox['e_ji']
        sigma_ji = torch.sigmoid(e_ji)
        h_forward = torch.sum(sigma_ji * A2h_j, dim=1) / (torch.sum(sigma_ji, dim=1) + 1e-6)
        return {'h_forward': h_forward}

    def message_backward(self, edges):
        A3h_k = edges.src['A3h']
        e_ik = edges.dst['B1h'] + edges.src['B2h'] + edges.data['B3e']  # e_ik = B_1*h_i + B_2*h_k + B_3*e_ik
        # Add relu, batchnorm, residual connection
        return {'A3h_k': A3h_k, 'e_ik': e_ik}

    def reduce_backward(self, nodes):
        A3h_k = nodes.mailbox['A3h_k']
        e_ik = nodes.mailbox['e_ik']
        sigma_ik = torch.sigmoid(e_ik)
        h_backward = torch.sum(sigma_ik * A3h_k, dim=1) / (torch.sum(sigma_ik, dim=1) + 1e-6)
        return {'h_backward': h_backward}

    def forward(self, g, h, e):
        h_in = h
        e_in = e

        g.ndata['h'] = h
        g.edata['e'] = e
        g.ndata['A1h'] = h * 10
        g.ndata['A2h'] = h * 100
        g.ndata['A3h'] = h * -1
        g.ndata['B1h'] = h * .1
        g.ndata['B2h'] = h * -10
        g.edata['B3e'] = e

        print('FORWARD')
        g.update_all(self.message_forward, self.reduce_forward)
        print('BACKWARD')
        gg = dgl.reverse(g, copy_ndata=True, copy_edata=True)
        gg.update_all(self.message_backward, self.reduce_backward)

        h = g.ndata['A1h'] + g.ndata['h_forward'] + gg.ndata['h_backward']
        
        if self.batch_norm:
            h = self.bn_h(h)

        h = F.relu(h)

        if self.residual:
            h += h_in

        h = F.dropout(h, self.dropout, training=self.training)
        


net = Net(1, 1)
g = dgl.graph(([0, 1, 2, 2], [2, 2, 3, 4]))
h = torch.tensor([1, 2, 3, 4, 5]).float().unsqueeze(-1)
e = torch.tensor([1, 2, 3, 4]).float().unsqueeze(-1)

print(h)
print(e)

g.ndata['h'] = h
g.edata['e'] = e
print('graph 1')
print(g)
print(g.ndata)
print(g.edata)

net(g, h, e)

# gg = dgl.reverse(g, copy_ndata=True, copy_edata=True)
# print('graph 2')
# print(gg)
# print(gg.ndata)
# print(gg.edata)
