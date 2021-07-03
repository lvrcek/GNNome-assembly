import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class GatedGCN(nn.Module):
    """
    GatedGCN layer, idea based on 'Residual Gated Graph ConvNets'
    paper by Xavier Bresson and Thomas Laurent, ICLR 2018.
    https://arxiv.org/pdf/1711.07553v2.pdf

    Attributes
    ----------
    dropout : bool
        Flag indicating whether to use dropout
    batch_norm : bool
        Flag indicating whether to use batch normalization.
    residual : bool
        Flag indicating whether to use node information from
        the previous iteration.
    A_n : torch.nn.Linear
        Linear layer used to update node representations
    B_n : torch.nn.Linear
        Linear layer used to update edge representations
    bn_h : torch.nn.BatchNorm1d
        Batch normalization layer used on node representations
    bn_e : torch.nn.BatchNorm1d
        Batch normalization layer used on edge representations
    """
    
    def __init__(self, in_channels, out_channels, dropout=0, batch_norm=True, residual=True):
        """
        
        """
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        self.A_1 = nn.Linear(in_channels, out_channels)
        self.A_2 = nn.Linear(in_channels, out_channels)
        self.A_3 = nn.Linear(in_channels, out_channels)
        self.B_1 = nn.Linear(in_channels, out_channels)
        self.B_2 = nn.Linear(in_channels, out_channels)
        self.B_3 = nn.Linear(in_channels, out_channels)

        self.bn_h = nn.BatchNorm1d(out_channels)
        self.bn_e = nn.BatchNorm1d(out_channels)

    def message_forward(self, edges):
        """Message function used on the original graph."""
        A2h_j = edges.src['A2h']
        e_ji = edges.src['B1h'] + edges.dst['B2h'] + edges.data['B3e']  # e_ji = B_1*h_j + B_2*h_i + B_3*e_ji

        if self.batch_norm:
            e_ji = self.bn_e(e_ji)

        e_ji = F.relu(e_ji)

        if self.residual:
            e_ji = e_ji + edges.data['e']

        return {'A2h_j': A2h_j, 'e_ji': e_ji}

    def reduce_forward(self, nodes):
        """Reduce function used on the original graph."""
        A2h_j = nodes.mailbox['A2h_j']
        e_ji = nodes.mailbox['e_ji']
        sigma_ji = torch.sigmoid(e_ji)
        h_forward = torch.sum(sigma_ji * A2h_j, dim=1) / (torch.sum(sigma_ji, dim=1) + 1e-6)
        return {'h_forward': h_forward}

    def message_backward(self, edges):
        """Message function used on the reverse graph."""
        A3h_k = edges.src['A3h']
        e_ik = edges.dst['B1h'] + edges.src['B2h'] + edges.data['B3e']  # e_ik = B_1*h_i + B_2*h_k + B_3*e_ik

        if self.batch_norm:
            e_ik = self.bn_e(e_ik)

        e_ik = F.relu(e_ik)

        if self.residual:
            e_ik = e_ik + edges.data['e']
        
        return {'A3h_k': A3h_k, 'e_ik': e_ik}

    def reduce_backward(self, nodes):
        """Reduce function used on the reverse graph."""
        A3h_k = nodes.mailbox['A3h_k']
        e_ik = nodes.mailbox['e_ik']
        sigma_ik = torch.sigmoid(e_ik)
        h_backward = torch.sum(sigma_ik * A3h_k, dim=1) / (torch.sum(sigma_ik, dim=1) + 1e-6)
        return {'h_backward': h_backward}

    def forward(self, g, h, e):
        """Return updated node representations."""
        h_in = h
        e_in = e

        g.ndata['h'] = h
        g.edata['e'] = e
        g.ndata['A1h'] = self.A_1(h)
        g.ndata['A2h'] = self.A_2(h)
        g.ndata['A3h'] = self.A_3(h)
        g.ndata['B1h'] = self.B_1(h)
        g.ndata['B2h'] = self.B_2(h)
        g.edata['B3e'] = self.B_3(e)

        g_reverse = dgl.reverse(g, copy_ndata=True, copy_edata=True)
        g.update_all(self.message_forward, self.reduce_forward)
        g_reverse.update_all(self.message_backward, self.reduce_backward)

        h = g.ndata['A1h'] + g.ndata['h_forward'] + g_reverse.ndata['h_backward']
        
        if self.batch_norm:
            h = self.bn_h(h)

        h = F.relu(h)

        if self.residual:
            h = h + h_in

        h = F.dropout(h, self.dropout, training=self.training)

        # TODO: I should also return e, since that is also being updated (and residual connection is used)
        return h
