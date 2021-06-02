import torch
import torch.nn as nn
import torch.functional as F
import torch_geometric.nn as nng
from torch_geometric.utils import add_self_loops

from layers import MPNN, EncoderNetwork, DecoderNetwork


class GCNModel(nn.Module):

    def __init__(self, node_features, edge_features, latent_features, processor_type='MPNN', bias=False):
        super(GCNModel, self).__init_()
        self.gcn1 = nng.GCNConv(in_channels=node_features, out_channels=32)
        self.gcn2 = nng.GCNConv(in_channels=32, out_channels=2)
        pass

    def forward(self, node_features, edge_features, latent_features, edge_index, device):
        pass
