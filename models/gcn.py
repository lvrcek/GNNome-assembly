import torch
import torch.nn as nn
import torch.functional as F
from dgl.nn import GraphConv

from layers import MPNN, EncoderNetwork, DecoderNetwork


class GCNModel(nn.Module):

    def __init__(self, dim_node, dim_edge, dim_latent, bias=False):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(dim_node, dim_latent, allow_zero_in_degree=True)
        self.classifier = nn.Linear(dim_latent, 1)

    def forward(self, graph, latent_features, device, mode):
        node_features = graph.ndata['read_length'].clone().unsqueeze(-1).to(device) / 20000  # Kind of normalization
        # edge_features = graph.edata['overlap_similarity'].clone().unsqueeze(-1).to(device)
        if mode == 'embed':
            return self.conv1(graph, node_features)
        else:
            return self.classifier(latent_features)
