import torch
import torch.nn as nn
import torch.functional as F
import torch_geometric.nn as nng
from torch_geometric.utils import add_self_loops

from layers import MPNN, EncoderNetwork, DecoderNetwork


class GCNModel(nn.Module):

    def __init__(self, dim_node, dim_edge, dim_latent, bias=False):
        super(GCNModel, self).__init__()
        self.embedder = nng.Sequential('x, edge_index', [(nng.GCNConv(1, 4), 'x, edge_index -> x'),
                                                         (nng.GCNConv(4, 8), 'x, edge_index -> x'),
                                                         (nng.GCNConv(8, 16), 'x, edge_index -> x'),
                                                         (nng.GCNConv(16, 32), 'x, edge_index -> x')])
        self.classifier = nn.Sequential(nn.Linear(32, 8),
                                        nn.Linear(8, 1))


    def forward(self, graph, latent_features, device, mode):
        if mode == 'embed':
            x = graph.read_length.clone().to(device) / 20000
            x = x.unsqueeze(-1).float()
            edge_index = graph.edge_index
            return self.embedder(x, edge_index)
        else:
            return self.classifier(latent_features)
