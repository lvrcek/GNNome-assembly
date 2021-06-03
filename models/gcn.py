import torch
import torch.nn as nn
import torch.functional as F
import torch_geometric.nn as nng
from torch_geometric.utils import add_self_loops

from layers import MPNN, EncoderNetwork, DecoderNetwork


class GCNModel(nn.Module):

    def __init__(self, dim_node, dim_edge, dim_latent, bias=False):
        super(GCNModel, self).__init_()
        pass

    def forward(self, graph, latent_features, device):
        pass
