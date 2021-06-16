import torch
import torch.nn as nn
import torch.functional as F
# from torch_geometric.utils import add_self_loops
import dgl

from layers import MPNN, EncoderNetwork, DecoderNetwork


class SequentialModel(nn.Module):

    def __init__(self, dim_node, dim_edge, dim_latent, processor_type='MPNN', bias=False):
        super(SequentialModel, self).__init__()
        self.node_encoder = EncoderNetwork(dim_node + dim_latent, dim_latent, bias=bias)
        self.edge_encoder = EncoderNetwork(dim_edge, dim_latent, bias=bias)
        self.processor = MPNN(dim_latent, dim_latent, dim_latent, bias=False)
        self.decoder = DecoderNetwork(2 * dim_latent, 1, bias=bias)

    def forward(self, graph, latent_features, device):
        node_features = graph.ndata['read_length'].clone().to(device) / 20000  # Kind of normalization
        edge_features = graph.edata['overlap_similarity'].clone().to(device)
        edge_index = graph.edges()

        # edge_index, edge_features = add_self_loops(edge_index, edge_weight=edge_features)  # fill_value = 1.0
        node_features = node_features.unsqueeze(-1).float().to(device)
        latent_features = latent_features.float().to(device)
        edge_features = edge_features.unsqueeze(-1).float().to(device)
        t = torch.cat((node_features, latent_features), dim=1).to(device)
        node_enc = self.node_encoder(t).to(device)
        edge_enc = self.edge_encoder(edge_features).to(device)
        print('RIGHT BEFORE SEQ!!!!!!!')
        latent_features = self.processor(graph, node_enc).to(device)
        print('SEQUENTIAL MODEL HERE')
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1)).to(device)
        return output, latent_features
