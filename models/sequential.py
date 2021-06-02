import torch
import torch.nn as nn
import torch.functional as F
from torch_geometric.utils import add_self_loops

from layers import MPNN, EncoderNetwork, DecoderNetwork


class SequentialModel(nn.Module):

    def __init__(self, node_features, edge_features, latent_features, processor_type='MPNN', bias=False):
        super(SequentialModel, self).__init__()
        self.node_encoder = EncoderNetwork(node_features + latent_features, latent_features, bias=bias)
        self.edge_encoder = EncoderNetwork(edge_features, latent_features, bias=bias)
        self.processor = MPNN(latent_features, latent_features, latent_features, bias=False)
        self.decoder = DecoderNetwork(2 * latent_features, 1, bias=bias)

    def forward(self, node_features, edge_features, latent_features, edge_index, device):
        # print('\n\t-----inside NN-----')
        # print('\tedge features:\t', edge_features)
        edge_index, edge_features = add_self_loops(edge_index, edge_weight=edge_features)  # fill_value = 1.0
        # print('\tedge features:\t', edge_features)
        # print('\tnode features:\t', node_features)
        node_features = node_features.unsqueeze(-1).float().to(device)
        latent_features = latent_features.float().to(device)
        # print('\tlatent before:\t', latent_features)
        edge_features = edge_features.unsqueeze(-1).float().to(device)
        t = torch.cat((node_features, latent_features), dim=1).to(device)
        node_enc = self.node_encoder(t).to(device)
        # print('\tnode encoded:\t', node_enc)
        edge_enc = self.edge_encoder(edge_features).to(device)
        # print('\tedge encoded:\t', edge_enc)
        latent_features = self.processor(node_enc, edge_enc, edge_index).to(device)  # Should I put clone here?
        # print('\tlatent after:\t', latent_features)
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1)).to(device)
        # print('\toutput:\t\t', output)
        return output, latent_features
