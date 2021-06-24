import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.utils import Sequential

from layers import GatedGCN, SeqEmbedder, EdgeDecoder


class NonAutoRegressive(nn.Module):
    def __init__(self, dim_latent, kernel_size):
        super().__init__()
        self.seq_encoder = SeqEmbedder(4, dim_latent, kernel_size)
        self.edge_encoder = nn.Linear(1, dim_latent)
        self.gcn_1 = GatedGCN(dim_latent, dim_latent,)  # non-linearity included in the GatedGCN (ReLU)
        self.gcn_2 = GatedGCN(dim_latent, dim_latent)
        self.gcn_3 = GatedGCN(dim_latent, dim_latent)
        self.gcn_4 = GatedGCN(dim_latent, dim_latent)
        self.decoder = EdgeDecoder(dim_latent, 1)

    def forward(self, graph, reads):
        h = self.seq_encoder(reads)
        e = self.edge_encoder(graph.edata['overlap_similarity'].unsqueeze(-1))
        h = self.gcn_1(graph, h, e)
        h = self.gcn_2(graph, h, e)
        h = self.gcn_3(graph, h, e)
        h = self.gcn_4(graph, h, e)
        p = self.decoder(graph, h, e)
        return p
