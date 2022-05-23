import dgl
import torch.nn as nn
import torch.nn.functional as F

import layers


class GraphGatedGCN(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GatedGCN_1d(hidden_features, hidden_features, batch_norm) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
            # h = F.relu(h)
            # e = F.relu(e)
        return h, e

