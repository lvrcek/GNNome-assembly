import dgl
import torch.nn as nn
import torch.nn.functional as F

import layers


# Full graph processors
class GraphGCN(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(hidden_features, hidden_features) for _ in range(num_layers)
        ])

    def forward(self, graph, x):
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](graph, x))
        return x


class GraphGatedGCN(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GatedGCN_1d(hidden_features, hidden_features) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
            # h = F.relu(h)
            # e = F.relu(e)
        return h, e


# Block graph processors
class BlockGCN(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(hidden_features, hidden_features) for _ in range(num_layers)
        ])

    def forward(self, blocks, x):
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](blocks[i], x))
        return x


class BlockGatedGCN(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GatedGCN_forwards(hidden_features, hidden_features) for _ in range(num_layers)
        ])

    def forward(self, blocks, h, e):
        for i in range(len(self.convs)):
            e = e[:blocks[i].num_edges()]
            h, e = self.convs[i](blocks[i], h, e)
            # h = F.relu(h)
            # e = F.relu(e)
        return h, e
