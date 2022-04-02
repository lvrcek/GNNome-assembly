import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import layers


class GraphGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.gnn = layers.GraphGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        x = self.gnn(graph, x)
        scores = self.predictor(graph, x, e)
        return scores


class GraphGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, hidden_features)
        self.edge_encoder = nn.Linear(edge_features, hidden_features)
        self.gnn = layers.GraphGatedGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores
