import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import layers


class BlockGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.gnn = layers.BlockGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features)

    def forward(self, edge_subgraph, blocks, x, e, e_subgraph):
        h = self.node_encoder(x)
        e = self.edge_encoder(e)
        h = self.gnn(blocks, h)
        scores = self.predictor(edge_subgraph, h, e_subgraph)
        return scores


class BlockGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.gnn =  layers.BlockGatedGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictorNoEdge(hidden_features)

    def forward(self, edge_subgraph, blocks, x, e, e_subgraph):
        h = self.node_encoder(x)
        e = self.edge_encoder(e)
        h, e = self.gnn(blocks, h, e)
        # e = e[:edge_subgraph.num_nodes()]  # TODO: This will not work. It works for blocks, but not for edge_subgraph
        scores = self.predictor(edge_subgraph, h)
        return scores
