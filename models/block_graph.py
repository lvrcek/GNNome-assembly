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
        e_subgraph = self.edge_encoder(e_subgraph)
        h = self.gnn(blocks, h)
        scores = self.predictor(edge_subgraph, h, e_subgraph)
        return scores


class BlockGatedGCNModel(nn.Module):
    """
    This modes uses edge features updated with the stack of GatedGCNs.
    """
    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.gnn =  layers.BlockGatedGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features)

    def forward(self, edge_subgraph, blocks, x, e, e_subgraph):
        h = self.node_encoder(x)
        e = self.edge_encoder(e)
        h, e = self.gnn(blocks, h, e)
        ids = [blocks[-1].edge_ids(src, dst) for src, dst in zip(*edge_subgraph.edges())]  # Find the edges in the last block that you're gonna predict on
        e = e[ids]  # Use feauters of those edges for predictions
        scores = self.predictor(edge_subgraph, h, e)
        return scores


class BlockGatedGCNModel_noEupdate(nn.Module):
    """
    This modes uses non-updated edge features (take the original edge_subgraph features and push them through the encoder).
    """
    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.gnn =  layers.BlockGatedGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features)

    def forward(self, edge_subgraph, blocks, x, e, e_subgraph):
        h = self.node_encoder(x)
        e = self.edge_encoder(e)
        h, e = self.gnn(blocks, h, e)
        e = self.edge_encoder(e_subgraph)
        scores = self.predictor(edge_subgraph, h, e)
        return scores

