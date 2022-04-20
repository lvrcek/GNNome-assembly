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
        #self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        nb_pe = 10 # TODO : make as input argument
        self.linear_pe = nn.Linear(nb_pe, hidden_features) 
        #self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        hidden_edge_features = 16 # TODO : make as input argument
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GraphGatedGCN(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features)

    def forward(self, graph, x, e, pe):
        #x = self.node_encoder(x) # x=1
        x = self.linear_pe(pe) 
        #e = self.edge_encoder(e) # e=[len,sim] default
        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores
