import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class CustomGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)

    def forward(self, block, h):
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))

            return self.W(torch.cat(
                [block.dstdata['h'], block.dstdata['h_neigh']], dim=1))  # This cat is bullshit
            

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


class ScorePredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Linear(3 * in_features, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        return {'score': self.W(data)}

    def forward(self, edge_subgraph, x, e):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.edata['e'] = e
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['score']


class BlockModel(nn.Module):

    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, hidden_features)
        self.edge_encoder = nn.Linear(edge_features, hidden_features)
        self.gnn = BlockGCN(num_layers, hidden_features)
        self.predictor = ScorePredictor(hidden_features)

    def forward(self, edge_subgraph, blocks, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        x = self.gnn(blocks, x)
        scores = self.predictor(edge_subgraph, x, e)
        return scores


class GraphModel(nn.Module):

    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, hidden_features)
        self.edge_encoder = nn.Linear(edge_features, hidden_features)
        self.gnn = GraphGCN(num_layers, hidden_features)
        self.predictor = ScorePredictor(hidden_features)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        x = self.gnn(graph, x)
        scores = self.predictor(graph, x, e)
        return scores
