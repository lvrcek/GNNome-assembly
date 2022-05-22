import torch
import torch.nn as nn


class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores):
        super().__init__()
        #self.W = nn.Linear(3 * in_features, 1)
        self.W1 = nn.Linear(3 * in_features, hidden_edge_scores) 
        self.W2 = nn.Linear(hidden_edge_scores, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        #score = self.W(data) 
        h = self.W1(data) 
        h = torch.relu(h) 
        score = self.W2(h) 
        return {'score': score}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


# To go with GatedGCN_2d for two-directional edge representation updates
class ScorePredictor_2d(nn.Module):
    def __init__(self, in_features, hidden_edge_scores):
        super().__init__()
        #self.W = nn.Linear(3 * in_features, 1)
        self.W1 = nn.Linear(4 * in_features, hidden_edge_scores)
        self.W2 = nn.Linear(hidden_edge_scores, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e_f'], edges.data['e_b']), dim=1)
        #score = self.W(data)
        h = self.W1(data)
        h = torch.relu(h)
        score = self.W2(h)
        return {'score': score}

    def forward(self, graph, x, e_f, e_b):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e_f'] = e_f
            graph.edata['e_b'] = e_b
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class ScorePredictorNoEdge(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Linear(2 * in_features, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x']), dim=1)
        return {'score': self.W(data)}

    def forward(self, graph, x):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
