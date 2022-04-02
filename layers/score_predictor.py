import torch
import torch.nn as nn


class ScorePredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Linear(3 * in_features, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        return {'score': self.W(data)}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
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
