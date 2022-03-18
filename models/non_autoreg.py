import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

# import layers
from layers import *
# from hyperparameters import get_hyperparameters
# from layers.node_decoder import NodeDecoder



class GCN(nn.Module):

    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(hidden_features, hidden_features) for _ in range(num_layers)
            ])

    def forward(self, blocks, x):
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](blocks[i], x))
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


class Model(nn.Module):

    def __init__(self, node_features, edge_features, hidden_features, num_layers):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, hidden_features)
        self.edge_encoder = nn.Linear(edge_features, hidden_features)
        self.gnn = GCN(num_layers, hidden_features)
        self.predictor = ScorePredictor(hidden_features)

    def forward(self, edge_subgraph, blocks, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        x = self.gnn(blocks, x)
        scores = self.predictor(edge_subgraph, x, e)
        return scores


# class NonAutoRegressive(nn.Module):
#     """
#     Non-autoregressive model used to predict the best next neighbor.
#     It encodes the entire graph, processes it, and returns the
#     conditional probability for each edge. Consists of sequence
#     encoder (node encoder), edge encoder, a variable number of GatedGCN
#     layers, and an edge decoder.

#     Attributes
#     ----------
#     seq_encoder : torch.nn.Module
#         Module that encodes genomic sequences into vectors
#     edge_encoder : torch.nn.Module
#         Module that encodes the edge information into vectors
#     layers : torch.nn.ModuleList
#         Variable number of GatedGCN layers to obtain node
#         representations
#     decoder : torch.nn.Module
#         Module that decodes node and edge representations and
#         returns conditional probability for each edge in the graph
#     """

#     def __init__(self, dim_latent, num_gnn_layers, encode='node', dim_linear_emb=3, kernel_size=20, num_conv_layers=1):
#         """
#         Parameters
#         ----------
#         dim_latent : int
#             Latent dimensions used for node and edge representations
#         dim_linear_emb : int, optional
#             Dimension of linear embedding used to represent A, C, G,
#             and T in a continuous space
#         kernel_size : int, optional
#             Size of the convolutional kernel used to represent
#             sequences
#         num_conv_layers : int, optional
#             Number of convolutional layers used to represent sequences
#         num_gnn_layers : int, optional
#             Number of GNN layers (in this case GatedGCN) used to obtain
#             node representations
#         """
#         super().__init__()
#         # self.seq_encoder = SequenceEncoder_noCNN(dim_hidden=dim_latent)
#         self.hyperparams = get_hyperparameters()
#         # self.encode = 'none'  # encode
#         # self.node_encoder = NodeEncoder(1, dim_latent)
#         self.edge_encoder = EdgeEncoder(2, dim_latent)
#         self.layers = nn.ModuleList([GatedGCN_1d(dim_latent, dim_latent) for _ in range(num_gnn_layers)])
#         self.decoder = EdgeDecoder(dim_latent, 1)

#     def forward(self, graph, reads, norm=None):
#         """Return the conditional probability for each edge."""
#         self.encode = self.hyperparams['encode']
#         use_reads = self.hyperparams['use_reads']
#         if self.encode == 'sequence' and use_reads:
#             h = self.seq_encoder(reads)
#         elif self.encode == 'node':
#             h = torch.ones((graph.num_nodes(), 1)).to(self.hyperparams['device'])
#             h = self.node_encoder(h)
#         else:
#             h = torch.ones((graph.num_nodes(), self.hyperparams['dim_latent'])).to(self.hyperparams['device'])
#         # h = h.type(torch.float16)

#         # norm = self.hyperparams['norm']
#         if norm is not None:
#             e_tmp = (graph.edata['overlap_length'] - norm[0] ) / norm[1]
#         else:
#             e_tmp = graph.edata['overlap_length'].float() 
#             e_tmp = (e_tmp - torch.mean(e_tmp)) / torch.std(e_tmp)
#         e = self.edge_encoder(graph.edata['overlap_similarity'], e_tmp)
#         #  e = e.type(torch.float16)
#         for conv in self.layers:
#             h, e = conv(graph, h, e)
#             # h = h.type(torch.float16)
#             # e = e.type(torch.float16)
        
#         # This might take a lot of memory
#         # e_f = e.clone()
#         # e_b = e.clone()
#         # p = self.decoder(graph, h, e_f, e_b)

#         p = self.decoder(graph, h, e, e)
#         return p


# class NonAutoRegressive_gt_graph(nn.Module):

#     def __init__(self, dim_latent, num_gnn_layers, encode='node', dim_linear_emb=3, kernel_size=20, num_conv_layers=1):

#         super().__init__()
#         # self.seq_encoder = SequenceEncoder_noCNN(dim_hidden=dim_latent)
#         self.hyperparams = get_hyperparameters()
#         # self.encode = 'none'  # encode
#         # self.node_encoder = NodeEncoder(1, dim_latent)
#         self.edge_encoder = EdgeEncoder(2, dim_latent)
#         self.layers = nn.ModuleList([GatedGCN_1d(dim_latent, dim_latent) for _ in range(num_gnn_layers)])
#         # self.node_decoder = NodeDecoder(dim_latent, 1)
#         self.edge_decoder = EdgeDecoder(dim_latent, 1)

#     def forward(self, graph, reads, norm=None):
#         """Return the conditional probability for each edge."""
#         self.encode = self.hyperparams['encode']
#         use_reads = self.hyperparams['use_reads']
#         if self.encode == 'sequence' and use_reads:
#             h = self.seq_encoder(reads)
#         elif self.encode == 'node':
#             h = torch.ones((graph.num_nodes(), 1)).to(self.hyperparams['device'])
#             h = self.node_encoder(h)
#         else:
#             h = torch.ones((graph.num_nodes(), self.hyperparams['dim_latent'])).to(self.hyperparams['device'])

#         # norm = self.hyperparams['norm']
#         if norm is not None:
#             e_tmp = (graph.edata['overlap_length'] - norm[0] ) / norm[1]
#         else:
#             e_tmp = graph.edata['overlap_length'].float() 
#             e_tmp = (e_tmp - torch.mean(e_tmp)) / torch.std(e_tmp)
#         e = self.edge_encoder(graph.edata['overlap_similarity'], e_tmp)
#         #  e = e.type(torch.float16)
#         for conv in self.layers:
#             h, e = conv(graph, h, e)

#         # p_n = self.node_decoder(graph, h)
#         p_e = self.edge_decoder(graph, h, e, e)
#         return p_e
