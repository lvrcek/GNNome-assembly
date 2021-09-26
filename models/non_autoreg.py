import torch
import torch.nn as nn

from layers import GatedGCN, SequenceEncoder, EdgeEncoder, EdgeDecoder, NodeEncoder
from hyperparameters import get_hyperparameters


class NonAutoRegressive(nn.Module):
    """
    Non-autoregressive model used to predict the best next neighbor.
    It encodes the entire graph, processes it, and returns the
    conditional probability for each edge. Consists of sequence
    encoder (node encoder), edge encoder, a variable number of GatedGCN
    layers, and an edge decoder.

    Attributes
    ----------
    seq_encoder : torch.nn.Module
        Module that encodes genomic sequences into vectors
    edge_encoder : torch.nn.Module
        Module that encodes the edge information into vectors
    layers : torch.nn.ModuleList
        Variable number of GatedGCN layers to obtain node
        representations
    decoder : torch.nn.Module
        Module that decodes node and edge representations and
        returns conditional probability for each edge in the graph
    """

    def __init__(self, dim_latent, num_gnn_layers, encode='node', dim_linear_emb=3, kernel_size=20, num_conv_layers=1):
        """
        Parameters
        ----------
        dim_latent : int
            Latent dimensions used for node and edge representations
        dim_linear_emb : int, optional
            Dimension of linear embedding used to represent A, C, G,
            and T in a continuous space
        kernel_size : int, optional
            Size of the convolutional kernel used to represent
            sequences
        num_conv_layers : int, optional
            Number of convolutional layers used to represent sequences
        num_gnn_layers : int, optional
            Number of GNN layers (in this case GatedGCN) used to obtain
            node representations
        """
        super().__init__()
        # self.seq_encoder = SequenceEncoder(dim_linear_emb=dim_linear_emb, dim_conv_emb=dim_latent,
        #                                     kernel_size=kernel_size, num_conv_layers=num_conv_layers)
        self.hyperparams = get_hyperparameters()
        # self.encode = 'none'  # encode
        # self.node_encoder = NodeEncoder(1, dim_latent)
        self.edge_encoder = EdgeEncoder(2, dim_latent)
        self.layers = nn.ModuleList([GatedGCN(dim_latent, dim_latent) for _ in range(num_gnn_layers)])
        self.decoder = EdgeDecoder(dim_latent, 1)

    def forward(self, graph, reads, norm=None):
        """Return the conditional probability for each edge."""
        self.encode = 'none'
        if self.encode == 'sequence':
            h = self.seq_encoder(reads)
        elif self.encode == 'node':
            h = torch.ones((graph.num_nodes(), 1)).to(self.hyperparams['device'])
            # h = self.node_encoder(h)
        else:
            h = torch.ones((graph.num_nodes(), self.hyperparams['dim_latent'])).to(self.hyperparams['device'])

        norm = None
        if norm is not None:
            e_tmp = (graph.edata['overlap_length'] - norm[0] ) / norm[1]
        else:
            e_tmp = graph.edata['overlap_length'].float() 
            e_tmp = (e_tmp - torch.mean(e_tmp)) / torch.std(e_tmp)
        e = self.edge_encoder(graph.edata['overlap_similarity'], e_tmp)
        for conv in self.layers:
            h, e = conv(graph, h, e)
        p = self.decoder(graph, h, e)
        return p

