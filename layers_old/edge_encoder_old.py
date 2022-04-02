import torch
import torch.nn as nn

from hyperparameters import get_hyperparameters


class EdgeEncoder(nn.Module):
    """
    Module that normalizes and encodes edge attributes
    (overlap length and similarity) into vectors.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer used to encode the edge attributes
    """

    def __init__(self, in_channels, out_channels, bias=True):
        """
        Parameters:
        in_channels : int
            Dimension of the input vectors
        out_channels : int
            Dimension of the output (encoded) vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def normalize(self, tensor):
        """Normalize the tensor to mean = 0, std = 1."""
        tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
        return tensor

    def forward(self, overlap_similarity, overlap_length):
        """Return the encoded edge attributes."""
        overlap_similarity = overlap_similarity.unsqueeze(-1)  # Can't normalize if all are 1.0!
        # overlap_length = self.normalize(overlap_length.float()).unsqueeze(-1)
        overlap_length = overlap_length.unsqueeze(-1)
        e = torch.cat((overlap_similarity, overlap_length), dim=1)
        return self.linear(e)


class EdgeEncoder_overlap(nn.Module):
    """
    Module that normalizes and encodes edge attributes
    (overlap length and similarity) into vectors.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear layer used to encode the edge attributes
    """

    def __init__(self, in_channels, out_channels, bias=True):
        """
        Parameters:
        in_channels : int
            Dimension of the input vectors
        out_channels : int
            Dimension of the output (encoded) vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.embedding = nn.Embedding(4, out_channels)

    def normalize(self, tensor):
        """Normalize the tensor to mean = 0, std = 1."""
        tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
        return tensor

    def forward(self, overlap_similarity, overlap_length, reads, edges, graph):
        device = get_hyperparameters()['device']
        """Return the encoded edge attributes."""
        # overlap_length = self.normalize(overlap_length.float()).unsqueeze(-1)
        overlap_length = overlap_length.unsqueeze(-1)

        edge_sequences = []
        for key, value in edges.items():
            src, dst = key
            edge_idx = value
            prefix = graph.edata['prefix_length'][edge_idx]
            seq = reads[src][prefix:]
            # edge_sequences[edge_idx] = seq

            read = seq.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
            read = ' '.join(read).split()
            read = torch.tensor(list(map(int, read)), device=device)
            read = self.embedding(read)
            read = torch.mean(read, dim=0)
            edge_sequences.append(read)

        edge_sequences = torch.stack(edge_sequences)
        return

        # e = torch.cat((overlap_similarity, overlap_length), dim=1)
        # return self.linear(e)
