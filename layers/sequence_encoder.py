import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import get_hyperparameters


class SequenceEncoder(nn.Module):
    """
    Module that encodes the genomic sequences into vectors.
    
    It consists of a linear embedding layer that first represents
    the sequences in a continuous space, and of a vairable number
    of convolutional layers that extract information from the
    sequences.

    Attributes
    ----------
    embedding : torch.nn.Embedding
        Module that represents the sequences in a continuous space
    conv1 : torch.nn.Conv1
        First CNN layer that extracts information from the sequences
    conv_rest : torch.nn.ModuleList
        List of the additional CNN layers, where number if input and
        output channels is the same
    weighted : bool
        Flag indicating whether to use weighted mean of the obtained
        feature maps
    W_q : torch.nn.Linear
        Trainable querry matrix, used in weighted mean
    W_k : torch.nn.Linear
        Trainable key matrix, used in weighted mean
    W_v : torch.nn.Linear
        Trainable value matrix, used in weighted mean
    """

    def __init__(self, dim_linear_emb, dim_conv_emb, kernel_size, num_conv_layers, weighted=True):
        """
        Parameters
        ----------
        dim_linear_emb : int
            Dimension of linear embedding used to represent A, C, G,
            and T in a continuous space
        dim_conv_emb : int
            Dimension of the output channels of the CNN layers
        kernel_size : int
            Size of the convolutional kernel used to represent
            sequences
        num_conv_layers : int
            Number of CNN layers used
        weighted : bool, optional
            Flag indicating whether to use weighted mean
        """
        super().__init__()
        self.embedding = nn.Embedding(4, dim_linear_emb)
        self.conv1 = nn.Conv1d(dim_linear_emb, dim_conv_emb, kernel_size)
        self.conv_rest = nn.ModuleList([nn.Conv1d(dim_conv_emb, dim_conv_emb, kernel_size) 
                                        for _ in range(num_conv_layers - 1)])
        self.weighted = weighted

        self.W_q = nn.Linear(dim_conv_emb, dim_conv_emb)
        self.W_k = nn.Linear(dim_conv_emb, dim_conv_emb)
        self.W_v = nn.Linear(dim_conv_emb, dim_conv_emb)

    def weighted_mean(self, read):
        """Calculate the weighted mean based on the local attention."""
        read_mean = read.mean(dim=1).unsqueeze(0)  # 1 x dim_conv_emb
        read = read.transpose(1, 0)  # len x dim_conv_emb
        q = self.W_q(read_mean)  # 1 x dim_conv_emb
        k = self.W_k(read)  # len x dim_conv_emb
        v = self.W_v(read)  # len x dim_conv_emb
        h = (q @ k.t()) / math.sqrt(q.shape[1])  # 1 x len
        h = F.softmax(h, dim=1)
        h = h @ v  # 1 x dim_conv_emb
        h = h.squeeze(0)  # dim_conv_emb
        return h     

    def forward(self, reads):
        """Return the list of the encoded reads."""
        device = get_hyperparameters()['device']
        embedded_reads = []
        reads = dict(sorted(reads.items(), key=lambda x: x[0]))
        for idx, read in reads.items():
            read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
            read = ' '.join(read).split()
            read = torch.tensor(list(map(int, read)), device=device)
            read = self.embedding(read)  # len(read) x dim_linear_emb
            read = read.transpose(1, 0).unsqueeze(0)  # 1 x dim_linear_emb x len(read)
            read = self.conv1(read)  # 1 x dim_conv_emb x len(feature_map)
            if len(self.conv_rest) > 0:
                for conv in self.conv_rest:
                    read = conv(read)
            read = read.squeeze(0)  # dim_conv_emb x len(feature_map)
            if self.weighted:
                h = self.weighted_mean(read)  # dim_conv_emb
            else:
                h = read.mean(dim=1)  # dim_conv_emb
            embedded_reads.append(h)

        embedded_reads = torch.stack(embedded_reads)
        return embedded_reads

        
class SequenceEncoder_noCNN(nn.Module):

    def __init__(self, dim_hidden):
        super().__init__()
        self.embedding = nn.Embedding(4, dim_hidden)

    def forward(self, reads):
        """Return the list of the encoded reads."""
        device = get_hyperparameters()['device']
        embedded_reads = []
        reads = dict(sorted(reads.items(), key=lambda x: x[0]))
        for idx, read in reads.items():
            # read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
            # read = ' '.join(read).split()
            # read = torch.tensor(list(map(int, read)), device=device)
            read = self.embedding(read)  # len(read) x dim_linear_emb
            read = torch.mean(read, dim=0)
            embedded_reads.append(read)

        embedded_reads = torch.stack(embedded_reads)
        return embedded_reads
