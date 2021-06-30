import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    def __init__(self, dim_linear, dim_conv, kernel_size, stride=1, weighted=True):
        super().__init__()
        self.embedding = nn.Embedding(4, dim_linear)
        self.conv = nn.Conv1d(dim_linear, dim_conv, kernel_size, stride=stride)
        self.weighted = weighted

        self.W_q = nn.Linear(dim_conv, dim_conv)
        self.W_k = nn.Linear(dim_conv, dim_conv)
        self.W_v = nn.Linear(dim_conv, dim_conv)

    def weighted_mean(self, read):
        read_mean = read.mean(dim=1).unsqueeze(0)  # 1 x dim_conv
        read = read.transpose(1, 0)  # len x dim_conv
        q = self.W_q(read_mean)  # 1 x dim_conv
        k = self.W_k(read)  # len x dim_conv
        v = self.W_v(read)  # len x dim_conv
        h = (q @ k.t()) / math.sqrt(q.shape[1])  # 1 x len
        h = F.softmax(h, dim=1)
        h = h @ v  # 1 x dim_conv
        h = h.squeeze(0)  # dim_conv
        return h     

    def forward(self, reads):
        embedded_reads = []
        reads = dict(sorted(reads.items(), key=lambda x: x[0]))
        for idx, read in reads.items():
            read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
            read = ' '.join(read).split()
            read = torch.tensor(list(map(int, read)))
            read = self.embedding(read)  # len(read) x dim_linear
            read = read.transpose(1, 0).unsqueeze(0)  # 1 x dim_linear x len(read)
            read = self.conv(read)  # 1 x dim_conv x len(feature_map)
            read = read.squeeze(0)  # dim_conv x len(feature_map)
            if self.weighted_mean:
                h = self.weighted_mean(read)  # dim_conv
            else:
                h = read.mean(dim=1)  # dim_conv
            embedded_reads.append(h)

        embedded_reads = torch.stack(embedded_reads)
        return embedded_reads    
