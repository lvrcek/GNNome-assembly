import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqEmbedder(nn.Module):
    def __init__(self, dim_linear, dim_conv, kernel_size, stride=1, bias=False):
        super().__init__()
        self.linear = nn.Linear(4, dim_linear, bias=bias)
        self.conv = nn.Conv1d(dim_linear, dim_conv, kernel_size, stride=stride, bias=bias)

    def forward(self, reads):
        embedded_reads = []
        reads = dict(sorted(reads.items(), key=lambda x: x[0]))
        for idx, read in reads.items():
            read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
            read = ' '.join(read).split()
            read = torch.tensor(list(map(int, read)))
            read = F.one_hot(read)
            read = self.linear(read.float())  # len(read) x dim_linear
            read = read.transpose(1, 0).unsqueeze(0)  # 1 x dim_linear x len(read)
            read = self.conv(read)  # 1 x dim_conv x len(feature_map)
            read = read.squeeze(0)  # dim_conv x len(feature_map)
            read = read.mean(dim=1)  # dim_conv
            embedded_reads.append(read)

        embedded_reads = torch.stack(embedded_reads)
        return embedded_reads    
