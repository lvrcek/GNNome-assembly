import torch
import torch.nn as nn


class EdgeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def normalize(self, tensor):
        tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
        return tensor

    def forward(self, overlap_similarity, overlap_length):
        overlap_similarity = self.normalize(overlap_similarity).unsqueeze(-1)
        overlap_length = self.normalize(overlap_length.float()).unsqueeze(-1)
        e = torch.cat((overlap_similarity, overlap_length), dim=1)
        return self.linear(e)
