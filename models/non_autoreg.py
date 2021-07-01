import torch.nn as nn

from layers import GatedGCN, SequenceEncoder, EdgeEncoder, EdgeDecoder


class NonAutoRegressive(nn.Module):
    def __init__(self, dim_latent, dim_linear_emb=3, kernel_size=20, num_conv_layers=1, num_gnn_layers=4):
        super().__init__()
        self.seq_encoder = SequenceEncoder(dim_linear_emb=dim_linear_emb, dim_conv_emb=dim_latent,
                                           kernel_size=kernel_size, num_conv_layers=num_conv_layers)
        self.edge_encoder = EdgeEncoder(2, dim_latent)
        self.layers = nn.ModuleList([GatedGCN(dim_latent, dim_latent) for _ in range(num_gnn_layers)])
        self.decoder = EdgeDecoder(dim_latent, 1)

    def forward(self, graph, reads):
        h = self.seq_encoder(reads)
        e = self.edge_encoder(graph.edata['overlap_similarity'], graph.edata['overlap_length'])
        for conv in self.layers:
            h = conv(graph, h, e)
        p = self.decoder(graph, h, e)
        return p
