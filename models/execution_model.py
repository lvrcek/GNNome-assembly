import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import mappy as mp
from Bio.Seq import Seq
import torch.optim as optim

from layers import MPNN, EncoderNetwork, DecoderNetwork
import graph_parser


class ExecutionModel(nn.Module):

    def __init__(self, node_features, edge_features, latent_features, processor_type='MPNN', bias=False):
        super(ExecutionModel, self).__init__()
        # TODO: Create different kinds of processors - MPNN, GAT, PNA, ...
        self.node_encoder = EncoderNetwork(node_features + latent_features, latent_features, bias=bias)
        self.edge_encoder = EncoderNetwork(edge_features, latent_features, bias=bias)
        self.processor = MPNN(latent_features, latent_features, latent_features, bias=False)
        self.decoder = DecoderNetwork(2 * latent_features, latent_features, bias=bias)

    def process(self, graph, optimizer, device='cpu'):
        print('Processing graph!')
        node_features = graph.read_length.clone().detach().to(device)
        edge_features = graph.overlap_similarity.clone().detach().to(device)
        last_latent = self.processor.zero_hidden(graph.num_nodes)  # TODO: Could this potentially be a problem?

        start = random.randint(0, graph.num_nodes - 1)
        neighbors = graph_parser.get_neighbors(graph)
        current = start
        walk = []
        reference = '../data/references/lambda_reference.fasta'
        aligner = mp.Aligner(reference, preset='map_ont', best_n=5)
        print('Iterating through neighbors!')

        while True:
            print(f'\nCurrent node: {current}')
            walk.append(current)
            if len(neighbors[current]) == 0:
                break

            # TODO: Maybe put masking before predictions, mask node features and edges?
            mask = torch.tensor([1 if n in neighbors[current] else 0 for n in range(graph.num_nodes)])

            # Get prediction for the next node out of those in list of neighbors (run the model)
            predict_actions = self.predict(node_features=node_features, edge_features=edge_features,
                                           latent_features=last_latent, edge_index=graph.edge_index)

            actions = predict_actions.squeeze(1) * mask
            # value, index = torch.topk(actions, k=1, dim=0)  # Check dimensions!
            # TODO: Do I really need topk since I am performing teacher forcing?
            best_score = -1
            best_neighbor = -1

            # ---- GET CORRECT -----
            print('neighbors:', neighbors[current])
            for neighbor in neighbors[current]:
                # Get mappings for all the neighbors
                print(f'walk = {walk}')
                print(f'current neighbor {neighbor}')
                node_tr = walk[-min(3, len(walk)):] + [neighbor]
                sequence = graph_parser.translate_nodes_into_sequence2(graph, node_tr)
                alignment = aligner.map(sequence)
                hits = list(alignment)
                print(f'length node_tr: {len(node_tr)}')
                print(f'length hits: {len(hits)}')
                try:
                    quality_score = graph_parser.get_quality(hits, len(sequence))
                except:
                    quality_score = 0
                if quality_score > best_score:
                    best_neighbor = neighbor
                    best_score = quality_score
            # ----------------------

            # I take the best neighbor out of reference, which is teacher forcing - good or not?
            current = best_neighbor

            # Evaluate your choice - calculate loss
            criterion = nn.CrossEntropyLoss()
            actions = actions.unsqueeze(0)  # Dimensions need to be batch_size x number_of_actions
            best = torch.tensor([best_neighbor])
            loss = criterion(actions, best)

            # Update weights
            # TODO: Probably I will need to accumulate losses and then do backprop once I'm done with the graph
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, node_features, edge_features, latent_features, edge_index):
        node_features = node_features.unsqueeze(-1).float()
        latent_features = latent_features.float()
        edge_features = edge_features.unsqueeze(-1).float()
        t = torch.cat((node_features, latent_features), dim=1)
        node_enc = self.node_encoder(t)
        edge_enc = self.edge_encoder(edge_features)
        latent_features = self.processor(node_enc, edge_enc, edge_index)  # Should I put clone here?
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1))
        return output


if __name__ == '__main__':
    graph_nx, graph_torch = graph_parser.from_csv('../data/debug/lambda/graph_before.csv')
    model = ExecutionModel(1, 1, 1)
    params = list(model.parameters())
    opt = optim.Adam(params, lr=1e-5)
    model.process(graph_torch, opt)
