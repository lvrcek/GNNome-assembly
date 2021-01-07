import random
from collections import defaultdict
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import mappy as mp
from Bio.Seq import Seq

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

    @staticmethod
    def get_neighbors(graph):
        neighbor_dict = defaultdict(list)
        for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
            neighbor_dict[src.item()].append(dst.item())
        return neighbor_dict

    @staticmethod
    def get_predecessors(graph):
        predecessor_dict = defaultdict(list)
        for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
            predecessor_dict[dst].append(src)
        return predecessor_dict

    # @staticmethod
    # def translate_nodes_into_sequence(graph, node_tr):
    #     seq = graph[node_tr[0]]
    #     for src, dst in zip(node_tr[:-1], node_tr[1:]):
    #         idx = graph_parser.find_edge_index(graph, src, dst)
    #         overlap_length = graph.overlap_length[idx]
    #         seq += graph[dst][overlap_length:]
    #     return seq

    @staticmethod
    def get_quality(blen, mlen):
        print(mlen, blen, mlen/blen)
        return mlen/blen

    def process(self, graph, optimizer=None):
        print('Starting process function')
        # TODO: Function which will take the whole graph and then process it node by node
        num_steps = graph.num_nodes

        overlap_lengths = graph.overlap_length.clone()
        overlap_similarity = graph.overlap_similarity.clone()

        node_features = graph.read_length.clone()
        edge_features = graph.overlap_similarity.clone()

        last_latent = self.processor.zero_hidden(graph.num_nodes)
        start = random.randint(0, graph.num_nodes - 1)
        neighbors = self.get_neighbors(graph)
        current = start
        walk = []
        reference = '../data/references/lambda_reference.fasta'
        aligner = mp.Aligner(reference, preset='map_ont', best_n=1)
        while True:
            print('\nEntering while loop!')
            print(f'current = {current}')
            walk.append(current)
            if len(neighbors[current]) == 0:
                break
            mask = torch.tensor([1 if n in neighbors[current] else 0 for n in range(graph.num_nodes)])
            # Get prediction for the next node out of those in list of neighbors (run the model)
            predict_actions = self.predict(node_features=node_features, edge_features=edge_features,
                                           latent_features=last_latent, edge_index=graph.edge_index)

            actions = predict_actions.squeeze(1) * mask
            value, index = torch.topk(actions, k=1, dim=0)  # Check dimensions!
            best_score = -1
            best_neighbor = -1
            # ---- GET CORRECT -----
            print('neighbors:', neighbors[current])
            for neighbor in neighbors[current]:
                # Get mappings for all the neighbors
                print(f'walk = {walk}')
                print(f'current neighbor {neighbor}')
                node_tr = walk[-min(3, len(walk)):] + [neighbor]
                #   print('entering translate func')
                sequence = graph_parser.translate_nodes_into_sequence(graph, node_tr)
                # sequence = graph.read_sequence[walk[-1]]
                print(walk[-1])
                alignment = aligner.map(sequence)
                # print(sequence)
                hits = list(alignment)
                # print(hits)
                try:
                    hit = hits[0]
                    blen = int(hit.blen)
                    mlen = int(hit.mlen)
                    quality_score = self.get_quality(blen, mlen)
                except:
                    quality_score = 0
                if quality_score > best_score:
                    best_neighbor = neighbor
                    best_score = quality_score
            # ----------------------
            current = best_neighbor
            # Evaluate your choice - calculate loss
            criterion = nn.CrossEntropyLoss()
            correct = torch.zeros_like(actions)
            correct[best_neighbor] = 1.
            actions = F.softmax(actions, dim=0)
            actions = actions.unsqueeze(0)
            # TODO: remove this
            best_neighbor = 5

            loss = criterion(actions, torch.tensor([best_neighbor]))

            # Update weights
            loss.backward()
            print('DONE')

    def predict(self, node_features, edge_features, latent_features, edge_index):
        # Here I will probably again do some kind of masking
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
    graph_nx, graph_torch = graph_parser.from_csv('../data/debug/graph_before.csv')
    model = ExecutionModel(1, 1, 1)
    model.process(graph_torch)
