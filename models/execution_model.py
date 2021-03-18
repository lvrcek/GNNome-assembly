import os
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
        self.decoder = DecoderNetwork(2 * latent_features, 1, bias=bias)

    def process(self, graph, optimizer, mode, device='cpu'):
        print('Processing graph!')
        node_features = graph.read_length.clone().detach().to(device)
        edge_features = graph.overlap_similarity.clone().detach().to(device)
        last_latent = self.processor.zero_hidden(graph.num_nodes)  # TODO: Could this potentially be a problem?
        visited = set()

        start = random.randint(0, graph.num_nodes - 1)
        neighbors = graph_parser.get_neighbors(graph)
        current = start
        walk = []
        loss_list = []
        reference = 'data/references/chm13/chr20.fasta'
        # criterion = nn.CrossEntropyLoss()
        # print(os.path.relpath())
        # print(os.path.relpath(__file__))
        # if not os.path.isfile(reference):
        #     raise Exception("Reference does not exist!!")
        # else:
        #     raise Exception("Reference exists!")
        aligner = mp.Aligner(reference, preset='map_pb', best_n=5)
        print('Iterating through neighbors!')

        total = 0
        correct = 0

        while True:
            # print(f'\nCurrent node: {current}')
            walk.append(current)
            if current in visited:
                break
            visited.add(current)
            total += 1
            if len(neighbors[current]) == 0:
                break
            if len(neighbors[current]) == 1:
                current = neighbors[current][0]
                continue

            # TODO: Maybe put masking before predictions, mask node features and edges?
            mask = torch.tensor([1 if n in neighbors[current] else 0 for n in range(graph.num_nodes)]).to(device)

            # Get prediction for the next node out of those in list of neighbors (run the model)
            predict_actions = self.predict(node_features=node_features, edge_features=edge_features,
                                           latent_features=last_latent, edge_index=graph.edge_index, device=device)

            # print(mask.shape)
            # print(predict_actions.shape)
            actions = predict_actions.squeeze(1) * mask

            # --- loss aggregation? ----
            # neighborhood_losses = {}
            # for idx, action in enumerate(actions):
            #     if action > 0:
            #         neighborhood_losses[idx] = action

            value, index = torch.topk(actions, k=1, dim=0)  # For evaluation
            best_score = -1
            best_neighbor = -1

            # ---- GET CORRECT -----
            print('previous:', None if len(walk)<2 else walk[-2])
            print('current:', current)
            print('neighbors:', neighbors[current])
            for neighbor in neighbors[current]:
                # Get mappings for all the neighbors
                # print(f'walk = {walk}')
                print(f'\tcurrent neighbor {neighbor}')
                node_tr = walk[-min(3, len(walk)):] + [neighbor]
                ####
                sequence = graph_parser.translate_nodes_into_sequence2(graph, node_tr)
                ll = min(len(sequence), 50000)
                sequence = sequence[-ll:]
                sequence *= 10
                name = '_'.join(map(str, node_tr)) + '.fasta'
                with open(f'concat_reads/{name}', 'w') as fasta:
                    fasta.write(f'>{name}\n')
                    fasta.write(f'{str(sequence)}\n')
                ####
                alignment = aligner.map(sequence)
                hits = list(alignment)
                # print(f'length node_tr: {len(node_tr)}')
                # print(f'length hits: {len(hits)}')
                try:
                    quality_score = graph_parser.get_quality(hits, len(sequence))
                except:
                    quality_score = 0
                print(f'\t\tquality score:', quality_score)
                if quality_score > best_score:
                    best_neighbor = neighbor
                    best_score = quality_score
            # ----------------------

            # I take the best neighbor out of reference, which is teacher forcing - good or not?
            # So far probably good, later I will do DFS-like search
            print('chosen:', best_neighbor)
            current = best_neighbor

            # Evaluate your choice - calculate loss
            criterion = nn.CrossEntropyLoss()

            actions = actions.unsqueeze(0)  # Dimensions need to be batch_size x number_of_actions
            best = torch.tensor([best_neighbor]).to(device)
            loss = criterion(actions, best)
            loss_list.append(loss.item())

            if mode == 'train':
                # Update weights
                # TODO: Probably I will need to accumulate losses and then do backprop once I'm done with the graph
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            index = index.item()
            if index == best_neighbor:
                correct += 1

        accuracy = correct / total
        return loss_list, accuracy

    def predict(self, node_features, edge_features, latent_features, edge_index, device):
        node_features = node_features.unsqueeze(-1).float().to(device)
        latent_features = latent_features.float().to(device)
        edge_features = edge_features.unsqueeze(-1).float().to(device)
        t = torch.cat((node_features, latent_features), dim=1).to(device)
        node_enc = self.node_encoder(t).to(device)
        edge_enc = self.edge_encoder(edge_features).to(device)
        latent_features = self.processor(node_enc, edge_enc, edge_index).to(device)  # Should I put clone here?
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1)).to(device)
        return output


if __name__ == '__main__':
    graph_nx, graph_torch = graph_parser.from_csv('../data/debug/lambda/graph_before.csv')
    model = ExecutionModel(1, 1, 1)
    params = list(model.parameters())
    opt = optim.Adam(params, lr=1e-5)
    losses, acc = model.process(graph_torch, opt, 'train')
    print('losses:', losses)
    print('accuracy:', acc)
    # accuracy = model.process(graph_torch, opt, 'eval')
    # print('accuracy:', accuracy)
