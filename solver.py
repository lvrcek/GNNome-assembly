import os
import random

from Bio import SeqIO
from Bio.Seq import Seq
import edlib
import mappy as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import add_self_loops

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

    @staticmethod
    def anchor(graph, current, aligner):
        if not hasattr(graph, 'batch'):
            sequence = graph.read_sequence[current]
        else:
            sequence = graph.read_sequence[0][current]
        alignment = aligner.map(sequence)
        hit = list(alignment)[0]
        r_st, r_en, strand = hit.r_st, hit.r_en, hit.strand
        return r_st, r_en, strand

    @staticmethod
    def get_overlap_length(graph, current, neighbor):
        idx = graph_parser.find_edge_index(graph, current, neighbor)
        if not hasattr(graph, 'batch'):
            overlap_length = len(graph.read_sequence[current]) - graph.prefix_length[idx]
        else:
            overlap_length = len(graph.read_sequence[0][current]) - graph.prefix_length[idx]
        return overlap_length

    @staticmethod
    def get_suffix(graph, node, overlap_length):
        if not hasattr(graph, 'batch'):
            return graph.read_sequence[node][overlap_length:]
        else:
            return graph.read_sequence[0][node][overlap_length:]

    @staticmethod
    def get_edlib_best(graph, current, neighbors, reference, aligner, visited):
        ref_start, ref_end = ExecutionModel.anchor(graph, current, aligner)
        edlib_start = ref_end
        distances = []
        for neighbor in neighbors[current]:
            overlap_length = ExecutionModel.get_overlap_length(graph, current, neighbor)
            suffix = ExecutionModel.get_suffix(graph, neighbor, overlap_length)
            reference_seq = next(SeqIO.parse(reference, 'fasta'))  # TODO: put this outside of the function
            edlib_end = edlib_start + len(suffix)  # Should I not also put overlap_length here? Nope
            reference_query = reference_seq[edlib_start:edlib_end]
            distance = edlib.align(reference_query, suffix)['editDistance']
            # This is why you don't do edlib with just one node
            # I need at least one or a few more otherwise I won't be able to map anything
            try:
                score = distance / (edlib_end - edlib_start)
            except ZeroDivisionError:
                print('edlib start and end:', edlib_start, edlib_end)
                print('current and neighbor', current, neighbor)
                print('overlap length:', overlap_length)
                print(len(graph.read_sequence[0][current]))
                print(graph.prefix_length[graph_parser.find_edge_index(graph, current, neighbor)])
                print(len(graph.read_sequence[0][neighbor]))
                print('Somehow we are dividing with zero!')
                # TODO: I didn't solve this still!
                # TODO: We divide by zero because the reads appear to be contained, but they are not!
                # TODO: If overlap > len(read_2) just take read 2, not overlap
                raise

            distances.append((neighbor, distance))
        best_neighbor, min_distance = min(distances, key=lambda x: x[1])
        return best_neighbor

    @staticmethod
    def get_edlib_best2(idx, graph, current, neighbors, reference, aligner, visited):
        ref_start, ref_end, strand = ExecutionModel.anchor(graph, current, aligner)
        edlib_start = ref_start
        reference_seq = next(SeqIO.parse(reference, 'fasta'))
        paths = [path[::-1] for path in ExecutionModel.get_paths(current, neighbors, num_nodes=4)]
        distances = []
        for path in paths:
            _, _, next_strand = ExecutionModel.anchor(graph, path[1], aligner)
            if next_strand != strand:
                continue
            sequence = graph_parser.translate_nodes_into_sequence2(graph, path[1:])
            if strand == -1:
                sequence = sequence.reverse_complement()
            # print(len(sequence))
            edlib_start = ref_start + graph.prefix_length[graph_parser.find_edge_index(graph, path[0], path[1])].item()
            edlib_end = edlib_start + len(sequence)
            reference_query = reference_seq[edlib_start:edlib_end]
            distance = edlib.align(reference_query, sequence)['editDistance']
            score = distance / (edlib_end - edlib_start)
            distances.append((path, score))
        try:
            best_path, min_distance = min(distances, key=lambda x: x[1])
            best_neighbor = best_path[1]
            # print(paths)
            # print(distances)
            return best_neighbor
        except ValueError:
            print('\nAll the next neighbors have an opposite strand')
            print('Graph index:', idx)
            print('current:,', current)
            print(paths)
            return None

    @staticmethod
    def get_paths(start, neighbors, num_nodes):
        if num_nodes == 0:
            # new_path = []
            # new_path.append([start])
            return [[start]]
        paths = []
        for neighbor in neighbors[start]:
            next_paths = ExecutionModel.get_paths(neighbor, neighbors, num_nodes-1)
            for path in next_paths:
                path.append(start)
                paths.append(path)
        return paths
            
    @staticmethod
    def get_minimap_best(graph, current, neighbors, walk, aligner):
        scores = []
        for neighbor in neighbors[current]:
            # Get mappings for all the neighbors
            # print(f'walk = {walk}')
            print(f'\tcurrent neighbor {neighbor}')
            node_tr = walk[-min(3, len(walk)):] + [neighbor]
            # print(node_tr)
            ####
            sequence = graph_parser.translate_nodes_into_sequence2(graph, node_tr)
            ll = min(len(sequence), 50000)
            sequence = sequence[-ll:]
            # sequence *= 10
            name = '_'.join(map(str, node_tr)) + '.fasta'
            with open(f'concat_reads/{name}', 'w') as fasta:
                fasta.write(f'>{name}\n')
                fasta.write(f'{str(sequence)*10}\n')
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
            scores.append((neighbor, quality_score))
        best_neighbor, quality_score = max(scores, key=lambda x: x[1])
        return best_neighbor

    @staticmethod
    def get_loss(actions, best_neighbor, criterion, device):
        indices = torch.nonzero(actions).squeeze(-1)
        new_best = torch.nonzero(indices == best_neighbor).item()
        actions = actions[indices].unsqueeze(0)
        best = torch.tensor([new_best]).to(device)
        loss = criterion(actions, best)
        return loss

    @staticmethod
    def get_loss2(actions, best_neighbor, curr_neighbors, criterion, device):
        print(best_neighbor)
        print(curr_neighbors)
        idx = curr_neighbors.index(best_neighbor)
        best = torch.tensor([idx]).to(device)
        loss = criterion(actions, best)
        return loss

    @staticmethod
    def print_prediction(walk, current, neighbors, actions, index, value, choice, best_neighbor):
        print('\n-----predicting-----')
        print('previous:\t', None if len(walk) < 2 else walk[-2])
        print('current:\t', current)
        print('neighbors:\t', neighbors[current])
        print('actions:\t', actions.tolist())
        # print('index:\t\t', index)
        # print('value:\t\t', value)
        print('choice:\t\t', choice)
        print('ground truth:\t', best_neighbor)

    def process(self, idx, graph, pred, succ, reference, optimizer, mode, device='cpu'):
        print('Processing graph!')
        node_features = graph.read_length.clone().detach().to(device) / 20000  # Kind of normalization
        edge_features = graph.overlap_similarity.clone().detach().to(device)
        last_latent = self.processor.zero_hidden(graph.num_nodes)  # TODO: Could this potentially be a problem?
        visited = set()

        # start = random.randint(0, graph.num_nodes - 1)  # TODO: find a better way to start, maybe from pred = []
        start_nodes = list(set(range(graph.num_nodes)) - set(pred.keys()))
        start = start_nodes[0]  # Not really good, maybe iterate over all the start nodes?
        # start = 901  # Test get_edlib_best2 function
        # neighbors = graph_parser.get_neighbors(graph)
        
        neighbors = succ

        current = start
        walk = []
        loss_list = []
        criterion = nn.CrossEntropyLoss()
        # print(os.path.relpath())
        # print(os.path.relpath(__file__))
        # if not os.path.isfile(reference):
        #     raise Exception("Reference does not exist!!")
        aligner = mp.Aligner(reference, preset='map_pb', best_n=1)
        print('Iterating through neighbors!')

        total = 0
        correct = 0

        while True:
            walk.append(current)
            if current in visited:
                break
            visited.add(current)  # current node
            visited.add(current ^ 1)  # virtual pair of the current node
            try:
                if len(neighbors[current]) == 0:
                    break
            except KeyError:
                print(current)
                raise
            if len(neighbors[current]) == 1:
                # if not, start with anchoring and probing
                current = neighbors[current][0]
                continue

            # TODO: Maybe put masking before predictions, mask node features and edges?
            mask = torch.tensor([1 if n in neighbors[current] else 0 for n in range(graph.num_nodes)]).to(device)

            # Get prediction for the next node out of those in list of neighbors (run the model)
            predict_actions, last_latent = self.predict(node_features=node_features, edge_features=edge_features,
                                           latent_features=last_latent, edge_index=graph.edge_index, device=device)

            # old_actions = predict_actions.squeeze(1) * mask
            actions = predict_actions.squeeze(1)[neighbors[current]]

            value, index = torch.topk(actions, k=1, dim=0)  # For evaluation

            best_score = -1
            best_neighbor = -1

            choice = neighbors[current][index]

            # We are out of the chain, so, this is the first node with more than 1 successor
            # Which means this node is an anchor
            # I can start with probing now - do edlib stuff here
            # -----------------------

            best_neighbor = ExecutionModel.get_edlib_best2(idx, graph, current, neighbors, reference, aligner, visited)
            # best_neighbor = ExecutionModel.get_minimap_best(graph, current, neighbors, walk, aligner)

            # ExecutionModel.print_prediction(walk, current, neighbors, actions, index, value, choice, best_neighbor)

            if best_neighbor is None:
                break

            # Evaluate your choice - calculate loss
            # Might need to modify loss for batch_size > 1
            # loss = ExecutionModel.get_loss2(actions, best_neighbor, neighbors[current], criterion, device)
            loss = criterion(actions.unsqueeze(0), index.to(device))
            loss_list.append(loss.item())

            # Teacher forcing
            current = best_neighbor

            if mode == 'train':
                # Update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            index = index.item()
            if choice == best_neighbor:
                correct += 1
            total += 1

        accuracy = correct / total
        return loss_list, accuracy

    def predict(self, node_features, edge_features, latent_features, edge_index, device):
        # print('\n\t-----inside NN-----')
        # print('\tedge features:\t', edge_features)
        edge_index, edge_features = add_self_loops(edge_index, edge_weight=edge_features)  # fill_value = 1.0
        # print('\tedge features:\t', edge_features)
        # print('\tnode features:\t', node_features)
        node_features = node_features.unsqueeze(-1).float().to(device)
        latent_features = latent_features.float().to(device)
        # print('\tlatent before:\t', latent_features)
        edge_features = edge_features.unsqueeze(-1).float().to(device)
        t = torch.cat((node_features, latent_features), dim=1).to(device)
        node_enc = self.node_encoder(t).to(device)
        # print('\tnode encoded:\t', node_enc)
        edge_enc = self.edge_encoder(edge_features).to(device)
        # print('\tedge encoded:\t', edge_enc)
        latent_features = self.processor(node_enc, edge_enc, edge_index).to(device)  # Should I put clone here?
        # print('\tlatent after:\t', latent_features)
        output = self.decoder(torch.cat((node_enc, latent_features), dim=1)).to(device)
        # print('\toutput:\t\t', output)
        return output, latent_features


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
