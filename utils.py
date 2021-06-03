import math

from Bio import SeqIO
import edlib
import mappy as mp
import torch
import torch.nn as nn

import graph_parser
from hyperparameters import get_hyperparameters


def anchor(graph, current, aligner):
    if not hasattr(graph, 'batch'):
        sequence = graph.read_sequence[current]
    else:
        sequence = graph.read_sequence[0][current]
    alignment = aligner.map(sequence)
    hit = list(alignment)[0]
    r_st, r_en, strand = hit.r_st, hit.r_en, hit.strand
    return r_st, r_en, strand


def get_overlap_length(graph, current, neighbor):
    idx = graph_parser.find_edge_index(graph, current, neighbor)
    if not hasattr(graph, 'batch'):
        overlap_length = len(graph.read_sequence[current]) - graph.prefix_length[idx]
    else:
        overlap_length = len(graph.read_sequence[0][current]) - graph.prefix_length[idx]
    return overlap_length


def get_suffix(graph, node, overlap_length):
    if not hasattr(graph, 'batch'):
        return graph.read_sequence[node][overlap_length:]
    else:
        return graph.read_sequence[0][node][overlap_length:]


def get_paths(start, neighbors, num_nodes):
    if num_nodes == 0:
        return [[start]]
    paths = []
    for neighbor in neighbors[start]:
        next_paths = get_paths(neighbor, neighbors, num_nodes-1)
        for path in next_paths:
            path.append(start)
            paths.append(path)
    return paths


def get_edlib_best(idx, graph, current, neighbors, reference_seq, aligner, visited):
    ref_start, ref_end, strand = anchor(graph, current, aligner)
    edlib_start = ref_start
    paths = [path[::-1] for path in get_paths(current, neighbors, num_nodes=4)]
    distances = []
    for path in paths:
        _, _, next_strand = anchor(graph, path[1], aligner)
        if next_strand != strand:
            continue
        sequence = graph_parser.translate_nodes_into_sequence2(graph, path[1:])
        if strand == -1:
            sequence = sequence.reverse_complement()
        edlib_start = ref_start + graph.prefix_length[graph_parser.find_edge_index(graph, path[0], path[1])].item()
        edlib_end = edlib_start + len(sequence)
        reference_query = reference_seq[edlib_start:edlib_end]
        distance = edlib.align(reference_query, sequence)['editDistance']
        score = distance / (edlib_end - edlib_start)
        distances.append((path, score))
    try:
        best_path, min_distance = min(distances, key=lambda x: x[1])
        best_neighbor = best_path[1]
        return best_neighbor
    except ValueError:
        print('\nAll the next neighbors have an opposite strand')
        print('Graph index:', idx)
        print('current:,', current)
        print(paths)
        return None
        

def get_minimap_best(graph, current, neighbors, walk, aligner):
    scores = []
    for neighbor in neighbors[current]:
        print(f'\tcurrent neighbor {neighbor}')
        node_tr = walk[-min(3, len(walk)):] + [neighbor]
        sequence = graph_parser.translate_nodes_into_sequence2(graph, node_tr)
        ll = min(len(sequence), 50000)
        sequence = sequence[-ll:]
        name = '_'.join(map(str, node_tr)) + '.fasta'
        with open(f'concat_reads/{name}', 'w') as fasta:
            fasta.write(f'>{name}\n')
            fasta.write(f'{str(sequence)*10}\n')
        alignment = aligner.map(sequence)
        hits = list(alignment)
        try:
            quality_score = graph_parser.get_quality(hits, len(sequence))
        except:
            quality_score = 0
        print(f'\t\tquality score:', quality_score)
        scores.append((neighbor, quality_score))
    best_neighbor, quality_score = max(scores, key=lambda x: x[1])
    return best_neighbor


def print_prediction(walk, current, neighbors, actions, choice, best_neighbor):
    print('\n-----predicting-----')
    print('previous:\t', None if len(walk) < 2 else walk[-2])
    print('current:\t', current)
    print('neighbors:\t', neighbors[current])
    print('actions:\t', actions.tolist())
    print('choice:\t\t', choice)
    print('ground truth:\t', best_neighbor)


def process(model, idx, graph, pred, neighbors, reference, optimizer, mode, device='cpu'):
    hyperparameters = get_hyperparameters()
    dim_latent = hyperparameters['dim_latent']
    last_latent = torch.zeros((graph.num_nodes, dim_latent)).to(device).detach()
    start_nodes = list(set(range(graph.num_nodes)) - set(pred.keys()))
    start = start_nodes[0]  # TODO: Maybe iterate over all the start nodes?

    criterion = nn.CrossEntropyLoss()
    aligner = mp.Aligner(reference, preset='map_pb', best_n=1)
    reference_seq = next(SeqIO.parse(reference, 'fasta'))

    current = start
    visited = set()
    walk = []
    loss_list = []
    total = 0
    correct = 0
    print('Iterating through nodes!')

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
            current = neighbors[current][0]
            continue

        mask = torch.tensor([1 if n in neighbors[current] else -math.inf for n in range(graph.num_nodes)]).to(device)

        # Get prediction for the next node out of those in list of neighbors (run the model)
        predict_actions, last_latent = model(graph, latent_features=last_latent, device=device)
        actions = predict_actions.squeeze(1)[neighbors[current]]
        value, index = torch.topk(actions, k=1, dim=0)  # For evaluation
        choice = neighbors[current][index]

        # Branching found - find the best neighbor with edlib
        best_neighbor = get_edlib_best(idx, graph, current, neighbors, reference_seq, aligner, visited)
        print_prediction(walk, current, neighbors, actions, choice, best_neighbor)
        if best_neighbor is None:
            break

        # Calculate loss
        # TODO: Modify for batch_size > 1
        loss = criterion(actions.unsqueeze(0), index.to(device))
        loss_list.append(loss.item())

        # Update weights
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if choice == best_neighbor:
            correct += 1
        total += 1

        # Teacher forcing
        current = best_neighbor

    accuracy = correct / total
    return loss_list, accuracy
