from algorithms import ground_truth
import math

from Bio import SeqIO
import edlib
import mappy as mp
import torch
import torch.nn as nn

import graph_parser
import models
import algorithms
from hyperparameters import get_hyperparameters


def anchor(reads, current, aligner):
    sequence = reads[current]
    alignment = aligner.map(sequence)
    hit = list(alignment)[0]
    r_st, r_en, strand = hit.r_st, hit.r_en, hit.strand
    return r_st, r_en, strand


def get_overlap_length(graph, reads, current, neighbor):
    idx = graph_parser.find_edge_index(graph, current, neighbor)
    overlap_length = len(reads[current]) - graph.prefix_length[idx]
    return overlap_length


def get_suffix(reads, node, overlap_length):
    return reads[node][overlap_length:]


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


def get_edlib_best(idx, graph, reads, current, neighbors, reference_seq, aligner, visited):
    ref_start, ref_end, strand = anchor(reads, current, aligner)
    edlib_start = ref_start
    paths = [path[::-1] for path in get_paths(current, neighbors, num_nodes=4)]
    distances = []
    for path in paths:
        _, _, next_strand = anchor(reads, path[1], aligner)
        if next_strand != strand:
            continue
        sequence = graph_parser.translate_nodes_into_sequence2(graph, reads, path[1:])
        if strand == -1:
            sequence = sequence.reverse_complement()
        edlib_start = ref_start + graph.edata['prefix_length'][graph_parser.find_edge_index(graph, path[0], path[1])].item()
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
        

def get_minimap_best(graph, reads, current, neighbors, walk, aligner):
    scores = []
    for neighbor in neighbors[current]:
        print(f'\tcurrent neighbor {neighbor}')
        node_tr = walk[-min(3, len(walk)):] + [neighbor]
        sequence = graph_parser.translate_nodes_into_sequence2(graph, reads, node_tr)
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


def process(model, idx, graph, pred, neighbors, reads, reference, optimizer, mode, device='cpu'):
    hyperparameters = get_hyperparameters()
    dim_latent = hyperparameters['dim_latent']
    last_latent = torch.zeros((graph.num_nodes(), dim_latent)).to(device).detach()
    start_nodes = list(set(range(graph.num_nodes())) - set(pred.keys()))
    start = start_nodes[0]  # TODO: Maybe iterate over all the start nodes?

    criterion = nn.CrossEntropyLoss()
    aligner = mp.Aligner(reference, preset='map_pb', best_n=1)
    reference_seq = next(SeqIO.parse(reference, 'fasta'))

    current = start
    visited = set()
    walk = []
    loss_list = []
    total_loss = 0
    total = 0
    correct = 0

    cond_prob = model(graph, reads)
    ground_truth, _ = algorithms.ground_truth(graph, start, neighbors)
    ground_truth = {n1: n2 for n1, n2 in zip(ground_truth[:-1], ground_truth[1:])}
    
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

        # Currently not used, but could be used for calculating loss
        # mask = torch.tensor([1 if n in neighbors[current] else -math.inf for n in range(graph.num_nodes())]).to(device)

        neighbor_edges = [graph_parser.find_edge_index(graph, current, n) for n in neighbors[current]]
        neighbor_logits = cond_prob.squeeze(1)[neighbor_edges]
        value, index = torch.topk(neighbor_logits, k=1, dim=0)
        choice = neighbors[current][index]

        best_neighbor = ground_truth[current]
        best_idx = neighbors[current].index(best_neighbor)

        print_prediction(walk, current, neighbors, neighbor_logits, choice, best_neighbor)

        best_idx = torch.tensor([best_idx], dtype=torch.long, device=device)
        loss = criterion(neighbor_logits.unsqueeze(0), best_idx)

        # Calculate loss
        if mode == 'train':
            current = best_neighbor
            total_loss += loss

        if choice == best_neighbor:
            correct += 1
        total += 1

        # Teacher forcing
        current = best_neighbor

    if mode == 'train':
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    accuracy = correct / total
    return loss_list, accuracy
