import pickle

import edlib
# import mappy as mp
import torch
import torch.nn as nn

import graph_parser
import algorithms


def anchor(reads, current, aligner):
    """Find where the read is mapped to the reference - DEPRECATED
    
    Parameters
    ----------
    reads : dict
        A dictionary with reads for all the nodes in a graph
    current : int
        Index of the current node in the walk
    aligner : mappy.Aligner
        Minimap aligner used to map the read to the refernce

    Returns
    -------
    int
        position on the reference where the mapping starts
    int
        position on the reference where the mapping ends
    int
        is the read mapped regularly or as its reverse-complement
    """
    sequence = reads[current]
    alignment = aligner.map(sequence)
    hit = list(alignment)[0]
    r_st, r_en, strand = hit.r_st, hit.r_en, hit.strand
    return r_st, r_en, strand


def get_overlap_length(graph, reads, current, neighbor):
    """Get length of the overlap between two reads - DEPRECATED"""
    idx = graph_parser.find_edge_index(graph, current, neighbor)
    overlap_length = len(reads[current]) - graph.ndata['prefix_length'][idx]
    return overlap_length


def get_walks(start, neighbors, num_nodes):
    """Return all the possible walks from a current node of length
    num_nodes.
    
    Parameters
    ----------
    start : int
        Index of the starting node
    neighbors : dict
        Dictionary with the list of neighbors for each node
    num_nodes : int
        Length of the walks to be returned

    Returns
    -------
    list
        a list of all the possible walks, where each walk is also
        stored in a list with num_nodes consecutive nodes
    """
    if num_nodes == 0:
        return [[start]]
    paths = []
    for neighbor in neighbors[start]:
        next_paths = get_walks(neighbor, neighbors, num_nodes-1)
        for path in next_paths:
            path.append(start)
            paths.append(path)
    return paths


def get_edlib_best(idx, graph, reads, current, neighbors, reference_seq, aligner, visited):
    """Get the ground-truth for the next node with edlib - DEPRECATED"""
    ref_start, ref_end, strand = anchor(reads, current, aligner)
    edlib_start = ref_start
    paths = [path[::-1] for path in get_walks(current, neighbors, num_nodes=4)]
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
        

# def get_minimap_best(graph, reads, current, neighbors, walk, aligner):
#     """Get the ground-truth for the next node with minimap - DEPRECATED"""
#     scores = []
#     for neighbor in neighbors[current]:
#         print(f'\tcurrent neighbor {neighbor}')
#         node_tr = walk[-min(3, len(walk)):] + [neighbor]
#         sequence = graph_parser.translate_nodes_into_sequence2(graph, reads, node_tr)
#         ll = min(len(sequence), 50000)
#         sequence = sequence[-ll:]
#         name = '_'.join(map(str, node_tr)) + '.fasta'
#         with open(f'concat_reads/{name}', 'w') as fasta:
#             fasta.write(f'>{name}\n')
#             fasta.write(f'{str(sequence)*10}\n')
#         alignment = aligner.map(sequence)
#         hits = list(alignment)
#         try:
#             quality_score = graph_parser.get_quality(hits, len(sequence))
#         except:
#             quality_score = 0
#         print(f'\t\tquality score:', quality_score)
#         scores.append((neighbor, quality_score))
#     best_neighbor, quality_score = max(scores, key=lambda x: x[1])
#     return best_neighbor


def print_prediction(walk, current, neighbors, actions, choice, best_neighbor):
    """Print summary of the prediction for the current position."""
    print('\n-----predicting-----')
    print('previous:\t', None if len(walk) < 2 else walk[-2])
    print('current:\t', current)
    print('neighbors:\t', neighbors[current])
    print('actions:\t', actions.tolist())
    print('choice:\t\t', choice)
    print('ground truth:\t', best_neighbor)


def process(model, idx, graph, pred, neighbors, reads, reference, edges, optimizer, mode, device='cpu'):
    """Process the graph by predicting the correct next neighbor.
    
    A graph is processed by simulating a walk over it where the 
    best next neighbor is predicted any time a branching occurs.
    The choices are compared tothe ground truth and loss is calculated.
    The list of losses and accuracy for the given graph are returned.

    Parameters
    ----------
    model : torch.nn.Module
        A model which will predict the following node
    idx : int
        Index of the processed graph
    graph : dgl.DGLGraph
        The processed graph
    pred : dict
        A dictionary with predecessors for all the nodes in the graph
    neighbors : dict
        A dictionary with neighbors for all the nodes in the graph
    reads : dict
        A dictionary with reads for all the nodes in the graph
    reference : str
        A path to the reference for the current graph
    optimizer : torch.optim.Optimizer
        An optimizer which will update the model's parameters
    mode : str
        Whether training or evaluation is performed
    device : str, optional
        On which device is the computation performed (cpu/cuda)

    Returns
    -------
    list
        a list of all the losses during processing the graph
    float
        accuracy of the preictions for the given graph
    """
    # start_nodes = [k for k, v in pred.items() if len(v)==0]
    start = 0  # A very naive approach, but good for now

    criterion = nn.CrossEntropyLoss()
    # aligner = mp.Aligner(reference, preset='map_pb', best_n=1)
    # reference_seq = next(SeqIO.parse(reference, 'fasta'))

    current = start
    visited = set()
    walk = []
    loss_list = []
    total_loss = 0
    total = 0
    correct = 0

    logits = model(graph, reads)
    # ground_truth, _ = algorithms.greedy(graph, start, neighbors, option='ground-truth')
    ground_truth = pickle.load(open(f'data/train/solutions/{idx}_gt.pkl', 'rb'))
    total_steps = len(ground_truth) - 1
    steps = 0
    ground_truth = {n1: n2 for n1, n2 in zip(ground_truth[:-1], ground_truth[1:])}
    
    print('Iterating through nodes!')

    while True:
        if steps == total_steps:
            break
        steps += 1
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

        neighbor_edges = [edges[current, n] for n in neighbors[current]]
        neighbor_logits = logits.squeeze(1)[neighbor_edges]
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

    # TODO: Should return the total_loss, not loss_list
    accuracy = correct / total
    return loss_list, accuracy
