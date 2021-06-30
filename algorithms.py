import torch
import dgl

from graph_parser import find_edge_index


def ground_truth(graph, start, neighbors):
    walk = []
    read_idx_walk = []
    current = start
    visited = set()

    while True:
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        read_idx_walk.append(graph.ndata['read_idx'][current])
        candidates = []
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue
        for neighbor in neighbors[current]:
            if neighbor in visited:
                continue
            if graph.ndata['read_strand'][neighbor] != graph.ndata['read_strand'][current]:
                continue
            candidates.append((neighbor, abs(graph.ndata['read_start'][neighbor] - graph.ndata['read_start'][current])))
            # candidates.append((neighbor, graph.ndata['read_idx'][neighbor], graph.ndata['read_strand'][neighbor], 
            #                    graph.ndata['read_start'][neighbor], graph.ndata['read_end'][neighbor]))
            # idx = find_edge_index(graph, current, neighbor)
            # candidates.append((neighbor, graph.edata['overlap_similarity'][idx], graph.edata['prefix_length'][idx]))
        candidates.sort(key=lambda x: x[1])
        current = candidates[0][0]

    return walk, read_idx_walk


def baseline_greedy(graph, start, neighbors):
    visited = set()
    current = start
    walk = []

    while True:
        if current in visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        candidates = []
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue
        for neighbor in neighbors[current]:
            idx = find_edge_index(graph, current, neighbor)
            candidates.append((neighbor, graph.edata['overlap_similarity'][idx], graph.edata['overlap_length'][idx]))
        candidates.sort(key=lambda x: (-x[1], x[2]))
        current = candidates[0][0]

    return walk


def decode_greedy(graph, start, neighbors):
    visited = set()
    current = start
    walk = []

    while True:
        if current in visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        candidates = []
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue
        for neighbor in neighbors[current]:
            candidates.append((neighbor, graph.ndata['p']))
        candidates.sort(key=lambda x: x[1], reverse=True)
        current = candidates[0][0]

    return walk
