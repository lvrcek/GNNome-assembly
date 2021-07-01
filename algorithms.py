from graph_parser import find_edge_index


def greedy_ground_truth(graph, current, neighbors, visited):
    candidates = []
    for neighbor in neighbors[current]:
        if neighbor in visited:
            continue
        if graph.ndata['read_strand'][neighbor] != graph.ndata['read_strand'][current]:
            continue
        candidates.append((neighbor, abs(graph.ndata['read_start'][neighbor] - graph.ndata['read_start'][current])))
    candidates.sort(key=lambda x: x[1])
    choice = candidates[0][0]
    return choice

def greedy_baseline(graph, current, neighbors):
    candidates = []
    for neighbor in neighbors[current]:
        idx = find_edge_index(graph, current, neighbor)
        candidates.append((neighbor, graph.edata['overlap_similarity'][idx], graph.edata['overlap_length'][idx]))
    candidates.sort(key=lambda x: (-x[1], x[2]))
    choice = candidates[0][0]
    return choice


def greedy_decode(graph, current, neighbors):
    candidates = []
    for neighbor in neighbors[current]:
        candidates.append((neighbor, graph.ndata['p']))
    candidates.sort(key=lambda x: x[1], reverse=True)
    choice = candidates[0][0]
    return choice


# def ground_truth(graph, start, neighbors):
#     walk = []
#     read_idx_walk = []
#     current = start
#     visited = set()

#     while True:
#         walk.append(current)
#         visited.add(current)
#         visited.add(current ^ 1)
#         read_idx_walk.append(graph.ndata['read_idx'][current])
#         candidates = []
#         if len(neighbors[current]) == 0:
#             break
#         if len(neighbors[current]) == 1:
#             current = neighbors[current][0]
#             continue
#         for neighbor in neighbors[current]:
#             # -------------------------
#             if neighbor in visited:
#                 continue
#             if graph.ndata['read_strand'][neighbor] != graph.ndata['read_strand'][current]:
#                 continue
#             candidates.append((neighbor, abs(graph.ndata['read_start'][neighbor] - graph.ndata['read_start'][current])))

#         candidates.sort(key=lambda x: x[1])
#         current = candidates[0][0]

#     return walk, read_idx_walk


# def baseline_greedy(graph, start, neighbors):
#     visited = set()
#     current = start
#     walk = []

#     while True:
#         if current in visited:
#             break
#         walk.append(current)
#         visited.add(current)
#         visited.add(current ^ 1)
#         if len(neighbors[current]) == 0:
#             break
#         if len(neighbors[current]) == 1:
#             current = neighbors[current][0]
#             continue
#         current = do_baseline(graph, current, neighbors)

#     return walk


def greedy(graph, start, neighbors, option):
    visited = set()
    current = start
    walk = []
    read_idx_walk = []

    assert option in ('ground-truth', 'baseline', 'decode'), \
        "Argument option has to be either 'ground-truth', 'baseline', or 'decode'"

    while True:
        if current in visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        read_idx_walk.append(graph.ndata['read_idx'][current])
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue
        if option == 'ground-truth':
            current = greedy_ground_truth(graph, current, neighbors, visited)
        if option == 'baseline':
            current = greedy_baseline(graph, current, neighbors)
        if option == 'decode':
            current = greedy_decode(graph, current, neighbors)

    return walk, read_idx_walk
