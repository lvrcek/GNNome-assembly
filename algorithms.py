import dgl
import torch

from graph_parser import find_edge_index


def greedy_ground_truth(graph, current, neighbors, visited):
    """Get the ground truth neighbor for a given node.

    Functions that compares the starting positions of reads for all the
    neighboring nodes to the current node, and returns the neighbor
    whose starting read position is the closest to the current node.
    It also takes into account that the strand of the nieghboring node
    has to be the same as of the current node, and that the neighboring
    node hasn't been visited before.

    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph on which the computation is performed
    current : int
        Index of the current node for which the best neighbor is found
    neighbors : dict
        A dictionary with a list of neighbors for each node
    visited : set
        A set of all the previsouly visited nodes
    
    Returns
    -------
    int
        index of the best neighbor
    """
    candidates = []
    for neighbor in neighbors[current]:
        if neighbor in visited:
            continue
        if graph.ndata['read_strand'][neighbor] != graph.ndata['read_strand'][current]:
            continue
        candidates.append((neighbor, abs(graph.ndata['read_start'][neighbor] - graph.ndata['read_start'][current])))
    candidates.sort(key=lambda x: x[1])
    choice = candidates[0][0] if len(candidates) > 0 else None
    return choice


def greedy_baseline(graph, current, neighbors):
    """Return the best neighbor for the greedy baseline scenario.

    Greedy algorithm that takes the best neighbor while taking into
    account the overlap similarity and overlap length. It chosses
    the neighbor with the highest similarity, and if the similarities
    are the same, then prefers the one with the lower overlap length.

    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph on which the computation is performed
    current : int
        Index of the current node for which the best neighbor is found
    neighbors : dict
        A dictionary with a list of neighbors for each node
    
    Returns
    -------
    int
        index of the best neighbor
    """
    candidates = []
    for neighbor in neighbors[current]:
        idx = find_edge_index(graph, current, neighbor)
        candidates.append((neighbor, graph.edata['overlap_similarity'][idx], graph.edata['overlap_length'][idx]))
    candidates.sort(key=lambda x: (-x[1], x[2]))
    choice = candidates[0][0] if len(candidates) > 0 else None
    return choice


def greedy_decode(graph, current, neighbors):
    """Return the best neighbor for the greedy decoding scenario.

    Greedy algorithm that takes the best neighbor while taking into
    account only the conditional probabilities obtained from the
    network. Useful for inference.

    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph on which the computation is performed
    current : int
        Index of the current node for which the best neighbor is found
    neighbors : dict
        A dictionary with a list of neighbors for each node
    
    Returns
    -------
    int
        index of the best neighbor
    """
    candidates = []
    for neighbor in neighbors[current]:
        candidates.append((neighbor, graph.ndata['p']))
    candidates.sort(key=lambda x: x[1], reverse=True)
    choice = candidates[0][0] if len(candidates) > 0 else None
    return choice


def greedy(graph, start, neighbors, option):
    """Greedy walk over the graph starting from the given node.

    Greedy algorithm that specifies the best neighbor according to a
    certain criterium, which is specified in option. Option can be
    either 'ground-truth', 'baseline', or 'decode'. This way algorithm
    creates a walk, starting from the start node and until the dead end
    is found of all the neighbors have already been visited. It returns
    two lists, first one is walk where node IDs are given, the second
    is the same walk but with read IDs instead of node IDs.

    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph on which the computation is performed
    start : int
        Index of the starting node.
    neighbors : dict
        A dictionary with a list of neighbors for each node
    option : str
        A string which specifies which criterium to take
        (can be 'ground-truth', 'baseline' or 'decode')

    Returns
    -------
    list
        a walk with node IDs given in the order of visiting
    list
        a walk with read IDs given in the order of visiting
    """
    visited = set()
    current = start
    walk = []
    read_idx_walk = []

    assert option in ('ground-truth', 'baseline', 'decode'), \
        "Argument option has to be 'ground-truth', 'baseline', or 'decode'"

    while current is not None:
        if current in visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        read_idx_walk.append(graph.ndata['read_idx'][current].item())
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


def assert_strand(graph, walk):
    for idx, node in enumerate(walk):
        strand = graph.ndata['read_strand'][node].item()
        if strand == -1:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'node index: {node}')


def assert_overlap(graph, walk):
    for idx, (src, dst) in enumerate(zip(walk[:-1], walk[1:])):
        start = graph.ndata['read_start'][dst].item()
        end = graph.ndata['read_end'][src].item()
        if start > end:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'start: {start}, end: {end}')


def to_csv(name):
    graph = dgl.load_graphs(f'data/graphs_1.0/processed/{name}.dgl')[0][0]
    with open(f'test_cases/{name}.csv', 'w') as f:
        f.write('node_id,read_strand,read_start,read_end\n')
        for n in range(graph.num_nodes()):
            strand = graph.ndata['read_strand'][n].item()
            start = graph.ndata['read_start'][n].item()
            end = graph.ndata['read_end'][n].item()
            f.write(f'{n},{strand},{start},{end}\n')




