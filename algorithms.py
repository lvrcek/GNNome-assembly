import os
import pickle
import subprocess
from collections import deque

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


def greedy_baseline(graph, current, neighbors, edges):
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
        idx = edges[(current, neighbor)]
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


def greedy(graph, start, neighbors, edges, option):
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
            current = greedy_baseline(graph, current, neighbors, edges)
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


def to_csv(name, root, print_strand=True, save_dir='test_cases'):
    graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(f'{save_dir}/{name}_info.csv', 'w') as f:
        if print_strand:
            f.write('node_id,read_strand,read_start,read_end\n')
        else:
            f.write('node_id,read_start,read_end\n')
        for n in range(graph.num_nodes()):
            strand = graph.ndata['read_strand'][n].item()
            start = graph.ndata['read_start'][n].item()
            end = graph.ndata['read_end'][n].item()
            if print_strand:
                f.write(f'{n},{strand},{start},{end}\n')
            else:
                if strand == 1:
                    f.write(f'{n},{start},{end}\n')


def to_positive_pairwise(name, root, save_dir='test_cases'):
    graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(f'{save_dir}/{name}_edges.txt', 'w') as f:
        f.write('src\tdst\n')
        for src, dst in zip(graph.edges()[0], graph.edges()[1]):
            src = src.item()
            dst = dst.item()
            if graph.ndata['read_strand'][src] == 1 and graph.ndata['read_strand'][dst] == 1:
                f.write(f'{src}\t{dst}\n')


def get_solutions_for_all_cpp(root, save_dir='test_cases'):
    processed_path = os.path.join(root, 'processed')

    for filename in os.listdir(processed_path):
        name = filename[:-4]
        print(f'Finding walk for... {name}')
        to_csv(name, root, print_strand=False, save_dir=save_dir)
        to_positive_pairwise(name, root, save_dir=save_dir)
        subprocess.run(f'./longestContinuousSequence {name}_info.csv > {name}.out', shell=True, cwd=save_dir)
        with open(f'{save_dir}/{name}.out') as f:
           lines = f.readlines()
           walk = lines[-1].strip().split(' -> ')
           walk = list(map(int, walk))
           pickle.dump(walk, open(f'{save_dir}/{name}_walk.pkl', 'wb'))


def interval_union(name, root):
    graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
    intervals = []
    for strand, start, end in zip(graph.ndata['read_strand'], graph.ndata['read_start'], graph.ndata['read_end']):
        if strand.item() == 1:
            intervals.append([start.item(), end.item()])
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]

    for interval in intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)

    return result


def dfs(graph, start, neighbors):
    visited = set()
    stack = deque()
    stack.append(start)
    walk = []

    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        walk.append(curr)
        stack.extend(neighbors.get(curr, []))

    return walk


def dfs_gt(graph, start, neighbors, threshold):
    execution = deque()
    walk = [start]
    execution.append(walk)
    max_reach = walk.copy()

    try:
        while execution:
            walk = execution.pop()
            visited = set(walk)
            last_node = walk[-1]
            
            if graph.ndata['read_end'][last_node] > graph.ndata['read_end'][max_reach[-1]]:
                max_reach = walk.copy()

            if len(neighbors[last_node]) == 0 and graph.ndata['read_end'][last_node] > threshold:
                break

            tmp = []
            for node in neighbors.get(last_node, []):
                if node in visited:
                    continue
                if graph.ndata['read_strand'][node] == -1:
                    continue
                if graph.ndata['read_start'][node] > graph.ndata['read_end'][last_node]:
                    continue
                tmp.append(node)
            
            tmp.sort(key=lambda x: -graph.ndata['read_start'][x])
            for node in tmp:
                execution.append(walk + [node])

        return max_reach
    
    except KeyboardInterrupt:
        return max_reach


def get_solutions_for_all():
    processed_path = 'data/train/processed'
    neighbors_path = 'data/train/info'
    solutions_path = 'data/train/solutions'
    start = 0
    for name in os.listdir(processed_path):
        idx = name[:-4]
        print(idx)
        graph = dgl.load_graphs(os.path.join(processed_path, name))[0][0]
        neighbors = pickle.load(open(os.path.join(neighbors_path, idx + '_succ.pkl'), 'rb'))
        walk = dfs_gt(graph, 0, neighbors, threshold=1995000)
        pickle.dump(walk, open(os.path.join(solutions_path, idx + '_gt.pkl'), 'wb'))

