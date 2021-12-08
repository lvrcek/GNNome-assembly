import os
import pickle
import subprocess
from collections import deque
from datetime import datetime

import dgl
import torch


def baseline(graph, start, neighbors, preds, edges):

    starts = [k for k,v in preds.items() if len(v)==0 and graph.ndata['read_strand'][k]==1]
    walks = []
    # best_read_idx_walks = []
    for start in starts:
        visited = set()
        current = start
        walk = []
        # read_idx_walk = []

        while current is not None:
            if current in visited:
                break
            walk.append(current)
            visited.add(current)
            visited.add(current ^ 1)
            # read_idx_walk.append(graph.ndata['read_idx'][current].item())
            if len(neighbors[current]) == 0:
                break
            if len(neighbors[current]) == 1:
                current = neighbors[current][0]
                continue
            neighbor_edges = [edges[(current, n)] for n in neighbors[current]]
            neighbor_lengths = graph.edata['overlap_length'][neighbor_edges]
            _, index = torch.topk(neighbor_lengths, k=1, dim=0)
            current = neighbors[current][index]

        walks.append(walk.copy())
        # best_read_idx_walks.append(read_idx_walk.copy())

    longest_walk = max(walks, key=lambda x: len(x))
    # sorted(best_read_idx_walks, key=lambda x: len(x))

    return longest_walk


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


def dfs(graph, neighbors, start=None):
    # TODO: Take only those with in-degree 0
    # TODO: Take only those with strand == 1, this is not working
    if start is None:
        min_value, idx = torch.topk(graph.ndata['read_start'], k=1, largest=False)
        start = idx.item()

    threshold, _ = torch.topk(graph.ndata['read_start'][graph.ndata['read_strand']==1], k=1)
    threshold = threshold.item()

    stack = deque()
    stack.append(start)

    visited = [False for i in range(graph.num_nodes())]
    visited[start] = True

    path = {start: None}
    max_node = start
    max_value = graph.ndata['read_end'][start]

    try:
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            
            if graph.ndata['read_end'][current] == threshold:
                break

            if graph.ndata['read_end'][current] > max_value:
                max_value = graph.ndata['read_end'][current]
                max_node = current

            visited[current] = True
            tmp = []
            for node in neighbors.get(current, []):
                if visited[node]:
                    continue
                if graph.ndata['read_strand'] == -1:
                    continue
                if graph.ndata['read_start'][node] > graph.ndata['read_end'][current]:
                    continue
                tmp.append(node)

            tmp.sort(key=lambda x: -graph.ndata['read_start'][x])
            for node in tmp:
                stack.append(node)
                path[node] = current

        walk = []
        current = max_node
        while current is not None:
            walk.append(current)
            current = path[current]
        walk.reverse()
        return walk


    except KeyboardInterrupt:
        return walk


def dfs_gt_forwards(graph, neighbors, threshold):
    min_value, idx = torch.topk(graph.ndata['read_start'][graph.ndata['read_strand']==1], k=1, largest=False)
    # assert graph.ndata['read_strand'][idx] == 1
    start = idx.item()

    threshold, _ = torch.topk(graph.ndata['read_start'][graph.ndata['read_strand']==1], k=1)
    threshold = threshold.item() // 100 * 100

    execution = deque()
    walk = [start]
    execution.append(walk)
    max_reach = walk.copy()
    time_start = datetime.now()

    try:
        while execution:
            time_now = datetime.now()
            if (time_now-time_start).seconds > 300:
                print(graph.ndata['read_end'][max_reach[-1]])
                # break
            walk = execution.pop()
            visited = set(walk)
            last_node = walk[-1]
            
            if graph.ndata['read_end'][last_node] > graph.ndata['read_end'][max_reach[-1]]:
                max_reach = walk.copy()

            if len(neighbors[last_node]) == 0 and graph.ndata['read_end'][last_node] >= threshold:
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


def dfs_gt_backwards(graph, neighbors, threshold):
    max_value, idx = torch.topk(graph.ndata['read_start'][graph.ndata['read_strand']==-1], k=1)
    # assert graph.ndata['read_strand'][idx] == -1
    start = idx.item()
    threshold, _ = torch.topk(graph.ndata['read_end'][graph.ndata['read_strand']==-1], k=1, largest=False)
    threshold = threshold.item()

    execution = deque()
    walk = [start]
    execution.append(walk)
    max_reach = walk.copy()
    time_start = datetime.now()

    try:
        while execution:
            time_now = datetime.now()
            if (time_now-time_start).seconds > 300:
                print(graph.ndata['read_end'][max_reach[-1]])
                # break
            walk = execution.pop()
            visited = set(walk)
            last_node = walk[-1]

            if graph.ndata['read_end'][last_node] < graph.ndata['read_end'][max_reach[-1]]:
                max_reach = walk.copy()

            if len(neighbors[last_node]) == 0 and graph.ndata['read_end'][last_node] <= threshold:
                break

            tmp = []
            for node in neighbors.get(last_node, []):
                if node in visited:
                    continue
                if graph.ndata['read_strand'][node] == 1:
                    continue
                if graph.ndata['read_start'][node] < graph.ndata['read_end'][last_node]:
                    continue
                tmp.append(node)

            tmp.sort(key=lambda x: graph.ndata['read_start'][x])
            for node in tmp:
                execution.append(walk + [node])

        return max_reach

    except KeyboardInterrupt:
        return max_reach


def get_solutions_for_all(data_path, threshold):
    processed_path = f'{data_path}/processed'
    neighbors_path = f'{data_path}/info'
    solutions_path = f'{data_path}/solutions'
    if not os.path.isdir(solutions_path):
        os.mkdir(solutions_path)
    for name in os.listdir(processed_path):
        idx = name[:-4]
        print(idx)
        graph = dgl.load_graphs(os.path.join(processed_path, name))[0][0]
        neighbors = pickle.load(open(os.path.join(neighbors_path, idx + '_succ.pkl'), 'rb'))
        walk = dfs_gt_forwards(graph, neighbors, threshold=threshold)
        pickle.dump(walk, open(os.path.join(solutions_path, idx + '_gt.pkl'), 'wb'))
