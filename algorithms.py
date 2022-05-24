import os
import math
import pickle
import subprocess
from collections import deque
from datetime import datetime

import dgl
import torch


def assert_strand(graph, walk):
    org_strand = graph.ndata['read_strand'][walk[0]].item()
    for idx, node in enumerate(walk[1:]):
        curr_strand = graph.ndata['read_strand'][node].item()
        if curr_strand != org_strand:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'node index: {node}')


def assert_overlap(graph, walk):
    for idx, (src, dst) in enumerate(zip(walk[:-1], walk[1:])):
        src_start = graph.ndata['read_start'][src].item()
        dst_start = graph.ndata['read_start'][dst].item()
        src_end = graph.ndata['read_end'][src].item()
        dst_end = graph.ndata['read_end'][dst].item()
        src_strand = graph.ndata['read_strand'][src].item()
        dst_strand = graph.ndata['read_strand'][dst].item()
        if src_strand == dst_strand == 1 and dst_start > src_end:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'end: {src_end}, start: {dst_start}')
        if src_strand == dst_strand == -1 and dst_end < src_start:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'end: {src_start}, start: {dst_end}')


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


def dfs(graph, neighbors, start=None, avoid={}):
    if start is None:
        min_value, idx = torch.topk(graph.ndata['read_start'], k=1, largest=False)
        start = idx.item()

    stack = deque()
    stack.append(start)

    visited = [True if i in avoid else False for i in range(graph.num_nodes())]

    path = {start: None}
    max_node = start
    max_value = graph.ndata['read_end'][start]

    try:
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            
            if graph.ndata['read_end'][current] > max_value:
                max_value = graph.ndata['read_end'][current]
                max_node = current

            visited[current] = True
            tmp = []
            for node in neighbors.get(current, []):
                if visited[node]:
                    continue
                if graph.ndata['read_strand'][node] == -1:
                    continue
                if graph.ndata['read_start'][node] > graph.ndata['read_end'][current]:
                    continue
                if graph.ndata['read_start'][node] < graph.ndata['read_start'][current]:
                    continue
                tmp.append(node)

            if len(tmp) == 0:
                for node in neighbors.get(current, []):
                    if visited[node]:
                        continue
                    if graph.ndata['read_strand'][node] == -1:
                        continue
                    if graph.ndata['read_start'][node] < graph.ndata['read_start'][current]:
                        continue
                    if graph.ndata['read_start'][node] > graph.ndata['read_end'][current]:
                        tmp.append(node)

            tmp.sort(key=lambda x: -graph.ndata['read_start'][x])
            for node in tmp:
                stack.append(node)
                path[node] = current

    except KeyboardInterrupt:
        pass
    
    finally:
        walk = []
        current = max_node
        while current is not None:
            walk.append(current)
            current = path[current]
        walk.reverse()
        visited = {i for i in range(graph.num_nodes()) if visited[i]}
        return walk, visited


def get_correct_edges(graph, neighbors, edges, walk):
    pos_str_edges = set()
    neg_str_edges = set()
    for i, src in enumerate(walk[:-1]):
        for dst in walk[i+1:]:
            if dst in neighbors[src] and graph.ndata['read_start'][dst] < graph.ndata['read_end'][src]:
                try:
                    pos_str_edges.add(edges[(src, dst)])
                except KeyError:
                    print('Edge not found in the edge dictionary')
                    raise
                try:
                    neg_str_edges.add(edges[dst^1, src^1])
                except KeyError:
                    print('Negative strand edge not found in the edge dictionary')
                    raise
            else:
                break
    return pos_str_edges, neg_str_edges


def get_gt_graph(graph, neighbors, edges):
    all_nodes = {i for i in range(graph.num_nodes()) if graph.ndata['read_strand'][i] == 1}
    last_node = max(all_nodes, key=lambda x: graph.ndata['read_end'][x])

    largest_visited = -1
    all_walks = []
    pos_correct_edges, neg_correct_edges = set(), set()
    all_visited = set()

    while all_nodes:
        start = min(all_nodes, key=lambda x: graph.ndata['read_start'][x])
        walk, visited = dfs(graph, neighbors, start, avoid=all_visited)
        if graph.ndata['read_end'][walk[-1]] < largest_visited or len(walk) == 1:
            all_nodes = all_nodes - visited
            all_visited = all_visited | visited
            # print(f'\nDiscard component')
            # print(f'Start = {graph.ndata["read_start"][walk[0]]}\t Node = {walk[0]}')
            # print(f'End   = {graph.ndata["read_end"][walk[-1]]}\t Node = {walk[-1]}')
            # print(f'Walk length = {len(walk)}')
            continue
        else:
            largest_visited = graph.ndata['read_end'][walk[-1]]
            all_walks.append(walk)

        # print(f'\nInclude component')
        # print(f'Start = {graph.ndata["read_start"][walk[0]]}\t Node = {walk[0]}')
        # print(f'End   = {graph.ndata["read_end"][walk[-1]]}\t Node = {walk[-1]}')
        # print(f'Walk length = {len(walk)}')

        pos_str_edges, neg_str_edges = get_correct_edges(graph, neighbors, edges, walk)
        pos_correct_edges = pos_correct_edges | pos_str_edges
        neg_correct_edges = neg_correct_edges | neg_str_edges

        if largest_visited == graph.ndata['read_end'][last_node]:
            break
        all_nodes = all_nodes - visited
        all_visited = all_visited | visited

    return pos_correct_edges, neg_correct_edges

