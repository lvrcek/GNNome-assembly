import os
import re
from collections import deque, defaultdict

import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from matplotlib import pyplot as plt
from Bio.Seq import Seq


def draw_graph(graph_nx):
    nx.draw(graph_nx, node_size=6, width=.2, arrowsize=3)
    plt.show()


def get_neighbors(graph):
    neighbor_dict = defaultdict(list)
    for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
        neighbor_dict[src.item()].append(dst.item())
    return neighbor_dict


def get_predecessors(graph):
    predecessor_dict = defaultdict(list)
    for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
        predecessor_dict[dst.item()].append(src.item())
    return predecessor_dict


def find_edge_index(graph, src, dst):
    for idx, (node1, node2) in enumerate(zip(graph.edge_index[0], graph.edge_index[1])):
        if node1 == src and node2 == dst:
            return idx


def translate_nodes_into_sequence(graph, node_tr):
    seq = graph.read_sequence[node_tr[0]]
    # print(node_tr)
    for src, dst in zip(node_tr[:-1], node_tr[1:]):
        idx = find_edge_index(graph, src, dst)
        overlap_length = graph.overlap_length[idx]
        # print(graph.read_sequence[dst][overlap_length:])
        seq += graph.read_sequence[dst][overlap_length:]
    return seq


def translate_nodes_into_sequence2(graph, node_tr):
    seq = ''
    # seq = graph.read_sequence[node_tr[0]]
    # print(node_tr)
    for src, dst in zip(node_tr[:-1], node_tr[1:]):
        idx = find_edge_index(graph, src, dst)
        prefix_length = graph.overlap_length[idx]
        # print(graph.read_sequence[dst][overlap_length:])
        seq += graph.read_sequence[src][:prefix_length]
    seq += graph.read_sequence[node_tr[-1]]
    return seq


def from_gfa(graph_path):
    read_sequences = deque()
    with open(graph_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 5:
                tag, name, sequence, length, count = line
                sequence = Seq(sequence)
                read_sequences.append(sequence)
                read_sequences.append(sequence.reverse_complement())
            else:
                break
    return read_sequences


def from_csv(graph_path):
    graph_nx = nx.DiGraph()
    node_lengths = {}
    node_data = {}
    edge_ids, edge_lengths, edge_similarities = {}, {}, {}
    read_sequences = from_gfa(graph_path[:-3] + 'gfa')
    with open(graph_path) as f:
        for line in f.readlines():
            src, dst, flag, overlap = line.strip().split(',')
            src, dst = src.split(), dst.split()
            flag = int(flag)
            pattern = r':(\d+)'
            src_id, src_len = int(src[0]), int(re.findall(pattern, src[2])[0])
            dst_id, dst_len = int(dst[0]), int(re.findall(pattern, dst[2])[0])

            if flag == 0:
                if src_id not in node_lengths.keys():
                    node_lengths[src_id] = src_len
                    node_data[src_id] = read_sequences.popleft()
                    graph_nx.add_node(src_id)
                if dst_id not in node_lengths.keys():
                    node_lengths[dst_id] = dst_len
                    node_data[dst_id] = read_sequences.popleft()
                    graph_nx.add_node(dst_id)
            else:
                graph_nx.add_edge(src_id, dst_id)
                # ID, length, weight, similarity
                # weight is always zero for some reason
                # similarity = edit distance of prefix-suffix overlap divided by the length of overlap
                overlap = overlap.split()
                # len(overlap)
                [edge_id, overlap_len, weight], similarity = map(int, overlap[:3]), float(overlap[3])
                if (src_id, dst_id) not in edge_lengths.keys():
                    edge_ids[(src_id, dst_id)] = edge_id
                    edge_lengths[(src_id, dst_id)] = overlap_len
                    edge_similarities[(src_id, dst_id)] = similarity

    nx.set_node_attributes(graph_nx, node_lengths, 'read_length')
    nx.set_node_attributes(graph_nx, node_data, 'read_sequence')
    nx.set_edge_attributes(graph_nx, edge_lengths, 'overlap_length')
    nx.set_edge_attributes(graph_nx, edge_similarities, 'overlap_similarity')
    graph_torch = from_networkx(graph_nx)

    return graph_nx, graph_torch

    # TODO: This keep both strands, and I only need one. I should fix that - maybe through parsing GFA


def main():
    graph_path = os.path.abspath('data/raw/graph_before.csv')
    graph_nx, graph_torch = from_csv(graph_path)
    graph_torch.x = torch.zeros(graph_torch.num_nodes, dtype=int)
    # --- TESTING ---
    print(graph_torch)
    print(graph_torch.x)
    print(graph_torch.read_length[:10])
    print(graph_torch.edge_index[0][:10])
    print(graph_torch.edge_index[1][:10])
    print(graph_torch.overlap_length[:10])
    print(graph_torch.overlap_similarity[:10])
    draw_graph(graph_nx)
    # ---------------


if __name__ == '__main__':
    main()
