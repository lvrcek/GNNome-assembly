import os
import re

import torch
import torch_geometric
from torch_geometric.utils import from_networkx
import networkx as nx
from matplotlib import pyplot as plt


def draw_graph(graph):
    nx.draw(graph_nx)
    plt.show()


def from_csv(graph_path):
    graph_nx = nx.DiGraph()
    node_features = {}
    edge_features = {}
    with open(graph_path) as f:
        for line in f.readlines():
            src, dst, flag, overlap = line.strip().split(',')
            flag = int(flag)
            src, dst = src.split(), dst.split()
            pattern = ':(\d+)'
            src_id, src_len = int(src[0]), int(re.findall(pattern, src[2])[0])
            dst_id, dst_len = int(dst[0]), int(re.findall(pattern, dst[2])[0])
            if flag == 0:
                if src_id not in node_features.keys():
                    node_features[src_id] = src_len
                    graph_nx.add_node(src_id)
                if dst_id not in node_features.keys():
                    node_features[dst_id] = dst_len
                    graph_nx.add_node(dst_id)
            else:
                graph_nx.add_edge(src_id, dst_id)
                edge_id, overlap_len, something = map(int, overlap.split())
                if (src_id, dst_id) not in edge_features.keys():
                    edge_features[(src_id, dst_id)] = overlap_len
    print(edge_features)
    nx.set_node_attributes(graph_nx, node_features, "read_length")
    nx.set_edge_attributes(graph_nx, edge_features, "overlap_length")
    graph_torch = from_networkx(graph_nx)
    print(graph_torch.read_length[:10])
    print(graph_torch.edge_index[0][:10])
    print(graph_torch.edge_index[1][:10])
    print(graph_torch.overlap_length[:10])
    return graph_nx, graph_torch


if __name__ == '__main__':
    graph_path = os.path.abspath('data/graph.csv')
    graph_nx, graph_torch = from_csv(graph_path)

# A file with which I will generate new data
# This is going to be somewhat different than before, because I will need to have a more realistic dataset.
# This means that all the features Raven is using need to be available to me as well.
# This includes: length of reads, length of overlaps, ... ?
