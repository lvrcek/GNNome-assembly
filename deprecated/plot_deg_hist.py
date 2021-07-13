import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx

proc_pth = 'data/train/processed'
save_pth = '../scratch/figures'

name_dict = {0: 22,
             1: 21,
             2: 20,
             3: 19,
             4: 18}

for file in os.listdir(proc_pth):
    file_pth = os.path.join(proc_pth, file)
    chr = name_dict[int(file[:-3])]
    # print(f'Plotting: {chr} ...', end='\t')
    graph = torch.load(file_pth)
    # print(graph)
    print(f'chr{chr} metrics: num_nodes = {graph.read_length.shape[0]}, num_edges = {graph.edge_index.shape[1]} ...', end='\t')
    # print('Done!')
    # continue
    graph_nx = to_networkx(graph)
    # print(f'<indegree> = {np.mean([d for n, d in graph_nx.in_degree()])}, num_weakly = {nx.number_weakly_connected_components(graph_nx)}')
    
    indegree_seq = sorted([d for n,d in graph_nx.in_degree()], reverse=True)
    indegree_count = Counter(indegree_seq)
    deg, cnt = zip(*indegree_count.items())

    fig, ax = plt.subplots()
    plt.bar(deg, np.log(cnt), width=0.8)

    plt.title(f'Indegree histogram for chr{chr}')
    plt.xlabel('Indegree')
    plt.ylabel('log (Count)')
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    for d, c in zip(deg, cnt):
        plt.annotate(f'{c}', xy=(d,np.log(c)), xytext=(0,0.2), textcoords='offset points', ha='center', va='bottom')

    plt.savefig(os.path.join(save_pth, f'{chr}.png'))
    print('Done!')


