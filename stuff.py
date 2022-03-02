import pickle
import subprocess
import os

import dgl
import torch

import algorithms


def run_new_gt_algo():
    path = 'data/no_rep_chromo_bench/'
    with open('gt_graph.csv', 'w') as f:
        for i in range(1, 24):
            if i == 23:
                i = 'X'
            graph = dgl.load_graphs(f'{path}/chr{i}/processed/0.dgl')[0][0] 
            neighs = pickle.load(open(f'{path}/chr{i}/info/0_succ.pkl', 'rb'))
            edges = pickle.load(open(f'{path}/chr{i}/info/0_edges.pkl', 'rb'))
            n_p, e_p, n_n, e_n, all_walks = algorithms.dfs_gt_graph(graph, neighs, edges)
            line = f'chr{i},{len(all_walks)},{len(n_p)+len(n_n)},{graph.num_nodes()},\
                    {(len(n_p)+len(n_n))/graph.num_nodes()},{len(e_p)+len(e_n)},\
                    {graph.num_edges()},{(len(e_p)+len(e_n))/graph.num_edges()}\n'
            f.write(line)
            correct_nodes = n_p | n_n
            correct_edges = e_p | e_n
            if not os.path.isdir(f'{path}/chr{i}/solutions'):
                pickle.dump(correct_nodes, open(f'{path}/chr{i}/solutions/0_nodes.pkl', 'wb'))
                pickle.dump(correct_edges, open(f'{path}/chr{i}/solutions/0_edges.pkl', 'wb'))


if __name__ == '__main__':
    run_new_gt_algo()

