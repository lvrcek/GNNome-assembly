import os
import sys
import pickle
import subprocess

import dgl
import numpy as np

from inference import inference
from algorithms import get_solutions_for_all


assert len(sys.argv) == 2, 'Provide an argument 2, 5, or 10.'

mbp = int(sys.argv[1])

assert int(mbp) in (2, 5, 10), 'Argument needs to be either 2, 5, or 10.'

if mbp == 2:
    path = 'data/train'
if mbp == 5:
    path = 'data/chr10_5M'
if mbp == 10:
    path = 'data/chr12_10M'

path = os.path.abspath(path)
mbp *= int(1e6)

print(f'Running GNN and greedy')
inference(data_path=path)
print(f'Running exhaustive search')
print(f'Some graphs might take a while...')
get_solutions_for_all(path)

proc_path = os.path.join(path, 'processed')
info_path = os.path.join(path, 'info')
sols_path = os.path.join(path, 'solutions')
infr_path = os.path.join(path, 'inference')

gnn_len, greedy_len, exhaustive_len = [], [], []

for i in range(len(os.listdir(proc_path))):
    if mbp == 2e6 and i not in (3, 4, 5, 15, 16, 25, 30, 40, 45, 48):  # Only graphs from the test set!
        continue
    pred = pickle.load(open(f'{infr_path}/{i}_predict.pkl', 'rb'))
    base = pickle.load(open(f'{infr_path}/{i}_greedy.pkl', 'rb'))
    sols = pickle.load(open(f'{sols_path}/{i}_gt.pkl', 'rb'))
    graph = dgl.load_graphs(f'{proc_path}/{i}.dgl')[0][0]
    pred_len = graph.ndata['read_end'][pred[-1]] - graph.ndata['read_start'][0]
    base_len = graph.ndata['read_end'][base[-1]] - graph.ndata['read_start'][0]
    sols_len = graph.ndata['read_end'][sols[-1]] - graph.ndata['read_start'][0]

    print(i, 'gnn:', pred_len.item(), 'greedy:', base_len.item(), 'exhaustive:', sols_len.item())
    gnn_len.append(pred_len.item() / mbp)
    greedy_len.append(base_len.item() / mbp)
    exhaustive_len.append(sols_len.item() / mbp)

gnn_mean, gnn_std = np.mean(gnn_len), np.std(gnn_len)
greedy_mean, greedy_std = np.mean(greedy_len), np.std(greedy_len)
exhaustive_mean, exhaustive_std = np.mean(exhaustive_len), np.std(exhaustive_len)

print()
print(f'GNN mean and standard deviation: {gnn_mean}, {gnn_std}')
print(f'Greedy mean and standard deviation: {greedy_mean}, {greedy_std}')
print(f'Exhaustive mean and standard deviation: {exhaustive_mean}, {exhaustive_std}')
