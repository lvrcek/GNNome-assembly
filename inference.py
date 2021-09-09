import argparse
from datetime import datetime
import copy
import os
import pickle
import random
import time

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import dgl

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import algorithms
import utils
import train


def print_prediction(walk, current, neighbors, actions, choice, best_neighbor):
    """Print summary of the prediction for the current position."""
    print('\n-----predicting-----')
    print('previous:\t', None if len(walk) < 2 else walk[-2])
    print('current:\t', current)
    print('neighbors:\t', neighbors[current])
    print('actions:\t', actions.tolist())
    print('choice:\t\t', choice)
    print('ground truth:\t', best_neighbor)


def process(model, graph, neighbors, reads, solution, edges):
    start = 0

    current = start
    visited = set()
    walk = []

    logits = model(graph, reads)
    print('encoded')
    ground_truth_list = solution.copy()
    total_steps = len(ground_truth_list) - 1
    steps = 0
    ground_truth = {n1: n2 for n1, n2 in zip(ground_truth_list[:-1], ground_truth_list[1:])}
    ground_truth[ground_truth_list[-1]] = None
    
    print('Iterating through nodes!')

    while True:
        walk.append(current)
        steps += 1
        if current in visited:
            break
        visited.add(current)  # current node
        visited.add(current ^ 1)  # virtual pair of the current node
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue

        neighbor_edges = [edges[current, n] for n in neighbors[current]]
        neighbor_logits = logits.squeeze(1)[neighbor_edges]

        value, index = torch.topk(neighbor_logits, k=1, dim=0)
        choice = neighbors[current][index]
        # best_neighbor = ground_truth[current]
        # print_prediction(walk, current, neighbors, neighbor_logits, choice, best_neighbor)

        current = choice

    return walk


def inference():
    hyperparameters = get_hyperparameters()
    # device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']
    device = 'cpu'

    model_path = 'pretrained/model_1e-8.pt'
    model = models.NonAutoRegressive(dim_latent).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(type(model))
    print(model)


    data_path = 'data/example_2'
    ds = AssemblyGraphDataset(data_path)
    idx, graph = ds[0]
    graph = graph.to(device)
    
    num_graphs = 1
    info_all = train.load_graph_data(num_graphs, data_path)
    pred = info_all['preds'][0]
    succ = info_all['succs'][0]
    reads = info_all['reads'][0]
    edges = info_all['edges'][0]
    solution = info_all['walks'][0]

    walk = process(model, graph, succ, reads, solution, edges)

    # Compare the walk to ground_truth
    # Compare it to the baseline
    inference_path = os.path.join(data_path, 'inference')
    if not os.path.isdir(inference_path):
        os.mkdir(inference_path)
    pickle.dump(walk, open(f'{inference_path}/walk.pkl', 'wb'))

    baseline, _ = algorithms.greedy(graph, 0, succ, edges, 'baseline')
    pickle.dump(walk, open(f'{inference_path}/baseline.pkl', 'wb'))
    return walk, baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    inference()
