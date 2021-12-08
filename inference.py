import argparse
import os
import pickle

import torch

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import algorithms
from utils import load_graph_data


def predict(model, graph, pred, neighbors, reads, edges):
    starts = [k for k,v in pred.items() if len(v)==0 and graph.ndata['read_strand'][k]==1]
    # TODO: This already returns all the possible walks, I just need to refine it
    best_walks = []
    for start in starts:
        current = start
        visited = set()
        walk = []

        logits = model(graph, reads)
        print('Finding optimal walk!')

        while True:
            if current in visited:
                break
            walk.append(current)
            visited.add(current)
            visited.add(current ^ 1)
            if len(neighbors[current]) == 0:
                break
            if len(neighbors[current]) == 1:
                current = neighbors[current][0]
                continue

            neighbor_edges = [edges[current, n] for n in neighbors[current]]
            neighbor_logits = logits.squeeze(1)[neighbor_edges]

            _, index = torch.topk(neighbor_logits, k=1, dim=0)
            choice = neighbors[current][index]
            current = choice

        best_walks.append(walk.copy())

    sorted(best_walks, key=lambda x: len(x))
    return best_walks[0]


def inference(model_path=None, data_path=None):
    hyperparameters = get_hyperparameters()
    device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']
    num_gnn_layers = hyperparameters['num_gnn_layers']

    if model_path is None:
        model_path = 'pretrained/model_32d_8l.pt'  # Best performing model
    model = models.NonAutoRegressive(dim_latent, num_gnn_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    if data_path is None:
        data_path = 'data/train'
    ds = AssemblyGraphDataset(data_path)

    info_all = load_graph_data(len(ds), data_path)

    for i in range(len(ds)):
        idx, graph = ds[i]
        print(f'Graph index: {idx}')
        graph = graph.to(device)
        
        succ = info_all['succs'][idx]
        pred = info_all['preds'][idx]
        # reads = info_all['reads'][idx]
        reads = None
        edges = info_all['edges'][idx]

        walk = predict(model, graph, pred, succ, reads, edges)

        inference_path = os.path.join(data_path, 'inference')
        if not os.path.isdir(inference_path):
            os.mkdir(inference_path)
        pickle.dump(walk, open(f'{inference_path}/{idx}_predict.pkl', 'wb'))

        # TODO: Greedy will not be too relevant soon, most likely
        baseline, _ = algorithms.baseline(graph, 0, succ, pred, edges)
        pickle.dump(baseline, open(f'{inference_path}/{idx}_greedy.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    model_path = args.model
    data_path = args.data
    inference(model_path, data_path)
