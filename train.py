import argparse
from datetime import datetime
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import utils


def get_dataloaders(ds, batch_size, eval, ratio):
    """Load the dataset and initialize dataloaders.

    Given a path to data, first an AssemblyGraphDataset is initialized.
    This dataset is then split and a PyTorch DataLoader is returned for
    training, validation, and testing dataset.

    Parameters
    ----------
    data_path : str
        Path to directory where the graphs are stored
    batch_size : int
        Size of a batch for the dataloaders
    eval : bool
        True if only the evaluation perfomed and training is skipped
    ratio : float
        Ratio how to split the dataset into train/valid/test datasets

    Returns
    -------
    torch.DataLoader
        a dataloader for the training set, None if eval
    torch.DataLoader
        a dataloader for the validation set, None if eval
    torch.DataLoader
        a dataloader for the testing set
    """
    if eval:
        dl_train, dl_valid = None, None
        dl_test = GraphDataLoader(ds, batch_size=batch_size, shuffle=False)
    else:
        valid_size = test_size = int(len(ds) * ratio)
        train_size = len(ds) - valid_size - test_size
        ds_train, ds_valid, ds_test = random_split(ds, [train_size, valid_size, test_size])
        dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)
        dl_test = GraphDataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    return dl_train, dl_valid, dl_test


def process(model, graph, neighbors, reads, solution, edges, optimizer, epoch, device):
    """Process the graph by predicting the correct next neighbor.
    
    A graph is processed by simulating a walk over it where the 
    best next neighbor is predicted any time a branching occurs.
    The choices are compared tothe ground truth and loss is calculated.
    The list of losses and accuracy for the given graph are returned.
    """
    walk_length = get_hyperparameters()['walk_length']

    ground_truth_list = solution.copy()  # TODO: Consistent naming
    ground_truth = {n1: n2 for n1, n2 in zip(ground_truth_list[:-1], ground_truth_list[1:])}
    ground_truth[ground_truth_list[-1]] = None
    total_steps = len(ground_truth_list) - 1

    start = (epoch * walk_length) % total_steps if walk_length != -1 else 0
    current = ground_truth_list[start]

    criterion = nn.CrossEntropyLoss()

    visited = set()
    walk = []
    loss_list = []
    total_loss = 0
    total = 0
    correct = 0
    steps = 0

    logits = model(graph, reads)
    
    print('Iterating through nodes!')
    while True:
        steps += 1
        walk.append(current)
        if steps == walk_length:  # TODO: create function / reduce redundancy
            break
        if steps == total_steps:  # This one is probably redundant
            break
        if ground_truth[current] is None:  # Because this one is the same (last node in the solution walk)
            break
        if current in visited:  # Since I'm doing teacher forcing, this is not necessary. I will never end up in a visited node
            break
        visited.add(current)  # This is also unnecessary
        visited.add(current ^ 1)  # virtual pair
        if len(neighbors[current]) == 0:  # This should also be covered by the upper "gt[curr] is None" case
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue

        neighbor_edges = [edges[current, n] for n in neighbors[current]]
        neighbor_logits = logits.squeeze(1)[neighbor_edges]
        value, index = torch.topk(neighbor_logits, k=1, dim=0)
        choice = neighbors[current][index]

        best_neighbor = ground_truth[current]
        best_idx = neighbors[current].index(best_neighbor)

        # utils.print_prediction(walk, current, neighbors, neighbor_logits, choice, best_neighbor)

        # Calculate loss
        best_idx = torch.tensor([best_idx], dtype=torch.long, device=device)
        loss = criterion(neighbor_logits.unsqueeze(0), best_idx)  # First squeeze, then unsqueeze - redundant?
        loss_list.append(loss.item())
        total_loss += loss

        if choice == best_neighbor:
            correct += 1
        total += 1
        current = best_neighbor  # Teacher forcing        

    if model.training:
        optimizer.zero_grad()
        total_loss.backward()  # Backprop summed losses
        optimizer.step()

    accuracy = correct / total
    return loss_list, accuracy


def train(args):
    """Training loop.
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the command line

    Returns
    -------
    None
    """
    hyperparameters = get_hyperparameters()
    num_epochs = hyperparameters['num_epochs']
    dim_node = hyperparameters['dim_nodes']
    dim_edge = hyperparameters['dim_edges']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience_limit = hyperparameters['patience_limit']
    learning_rate = hyperparameters['lr']
    device = hyperparameters['device']

    # utils.set_seed(0)

    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    data_path = os.path.abspath(args.data)
    out = args.out if args.out is not None else timestamp
    eval = args.eval

    ds = AssemblyGraphDataset(data_path)
    dl_train, dl_valid, dl_test = get_dataloaders(ds, batch_size, eval, ratio=0.2)
    num_graphs = len(ds)

    model = models.NonAutoRegressive(dim_latent).to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    model_path = os.path.abspath(f'pretrained/model_{out}.pt')

    best_model = models.NonAutoRegressive(dim_latent)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.to(device)
    best_model.eval()

    info_all = utils.load_graph_data(num_graphs, data_path)

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    if not eval:
        patience = 0
        loss_per_epoch_train, loss_per_epoch_valid = [], []
        accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

        # --- Training ---
        for epoch in range(num_epochs):
            # model.train()
            print(f'Epoch: {epoch}')
            patience += 1
            loss_per_graph = []
            accuracy_per_graph = []
            for data in dl_train:
                idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all)
                graph = graph.to(device)
                solution = utils.get_walks(idx, data_path)

                utils.print_graph_info(idx, graph)
                loss_list, accuracy = process(model, graph, succ, reads, solution, edges, optimizer, epoch, device=device)
                loss_per_graph.append(np.mean(loss_list))
                accuracy_per_graph.append(accuracy)

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'Processing graph {idx} done. Elapsed time: {elapsed}')

            loss_per_epoch_train.append(np.mean(loss_per_graph))
            accuracy_per_epoch_train.append(np.mean(accuracy_per_graph))
            elapsed = utils.timedelta_to_str(datetime.now() - time_start)
            print(f'\nTraining in epoch {epoch} done. Elapsed time: {elapsed}\n')

            # --- Validation ---
            with torch.no_grad():
                print('VALIDATION')
                model.eval()
                loss_per_graph = []
                accuracy_per_graph = []
                for data in dl_valid:
                    idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all)
                    graph = graph.to(device)
                    solution = utils.get_walks(idx, data_path)

                    utils.print_graph_info(idx, graph)
                    loss_list, accuracy = process(model, graph, succ, reads, solution, edges, optimizer, epoch, device=device)
                    loss_per_graph.append(np.mean(loss_list))
                    accuracy_per_graph.append(accuracy)

                    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                    print(f'Processing graph {idx} done. Elapsed time: {elapsed}')

                if len(loss_per_epoch_valid) > 0 and loss_per_graph[-1] < min(loss_per_epoch_valid):
                    patience = 0
                    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    best_model.to(device)
                    torch.save(best_model.state_dict(), model_path)
                elif patience >= patience_limit:
                    pass
                    # TODO: Enable early stopping, incrase patience_limit
                    # break

                loss_per_epoch_valid.append(np.mean(loss_per_graph))
                accuracy_per_epoch_valid.append(np.mean(accuracy_per_graph))
                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\nValidation in epoch {epoch} done. Elapsed time: {elapsed}\n')

        utils.draw_loss_plots(loss_per_epoch_train, loss_per_epoch_valid, out)
        utils.draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, out)

    torch.save(best_model.state_dict(), model_path)

    # --- Testing ---
    with torch.no_grad():
        test_accuracy = []
        print('TESTING')
        model.eval()
        for data in dl_test:
            idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all)
            graph = graph.to(device)
            solution = utils.get_walks(idx, data_path)

            utils.print_graph_info(idx, graph)
            loss_list, accuracy = process(best_model, graph, succ, reads, solution, edges, optimizer, epoch, device=device)
            test_accuracy.append(accuracy)

        elapsed = utils.timedelta_to_str(datetime.now() - time_start)
        print(f'\nTesting done. Elapsed time: {elapsed}')
        print(f'Average accuracy on the test set:', np.mean(test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train', help='path to directory with training data')
    parser.add_argument('--out', type=str, default=None, help='Output name for figures and models')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    train(args)
