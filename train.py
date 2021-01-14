import argparse
from datetime import datetime
import copy
import os


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import random_split

import dataset
import models
import hyperparameters


# NUM_EPOCHS = 5


def train():

    hyperparams = hyperparameters.get_hyperparameters()
    num_epochs = hyperparams['num_epochs']
    dim_node = hyperparams['dim_nodes']
    dim_edge = hyperparams['dim_edges']
    dim_latent = hyperparams['dim_latent']
    batch_size = hyperparams['batch_size']
    patience_limit = hyperparams['patience_limit']
    learning_rate = hyperparams['lr']

    # --- DEBUGGING ---
    num_epochs = 1
    dim_node = 1
    dim_edge = 1
    dim_latent = 1
    batch_size = 1
    patience_limit = 10
    learning_rate = 1e-5
    # -----------------

    mode = 'train'

    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    train_path = os.path.abspath('data/train')
    # valid_path = os.path.abspath('data/train/processed')
    test_path = os.path.abspath('data/test')

    # TODO: Discuss with Mile how to train this thing - maybe through generated reads by some tools?
    # First with real data just to check if the thing works, then probably with the generated graphs
    # The problem is that generated graphs don't have chimeric reads
    ds_train = dataset.GraphDataset(train_path)
    # ds_valid = dataset.GraphDataset(valid_path)
    ds_test = dataset.GraphDataset(test_path)

    ratio = 0.5
    valid_size = int(len(ds_train) * ratio)
    train_size = len(ds_train) - valid_size
    ds_train, ds_valid = random_split(ds_train, [train_size, valid_size])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    processor = models.ExecutionModel(dim_node, dim_edge, dim_latent)
    params = list(processor.parameters())
    model_path = os.path.abspath(f'trained_models/{time_now}.pt)')

    optimizer = optim.Adam(params, lr=learning_rate)

    patience = 0
    best_model = models.ExecutionModel(dim_node, dim_edge, dim_latent)
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    if mode == 'train':
        loss_per_epoch = []
        accuracy_per_epoch = []
        # Training
        for epoch in range(num_epochs):
            processor.train()
            print(f'Epoch: {epoch}')
            loss_per_graph = []
            for data in dl_train:
                print(data)
                graph_loss = processor.process(data, optimizer, 'train')  # Returns list of losses for each step in path finding
                loss_per_graph.append(np.mean(graph_loss))  # Take the mean of that for each graph

            loss_per_epoch.append(np.mean(loss_per_graph))

            # Patience is a bit different than this
            # if len(loss_per_graph) >= 10:
            #     patience = loss_per_graph[-10:].copy()
            # else:
            #     patience = loss_per_graph.copy()
            # if loss_per_graph > max(patience):
            #     break

            # Validation
            with torch.no_grad():
                processor.eval()
                for data in dl_valid:

                    graph_acc = processor.process(data, optimizer, 'eval')
                    accuracy_per_epoch.append(graph_acc)
                    if graph_acc > max(accuracy_per_epoch):
                        best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    torch.save(best_model.state_dict(), model_path)

    # Testing
    if mode == 'test':  # TODO: put validation/testing into different functions
        with torch.no_grad():
            processor.eval()
            for data in dl_test:
                graph_acc = best_model.process(data, optimizer, 'eval')


if __name__ == '__main__':
    train()
