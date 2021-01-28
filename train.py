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
from hyperparameters import get_hyperparameters


# NUM_EPOCHS = 5


def draw_loss_plot(train_loss, valid_loss, timestamp):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'figures/loss_{timestamp}.png')
    plt.show()


def draw_accuracy_plots(train_acc, valid_acc, timestamp):
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'figures/train_accuracy_{timestamp}.png')
    plt.show()


def train():

    hyperparameters = get_hyperparameters()
    num_epochs = hyperparameters['num_epochs']
    dim_node = hyperparameters['dim_nodes']
    dim_edge = hyperparameters['dim_edges']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience_limit = hyperparameters['patience_limit']
    learning_rate = hyperparameters['lr']
    device = hyperparameters['device']


    # --- DEBUGGING ---
    num_epochs = 3
    dim_node = 1
    dim_edge = 1
    dim_latent = 1
    batch_size = 1
    patience_limit = 10
    learning_rate = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------

    mode = 'train'

    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    train_path = os.path.abspath('data/train')
    test_path = os.path.abspath('data/test')

    # TODO: Discuss with Mile how to train this thing - maybe through generated reads by some tools?
    # First with real data just to check if the thing works, then probably with the generated graphs
    # The problem is that generated graphs don't have chimeric reads
    ds_train = dataset.GraphDataset(train_path)
    ds_test = dataset.GraphDataset(test_path)

    ratio = 0.5
    valid_size = int(len(ds_train) * ratio)
    train_size = len(ds_train) - valid_size
    ds_train, ds_valid = random_split(ds_train, [train_size, valid_size])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    processor = models.ExecutionModel(dim_node, dim_edge, dim_latent).to(device)
    params = list(processor.parameters())
    model_path = os.path.abspath(f'trained_models/{time_now}.pt)')

    optimizer = optim.Adam(params, lr=learning_rate)

    patience = 0
    best_model = models.ExecutionModel(dim_node, dim_edge, dim_latent)
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    if mode == 'train':
        loss_per_epoch_train, loss_per_epoch_valid = [], []
        accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

        # Training
        for epoch in range(num_epochs):
            processor.train()
            print(f'Epoch: {epoch}')
            patience += 1
            loss_per_graph = []
            acc_per_graph = []
            for data in dl_train:
                print(data)
                graph_loss, graph_accuracy = processor.process(data, optimizer, 'train')  # Returns list of losses for each step in path finding
                loss_per_graph.append(np.mean(graph_loss))  # Take the mean of that for each graph
                acc_per_graph.append(graph_accuracy)

            loss_per_epoch_train.append(np.mean(loss_per_graph))
            accuracy_per_epoch_train.append(np.mean(acc_per_graph))

            # Validation
            with torch.no_grad():
                processor.eval()
                loss_per_graph = []
                acc_per_graph = []
                for data in dl_valid:
                    graph_loss, graph_acc = processor.process(data, optimizer, 'eval')
                    current_loss = np.mean(graph_loss)
                    loss_per_graph.append(current_loss)
                    acc_per_graph.append(graph_acc)

                if len(loss_per_epoch_valid) > 0 and current_loss < min(loss_per_epoch_valid):
                    patience = 0
                    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
                    torch.save(best_model.state_dict(), model_path)
                elif patience >= patience_limit:
                    break

                loss_per_epoch_valid.append(np.mean(loss_per_graph))
                accuracy_per_epoch_valid.append(np.mean(graph_acc))

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, time_now)
        draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, time_now)

    torch.save(best_model.state_dict(), model_path)

    # Testing
    if mode == 'test':  # TODO: put validation/testing into different functions
        with torch.no_grad():
            processor.eval()
            for data in dl_test:
                graph_loss, graph_acc = best_model.process(data, optimizer, 'eval')

            average_test_accuracy = np.mean(graph_acc)
            print(f'Average accuracy on the test set:', average_test_accuracy)


if __name__ == '__main__':
    train()
