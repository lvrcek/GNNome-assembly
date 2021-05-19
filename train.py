import argparse
from datetime import datetime
import copy
import os
import pickle
import random
import time


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import random_split

import dataset
from solver import ExecutionModel
from hyperparameters import get_hyperparameters


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    # num_epochs = 3
    # dim_node = 1
    # dim_edge = 1
    # dim_latent = 1
    # batch_size = 1
    # patience_limit = 10
    # learning_rate = 1e-5
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    # -----------------

    mode = 'train'

    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    train_path = os.path.abspath('data/train')
    test_path = os.path.abspath('data/train')

    # TODO: Discuss with Mile how to train this thing - maybe through generated reads by some tools?
    # First with real data just to check if the thing works, then probably with the generated graphs
    # The problem is that generated graphs don't have chimeric reads
    ds = dataset.GraphDataset(train_path)
    # ds_test = dataset.GraphDataset(test_path)
    # exit()

    ratio = 0.2
    valid_size = test_size = int(len(ds) * ratio)
    train_size = len(ds) - valid_size - test_size
    ds_train, ds_valid, ds_test = random_split(ds, [train_size, valid_size, test_size])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)  # Change later to True
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    processor = ExecutionModel(dim_node, dim_edge, dim_latent)

    # Multi-GPU training not available as batch_size = 1
    # Therefore, samples in a batch cannot be distributed over GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f'We use {torch.cuda.device_count()} GPUs!')
    #     processor = nn.DataParallel(processor)

    processor.to(device)
    params = list(processor.parameters())
    model_path = os.path.abspath(f'pretrained/{time_now}.pt')

    optimizer = optim.Adam(params, lr=learning_rate)

    patience = 0
    best_model = ExecutionModel(dim_node, dim_edge, dim_latent)
    # if torch.cuda.device_count() > 1:
    #     best_model = nn.DataParallel(best_model)
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
    best_model.to(device)

    mode = 'train'

    if mode == 'train':
        loss_per_epoch_train, loss_per_epoch_valid = [], []
        accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

        # Training
        start_time = time.time()
        for epoch in range(num_epochs):
            processor.train()
            print(f'Epoch: {epoch}')
            patience += 1
            loss_per_graph = []
            acc_per_graph = []
            for data in dl_train:
                idx, graph = data
                idx = idx.item()
                pred, succ = get_neighbors_dicts(idx)
                reference = get_reference(idx)
                print(idx)
                print(graph)
                # print(pred)
                # print(type(pred))
                graph = graph.to(device)
                # Return list of losses for each step in path finding
                graph_loss, graph_accuracy = processor.process(graph, pred, succ, reference, optimizer, 'train', device=device)
                loss_per_graph.append(np.mean(graph_loss))  # Take the mean of that for each graph
                acc_per_graph.append(graph_accuracy)

            loss_per_epoch_train.append(np.mean(loss_per_graph))
            accuracy_per_epoch_train.append(np.mean(acc_per_graph))
            print(f'Training in epoch {epoch} done. Elapsed time: {time.time()-start_time}s')

            # Validation
            with torch.no_grad():
                print('VALIDATION')
                processor.eval()
                loss_per_graph = []
                acc_per_graph = []
                for data in dl_valid:
                    idx, graph = data
                    idx = idx.item()
                    pred, succ = get_neighbors_dicts(idx)
                    reference = get_reference(idx)
                    print(idx)
                    graph = graph.to(device)
                    graph_loss, graph_acc = processor.process(graph, pred, succ, reference, optimizer, 'eval', device=device)
                    current_loss = np.mean(graph_loss)
                    loss_per_graph.append(current_loss)
                    acc_per_graph.append(graph_acc)

                if len(loss_per_epoch_valid) > 0 and current_loss < min(loss_per_epoch_valid):
                    patience = 0
                    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
                    best_model.to(device)
                    torch.save(best_model.state_dict(), model_path)
                elif patience >= patience_limit:
                    break

                loss_per_epoch_valid.append(np.mean(loss_per_graph))
                accuracy_per_epoch_valid.append(np.mean(graph_acc))
                print(f'Validation in epoch {epoch} done. Elapsed time: {time.time()-start_time}s')

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, time_now)
        draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, time_now)

    torch.save(best_model.state_dict(), model_path)

    just_for_testing = DataLoader(ds, batch_size=1, shuffle=False)

    # Testing
    mode = 'test'  # Make this more sophisticated, separate inference into a separate file
    if mode == 'test':  # TODO: put validation/testing into different functions
        with torch.no_grad():
            print('TESTING')
            processor.eval()
            for data in dl_test:
                idx, graph = data
                idx = idx.item()
                pred, succ = get_neighbors_dicts(idx)
                reference = get_reference(idx)
                # if idx != 2:
                #     continue
                print(idx)
                print(graph)
                graph = graph.to(device)
                graph_loss, graph_acc = best_model.process(graph, pred, succ, reference, optimizer, 'eval', device=device)
                break

            average_test_accuracy = np.mean(graph_acc)
            print(f'Average accuracy on the test set:', average_test_accuracy)

def get_neighbors_dicts(idx):
    pred = pickle.load(open(f'data/train/processed/{idx}_pred.pkl', 'rb'))
    succ = pickle.load(open(f'data/train/processed/{idx}_succ.pkl', 'rb'))
    return pred, succ


def get_reference(idx):
    reference = f'data/train/references/{idx}.fasta'
    return reference


if __name__ == '__main__':
    train()

