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
from hyperparameters import get_hyperparameters
import models
import utils


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


def get_neighbors_dicts(idx, data_path):
    pred_path = os.path.join(data_path, f'processed/{idx}_pred.pkl')
    succ_path = os.path.join(data_path, f'processed/{idx}_succ.pkl')
    pred = pickle.load(open(pred_path, 'rb'))
    succ = pickle.load(open(succ_path, 'rb'))
    return pred, succ


def get_reference(idx, data_path):
    ref_path = os.path.join(data_path, f'references/{idx}.fasta')
    return ref_path


def get_dataloaders(data_path, batch_size, eval, ratio):
    ds = dataset.GraphDataset(data_path)
    if eval:
        dl_train, dl_valid = None, None
        dl_test = DataLoader(ds, batch_size=batch_size, shuffle=False)
    else:
        valid_size = test_size = int(len(ds) * ratio)
        train_size = len(ds) - valid_size - test_size
        ds_train, ds_valid, ds_test = random_split(ds, [train_size, valid_size, test_size])
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    return dl_train, dl_valid, dl_test


def unpack_data(data, data_path, device):
    idx, graph = data
    idx = idx.item()
    pred, succ = get_neighbors_dicts(idx, data_path)
    reference = get_reference(idx, data_path)
    graph = graph.to(device)
    return idx, graph, pred, succ, reference


def print_graph_info(idx, graph):
    print('Graph index:', idx)
    print('Number of nodes:', graph.num_nodes)
    print('Number of edges:', len(graph.edge_index[0]))


def train(args):
    hyperparameters = get_hyperparameters()
    num_epochs = hyperparameters['num_epochs']
    dim_node = hyperparameters['dim_nodes']
    dim_edge = hyperparameters['dim_edges']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience_limit = hyperparameters['patience_limit']
    learning_rate = hyperparameters['lr']
    device = hyperparameters['device']

    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    data_path = os.path.abspath(args.data_path)
    eval = args.eval

    dl_train, dl_valid, dl_test = get_dataloaders(data_path, batch_size, eval, ratio=0.2)

    model = models.SequentialModel(dim_node, dim_edge, dim_latent).to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    model_path = os.path.abspath(f'pretrained/{time_now}.pt')

    best_model = models.SequentialModel(dim_node, dim_edge, dim_latent)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.to(device)

    if not eval:
        patience = 0
        loss_per_epoch_train, loss_per_epoch_valid = [], []
        accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

        # --- Training ---
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            print(f'Epoch: {epoch}')
            patience += 1
            loss_per_graph = []
            accuracy_per_graph = []
            for data in dl_train:
                idx, graph, pred, succ, reference = unpack_data(data, data_path, device)
                print_graph_info(idx, graph)
                loss_list, accuracy = utils.process(model, idx, graph, pred, succ, reference, optimizer, 'train', device=device)
                loss_per_graph.append(np.mean(loss_list))
                accuracy_per_graph.append(accuracy)

            loss_per_epoch_train.append(np.mean(loss_per_graph))
            accuracy_per_epoch_train.append(np.mean(accuracy_per_graph))
            print(f'Training in epoch {epoch} done. Elapsed time: {time.time()-start_time}s')

            # --- Validation ---
            with torch.no_grad():
                print('VALIDATION')
                model.eval()
                loss_per_graph = []
                accuracy_per_graph = []
                for data in dl_valid:
                    idx, graph, pred, succ, reference = unpack_data(data, data_path, device)
                    print_graph_info(idx, graph)
                    loss_list, accuracy = utils.process(model, idx, graph, pred, succ, reference, optimizer, 'eval', device=device)
                    loss_per_graph.append(np.mean(loss_list))
                    accuracy_per_graph.append(accuracy)

                if len(loss_per_epoch_valid) > 0 and loss_per_graph[-1] < min(loss_per_epoch_valid):
                    patience = 0
                    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    best_model.to(device)
                    torch.save(best_model.state_dict(), model_path)
                elif patience >= patience_limit:
                    break

                loss_per_epoch_valid.append(np.mean(loss_per_graph))
                accuracy_per_epoch_valid.append(np.mean(accuracy_per_graph))
                print(f'Validation in epoch {epoch} done. Elapsed time: {time.time()-start_time}s')

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, time_now)
        draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, time_now)

    torch.save(best_model.state_dict(), model_path)

    # --- Testing ---
    with torch.no_grad():
        test_accuracy = []
        print('TESTING')
        model.eval()
        for data in dl_test:
            idx, graph, pred, succ, reference = unpack_data(data, data_path, device)
            print_graph_info(idx, graph)
            loss_list, accuracy = utils.process(best_model, idx, graph, pred, succ, reference, optimizer, 'eval', device=device)
            test_accuracy.append(accuracy)

        print(f'Average accuracy on the test set:', np.mean(test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/train', help='path to directory with training data')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    train(args)
