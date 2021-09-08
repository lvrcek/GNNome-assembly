import argparse
from datetime import datetime
import copy
import os
import pickle
import random
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
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


def draw_loss_plots(train_loss, valid_loss, timestamp):
    """Draw and save plot of train and validation loss over epochs.

    Parameters
    ----------
    train_loss : list
        List of training loss for each epoch
    valid_loss : list
        List of validation loss for each epoch
    timestamp : str
        A timestep used for naming the file

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'figures/loss_{timestamp}.png')


def draw_accuracy_plots(train_acc, valid_acc, timestamp):
    """Draw and save plot of train and validation accuracy over epochs.

    Parameters
    ----------
    train_loss : list
        List of training accuracy for each epoch
    valid_loss : list
        List of validation accuracy for each epoch
    timestamp : str
        A timestep used for naming the file

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'figures/accuracy_{timestamp}.png')


def set_seed(seed=42):
    """Set random seed to enable reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        A number used to set the random seed

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_neighbors_dicts(idx, data_path):
    """Return dictionaries with predecessor and successor information.
    
    Parameters
    ----------
    idx : int
        Index of the graph for which the information will be loaded
    data_path : str
        Path to where the information data of a graph is stored
    
    Returns
    -------
    dict
        a dictionary with a list of predecessors for each node
    dict
        a dictionary with a list of successors for each node
    """
    pred_path = os.path.join(data_path, f'info/{idx}_pred.pkl')
    succ_path = os.path.join(data_path, f'info/{idx}_succ.pkl')
    pred = pickle.load(open(pred_path, 'rb'))
    succ = pickle.load(open(succ_path, 'rb'))
    return pred, succ


def get_reads(idx, data_path):
    """Return dictionary with sequence information for a graph.
    
    Parameters
    ----------
    idx : int
        Index of the graph for which the information will be loaded
    data_path : str
        Path to where the information data of a graph is stored

    Returns
    -------
    dict
        a dictionary with a sequence for each node in the graph
    """
    reads_path = os.path.join(data_path, f'info/{idx}_reads.pkl')
    reads = pickle.load(open(reads_path, 'rb'))
    return reads


def get_reference(idx, data_path):
    """Get path for the reference (ground-truth) for a graph.
    
    Parameters
    ----------
    idx : int
        Index of the graph for which the information will be loaded
    data_path : str
        Path to where the information data of a graph is stored

    Returns
    -------
    str
       a path to the reference associated with the graph with index idx 
    """
    ref_path = os.path.join(data_path, f'references/{idx}.fasta')
    return ref_path


def get_edges(idx, data_path):
    """Return dictionary with edge indices of the graph

    Parameters
    ----------
    idx : int
        Index of the graph for which the information will be loaded
    data_path : str
        Path to where the information data of a graph is stored

    Returns
    -------
    dict
        a dictionary with edge indices for each edge in the graph
    """
    edges_path = os.path.join(data_path, f'info/{idx}_edges.pkl')
    edges = pickle.load(open(edges_path, 'rb'))
    return edges



def get_dataloaders(data_path, batch_size, eval, ratio):
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
    ds = AssemblyGraphDataset(data_path)
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


def unpack_data(data, data_path, device):
    """Unpacks the data loaded by the dataloader.
    
    Parameters
    ----------
    data : tuple
        A tuple containing index of a graph and the associated graph
    data_path : str
        A path to directory where an additional data is stored for
        the graph
    device : str
        On which device will the copmutation be performed (cpu/cuda)
    
    Returns
    -------
    int
        index of the graph
    dgl.DGLGraph
        graph in the DGLGraph format
    dict
        a dictionary with predecessors for each node
    dict
        a dictionary with successors for each node
    dict
        a dictionary with reads for each node
    str
        a path to the reference associated with the graph
    """
    idx, graph = data
    idx = idx.item()
    pred, succ = get_neighbors_dicts(idx, data_path)
    reads = get_reads(idx, data_path)
    reference = get_reference(idx, data_path)
    edges = get_edges(idx, data_path)
    graph = graph.to(device)
    return idx, graph, pred, succ, reads, reference, edges


def print_graph_info(idx, graph):
    """Print the basic information for the graph with index idx."""
    print('\n---- GRAPH INFO ----')
    print('Graph index:', idx)
    print('Number of nodes:', graph.num_nodes())
    print('Number of edges:', len(graph.edges()[0]))


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

    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    data_path = os.path.abspath(args.data_path)
    eval = args.eval

    dl_train, dl_valid, dl_test = get_dataloaders(data_path, batch_size, eval, ratio=0.2)

    # exit()

    model = models.NonAutoRegressive(dim_latent).to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    model_path = os.path.abspath(f'pretrained/{time_now}.pt')

    best_model = model = models.NonAutoRegressive(dim_latent)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.to(device)

    if not eval:
        patience = 0
        loss_per_epoch_train, loss_per_epoch_valid = [], []
        accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

        # --- Training ---
        start_time = time.time()
        for epoch in range(num_epochs):
            # model.train()
            print(f'Epoch: {epoch}')
            patience += 1
            loss_per_graph = []
            accuracy_per_graph = []
            for data in dl_train:
                idx, graph, pred, succ, reads, reference, edges = unpack_data(data, data_path, device)
                print_graph_info(idx, graph)
                loss_list, accuracy = utils.process(model, idx, graph, pred, succ, reads, reference, edges, optimizer, 'train', epoch, device=device)
                loss_per_graph.append(np.mean(loss_list))
                accuracy_per_graph.append(accuracy)
                process_time = time.time()
                print(f'Processing graph {idx} done. Elapsed time: {process_time - start_time}')

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
                    idx, graph, pred, succ, reads, reference, edges = unpack_data(data, data_path, device)
                    print_graph_info(idx, graph)
                    loss_list, accuracy = utils.process(model, idx, graph, pred, succ, reads, reference, edges, optimizer, 'eval', epoch, device=device)
                    loss_per_graph.append(np.mean(loss_list))
                    accuracy_per_graph.append(accuracy)

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
                print(f'Validation in epoch {epoch} done. Elapsed time: {time.time()-start_time}s')

        draw_loss_plots(loss_per_epoch_train, loss_per_epoch_valid, time_now)
        draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, time_now)

    torch.save(best_model.state_dict(), model_path)

    # --- Testing ---
    with torch.no_grad():
        test_accuracy = []
        print('TESTING')
        model.eval()
        for data in dl_test:
            idx, graph, pred, succ, reads, reference, edges = unpack_data(data, data_path, device)
            print_graph_info(idx, graph)
            loss_list, accuracy = utils.process(best_model, idx, graph, pred, succ, reads, reference, edges, optimizer, 'eval', epoch, device=device)
            test_accuracy.append(accuracy)

        print(f'Average accuracy on the test set:', np.mean(test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/train', help='path to directory with training data')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    train(args)
