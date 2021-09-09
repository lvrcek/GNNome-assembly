import os
import pickle
import random

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hyperparameters import get_hyperparameters


def get_walks(start, neighbors, num_nodes):
    """Return all the possible walks from a current node of length
    num_nodes.
    
    Parameters
    ----------
    start : int
        Index of the starting node
    neighbors : dict
        Dictionary with the list of neighbors for each node
    num_nodes : int
        Length of the walks to be returned

    Returns
    -------
    list
        a list of all the possible walks, where each walk is also
        stored in a list with num_nodes consecutive nodes
    """
    if num_nodes == 0:
        return [[start]]
    paths = []
    for neighbor in neighbors[start]:
        next_paths = get_walks(neighbor, neighbors, num_nodes-1)
        for path in next_paths:
            path.append(start)
            paths.append(path)
    return paths


def draw_loss_plots(train_loss, valid_loss, out):
    """Draw and save plot of train and validation loss over epochs.

    Parameters
    ----------
    train_loss : list
        List of training loss for each epoch
    valid_loss : list
        List of validation loss for each epoch
    out : str
        A string used for naming the file

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
    plt.savefig(f'figures/loss_{out}.png')


def draw_accuracy_plots(train_acc, valid_acc, out):
    """Draw and save plot of train and validation accuracy over epochs.

    Parameters
    ----------
    train_loss : list
        List of training accuracy for each epoch
    valid_loss : list
        List of validation accuracy for each epoch
    out : str
        A string used for naming the file

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
    plt.savefig(f'figures/accuracy_{out}.png')


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


def get_walks(idx, data_path):
    walk_path = os.path.join(data_path, f'solutions/{idx}_gt.pkl')
    walk = pickle.load(open(walk_path, 'rb'))
    return walk


def get_info(idx, data_path, type):
    # TODO
    info_path = os.path.join(data_path, 'info', f'{idx}_{type}.pkl')
    info = pickle.load(open('info_path', 'rb'))
    return info


def unpack_data_2(data, data_path, device):
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


def unpack_data(data, info_all ,device):
    idx, graph = data
    idx = idx.item()
    graph = graph.to(device)
    pred = info_all['preds'][idx]
    succ = info_all['succs'][idx]
    reads = info_all['reads'][idx]
    edges = info_all['edges'][idx]
    reference = None
    return idx, graph, pred, succ, reads, reference, edges


def load_graph_data(num_graphs, data_path, load_solutions=False):
    info_all = {
        'preds': [],
        'succs': [],
        'reads': [],
        'edges': [],
        'walks': [],
    }
    for idx in range(num_graphs):
        p, s = get_neighbors_dicts(idx, data_path)
        info_all['preds'].append(p)
        info_all['succs'].append(s)
        r = get_reads(idx, data_path)
        info_all['reads'].append(r)
        e = get_edges(idx, data_path)
        info_all['edges'].append(e)

        if load_solutions:
            w = get_walks(idx, data_path)
            info_all['walks'].append(w)
        
    return info_all


def print_graph_info(idx, graph):
    """Print the basic information for the graph with index idx."""
    print('\n---- GRAPH INFO ----')
    print('Graph index:', idx)
    print('Number of nodes:', graph.num_nodes())
    print('Number of edges:', len(graph.edges()[0]))


def print_prediction(walk, current, neighbors, actions, choice, best_neighbor):
    """Print summary of the prediction for the current position."""
    print('\n-----predicting-----')
    print('previous:\t', None if len(walk) < 2 else walk[-2])
    print('current:\t', current)
    print('neighbors:\t', neighbors[current])
    print('actions:\t', actions.tolist())
    print('choice:\t\t', choice)
    print('ground truth:\t', best_neighbor)
