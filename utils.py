import os
import pickle
import random

import torch
# import matplotlib
# # matplotlib.use('Agg')
# matplotlib.interactive(True)
# import matplotlib.pyplot as plt
import numpy as np
import dgl


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
    pass
    # Removed because importing matplotlib causes terminal to hang in some situations
    # plt.figure()
    # plt.plot(train_loss, label='train')
    # plt.plot(valid_loss, label='validation')
    # plt.title('Loss over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig(f'figures/loss_{out}.png')


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
    pass
    # Removed because importing matplotlib causes terminal to hang in some situations
    # plt.figure()
    # plt.plot(train_acc, label='train')
    # plt.plot(valid_acc, label='validation')
    # plt.title('Accuracy over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig(f'figures/accuracy_{out}.png')


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
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


# def get_paths(start, neighbors, num_nodes):
#     """Return all the possible walks from a current node of length
#     num_nodes.
    
#     Parameters
#     ----------
#     start : int
#         Index of the starting node
#     neighbors : dict
#         Dictionary with the list of neighbors for each node
#     num_nodes : int
#         Length of the walks to be returned

#     Returns
#     -------
#     list
#         a list of all the possible walks, where each walk is also
#         stored in a list with num_nodes consecutive nodes
#     """
#     if num_nodes == 0:
#         return [[start]]
#     paths = []
#     for neighbor in neighbors[start]:
#         next_paths = get_paths(neighbor, neighbors, num_nodes-1)
#         for path in next_paths:
#             path.append(start)
#             paths.append(path)
#     return paths


def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'


def get_walks(idx, data_path):
    # TODO: This implies there is only 1 walk
    walk_path = os.path.join(data_path, f'solutions/{idx}_gt.pkl')
    walks = pickle.load(open(walk_path, 'rb'))
    return walks


def get_info(idx, data_path, type):
    info_path = os.path.join(data_path, 'info', f'{idx}_{type}.pkl')
    info = pickle.load(open(info_path, 'rb'))
    return info


def unpack_data(data, info_all, use_reads):
    idx, graph = data
    idx = idx if isinstance(idx, int) else idx.item()
    pred = info_all['preds'][idx]
    succ = info_all['succs'][idx]
    if use_reads:
        reads = info_all['reads'][idx]
    else:
        reads = None
    edges = info_all['edges'][idx]
    return idx, graph, pred, succ, reads, edges


def load_graph_data(num_graphs, data_path, use_reads):
    info_all = {
        'preds': [],
        'succs': [],
        'reads': [],
        'edges': [],
    }
    for idx in range(num_graphs):
        info_all['preds'].append(get_info(idx, data_path, 'pred'))
        info_all['succs'].append(get_info(idx, data_path, 'succ'))
        if use_reads:
            info_all['reads'].append(get_info(idx, data_path, 'reads'))
        info_all['edges'].append(get_info(idx, data_path, 'edges'))
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
