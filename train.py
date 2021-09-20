import argparse
from datetime import datetime
import copy
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader
import wandb

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


def save_checkpoint(epoch, model, optimizer, loss, out):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss': loss,
    }
    ckpt_path = f'checkpoints/{out}.pt'
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(out, model, optimizer):
    ckpt_path = f'checkpoints/{out}.pt'
    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def process(model, graph, neighbors, reads, solution, edges, criterion, optimizer, epoch, device):
    """Process the graph by predicting the correct next neighbor.
    
    A graph is processed by simulating a walk over it where the 
    best next neighbor is predicted any time a branching occurs.
    The choices are compared tothe ground truth and loss is calculated.
    The list of losses and accuracy for the given graph are returned.
    """
    walk_length = get_hyperparameters()['walk_length']

    ground_truth = {n1: n2 for n1, n2 in zip(solution[:-1], solution[1:])}
    ground_truth[solution[-1]] = None
    total_steps = len(solution) - 1

    # if model.training:
    #     start = (epoch * walk_length) % total_steps if walk_length != -1 else 0
    # else:
    #     start = 0
    #     walk_length = -1

    # current = solution[start]

    # loss_list = []
    # total_loss = 0
    # correct = 0
    # steps = 0

    if walk_length != -1:
        num_mini_batches = total_steps // walk_length + 1
    else:
        num_mini_batches = 1

    mini_batch_loss_list = []
    mini_batch_acc_list= []

    for mini_batch in range(num_mini_batches):

        start = mini_batch * walk_length

        current = solution[start]

        loss_list = []
        total_loss = 0
        correct = 0
        steps = 0

        logits = model(graph, reads)

        print('Iterating through nodes!')
        while True:
            if steps == walk_length or ground_truth[current] is None:
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
            current = best_neighbor  # Teacher forcing
            steps += 1

        if model.training and total_loss > 0:
            optimizer.zero_grad()
            total_loss.backward()  # Backprop summed losses
            optimizer.step()

        if len(loss_list) == 0:
            continue
        else:
            accuracy = correct / steps
            mini_batch_loss_list.append(np.mean(loss_list))
            mini_batch_acc_list.append(accuracy)

    accuracy = np.mean(mini_batch_acc_list)
    return mini_batch_loss_list, accuracy


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
    seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience_limit = hyperparameters['patience_limit']
    learning_rate = hyperparameters['lr']
    device = hyperparameters['device']

    utils.set_seed(seed)

    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    data_path = os.path.abspath(args.data)
    out = args.out if args.out is not None else timestamp
    eval = args.eval

    ds = AssemblyGraphDataset(data_path)
    dl_train, dl_valid, dl_test = get_dataloaders(ds, batch_size, eval, ratio=0.2)
    num_graphs = len(ds)

    model = models.NonAutoRegressive(dim_latent, num_gnn_layers).to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    model_path = os.path.abspath(f'pretrained/model_{out}.pt')

    best_model = models.NonAutoRegressive(dim_latent, num_gnn_layers)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.to(device)
    best_model.eval()

    info_all = utils.load_graph_data(num_graphs, data_path)

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    with wandb.init(project="assembly-walk", config=hyperparameters):

        criterion = nn.CrossEntropyLoss()
        wandb.watch(model, criterion, log='all', log_freq=1)

        patience = 0
        loss_per_epoch_train, loss_per_epoch_valid = [], []
        accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

        # --- Training ---
        for epoch in tqdm(range(num_epochs)):
            model.train()
            print(f'Epoch: {epoch}')
            patience += 1
            loss_per_graph = []
            accuracy_per_graph = []
            for data in dl_train:
                idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all)
                graph = graph.to(device)
                solution = utils.get_walks(idx, data_path)

                utils.print_graph_info(idx, graph)
                loss_list, accuracy = process(model, graph, succ, reads, solution, edges, criterion, optimizer, epoch, device=device)
                if loss_list is not None:
                    loss_per_graph.append(np.mean(loss_list))
                    accuracy_per_graph.append(accuracy)

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'Processing graph {idx} done. Elapsed time: {elapsed}')

            train_loss = np.mean(loss_per_graph)
            train_acc = np.mean(accuracy_per_graph)
            loss_per_epoch_train.append(np.mean(loss_per_graph))
            accuracy_per_epoch_train.append(np.mean(accuracy_per_graph))
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc}, step=epoch)

            elapsed = utils.timedelta_to_str(datetime.now() - time_start)
            print(f'\nTraining in epoch {epoch} done. Elapsed time: {elapsed}\n')

            # !!!!!!!!!!! Only for overfitting - REMOVE LATER !!!!!!!!!!!
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
            best_model.to(device)
            torch.save(best_model.state_dict(), model_path)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # !!!! This should probably go after validation !!!!
            save_checkpoint(epoch, model, optimizer, loss)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
                    loss_list, accuracy = process(model, graph, succ, reads, solution, edges, criterion, optimizer, epoch, device=device)
                    if loss_list is not None:
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

                if len(loss_per_graph) > 0:
                    valid_loss = np.mean(loss_per_graph)
                    valid_acc = np.mean(accuracy_per_graph)
                    loss_per_epoch_valid.append(np.mean(loss_per_graph))
                    accuracy_per_epoch_valid.append(np.mean(accuracy_per_graph))
                    wandb.log({'epoch': epoch, 'valid_loss': valid_loss, 'valid_accuracy': valid_acc}, step=epoch)

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\nValidation in epoch {epoch} done. Elapsed time: {elapsed}\n')

        utils.draw_loss_plots(loss_per_epoch_train, loss_per_epoch_valid, out)
        utils.draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, out)

        torch.save(best_model.state_dict(), model_path)
        wandb.save("model.onnx")

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
                loss_list, accuracy = process(best_model, graph, succ, reads, solution, edges, criterion, optimizer, epoch, device=device)
                test_accuracy.append(accuracy)

            if len(test_accuracy) > 0:
                test_accuracy = np.mean(test_accuracy)
                wandb.log({"test_accuracy": test_accuracy})
                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\nTesting done. Elapsed time: {elapsed}')
                print(f'Average accuracy on the test set:', test_accuracy)

        print('Testing done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train', help='path to directory with training data')
    parser.add_argument('--out', type=str, default=None, help='Output name for figures and models')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    train(args)
