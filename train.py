import argparse
from datetime import datetime
import copy
import os
from posixpath import split

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch.profiler import profile, record_function, ProfilerActivity
from dgl.dataloading import GraphDataLoader
import wandb

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import utils

import dgl

def get_dataloaders(ds, batch_size, is_eval, ratio):
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
    is_eval : bool
        True if only the evaluation perfomed and training is skipped
    ratio : float
        Ratio how to split the dataset into train/valid/test datasets

    Returns
    -------
    torch.DataLoader
        a dataloader for the training set, None if is_eval
    torch.DataLoader
        a dataloader for the validation set, None if is_eval
    torch.DataLoader
        a dataloader for the testing set
    """
    if is_eval:
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


def save_checkpoint(epoch, model, optimizer, loss_train, loss_valid, out):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
    }
    ckpt_path = f'checkpoints/{out}.pt'
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(out, model, optimizer):
    ckpt_path = f'checkpoints/{out}.pt'
    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    loss_train = checkpoint['loss_train']
    loss_valid = checkpoint['loss_valid']
    return epoch, model, optimizer, loss_train, loss_valid


def process(model, graph, neighbors, reads, walks, edges, criterion, optimizer, scaler, epoch, norm, device):
    """Process the graph by predicting the correct next neighbor.
    
    A graph is processed by simulating a walk over it where the 
    best next neighbor is predicted any time a branching occurs.
    The choices are compared tothe ground truth and loss is calculated.
    The list of losses and accuracy for the given graph are returned.
    """
    walk_length = get_hyperparameters()['walk_length']
    use_amp = get_hyperparameters()['use_amp']

    per_walk_loss = []
    per_walk_acc = []

    for solution in walks:
        if len(solution) < 10:
            continue
        ground_truth = {n1: n2 for n1, n2 in zip(solution[:-1], solution[1:])}
        ground_truth[solution[-1]] = None
        total_steps = len(solution) - 1

        if walk_length != -1:
            num_mini_batches = total_steps // walk_length + 1
        else:
            num_mini_batches = 1

        mini_batch_loss_list = []
        mini_batch_acc_list= []

        correct = 0

        for mini_batch in range(num_mini_batches):

            start = mini_batch * walk_length
            current = solution[start]

            loss_list = []
            total_loss = 0
            steps = 0

            # One forward pass per mini-batch
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(graph, reads, norm)

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

            # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            # prof.export_chrome_trace('trace.json')

            # TODO: Total loss should never be 0
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                # with record_function('model_backward_pass'):
            if model.training and total_loss > 0:
                total_loss /= steps
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()  # Backprop averaged (summed) losses for one mini-walk
                    optimizer.step()

            # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            # exit()

            if len(loss_list) == 0:
                continue
            else:
                # accuracy = correct / steps
                mini_batch_loss_list.append(np.mean(loss_list))  # List of mean loss per mini-walk
                # mini_batch_acc_list.append(accuracy)

        per_walk_loss.append(np.mean(mini_batch_loss_list))  # List of mean loss per solution-walk
        per_walk_acc.append(correct / total_steps)  # List of accuracies per solution-walk

    return per_walk_loss, per_walk_acc


def process_gt_graph(model, graph, neighbors, edges, criterion, optimizer, scaler, epoch, norm, device, nodes_gt, edges_gt):

    use_amp = get_hyperparameters()['use_amp']

    nodes_gt = torch.tensor([1 if i in nodes_gt else 0 for i in range(graph.num_nodes())], dtype=torch.float).to(device)
    edges_gt = torch.tensor([1 if i in edges_gt else 0 for i in range(graph.num_edges())], dtype=torch.float).to(device)

    losses = []
    accuracies = []
    
    node_criterion = nn.BCEWithLogitsLoss()
    edge_pos_weight = torch.tensor([1/25], device=device)
    edge_criterion = nn.BCEWithLogitsLoss(pos_weight=None)

    edges_p = model(graph, None)
    # start_end = slice(batch*batch_size, (batch+1)*batch_size)
    edge_loss = edge_criterion(edges_p.squeeze(-1), edges_gt)
    loss = edge_loss
    optimizer.zero_grad()
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        edge_loss.backward()
        optimizer.step()

    edges_predict = torch.round(torch.sigmoid(edges_p.squeeze(-1)))

    TP = torch.sum(torch.logical_and(edges_predict==1, edges_gt==1)).item()
    TN = torch.sum(torch.logical_and(edges_predict==0, edges_gt==0)).item()
    FP = torch.sum(torch.logical_and(edges_predict==1, edges_gt==0)).item()
    FN = torch.sum(torch.logical_and(edges_predict==0, edges_gt==1)).item()

    recall = TP / (TP + FP)
    precision = TP / (TP + FN)
    f1 = TP / (TP + 0.5 * (FP + FN) )
    # f1 = 2 * precision * recall / (precision + recall)

    edge_accuracy = (edges_predict == edges_gt).sum().item() / graph.num_edges()

    # accuracy = (node_accuracy + edge_accuracy) / 2
    accuracy = edge_accuracy
    losses.append(loss.item())
    accuracies.append(accuracy)
    wandb.log({'loss': loss.item(), 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
    print(f'{TP=}, {TN=}, {FP=}, {FN=}')

    return losses, accuracies


def process_gt_graph_batched(model, graph, neighbors, edges, criterion, optimizer, scaler, epoch, norm, device, nodes_gt, edges_gt):

    # use_amp = get_hyperparameters()['use_amp']

    nodes_gt = torch.tensor([1 if i in nodes_gt else 0 for i in range(graph.num_nodes())], dtype=torch.float).to(device)
    edges_gt = torch.tensor([1 if i in edges_gt else 0 for i in range(graph.num_edges())], dtype=torch.float).to(device)

    losses = []
    accuracies = []

    node_criterion = nn.BCEWithLogitsLoss()
    edge_pos_weight = torch.tensor([1/25], device=device)
    edge_criterion = nn.BCEWithLogitsLoss(pos_weight=None)

    batch_size = 128
    num_batches = graph.num_nodes() // batch_size
    for batch in range(num_batches-1):
        nodes_p, edges_p = model(graph, None)
        start_end = slice(batch*batch_size, (batch+1)*batch_size)
        # node_loss = node_criterion(nodes_p.squeeze(-1), nodes_gt)
        edge_loss = edge_criterion(edges_p.squeeze(-1), edges_gt)
        # loss = node_loss + edge_loss
        loss = edge_loss
        optimizer.zero_grad()
        edge_loss.backward()
        optimizer.step()
        # print(nodes_p)

        # node_accuracy = (torch.round(torch.sigmoid(nodes_p.squeeze(-1))) == nodes_gt).sum().item() / graph.num_nodes()
        edge_accuracy = (torch.round(torch.sigmoid(edges_p.squeeze(-1))) == edges_gt).sum().item() / graph.num_edges()

        # accuracy = (node_accuracy + edge_accuracy) / 2
        accuracy = edge_accuracy
        losses.append(loss.item())
        accuracies.append(accuracy)
        # wandb.log({'train_loss': loss.item(), 'train_accuracy': accuracy})

    return losses, accuracies


def process_reads(reads, device):
    processed_reads = {}
    for id, read in reads.items():
        read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
        read = ' '.join(read).split()
        read = torch.tensor(list(map(int, read)), device=device)
        processed_reads[id] = read
    return processed_reads



def process_graph(g):
    g.ndata['x'] = torch.ones(g.num_nodes(), node_features)
    g.edata['e'] = torch.cat((g.edata['overlap_length'], g.edata['overlap_similarity']), dim=1)
    pass

def train_new(args):
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    dim_latent = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience_limit = hyperparameters['patience_limit']
    lr = hyperparameters['lr']
    device = hyperparameters['device']
    use_reads = hyperparameters['use_reads']
    use_amp = hyperparameters['use_amp']

    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    data_path = os.path.abspath(args.data)
    out = args.out if args.out is not None else timestamp
    is_eval = args.eval
    is_split = args.split

    utils.set_seed(seed)
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_gnn_layers)
    
    if is_split:
        ds_train = graph_dataset.AssemblyGraphDataset(os.path.join(data_path, 'train'))
        ds_valid = graph_dataset.AssemblyGraphDataset(os.path.join(data_path, 'valid'))
        num_graphs = len(ds_train) + len(ds_valid)
    else:
        ds = AssemblyGraphDataset(data_path)
        # TODO: Only a temporary stupid fix, have to decide later how to make it proper
        ds_train = ds
        num_graphs = len(ds)

    overfit = num_graphs == 1

    node_features = 1
    edge_features = 2
    hidden_features = dim_latent

    model = models.Model(node_features, edge_features, hidden_features, num_gnn_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    best_model = models.Model(node_features, edge_features, hidden_features, num_gnn_layers)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.eval()

    # Don't need normalization here, do that in preprocessing

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')


    for epoch in range(num_epochs):

        loss_per_graph, acc_per_graph = [], []
        for data in ds_train:
            model.train()
            idx, g = data
            g = dgl.add_self_loop(g)
        
            dl = dgl.dataloading.EdgeDataLoader(
                g, torch.arange(g.num_edges()), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0)

            # TODO:
            # Remove this later, features should be determined in preprocessing,
            # Not in every training epoch
            g.ndata['x'] = torch.ones(g.num_nodes(), node_features)

            # First I need to normalize length and similarity, though
            # This should also be done in preprocessing
            g.edata['e'] = torch.cat((g.edata['overlap_length'].unsqueeze(-1), g.edata['overlap_similarity'].unsqueeze(-1)), dim=1)

            # This should also be done in preprocessing
            nodes_gt, edges_gt = utils. get_correct_ne(idx, data_path)
            g.edata['y'] = torch.tensor([1 if i in edges_gt else 0 for i in range(g.num_edges())], dtype=torch.float)

            step_loss, step_acc = [], []

            for input_nodes, edge_subgraph, blocks in tqdm(dl):
                blocks = [b.to(device) for b in blocks]
                edge_subgraph = edge_subgraph.to(device)
                x = blocks[0].srcdata['x']
                # For GNN edge feautre update, I need edge data from block[0]
                e = edge_subgraph.edata['e'].to(device)
                edge_labels = edge_subgraph.edata['y'].to(device)
                edge_predictions = model(edge_subgraph, blocks, x, e)

                edge_predictions = edge_predictions.squeeze(-1)
                print(edge_predictions.shape)
                print(edge_labels.shape)
                loss = criterion(edge_predictions, edge_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct = (edge_labels == torch.round(torch.sigmoid(edge_predictions))).sum().item()
                accuracy = correct / edge_labels.shape[0]
                step_loss.append(loss.item())
                step_acc.append(accuracy)

            loss_per_graph.append(np.mean(step_loss))
            acc_per_graph.append(np.mean(step_acc))

            elapsed = utils.timedelta_to_str(datetime.now() - time_start)
            print(f'\nTraining: epoch = {epoch}, graph = {idx}')
            print(f'Train loss: {train_loss:.4f},\tTrain accuracy: {train_acc:.4f}')
            print(f'Elapsed time: {elapsed}\n')


        if not overfit:
            with torch.no_grad():
                model.eval()
                ...


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
    use_reads = hyperparameters['use_reads']
    use_amp = hyperparameters['use_amp']

    utils.set_seed(seed)

    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    data_path = os.path.abspath(args.data)
    out = args.out if args.out is not None else timestamp
    is_eval = args.eval
    is_split = args.split

    # Get the dataloaders
    if is_split:
        ds_train = AssemblyGraphDataset(os.path.join(data_path, 'train'))
        ds_valid = AssemblyGraphDataset(os.path.join(data_path, 'valid'))
        ds_test = AssemblyGraphDataset(os.path.join(data_path, 'test'))
        dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)
        dl_test = GraphDataLoader(ds_test, batch_size=batch_size, shuffle=False)
        num_graphs = len(ds_train) + len(ds_valid)  # ???
    else:
        ds = AssemblyGraphDataset(data_path)
        dl_train, dl_valid, dl_test = get_dataloaders(ds, batch_size, is_eval, ratio=0.2)
        num_graphs = len(ds)

    overfit = num_graphs == 1

    # Initialize training specifications
    model = models.NonAutoRegressive_gt_graph(dim_latent, num_gnn_layers).to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    model_path = os.path.abspath(f'pretrained/model_{out}.pt')
    criterion = nn.CrossEntropyLoss()

    # Initialize best model obtained during the training process
    best_model = models.NonAutoRegressive_gt_graph(dim_latent, num_gnn_layers)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.to(device)
    best_model.eval()

    # TODO: For full chromosomes, this will probalby be too large to store in memory
    if is_split:
        info_all_train = utils.load_graph_data(len(ds_train), data_path+'/train', use_reads)
        info_all_valid = utils.load_graph_data(len(ds_valid), data_path+'/valid', use_reads)
    else:
        info_all = utils.load_graph_data(num_graphs, data_path, use_reads)

    # Normalization of the training set
    normalize_tensor = torch.cat([graph.edata['overlap_length'] for _, graph in dl_train]).float()
    norm_mean, norm_std = torch.mean(normalize_tensor), torch.std(normalize_tensor)
    norm_train = (norm_mean.item(), norm_std.item())

    # Normalize validation set if it exists
    if len(dl_valid) > 0:
        normalize_tensor = torch.cat([graph.edata['overlap_length'] for _, graph in dl_valid]).float()
        norm_mean, norm_std = torch.mean(normalize_tensor), torch.std(normalize_tensor)
        norm_valid = (norm_mean.item(), norm_std.item())
    else:
        norm_valid = None

    # Normalize testing set if it exists
    if len(dl_test) > 0:
        normalize_tensor = torch.cat([graph.edata['overlap_length'] for _, graph in dl_test]).float()
        norm_mean, norm_std = torch.mean(normalize_tensor), torch.std(normalize_tensor)
        norm_test = (norm_mean.item(), norm_std.item())
    else:
        norm_test = None

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    scaler = torch.cuda.amp.GradScaler()

    try:
        with wandb.init(project="assembly-walk-nips", config=hyperparameters):
            # wandb.watch(model, criterion, log='all', log_freq=1)

            patience = 0
            out_of_patience = False  # TODO: Remove, I probably don't need both patience and scheduler
            loss_per_epoch_train, loss_per_epoch_valid = [], []
            accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []

            # --- Training ---
            for epoch in range(num_epochs):
                model.train()
                print(f'Epoch: {epoch}')
                if out_of_patience:
                    print('Out of patience!')
                    break
                patience += 1
                loss_per_graph = []
                accuracy_per_graph = []
                for data in tqdm(dl_train):
                    if is_split:
                        idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all_train, use_reads)
                    else:
                        idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all, use_reads)
                    graph = graph.to(device)
                    if use_reads:
                        reads = process_reads(reads, device)
                    if is_split:
                        solution = utils.get_walks(idx, data_path + '/train')
                        nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path + '/train')
                    else:
                        solution = utils.get_walks(idx, data_path)
                        nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path)

                    utils.print_graph_info(idx, graph)
                    # loss_list, accuracy_list = process(model, graph, succ, reads, solution, edges, criterion, optimizer, scaler, epoch, norm_train, device=device)
                    loss_list, accuracy_list = process_gt_graph(model, graph, succ, edges, criterion, optimizer, scaler, epoch, norm_train, device, nodes_gt, edges_gt)
                    if loss_list is not None:
                        loss_per_graph.append(np.mean(loss_list))
                        accuracy_per_graph.append(np.mean(accuracy_list))

                    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                    print(f'Processing graph {idx} done. Elapsed time: {elapsed}')

                train_loss = np.mean(loss_per_graph)
                train_acc = np.mean(accuracy_per_graph)
                loss_per_epoch_train.append(train_loss)
                accuracy_per_epoch_train.append(train_acc)

                # try:
                #     wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc}, step=epoch)
                # except Exception:
                #     pass

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\nTraining in epoch {epoch} done. Elapsed time: {elapsed}')
                print(f'Train loss: {train_loss},\tTrain accuracy: {train_acc}\n')

                # Save the best model and the checkpoint
                # Only for overfitting, usually done after validation
                if overfit:
                    if len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        best_model.to(device)
                        torch.save(best_model.state_dict(), model_path)
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out)
                    scheduler.step(train_loss)

                    # Taken from the older training code, see if you can delete this
                    # last_losses = loss_per_epoch_train[-6:-1]
                    # if len(loss_per_epoch_train) > 10 and loss_per_epoch_train[-1] <= min(last_losses):
                    #     print(f'epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"] / 5}')
                    #     for g in optimizer.param_groups:
                    #         g['lr'] /= 5

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # !!!! This should probably go after validation !!!!
                    # save_checkpoint(epoch, model, optimizer, loss)
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                # --- Validation ---
                if not overfit:
                    with torch.no_grad():
                        print('VALIDATION')
                        model.eval()
                        loss_per_graph = []
                        accuracy_per_graph = []
                        for data in dl_valid:
                            if is_split:
                                idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all_valid, use_reads)
                            else:
                                idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all, use_reads)
                            graph = graph.to(device)
                            if use_reads:
                                reads = process_reads(reads, device)
                            if is_split:
                                solution = utils.get_walks(idx, data_path + '/valid')
                                nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path + '/valid')
                            else:
                                solution = utils.get_walks(idx, data_path)
                                nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path)

                            utils.print_graph_info(idx, graph)
                            loss_list, accuracy_list = process(model, graph, succ, reads, solution, edges, criterion, optimizer, scaler, epoch, norm_valid, device=device)
                            if loss_list is not None:
                                loss_per_graph.append(np.mean(loss_list))
                                accuracy_per_graph.append(np.mean(accuracy_list))

                            elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                            print(f'Processing graph {idx} done. Elapsed time: {elapsed}')

                        # Save model, plot loss, etc.
                        valid_loss = np.mean(loss_per_graph)
                        valid_acc = np.mean(accuracy_per_graph)
                        loss_per_epoch_valid.append(valid_loss)
                        accuracy_per_epoch_valid.append(valid_acc)

                        try:
                            wandb.log({'epoch': epoch, 'valid_loss': valid_loss, 'valid_accuracy': valid_acc}, step=epoch)
                        except Exception:
                            pass

                        if len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                            patience = 0
                            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                            best_model.to(device)
                            torch.save(best_model.state_dict(), model_path)

                        # TODO: Probably should be removed, I have scheduler for this
                        elif patience >= patience_limit:
                            # out_of_patience = True  # Delete later
                            # Not sure about this
                            #####
                            if learning_rate <= 5e-8:
                                out_of_patience = True
                            else:
                                patience = 0
                            #     learning_rate /= 10
                            ######

                        if len(loss_per_epoch_train) > 0 and len(loss_per_epoch_valid) > 0:
                            save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], loss_per_epoch_valid[-1], out)

                        scheduler.step(valid_loss)

                        elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                        print(f'\nValidation in epoch {epoch} done. Elapsed time: {elapsed}\n')
                        print(f'Valid loss: {valid_loss},\tValid accuracy: {valid_acc}\n')

            # TODO: Deprecated, see what you'll do with this
            utils.draw_loss_plots(loss_per_epoch_train, loss_per_epoch_valid, out)
            utils.draw_accuracy_plots(accuracy_per_epoch_train, accuracy_per_epoch_valid, out)
            
            try:
                # TODO: This doesn't seem to work
                wandb.save("model.onnx")
            except Exception:
                print("W&B Error: Did not save the model.onnx")

            # TODO: Why are you not loading the model anymore, wtf

            # --- Testing ---
            if not overfit and False:  # No testing needed now
                with torch.no_grad():
                    test_accuracy = []
                    print('TESTING')
                    model.eval()
                    for i, data in enumerate(dl_test):
                        idx, graph, pred, succ, reads, edges = utils.unpack_data(data, info_all, use_reads)
                        graph = graph.to(device)
                        if use_reads:
                            reads = process_reads(reads, device)
                        if is_split:
                            solution = utils.get_walks(idx, data_path + '/test')
                            nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path + '/test')
                        else:
                            solution = utils.get_walks(idx, data_path)
                            nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path)

                        utils.print_graph_info(idx, graph)
                        loss_list, accuracy_list = process(best_model, graph, succ, reads, solution, edges, criterion, optimizer, scaler, epoch, norm_test, device=device)
                        test_accuracy.append(np.mean(accuracy_list))

                        try:
                            wandb.log({'test_graph': idx, 'test_accuracy': np.mean(accuracy_list)}, step=i)
                        except Exception:
                            print(f'test_graph: {idx}, test_accuracy: {np.mean(accuracy_list)}')
                            pass

                    test_accuracy = np.mean(test_accuracy)
                    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                    print(f'\nTesting done. Elapsed time: {elapsed}')
                    print(f'Average accuracy on the test set:', test_accuracy)

                print('Testing done')

    except KeyboardInterrupt:
        # TODO: Implement this to do something, maybe evaluate on test set?
        print("Keyboard Interrupt...")
        print("Exiting...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train', help='Path to directory with training data')
    parser.add_argument('--out', type=str, default=None, help='Output name for figures and models')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--split', action='store_true', default=False, help='Is the dataset already split into train/valid/test')
    args = parser.parse_args()
    train_new(args)
