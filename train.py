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
import dgl
from dgl.dataloading import GraphDataLoader
import wandb

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import utils


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


def process_reads(reads, device):
    processed_reads = {}
    for id, read in reads.items():
        read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
        read = ' '.join(read).split()
        read = torch.tensor(list(map(int, read)), device=device)
        processed_reads[id] = read
    return processed_reads


def train(args):
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    batch_size = hyperparameters['batch_size']
    patience = hyperparameters['patience']
    lr = hyperparameters['lr']
    device = hyperparameters['device']
    use_reads = hyperparameters['use_reads']
    use_amp = hyperparameters['use_amp']

    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    decay = hyperparameters['decay']

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

    if batch_size == -1:
        model = models.GraphGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
        best_model = models.GraphGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
    else:
        model = models.BlockGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
        best_model = models.BlockGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)

    model.to(device)
    model_path = os.path.abspath(f'pretrained/model_{out}.pt')
    best_model.to(device)  # TODO: IF I really need to save on memory, maybe not do this
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=patience, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    acc_per_epoch_train, acc_per_epoch_valid = [], []

    try:
        with wandb.init(project="GeNNome-april", config=hyperparameters):
            wandb.watch(model, criterion, log='all', log_freq=1000)

            for epoch in range(num_epochs):

                loss_per_graph, acc_per_graph = [], []

                print('TRAINING')
                for data in ds_train:
                    model.train()
                    idx, g = data

                    if batch_size == -1:
                        g = g.to(device)
                        x = g.ndata['x'].to(device)
                        e = g.edata['e'].to(device)
                        edge_predictions = model(g, x, e)
                        edge_predictions = edge_predictions.squeeze(-1)
                        edge_labels = g.edata['y'].to(device)
                        loss = criterion(edge_predictions, edge_labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        step_loss = [loss.item()]
                        TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                        print(f'\t{TP=}, {TN=}, {FP=}, {FN=}')
                        acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                        step_acc = [acc]

                        fp_rate = FP / (FP + TN)
                        print(f'FP-rate = {fp_rate}')

                        try:
                            wandb.log({'train_loss': loss.item(), 'train_accuracy': acc, 'train_precision': precision, \
                                           'train_recall': recall, 'train_f1': f1})
                        except Exception:
                            print(f'WandB exception occured!')

                    else:
                        graph_ids = torch.arange(g.num_edges()).int()
                        dl = dgl.dataloading.EdgeDataLoader(
                            g, graph_ids, sampler,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=0)

                        step_loss, step_acc = [], []

                        for input_nodes, edge_subgraph, blocks in tqdm(dl):
                            blocks = [b.to(device) for b in blocks]
                            edge_subgraph = edge_subgraph.to(device)
                            x = blocks[0].srcdata['x']
                            # TODO: For GNN edge feature update, I need edge data from block[0]
                            e_0 = blocks[0].edata['e'].to(device)
                            e_subgraph = edge_subgraph.edata['e'].to(device)  # e = blocks[0].edata['e'].to(device)
                            # TODO: What I said above, read your own comments moron
                            edge_labels = edge_subgraph.edata['y'].to(device)
                            edge_predictions = model(edge_subgraph, blocks, x, e_0, e_subgraph)
                            edge_predictions = edge_predictions.squeeze(-1)
                            loss = criterion(edge_predictions, edge_labels)
                            optimizer.zero_grad()

                            if use_amp:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()

                            TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                            print(f'{TP=}, {TN=}, {FP=}, {FN=}')
                            acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)

                            fp_rate = FP / (FP + TN)
                            print(f'FP-rate = {fp_rate}')

                            step_loss.append(loss.item())
                            step_acc.append(acc)

                            try:
                                wandb.log({'train_loss': loss.item(), 'train_accuracy': acc, 'train_precision': precision, \
                                           'train_recall': recall, 'train_f1': f1})
                            except Exception:
                                print(f'WandB exception occured!')

                    loss_per_graph.append(np.mean(step_loss))
                    acc_per_graph.append(np.mean(step_acc))

                    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                    # print(f'\nTRAINING: Epoch = {epoch}, Graph = {idx}')
                    # print(f'Loss: {train_loss:.4f},\tAccuracy: {train_acc:.4f}', end='')
                    # print(f'Precision: {precision:.4f},\tRecall: {recall:.4f},\tF1: {f1:.4f}')
                    # print(f'Elapsed time: {elapsed}\n')

                train_loss = np.mean(loss_per_graph)
                train_acc = np.mean(acc_per_graph)
                loss_per_epoch_train.append(train_loss)
                acc_per_epoch_train.append(train_acc)

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\nTraining in epoch {epoch} done. Elapsed time: {elapsed}')
                print(f'Train loss mean: {train_loss:.4f},\tTrain accuracy mean: {train_acc:.4f}\n')

                if overfit:
                    if len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        torch.save(best_model.state_dict(), model_path)
                    # TODO: Check what's going on here
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out)
                    scheduler.step(train_loss)

                if not overfit:
                    with torch.no_grad():
                        print('VALIDATION')
                        model.eval()
                        for data in ds_valid:
                            idx, g = data
                            # TODO: Do I need to add self loops? Why yes or why not?
                            # g = dgl.add_self_loop(g)
                            
                            if batch_size == -1:
                                g = g.to(device)
                                x = g.ndata['x'].to(device)
                                e = g.edata['e'].to(device)
                                edge_predictions = model(g, x, e)
                                edge_predictions = edge_predictions.squeeze(-1)
                                edge_labels = g.edata['y'].to(device)
                                loss = criterion(edge_predictions, edge_labels)

                                step_loss = [loss.item()]
                                TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                                print(f'\t{TP=}, {TN=}, {FP=}, {FN=}')
                                acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                                step_acc = [acc]

                                fp_rate = FP / (FP + TN)
                                print(f'FP-rate = {fp_rate}')

                                try:
                                    wandb.log({'valid_loss': loss.item(), 'valid_accuracy': acc, 'valid_precision': precision, \
                                            'valid_recall': recall, 'valid_f1': f1})
                                except Exception:
                                    print(f'WandB exception occured!')

                            else:
                                graph_ids = torch.arange(g.num_edges()).int()
                                dl = dgl.dataloading.EdgeDataLoader(
                                    g, graph_ids, sampler,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=0)

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
                                    loss = criterion(edge_predictions, edge_labels)

                                    TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                                    print(f'{TP=}, {TN=}, {FP=}, {FN=}')
                                    acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)

                                    step_loss.append(loss.item())
                                    step_acc.append(acc)

                                    fp_rate = FP / (FP + TN)
                                    print(f'FP-rate = {fp_rate}')

                                    try:
                                        wandb.log({'valid_loss': loss.item(), 'valid_accuracy': acc, 'valid_precision': precision, \
                                                'valid_recall': recall, 'valid_f1': f1})
                                    except Exception:
                                        print(f'WandB exception occured!')
                            
                            loss_per_graph.append(np.mean(step_loss))
                            acc_per_graph.append(np.mean(step_acc))

                        valid_loss = np.mean(loss_per_graph)
                        valid_acc = np.mean(acc_per_graph)
                        loss_per_epoch_valid.append(valid_loss)
                        acc_per_epoch_valid.append(valid_acc)

                        elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                        print(f'\nValidation in epoch {epoch} done. Elapsed time: {elapsed}\n')
                        print(f'Valid loss: {valid_loss},\tValid accuracy: {valid_acc}\n')

                        if len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                            torch.save(best_model.state_dict(), model_path)
                        save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], loss_per_epoch_valid[-1], out)
                        scheduler.step(valid_loss)

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
    train(args)
