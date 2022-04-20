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

    batch_norm = hyperparameters['batch_norm']

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

    #overfit = False # DEBUG !!!!!!!!!!!!!
    #ds_valid = ds_train # DEBUG !!!!!!!!!!!!!

    if batch_size == -1:
        model = models.GraphGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
        best_model = models.GraphGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
    else:
        #model = models.BlockGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers, batch_norm=batch_norm)
        #best_model = models.BlockGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers, batch_norm=batch_norm)
        model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers) # GatedGCN 
        best_model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers) # GatedGCN 

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

                train_loss_all_graphs, train_fp_rate_all_graphs, train_fn_rate_all_graphs = [], [], []
                train_acc_all_graphs, train_precision_all_graphs, train_recall_all_graphs, train_f1_all_graphs = [], [], [], []

                print('TRAINING')
                for data in ds_train:
                    model.train()
                    idx, g = data

                    if batch_size == -1: # train with full graph 
                                         # XB : This part of the code must be updated !

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

                    else: # train with mini-batch

                        # remove Metis clusters to force new clusters
                        try:
                            os.remove('cluster_gcn.pkl')
                        except:
                            pass 

                        # Run Metis
                        num_parts = 1000 # TODO : make as input argument
                        #batch_size = 20
                        g = g.long()
                        sampler = dgl.dataloading.ClusterGCNSampler(g, num_parts) 
                        dataloader = dgl.dataloading.DataLoader(g, torch.arange(num_parts), sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4) # XB
                        #for subg in dataloader:
                        #    print('mini-batch',subg)

                        # For loop over all mini-batch in the graph
                        running_loss, running_fp_rate, running_fn_rate = [], [], []
                        running_acc, running_precision, running_recall, running_f1 = [], [], [], []
                        for sub_g in dataloader:
                            sub_g = sub_g.to(device)
                            x = sub_g.ndata['x'].to(device)
                            e = sub_g.edata['e'].to(device)
                            #pe = sub_g.ndata['pe'].to(device)
                            edge_predictions = model(sub_g, x, e) 
                            #edge_predictions = model(sub_g, x, e, pe) 
                            edge_predictions = edge_predictions.squeeze(-1)
                            edge_labels = sub_g.edata['y'].to(device)
                            loss = criterion(edge_predictions, edge_labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            running_loss.append(loss.item())
                            TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                            acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                            fp_rate = FP / (FP + TN)
                            fn_rate = FN / (FN + TP)
                            running_fp_rate = [fp_rate]
                            running_fn_rate = [fn_rate]

                        # Average over all mini-batch in the graph
                        train_loss = np.mean(running_loss)
                        train_fp_rate = np.mean(running_fp_rate)
                        train_fn_rate = np.mean(running_fn_rate)
                        train_acc = np.mean(running_acc)
                        train_precision = np.mean(running_precision)
                        train_recall = np.mean(running_recall)
                        train_f1 = np.mean(running_f1)
                        lr_value = optimizer.param_groups[0]['lr']

                        elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                        print(f'\nTRAINING (one training graph): Epoch = {epoch}, Graph = {idx}')
                        print(f'Loss: {train_loss:.4f}, fp_rate(GT=0): {train_fp_rate:.4f}, fn_rate(GT=1): {train_fn_rate:.4f}')
                        print(f'lr_value: {lr_value:.6f}, elapsed time: {elapsed}\n')

                    # Record after each epoch
                    train_loss_all_graphs.append(train_loss)
                    train_fp_rate_all_graphs.append(train_fp_rate)
                    train_fn_rate_all_graphs.append(train_fn_rate)
                    train_acc_all_graphs.append(train_acc)
                    train_precision_all_graphs.append(train_precision)
                    train_recall_all_graphs.append(train_recall)
                    train_f1_all_graphs.append(train_f1)

                # Average over all training graphs
                train_loss_all_graphs = np.mean(train_loss_all_graphs)
                train_fp_rate_all_graphs = np.mean(train_fp_rate_all_graphs)
                train_fn_rate_all_graphs = np.mean(train_fn_rate_all_graphs)
                train_acc_all_graphs = np.mean(train_acc_all_graphs)
                train_precision_all_graphs = np.mean(train_precision_all_graphs)
                train_recall_all_graphs = np.mean(train_recall_all_graphs)
                train_f1_all_graphs = np.mean(train_f1_all_graphs)
                lr_value = optimizer.param_groups[0]['lr']

                loss_per_epoch_train.append(train_loss_all_graphs)

                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\nTRAINING (all training graphs): Epoch = {epoch}')
                print(f'Loss: {train_loss_all_graphs:.4f}, fp_rate(GT=0): {train_fp_rate_all_graphs:.4f}, fn_rate(GT=1): {train_fn_rate_all_graphs:.4f}')
                print(f'lr_value: {lr_value:.6f}, elapsed time: {elapsed}\n')

                try:
                    wandb.log({'train_loss': train_loss_all_graphs, 'train_accuracy': train_acc_all_graphs, 'train_precision': train_precision_all_graphs, \
                            'train_recall': train_recall_all_graphs, 'train_f1': train_f1_all_graphs, 'fp-rate': train_fp_rate_all_graphs, 'fn-rate': train_fn_rate_all_graphs, 'lr_value': lr_value})
                except Exception:
                    print(f'WandB exception occured!')

                if overfit: # temp : one graph at the moment
                    if len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        torch.save(best_model.state_dict(), model_path)
                    # TODO: Check what's going on here
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out)
                    scheduler.step(train_loss_all_graphs)

                if not overfit:

                    val_loss_all_graphs, val_fp_rate_all_graphs, val_fn_rate_all_graphs = [], [], []
                    val_acc_all_graphs, val_precision_all_graphs, val_recall_all_graphs, val_f1_all_graphs = [], [], [], []

                    with torch.no_grad():
                        print('===> VALIDATION')
                        time_start_eval = datetime.now()
                        model.eval()
                        for data in ds_valid:
                            idx, g = data

                            if batch_size == -1: # full graph 
                                                 # XB : This part of the code must be updated !

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

                            else: # mini-batch

                                # remove Metis clusters to force new clusters
                                try:
                                    os.remove('cluster_gcn.pkl')
                                except:
                                    pass 

                                # Run Metis
                                num_parts = 1000 # TODO : make as input argument
                                #batch_size = 20
                                g = g.long()
                                sampler = dgl.dataloading.ClusterGCNSampler(g, num_parts) 
                                dataloader = dgl.dataloading.DataLoader(g, torch.arange(num_parts), sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4) # XB
                                #for subg in dataloader:
                                #    print('mini-batch',subg)

                                # For loop over all mini-batch in the graph
                                running_loss, running_fp_rate, running_fn_rate = [], [], []
                                running_acc, running_precision, running_recall, running_f1 = [], [], [], []
                                for sub_g in dataloader:
                                    sub_g = sub_g.to(device)
                                    x = sub_g.ndata['x'].to(device)
                                    e = sub_g.edata['e'].to(device)
                                    #pe = sub_g.ndata['pe'].to(device)
                                    edge_predictions = model(sub_g, x, e) 
                                    #edge_predictions = model(sub_g, x, e, pe) 
                                    edge_predictions = edge_predictions.squeeze(-1)
                                    edge_labels = sub_g.edata['y'].to(device)
                                    loss = criterion(edge_predictions, edge_labels)
                                    running_loss.append(loss.item())
                                    TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                                    acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                                    fp_rate = FP / (FP + TN)
                                    fn_rate = FN / (FN + TP)
                                    running_fp_rate = [fp_rate]
                                    running_fn_rate = [fn_rate]

                                # Average over all mini-batch in the graph
                                val_loss = np.mean(running_loss)
                                val_fp_rate = np.mean(running_fp_rate)
                                val_fn_rate = np.mean(running_fn_rate)
                                val_acc = np.mean(running_acc)
                                val_precision = np.mean(running_precision)
                                val_recall = np.mean(running_recall)
                                val_f1 = np.mean(running_f1)

                                elapsed = utils.timedelta_to_str(datetime.now() - time_start_eval)
                                print(f'\n===> VALIDATION (one validation graph): Epoch = {epoch}, Graph = {idx}')
                                print(f'Loss: {val_loss:.4f}, fp_rate(GT=0): {val_fp_rate:.4f}, fn_rate(GT=1): {val_fn_rate:.4f}')
                                print(f'elapsed time: {elapsed}\n')

                            # Record after each epoch
                            val_loss_all_graphs.append(val_loss)
                            val_fp_rate_all_graphs.append(val_fp_rate)
                            val_fn_rate_all_graphs.append(val_fn_rate)
                            val_acc_all_graphs.append(val_acc)
                            val_precision_all_graphs.append(val_precision)
                            val_recall_all_graphs.append(val_recall)
                            val_f1_all_graphs.append(val_f1)

                        # Average over all training graphs
                        val_loss_all_graphs = np.mean(val_loss_all_graphs)
                        val_fp_rate_all_graphs = np.mean(val_fp_rate_all_graphs)
                        val_fn_rate_all_graphs = np.mean(val_fn_rate_all_graphs)
                        val_acc_all_graphs = np.mean(val_acc_all_graphs)
                        val_precision_all_graphs = np.mean(val_precision_all_graphs)
                        val_recall_all_graphs = np.mean(val_recall_all_graphs)
                        val_f1_all_graphs = np.mean(val_f1_all_graphs)

                        loss_per_epoch_valid.append(val_loss_all_graphs)

                        elapsed = utils.timedelta_to_str(datetime.now() - time_start_eval)
                        print(f'===> VALIDATION (all validation graphs): Epoch = {epoch}')
                        print(f'Loss: {val_loss_all_graphs:.4f}, fp_rate(GT=0): {val_fp_rate_all_graphs:.4f}, fn_rate(GT=1): {val_fn_rate_all_graphs:.4f}')
                        print(f'elapsed time: {elapsed}\n')

                        try:
                            wandb.log({'val_loss': val_loss_all_graphs, 'val_accuracy': val_acc_all_graphs, 'val_precision': val_precision_all_graphs, \
                                    'val_recall': val_recall_all_graphs, 'val_f1': val_f1_all_graphs, 'fp-rate': val_fp_rate_all_graphs, 'fn-rate': val_fn_rate_all_graphs})
                        except Exception:
                            print(f'WandB exception occured!')

                        if len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                            torch.save(best_model.state_dict(), model_path)
                        save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], loss_per_epoch_valid[-1], out)
                        scheduler.step(val_loss_all_graphs)

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
