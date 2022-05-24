import argparse
from datetime import datetime
import copy
import os
# from posixpath import split
import pickle
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import random_split
from torch.profiler import profile, record_function, ProfilerActivity
import dgl
from dgl.dataloading import GraphDataLoader
import wandb

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import evaluate
import models
import utils


def save_checkpoint(epoch, model, optimizer, loss_train, loss_valid, out):
    """Save the state of the training process.

    Parameters
    ----------
    epoch : int
        Number of epoch in which the training loop is at the time of saving
    model : torch.nn.Module
        A PyTorch model used during training
    optimizer : torch.optim.Optimizer
        A PyTorch optimizer used during training
    loss_train : float
        Loss on the training dataset in the current epoch
    loss_valid : float
        Loss on the validation dataset in the current epoch
    out : str
        Name of the file in which the checkpoint is saved, not the full path

    Returns
    -------
    None
    """
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
    """Load the previously saved checkpoints.

    Parameters
    ----------
    out : string
        Name of the file from which the checkpoints will be loaded
    model : torch.nn.Module
        A PyTorch model into which the weights will be loaded
    optimizer : torch.nn.Optimizer
        A PyTorch optimizer into which the optimizer state will be loaded

    Returns
    -------
    int
        Epoch at which the checkpoint was saved
    torch.nn.Module
        A model with weights loaded from the checkpoint
    torch.nn.Optimizer
        An optimizer with state loaded from the checkpoint
    float
        Training loss at the epoch of saving the checkpoint
    float
        Validation loss at the epoch of saving the checkpoint
    """
    ckpt_path = f'checkpoints/{out}.pt'
    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    loss_train = checkpoint['loss_train']
    loss_valid = checkpoint['loss_valid']
    return epoch, model, optimizer, loss_train, loss_valid


def view_model_param(model):
    """Get the total number of parameters of the model.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model for which the number of parameters is calculated

    Returns
    -------
    int
        Number of parameters of the model
    """
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


def train(train_path, valid_path, out, overfit):
    """Training loop where the model learns to predict the edge labels.

    Parameters
    ----------
    data : str
        Path to where training and validation data is stored
    out : str
        Name used for saving auxiliary files and the trained model
    overfit : bool
        Whether to train in the overfitting mode

    Returns
    -------
    None
    """
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    #batch_size = hyperparameters['batch_size']
    batch_size_train = hyperparameters['batch_size_train']
    batch_size_eval = hyperparameters['batch_size_eval']
    nb_pos_enc = hyperparameters['nb_pos_enc']
    num_parts_metis_train = hyperparameters['num_parts_metis_train']
    num_parts_metis_eval = hyperparameters['num_parts_metis_eval']
    # num_decoding_paths = hyperparameters['num_decoding_paths']
    # num_contigs = hyperparameters['num_contigs']
    patience = hyperparameters['patience']
    lr = hyperparameters['lr']
    device = hyperparameters['device']
    # use_reads = hyperparameters['use_reads']
    # use_amp = hyperparameters['use_amp']
    batch_norm = hyperparameters['batch_norm']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    hidden_edge_features = hyperparameters['hidden_edge_features']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']
    decay = hyperparameters['decay']
    # pos_to_neg_ratio = hyperparameters['pos_to_neg_ratio']
    wandb_mode = hyperparameters['wandb_mode']

    utils.set_seed(seed)

    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    # data_path = os.path.abspath(data)
    
    if out is None:
        out = timestamp
    if train_path is None:
        train_path = os.path.join(data_path, f'train_{out}')
    if valid_path is None:
        valid_path = os.path.join(data_path, f'valid_{out}')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_gnn_layers)

    if not overfit:
        ds_train = AssemblyGraphDataset(train_path, nb_pos_enc=nb_pos_enc)
        ds_valid = AssemblyGraphDataset(valid_path, nb_pos_enc=nb_pos_enc)
    else:
        ds = AssemblyGraphDataset(train_path, nb_pos_enc=nb_pos_enc)
        ds_train = ds 
        ds_valid = ds_train # DEBUG !!!!!!!!!!!!!

    print(ds_train)
    print(ds_valid)
    pos_to_neg_ratio = sum([((g.edata['y']==1).sum() / (g.edata['y']==0).sum()).item() for idx, g in ds_train]) / len(ds_train)

#     if batch_size_train <= 1: # train with full graph 
#         # model = models.GraphGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
#         # best_model = models.GraphGCNModel(node_features, edge_features, hidden_features, num_gnn_layers)
#         model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc) # GatedGCN 
#         best_model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc) # GatedGCN 
#     else:
#         # model = models.BlockGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers, batch_norm=batch_norm)
#         # best_model = models.BlockGatedGCNModel(node_features, edge_features, hidden_features, num_gnn_layers, batch_norm=batch_norm)
#         model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc) # GatedGCN 
#         best_model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc) # GatedGCN 


    model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc) # GatedGCN
    best_model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc) # GatedGCN

    model.to(device)
    if not os.path.exists('pretrained'):
        os.makedirs('pretrained')
    model_path = os.path.abspath(f'pretrained/model_{out}.pt')
    best_model.to(device)
    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
    best_model.eval()

    print(f'\nNumber of network parameters: {view_model_param(model)}\n')
    print(f'Normalization type : Batch Normalization\n') if batch_norm else print(f'Normalization type : Layer Normalization\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([1 / pos_to_neg_ratio], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=patience, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    if not os.path.exists(os.path.join('checkpoints')):
        os.makedirs(os.path.join('checkpoints'))

    cluster_cache_path = f'checkpoints/{out}_cluster_gcn.pkl'
    if os.path.exists(cluster_cache_path):
        os.remove(cluster_cache_path)

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    acc_per_epoch_train, acc_per_epoch_valid = [], []

    try:
        with wandb.init(project="GeNNome", config=hyperparameters, mode=wandb_mode):
            wandb.watch(model, criterion, log='all', log_freq=1000)

            for epoch in range(num_epochs):

                train_loss_all_graphs, train_fp_rate_all_graphs, train_fn_rate_all_graphs = [], [], []
                train_acc_all_graphs, train_precision_all_graphs, train_recall_all_graphs, train_f1_all_graphs = [], [], [], []

                print('TRAINING')
                random.shuffle(ds_train.graph_list)
                for data in ds_train:
                    model.train()
                    idx, g = data

                    if batch_size_train <= 1: # train with full graph 

                        g = g.to(device)
                        x = g.ndata['x'].to(device)
                        e = g.edata['e'].to(device)
                        pe = g.ndata['pe'].to(device)
                        pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
                        pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
                        pe = torch.cat((pe_in, pe_out, pe), dim=1)
                        edge_predictions = model(g, x, e, pe)
                        edge_predictions = edge_predictions.squeeze(-1)
                        edge_labels = g.edata['y'].to(device)
                        loss = criterion(edge_predictions, edge_labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss = loss.item()
                        TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                        acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                        try:
                            fp_rate = FP / (FP + TN)
                        except ZeroDivisionError:
                            fp_rate = 0.0
                        try:
                            fn_rate = FN / (FN + TP)
                        except ZeroDivisionError:
                            fn_rate = 0.0
                        train_fp_rate = fp_rate
                        train_fn_rate = fn_rate
                        train_acc = acc
                        train_precision = precision
                        train_recall = recall
                        train_f1 = f1

                        elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                        print(f'\nTRAINING (one training graph): Epoch = {epoch}, Graph = {idx}')
                        print(f'Loss: {train_loss:.4f}, fp_rate(GT=0): {train_fp_rate:.4f}, fn_rate(GT=1): {train_fn_rate:.4f}')
                        print(f'elapsed time: {elapsed}\n')

                    else: # train with mini-batch
                        # remove Metis clusters to force new clusters
                        try:
                            os.remove(cluster_cache_path)
                        except:
                            pass 

                        # Run Metis
                        g = g.long()
                        num_clusters = torch.LongTensor(1).random_(num_parts_metis_train-100,num_parts_metis_train+100).item() # DEBUG!!!
                        sampler = dgl.dataloading.ClusterGCNSampler(g, num_clusters, cache_path=cluster_cache_path) 
                        dataloader = dgl.dataloading.DataLoader(g, torch.arange(num_clusters), sampler, batch_size=batch_size_train, shuffle=True, drop_last=False, num_workers=4)

                        # For loop over all mini-batch in the graph
                        running_loss, running_fp_rate, running_fn_rate = [], [], []
                        running_acc, running_precision, running_recall, running_f1 = [], [], [], []
                        for sub_g in dataloader:
                            sub_g = sub_g.to(device)
                            x = sub_g.ndata['x'].to(device)
                            e = sub_g.edata['e'].to(device)
                            pe = sub_g.ndata['pe'].to(device)
                            pe_in = sub_g.ndata['in_deg'].unsqueeze(1).to(device)
                            pe_out = sub_g.ndata['out_deg'].unsqueeze(1).to(device)
                            pe = torch.cat((pe_in, pe_out, pe), dim=1)
                            edge_predictions = model(sub_g, x, e, pe) 
                            edge_predictions = edge_predictions.squeeze(-1)
                            edge_labels = sub_g.edata['y'].to(device)
                            loss = criterion(edge_predictions, edge_labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            running_loss.append(loss.item())
                            TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                            acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                            try:
                                fp_rate = FP / (FP + TN)
                            except ZeroDivisionError:
                                fp_rate = 0.0
                            try:
                                fn_rate = FN / (FN + TP)
                            except ZeroDivisionError:
                                fn_rate = 0.0
                            running_fp_rate.append(fp_rate)
                            running_fn_rate.append(fn_rate)
                            running_acc.append(acc)
                            running_precision.append(precision)
                            running_recall.append(recall)
                            running_f1.append(f1)

                        # Average over all mini-batch in the graph
                        train_loss = np.mean(running_loss)
                        train_fp_rate = np.mean(running_fp_rate)
                        train_fn_rate = np.mean(running_fn_rate)
                        train_acc = np.mean(running_acc)
                        train_precision = np.mean(running_precision)
                        train_recall = np.mean(running_recall)
                        train_f1 = np.mean(running_f1)

                        elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                        print(f'\nTRAINING (one training graph): Epoch = {epoch}, Graph = {idx}')
                        print(f'Loss: {train_loss:.4f}, fp_rate(GT=0): {train_fp_rate:.4f}, fn_rate(GT=1): {train_fn_rate:.4f}')
                        print(f'elapsed time: {elapsed}\n')

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

                if overfit: # temp : one graph at the moment
                    if len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        torch.save(best_model.state_dict(), model_path)
                    # TODO: Check what's going on here
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out)
                    scheduler.step(train_loss_all_graphs)

                if True:  # TODO: if you're going to do validation every epoch just remove this
                # if not epoch % 3 and epoch > 0: # DEBUG !!!!!!!!!!!!!

                    val_loss_all_graphs, val_fp_rate_all_graphs, val_fn_rate_all_graphs = [], [], []
                    val_acc_all_graphs, val_precision_all_graphs, val_recall_all_graphs, val_f1_all_graphs = [], [], [], []

                    with torch.no_grad():
                        print('===> VALIDATION')
                        time_start_eval = datetime.now()
                        model.eval()
                        for data in ds_valid:
                            idx, g = data

                            if batch_size_eval <= 1: # full graph 

                                g = g.to(device)
                                x = g.ndata['x'].to(device)
                                e = g.edata['e'].to(device)
                                pe = g.ndata['pe'].to(device)
                                pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
                                pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
                                pe = torch.cat((pe_in, pe_out, pe), dim=1)
                                edge_predictions = model(g, x, e, pe)
                                edge_predictions = edge_predictions.squeeze(-1)
                                edge_labels = g.edata['y'].to(device)
                                loss = criterion(edge_predictions, edge_labels)
                                val_loss = loss.item()
                                TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                                acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                                try:
                                    fp_rate = FP / (FP + TN)
                                except ZeroDivisionError:
                                    fp_rate = 0.0
                                try:
                                    fn_rate = FN / (FN + TP)
                                except ZeroDivisionError:
                                    fn_rate = 0.0
                                val_fp_rate = fp_rate
                                val_fn_rate = fn_rate
                                val_acc = acc
                                val_precision = precision
                                val_recall = recall
                                val_f1 = f1

                                elapsed = utils.timedelta_to_str(datetime.now() - time_start_eval)
                                print(f'\n===> VALIDATION (one validation graph): Epoch = {epoch}, Graph = {idx}')
                                print(f'Loss: {val_loss:.4f}, fp_rate(GT=0): {val_fp_rate:.4f}, fn_rate(GT=1): {val_fn_rate:.4f}')
                                print(f'elapsed time: {elapsed}\n')

                            else: # mini-batch

                                # remove Metis clusters to force new clusters
                                try:
                                    os.remove(cluster_cache_path)
                                except:
                                    pass 

                                # Run Metis
                                g = g.long()
                                sampler = dgl.dataloading.ClusterGCNSampler(g, num_parts_metis_eval, cache_path=cluster_cache_path) 
                                dataloader = dgl.dataloading.DataLoader(g, torch.arange(num_parts_metis_eval), sampler, batch_size=batch_size_eval, shuffle=True, drop_last=False, num_workers=4)

                                # For loop over all mini-batch in the graph
                                running_loss, running_fp_rate, running_fn_rate = [], [], []
                                running_acc, running_precision, running_recall, running_f1 = [], [], [], []
                                for sub_g in dataloader:
                                    sub_g = sub_g.to(device)
                                    x = sub_g.ndata['x'].to(device)
                                    e = sub_g.edata['e'].to(device)
                                    pe = sub_g.ndata['pe'].to(device)
                                    pe_in = sub_g.ndata['in_deg'].unsqueeze(1).to(device)
                                    pe_out = sub_g.ndata['out_deg'].unsqueeze(1).to(device)
                                    pe = torch.cat((pe_in, pe_out, pe), dim=1)
                                    edge_predictions = model(sub_g, x, e, pe) 
                                    edge_predictions = edge_predictions.squeeze(-1)
                                    edge_labels = sub_g.edata['y'].to(device)
                                    loss = criterion(edge_predictions, edge_labels)
                                    running_loss.append(loss.item())
                                    TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                                    acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                                    try:
                                        fp_rate = FP / (FP + TN)
                                    except ZeroDivisionError:
                                        fp_rate = 0.0
                                    try:
                                        fn_rate = FN / (FN + TP)
                                    except ZeroDivisionError:
                                        fn_rate = 0.0
                                    running_fp_rate.append(fp_rate)
                                    running_fn_rate.append(fn_rate)
                                    running_acc.append(acc)
                                    running_precision.append(precision)
                                    running_recall.append(recall)
                                    running_f1.append(f1)

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
                            wandb.log({'train_loss': train_loss_all_graphs, 'train_accuracy': train_acc_all_graphs, \
                                       'train_precision': train_precision_all_graphs, 'lr_value': lr_value, \
                                       'train_recall': train_recall_all_graphs, 'train_f1': train_f1_all_graphs, \
                                       'train_fp-rate': train_fp_rate_all_graphs, 'train_fn-rate': train_fn_rate_all_graphs, \
                                       'val_loss': val_loss_all_graphs, 'val_accuracy': val_acc_all_graphs, \
                                       'val_precision': val_precision_all_graphs, \
                                       'val_recall': val_recall_all_graphs, 'val_f1': val_f1_all_graphs, \
                                       'val_fp-rate': val_fp_rate_all_graphs, 'val_fn-rate': val_fn_rate_all_graphs})
                        except Exception:
                            print(f'WandB exception occured!')

                        if len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                            torch.save(best_model.state_dict(), model_path)
                        save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], loss_per_epoch_valid[-1], out)
                        scheduler.step(val_loss_all_graphs)

    except KeyboardInterrupt:
        print("Keyboard Interrupt...")
        print("Exiting...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=None, help='Path to directory with training data')
    parser.add_argument('--valid', type=str, default=None, help='Path to directory with validation data')
    parser.add_argument('--out', type=str, default=None, help='Output name for figures and models')
    parser.add_argument('--overfit', action='store_true', default=False, help='Overfit on the chromosomes in the train directory')
    args = parser.parse_args()
    train(args.train, args.valid, args.out, args.overfit)

