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

    # TODO: This also assumes just one walk, should implement case if I have multiple (from each end-node)
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
            with torch.cuda.amp.autocast(enabled=use_amp), profile(activities=[ProfilerActivity.CUDA],
                                                                   profile_memory=True) as prof:
                with record_function('model_forward_pass'):
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

            print()
            # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            # print()
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            print()

            # TODO: Total loss should never be 0
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


            if len(loss_list) == 0:
                continue
            else:
                # accuracy = correct / steps
                mini_batch_loss_list.append(np.mean(loss_list))  # List of mean loss per mini-walk
                # mini_batch_acc_list.append(accuracy)

        per_walk_loss.append(np.mean(mini_batch_loss_list))  # List of mean loss per solution-walk
        per_walk_acc.append(correct / total_steps)  # List of accuracies per solution-walk

    return per_walk_loss, per_walk_acc


def process_reads(reads, device):
    processed_reads = {}
    for id, read in reads.items():
        read = read.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
        read = ' '.join(read).split()
        read = torch.tensor(list(map(int, read)), device=device)
        processed_reads[id] = read
    return processed_reads


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
    model = models.NonAutoRegressive(dim_latent, num_gnn_layers).to(device)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    model_path = os.path.abspath(f'pretrained/model_{out}.pt')
    criterion = nn.CrossEntropyLoss()

    # Initialize best model obtained during the training process
    best_model = models.NonAutoRegressive(dim_latent, num_gnn_layers)
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
        with wandb.init(project="assembly-walk-icml", config=hyperparameters):
            wandb.watch(model, criterion, log='all', log_freq=1)

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
                    else:
                        solution = utils.get_walks(idx, data_path)

                    utils.print_graph_info(idx, graph)
                    loss_list, accuracy_list = process(model, graph, succ, reads, solution, edges, criterion, optimizer, scaler, epoch, norm_train, device=device)
                    if loss_list is not None:
                        loss_per_graph.append(np.mean(loss_list))
                        accuracy_per_graph.append(np.mean(accuracy_list))

                    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                    print(f'Processing graph {idx} done. Elapsed time: {elapsed}')

                train_loss = np.mean(loss_per_graph)
                train_acc = np.mean(accuracy_per_graph)
                loss_per_epoch_train.append(train_loss)
                accuracy_per_epoch_train.append(train_acc)

                try:
                    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc}, step=epoch)
                except Exception:
                    pass

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
                            else:
                                solution = utils.get_walks(idx, data_path)

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
                        else:
                            solution = utils.get_walks(idx, data_path)

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
    train(args)
