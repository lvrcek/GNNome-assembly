import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'num_epochs': 1,
        'dim_latent': 4,
        'num_gnn_layers': 1,
        'batch_size': 1,
        'patience_limit': 1000,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-5,
        'walk_length': 10,
        'bias': False,
        'gnn_mode': 'builtin'
    }
