import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'num_epochs': 100,
        'dim_latent': 64,
        'num_gnn_layers': 4,
        'batch_size': 1,
        'patience_limit': 10,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-5,
        'walk_length': 10,
        'bias': False,
        'gnn_mode': 'builtin',
        'encode': 'none',
        'norm': None,
    }
