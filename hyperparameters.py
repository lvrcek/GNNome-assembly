import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'num_epochs': 100,
        'dim_latent': 32,
        'num_gnn_layers': 8,
        'batch_size': 1,
        'patience_limit': 5,
        'device': 'cpu',  # 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'walk_length': 10,
        'bias': False,
        'gnn_mode': 'builtin',
        'encode': 'none',
        'norm': 'all',
    }
