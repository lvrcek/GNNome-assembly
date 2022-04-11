import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'lr': 1e-3,
        'num_epochs': 300,
        'dim_latent': 128,
        'node_features': 1,
        'edge_features': 2,
        'num_gnn_layers': 4,
        'batch_size': 2048,
        'patience': 1,
        'decay': 0.9,
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  
        'bias': False,
        'gnn_mode': 'builtin',
        'encode': 'none',
        'norm': 'all',
        'batch_norm': False,
        'use_reads': False,
        'use_amp': False,
    }
