import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'lr': 1e-3,
        'num_epochs': 10,
        'dim_latent': 32,
        'node_features': 1,
        'edge_features': 2,
        'num_gnn_layers': 3,
        'batch_size': 1024,
        'patience': 10,
        'decay': 0.9,
        'device': 'cpu', # 'cuda:7' if torch.cuda.is_available() else 'cpu',
        
        'bias': False,
        'gnn_mode': 'builtin',
        'encode': 'none',
        'norm': 'all',
        'use_reads': False,
        'use_amp': True,
    }
