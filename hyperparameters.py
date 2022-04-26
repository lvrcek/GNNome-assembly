import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'lr': 1e-3,
        'num_epochs': 1000,
        'dim_latent': 128,
        'node_features': 1,
        'edge_features': 2,
        'hidden_edge_features': 16, 
        'hidden_edge_scores': 64, 
        'num_gnn_layers': 4,
        'nb_pos_enc': 10,
        'num_parts_metis_train': 1000,
        'num_parts_metis_eval': 1000,
        'batch_size_train': 20,
        'batch_size_eval': 20,
        'patience': 5,
        'decay': 0.95,
        'device': 'cuda:3' if torch.cuda.is_available() else 'cpu',
        'batch_norm': True,
        'use_amp': False,
        'bias': False,
        'gnn_mode': 'builtin',
        'encode': 'none',
        'norm': 'all',
        'use_reads': False,
    }
