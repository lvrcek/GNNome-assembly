import torch


def get_hyperparameters():
    return {
        'num_epochs': 100,
        'dim_latent': 16,
        'dim_nodes': 1,
        'dim_edges': 1,
        'batch_size': 1,
        'patience_limit': 100,
        'device': "cuda:1" if torch.cuda.is_available() else "cpu",
        'lr': 1e-8,
        'walk_length': 10,
        'weight_decay': 0,
        'bias': False,
    }
