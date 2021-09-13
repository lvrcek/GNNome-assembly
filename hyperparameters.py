import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'num_epochs': 2000,
        'dim_latent': 16,
        'num_gnn_layers': 1,
        'batch_size': 1,
        'patience_limit': 100,
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'lr': 1e-6,
        'walk_length': 5,
        'bias': False,
    }
