import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'num_epochs': 1500,
        'dim_latent': 4,
        'num_gnn_layers': 1,
        'batch_size': 1,
        'patience_limit': 100,
        'device': "cuda:2" if torch.cuda.is_available() else "cpu",
        'lr': 1e-6,
        'walk_length': 10,
        'bias': False,
    }
