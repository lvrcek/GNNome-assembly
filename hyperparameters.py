import torch


def get_hyperparameters():
    return {
        "num_epochs": 50,
        "dim_latent": 16,
        "dim_nodes": 1,
        # "dim_target": 2,
        "dim_edges": 1,
        "batch_size": 1,
        # "max_threshold": 10,
        "patience_limit": 10,
        # "sigmoid_offset": -300,
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "lr": 1e-7,
        "walk_length": -1,
        "weight_decay": 0,
        "bias": False,
    }
