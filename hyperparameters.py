import torch


def get_hyperparameters():
    return {
        "num_epochs": 5,
        "dim_latent": 5,
        "dim_nodes": 1,
        # "dim_target": 2,
        "dim_edges": 1,
        # "dim_bits": 8,
        "batch_size": 1,
        # "max_threshold": 10,
        "patience_limit": 10,
        # "growth_rate_sigmoid": 0.0020,
        # "sigmoid_offset": -300,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "calculate_termination_statistics": True,
        "lr": 5e-4,
        "weight_decay": 0,
        "bias": False,
    }
