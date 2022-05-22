import torch

def get_hyperparameters():
    return {
        'seed': 0,
        'lr': 1e-3,
        'num_epochs': 100,
        'dim_latent': 256,
        'node_features': 1,
        'edge_features': 2,
        'hidden_edge_features': 16, 
        'hidden_edge_scores': 64, 
        'num_gnn_layers': 16,
        'nb_pos_enc': 16,
        'num_parts_metis_train': 500,
        'num_parts_metis_eval': 500,
        'batch_size_train': 50,
        'batch_size_eval': 50,
        'num_decoding_paths': 25,
        # 'pos_to_neg_ratio': 16.5,
        'num_contigs': 10,
        'patience': 2,
        'decay': 0.95,
        'device': 'cuda:3' if torch.cuda.is_available() else 'cpu',
        'batch_norm': True,
        'wandb_mode': 'disabled',  # switch between 'online' and 'disabled'
        # 'use_amp': False,
        # 'bias': False,
        # 'gnn_mode': 'builtin',
        # 'encode': 'none',
        # 'norm': 'all',
        # 'use_reads': False,
    }

