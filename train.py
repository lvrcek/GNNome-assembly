import os
import copy
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

import dataset
import models


# NUM_EPOCHS = 5


def train():

    num_epochs = 5
    dim_node = 1
    dim_edge = 1
    dim_latent = 1
    batch_size = 1
    patience_limit = 10
    learning_rate = 1e-5

    mode = 'train'

    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    train_path = os.path.abspath('')
    test_path = os.path.abspath('')

    # TODO: Discuss with Mile how to train this thing - maybe through generated reads by some tools?
    # First with real data just to check if the thing works, then probably with the generated graphs
    # The problem is that generated graphs don't have chimeric reads
    ds_train = dataset.GraphDataset(train_path)
    ds_test = dataset.GraphDataset(test_path)

    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    processor = models.ExecutionModel(dim_node, dim_edge, dim_latent)
    params = list(processor.parameters())
    model_path = os.path.abspath(f'trained_models/{time_now}.pt)')

    optimizer = optim.Adam(params, lr=learning_rate)

    patience = 0
    best_model = models.ExecutionModel(dim_node, dim_edge, dim_latent)
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    if mode == 'train':
        loss = []
        # accuracy_per_epoch = []
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            processor.train()
            for data in dl_train:
                pass
                # processor.process_graph(data)


if __name__ == '__main__':
    train()
