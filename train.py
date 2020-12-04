import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

import dataset
import models


def main():

    num_epochs = 5
    dim_latent = 1
    batch_size = 1
    learning_rate = 1e-5

    mode = 'train'
    time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
    train_path = os.path.abspath('')
    test_path = os.path.abspath('')
    ds_train = dataset.GraphDataset(train_path)
    ds_test = dataset.GraphDataset(test_path)

    processor = models.AlgorithmProcessor(dim_latent)
    params = list(processor.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    
if __name__ == '__main__':
    main()