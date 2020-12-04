import os

import torch
from torch_geometric.data import Dataset

# TODO: This is pretty useless so far. Create a plan on how this will look and implement it
#       For example, raw can be a csv and then full processed would be a whole graph in .pt format.
#       Or out of one graph I can construct many smaller data points, such as:
#           - MPNN gets a small neighborhood and needs to make a decision

# TODO: Figure out if you even need the 'split' parameter here. Does it matter what is train/test?


class GraphDataset(Dataset):

    def __init__(self, root, device='cpu', split='train'):
        super(GraphDataset, self).__init__(root)
        self.device = device
        self.split = split

    def len(self):
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, str(idx) + '.pt'))

    @property
    def raw_file_names(self):
        dirname = self.raw_dir
        raw_files = os.listdir(dirname)
        return raw_files

    @property
    def processed_file_names(self):
        pass

    def download(self):
        # Maybe run raven here in order to generate the graph in csv?
        pass

    def process(self):
        pass

