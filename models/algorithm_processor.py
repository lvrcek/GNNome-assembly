import torch
import torch.nn as nn

from layers import MPNN


class AlgorithmProcessor(nn.Module):

    def __init__(self, latent_features, processor_type='MPNN'):
        super(AlgorithmProcessor, self).__init__()
        if processor_type == 'MPNN':
            self.processor = MPNN(latent_features, latent_features, latent_features, bias=False)
        self.algorithms = nn.ModuleDict()

    def add_algorithm(self, name, algorithm):
        self.algorithms[name] = algorithm

    def process(self, graph, optimizer, loss_list, accuracy_list, train=True, device='cpu'):
        pass
