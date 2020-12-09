import os
import subprocess

import torch
from torch_geometric.data import Dataset

import graph_parser


class GraphDataset(Dataset):
    # TODO: Figure out if you even need the 'split' parameter here. Does it matter what is train/test?
    def __init__(self, root, device='cpu', split='train'):
        if 'raw' not in os.listdir(root):
            print('here A')
            subprocess.run(f"mkdir 'raw'", shell=True, cwd=root)
        if 'processed' not in os.listdir(root):
            print('here B')
            subprocess.run(f"mkdir 'processed'", shell=True, cwd=root)

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
        return [str(n) + '.pt' for n in range(self.len())]

    def download(self):
        # TODO: Fix paths somehow, they can't be defined here
        reads_path = os.path.abspath('data/reads/lambda_reads.fastq')
        raven_path = os.path.abspath('vendor/raven/build/bin/raven')
        subprocess.run(f'{raven_path} -t 2 -p 0 {reads_path} > assembly.fasta', shell=True, cwd=self.raw_dir)

    def process(self):
        # TODO: Fix paths
        # print('----Processing----')
        cnt = 0
        raw_path = os.path.join(self.raw_dir, 'graph_before.csv')
        processed_path = os.path.join(self.processed_dir, str(cnt) + '.pt')
        _, graph = graph_parser.from_csv(raw_path)
        torch.save(graph, processed_path)


def main():
    ds = GraphDataset('data')
    ds.download()
    ds.process()
    graph = torch.load('data/processed/0.pt')

    assert graph.read_length is not None, \
        'Graph does not contain read_length field.'
    assert graph.overlap_length is not None, \
        'Graph does not contain overlap_length field.'
    assert graph.overlap_similarity is not None, \
        'Graph does not contain overlap_similarity field.'
    assert len(graph.overlap_length) == len(graph.overlap_similarity), \
        'Lengths do not match: overlap_length and overlap_similarity'
    assert graph.edge_index.shape[1] == graph.overlap_length.shape[0], \
        'Length do not match: edge_index and overlap_length'

    print('Passed all tests!')


if __name__ == '__main__':
    main()
