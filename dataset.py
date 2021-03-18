import os
import subprocess

import torch
from torch_geometric.data import Dataset

import graph_generator
import graph_parser


class GraphDataset(Dataset):

    def __init__(self, root, device='cpu', reads_path='data/reads/lambda_reads.fastq',
                 reference_path='data/references/lambda_reference.fasta', num_graphs=10):
        if 'raw' not in os.listdir(root):
            subprocess.run(f"mkdir 'raw'", shell=True, cwd=root)
        if 'tmp' not in os.listdir(root):
            subprocess.run(f"mkdir 'tmp'", shell=True, cwd=root)
        if 'processed' not in os.listdir(root):
            subprocess.run(f"mkdir 'processed'", shell=True, cwd=root)

        self.tmp_dir = os.path.join(root, 'tmp')
        self.reads_path = os.path.abspath(reads_path)
        self.reference_path = os.path.abspath(reference_path)
        self.raven_path = os.path.abspath('vendor/raven/build/bin/raven')
        self.num_graphs = num_graphs

        super(GraphDataset, self).__init__(root)
        self.device = device

    def len(self):
        return len(os.listdir(self.processed_dir)) -2 # if there are those other filter files

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
        graph_generator.generate_pacbio(self.num_graphs, self.reference_path, self.raw_dir)

    def process(self):
        print(os.getcwd())
        print(self.reads_path)
        print(self.reference_path)
        print(os.path.isfile(self.reads_path))
        sequences_path = os.path.join(root, 'sequences')
        if not os.path.isdir(sequences_path):
            os.mkdir(sequences_path)
        for cnt, reads in enumerate(os.listdir(self.raw_dir)):
            print(cnt, reads)
            reads_path = os.path.abspath(os.path.join(self.raw_dir, reads))
            print(reads_path)
            subprocess.run(f'{self.raven_path} -t32 -p0 {reads_path} > assembly.fasta', shell=True, cwd=self.tmp_dir)
            processed_path = os.path.join(self.processed_dir, str(cnt) + '.pt')
            _, graph = graph_parser.from_csv(os.path.join(self.tmp_dir, 'graph_before.csv'))
            torch.save(graph, processed_path)
            graph_path = os.path.join(sequences_path, f'graph_{cnt}')
            if not os.path.isdir(graph_path):
                os.mkdir(graph_path)
            graph_parser.print_fasta(graph, graph_path)


def main():
    ds = GraphDataset('data/debug')
    ds.download()
    ds.process()
    graph = torch.load('data/debug/processed/0.pt')

    assert hasattr(graph, 'read_length'), \
        'Graph does not contain read_length field.'
    assert hasattr(graph, 'prefix_length'), \
        'Graph does not contain prefix_length field.'
    assert hasattr(graph, 'overlap_similarity'), \
        'Graph does not contain overlap_similarity field.'
    assert len(graph.prefix_length) == len(graph.overlap_similarity), \
        'Lengths do not match: prefix_length and overlap_similarity'
    assert graph.edge_index.shape[1] == graph.prefix_length.shape[0], \
        'Length do not match: edge_index and prefix_length'

    print('Passed all tests!')


if __name__ == '__main__':
    main()
