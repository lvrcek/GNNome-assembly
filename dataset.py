import os
import pickle
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
        return (len(os.listdir(self.processed_dir)) -2) // 3 # if there are those other filter files

    def get(self, idx):
        graph = torch.load(os.path.join(self.processed_dir, str(idx) + '.pt'))
        # pred = pickle.load(open(os.path.join(self.processed_dir, f'{idx}_pred.pkl'), 'rb'))
        # succ = pickle.load(open(os.path.join(self.processed_dir, f'{idx}_succ.pkl'), 'rb'))
        return idx, graph

    @property
    def raw_file_names(self):
        dirname = self.raw_dir
        raw_files = os.listdir(dirname)
        return raw_files

    @property
    def processed_file_names(self):
        return [str(n) + '.pt' for n in range(self.len())]

    def download(self):
        # graph_generator.generate_pacbio(self.num_graphs, self.reference_path, self.raw_dir)
        pass

    def process(self):
        print(os.getcwd())
        print(self.reads_path)
        print(self.reference_path)
        print(os.path.isfile(self.reads_path))
        sequences_path = os.path.join(self.root, 'sequences')
        if not os.path.isdir(sequences_path):
            os.mkdir(sequences_path)
        graphia_dir = os.path.join(self.root, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        f = open(f'{self.root}/dataset_log.txt', 'w')
        for cnt, reads in enumerate(os.listdir(self.raw_dir)):
            print(cnt, reads)
            reads_path = os.path.abspath(os.path.join(self.raw_dir, reads))
            print(reads_path)
            subprocess.run(f'{self.raven_path} --weaken -t32 -p0 {reads_path} > assembly.fasta', shell=True, cwd=self.tmp_dir)
            processed_path = os.path.join(self.processed_dir, str(cnt) + '.pt')
            _, _, graph_dir, graph_und, pred, succ = graph_parser.from_csv(os.path.join(self.tmp_dir, 'graph_before.csv'), reads_path)
            torch.save(graph_und, processed_path)

            pickle.dump(pred, open(f'{self.processed_dir}/{cnt}_pred.pkl', 'wb'))  # print predecessors
            pickle.dump(succ, open(f'{self.processed_dir}/{cnt}_succ.pkl', 'wb'))  # print successors

            graphia_path = os.path.join(graphia_dir, f'{cnt}_graph.txt')
            graph_parser.print_pairwise(graph_dir, graphia_path)  # print pairwise-txt format for graphia visualization

            graph_path = os.path.join(sequences_path, f'graph_{cnt}')  # print sequences, useful for gepard
            if not os.path.isdir(graph_path):
                os.mkdir(graph_path)
            graph_parser.print_fasta(graph_und, graph_path)
            f.write(f'{cnt} - {reads}\n')

        f.close()


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
