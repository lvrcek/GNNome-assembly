import os
import pickle
import subprocess

import torch
import dgl
from dgl.data import DGLDataset

import graph_generator
import graph_parser

class GraphDataset(DGLDataset):

    def __init__(self, name='chr11_90-100M.fastq', root='data/test'):
        print('here 1')
        # super(GraphDataset, self).__init__(name='AssemblyGraph')
        print('here 2')
        self.root = root
        # self.raw_dir = root
        if 'raw' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'raw'", shell=True, cwd=self.root)
        if 'tmp' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'tmp'", shell=True, cwd=self.root)
        if 'processed' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'processed'", shell=True, cwd=self.root)

        self.raww_dir = os.path.join(self.root, 'raw')
        self.processed_dir = os.path.join(self.root, 'processed')
        self.tmp_dir = os.path.join(self.root, 'tmp')
        self.raven_path = os.path.abspath('vendor/raven/build/bin/raven')
        super().__init__(name='AssemblyGraph')


    def __len__(self):
        return len(os.listdir(self.processed_dir)) // 3# if there are those other filter files

    def __getitem__(self, idx):
        (graph,), _ = dgl.load_graphs(os.path.join(self.processed_dir, str(idx) + '.dgl'))
        # pred = pickle.load(open(os.path.join(self.processed_dir, f'{idx}_pred.pkl'), 'rb'))
        # succ = pickle.load(open(os.path.join(self.processed_dir, f'{idx}_succ.pkl'), 'rb'))
        return idx, graph

    # @property
    # def raw_file_names(self):
    #     dirname = self.raw_dir
    #     raw_files = os.listdir(dirname)
    #     return raw_files

    # @property
    # def processed_file_names(self):
    #     return [str(n) + '.pt' for n in range(self.len())]

    # def download(self):
    #     # graph_generator.generate_pacbio(self.num_graphs, self.reference_path, self.raw_dir)
    #     pass

    def process(self):
        print(self)
        print(self.root)
        sequences_path = os.path.join(self.root, 'sequences')
        if not os.path.isdir(sequences_path):
            os.mkdir(sequences_path)
        graphia_dir = os.path.join(self.root, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        with open(f'{self.root}/dataset_log.txt', 'w') as f:
            for cnt, reads in enumerate(os.listdir(self.raww_dir)):
                print(cnt, reads)
                reads_path = os.path.abspath(os.path.join(self.raww_dir, reads))
                print(reads_path)
                subprocess.run(f'{self.raven_path} --weaken -t32 -p0 {reads_path} > assembly.fasta', shell=True, cwd=self.tmp_dir)
                processed_path = os.path.join(self.processed_dir, str(cnt) + '.dgl')
                graph, pred, succ = graph_parser.from_csv_dgl(os.path.join(self.tmp_dir, 'graph_before.csv'), reads_path)
                dgl.save_graphs(processed_path, graph)
                # torch.save(graph_und, processed_path)

                pickle.dump(pred, open(f'{self.processed_dir}/{cnt}_pred.pkl', 'wb'))  # print predecessors
                pickle.dump(succ, open(f'{self.processed_dir}/{cnt}_succ.pkl', 'wb'))  # print successors

                graphia_path = os.path.join(graphia_dir, f'{cnt}_graph.txt')
                graph_parser.print_pairwise(graph, graphia_path)  # print pairwise-txt format for graphia visualization

                # graph_path = os.path.join(sequences_path, f'graph_{cnt}')  # print sequences, useful for gepard
                # if not os.path.isdir(graph_path):
                #     os.mkdir(graph_path)
                # graph_parser.print_fasta(graph_und, graph_path)
                f.write(f'{cnt} - {reads}\n')



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
