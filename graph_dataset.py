import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset

import graph_parser


class AssemblyGraphDataset(DGLDataset):

    def __init__(self, root):
        self.root = os.path.abspath(root)
        if 'raw' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'raw'", shell=True, cwd=self.root)
        if 'tmp' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'tmp'", shell=True, cwd=self.root)
        if 'processed' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'processed'", shell=True, cwd=self.root)
        if 'info' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'info'", shell=True, cwd=self.root)
        raw_dir = os.path.join(self.root, 'raw')
        save_dir = os.path.join(self.root, 'processed')
        self.tmp_dir = os.path.join(self.root, 'tmp')
        self.info_dir = os.path.join(self.root, 'info')
        self.raven_path = os.path.abspath('vendor/raven/build/bin/raven')
        super().__init__(name='assembly_graphs', raw_dir=raw_dir, save_dir=save_dir)

    def has_cache(self):
        return len(os.listdir(self.save_dir)) == len(os.listdir(self.raw_dir))

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        (graph,), _ = dgl.load_graphs(os.path.join(self.save_dir, str(idx) + '.dgl'))
        return idx, graph

    def process(self):
        graphia_dir = os.path.join(self.root, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        with open(f'{self.root}/dataset_log.txt', 'w') as f:
            for cnt, reads in enumerate(os.listdir(self.raw_dir)):
                print(cnt, reads)
                reads_path = os.path.abspath(os.path.join(self.raw_dir, reads))
                print(reads_path)
                subprocess.run(f'{self.raven_path} --weaken -t32 -p0 {reads_path} > assembly.fasta', shell=True, cwd=self.tmp_dir)
                processed_path = os.path.join(self.save_dir, str(cnt) + '.dgl')
                graph, pred, succ, reads = graph_parser.from_csv(os.path.join(self.tmp_dir, 'graph_before.csv'), reads_path)
                dgl.save_graphs(processed_path, graph)

                pickle.dump(pred, open(f'{self.info_dir}/{cnt}_pred.pkl', 'wb'))
                pickle.dump(succ, open(f'{self.info_dir}/{cnt}_succ.pkl', 'wb'))
                pickle.dump(reads, open(f'{self.info_dir}/{cnt}_reads.pkl', 'wb'))

                graphia_path = os.path.join(graphia_dir, f'{cnt}_graph.txt')
                graph_parser.print_pairwise(graph, graphia_path)
                f.write(f'{cnt} - {reads}\n')
