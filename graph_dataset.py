import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset

import graph_parser
from utils import preprocess_graph, add_positional_encoding

class AssemblyGraphDataset(DGLDataset):
    """
    A dataset to store the assembly graphs.

    A class that inherits from the DGLDataset and extends the
    functionality by adding additional attributes and processing the
    graph data appropriately.

    Attributes
    ----------
    root : str
        Root directory consisting of other directories where the raw
        data can be found (reads in FASTQ format), and where all the
        processing results are stored.
    tmp_dir : str
        Directory inside root where mid-results (output of the raven 
        assembler) is stored
    info_dir : str
        Directory where additional graph information is stored
    raven_path : str
        Path to the raven assembler
    """

    def __init__(self, root, nb_pos_enc=10, specs=None):
        """
        Parameters
        ----------
        root : str
            Root directory consisting of other directories where the raw
            data can be found (reads in FASTQ format), and where all the
            processing results are stored.
        """
        self.root = os.path.abspath(root)
        self.specs = specs
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

        self.graph_list = []
        for file in sorted(os.listdir(self.save_dir)):
            idx = int(file[:-4])
            graph = dgl.load_graphs(os.path.join(self.save_dir, file))[0][0]
            graph = preprocess_graph(graph, self.root, idx)
            graph = add_positional_encoding(graph, nb_pos_enc) 
            #graph, _ = dgl.khop_in_subgraph(graph, 390, k=20) # DEBUG !!!!
            print('DGL graph info:',graph)
            self.graph_list.append(graph)

    def has_cache(self):
        """Check if the raw data is already processed and stored."""
        # return len(os.listdir(self.save_dir)) > 0
        return len(os.listdir(self.save_dir)) >= len(os.listdir(self.raw_dir))

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        # (graph,), _ = dgl.load_graphs(os.path.join(self.save_dir, str(idx) + '.dgl'))
        graph = self.graph_list[idx]
        return idx, graph

    def process(self):
        """Process the raw data and save it on the disk."""
        if self.specs is None:
            threads = 32
            filter = 0.99
            out = 'assembly.fasta'
        else:
            threads = self.specs['threads']
            filter = self.specs['filter']
            out = self.specs['out']

        graphia_dir = os.path.join(self.root, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        print(f'====> FILTER = {filter}')

        with open(f'{self.root}/dataset_log.txt', 'w') as f:
            n_have = len(os.listdir(self.save_dir))
            n_need = len(os.listdir(self.raw_dir))
            n_diff = n_need - n_have
            files = sorted(os.listdir(self.raw_dir))  # [0.fasta, 1.fasta, ...]
            for cnt, idx in enumerate(range(n_have, n_need)):
            # for cnt, fastq in enumerate(os.listdir(self.raw_dir)):
                fastq = files[idx]  # have 4 [0-3] -> fastq = 4.fasta, 5.fasta, ...
                print(cnt, fastq)
                reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
                print(reads_path)
                subprocess.run(f'{self.raven_path} --filter {filter} --weaken -t{threads} -p0 {reads_path} > {out}', shell=True, cwd=self.tmp_dir)
                cnt = idx  # Just not to change original code too much yet. TODO: Fix later
                for j in range(1, 2):
                    print(f'graph {j}')
                    # processed_path = os.path.join(self.save_dir, f'd{cnt}_g{j}.dgl')
                    processed_path = os.path.join(self.save_dir, f'{cnt}.dgl')
                    graph, pred, succ, reads, edges = graph_parser.from_csv(os.path.join(self.tmp_dir, f'graph_{j}.csv'), reads_path)
                    dgl.save_graphs(processed_path, graph)

                    pickle.dump(pred, open(f'{self.info_dir}/{cnt}_pred.pkl', 'wb'))
                    pickle.dump(succ, open(f'{self.info_dir}/{cnt}_succ.pkl', 'wb'))
                    pickle.dump(reads, open(f'{self.info_dir}/{cnt}_reads.pkl', 'wb'))
                    pickle.dump(edges, open(f'{self.info_dir}/{cnt}_edges.pkl', 'wb'))

                    graphia_path = os.path.join(graphia_dir, f'{cnt}_graph.txt')
                    graph_parser.print_pairwise(graph, graphia_path)
                f.write(f'{cnt} - {fastq}\n')
