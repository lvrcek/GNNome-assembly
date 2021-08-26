import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset

import graph_parser


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

    def __init__(self, root):
        """
        Parameters
        ----------
        root : str
            Root directory consisting of other directories where the raw
            data can be found (reads in FASTQ format), and where all the
            processing results are stored.
        """
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
        self.raven_path = os.path.abspath('vendor/raven_filter/build/bin/raven')
        super().__init__(name='assembly_graphs', raw_dir=raw_dir, save_dir=save_dir)

    def has_cache(self):
        """Check if the raw data is already processed and stored."""
        return len(os.listdir(self.save_dir)) == len(os.listdir(self.raw_dir))

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        (graph,), _ = dgl.load_graphs(os.path.join(self.save_dir, str(idx) + '.dgl'))
        return idx, graph

    def process(self):
        """Process the raw data and save it on the disk."""
        graphia_dir = os.path.join(self.root, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        with open(f'{self.root}/dataset_log.txt', 'w') as f:
            for cnt, fastq in enumerate(os.listdir(self.raw_dir)):
                print(cnt, fastq)
                reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
                print(reads_path)
                subprocess.run(f'{self.raven_path} --filter 1.0 --weaken -t32 -p0 {reads_path} > assembly.fasta', shell=True, cwd=self.tmp_dir)
                for j in range(1, 7):
                    print(f'graph {j}')
                    processed_path = os.path.join(self.save_dir, f'd{cnt}_g{j}.dgl')  # d = dataset [0, 18], g = graph [1, 7]
                    graph, pred, succ, reads = graph_parser.from_csv(os.path.join(self.tmp_dir, f'graph_{j}.csv'), reads_path)
                    dgl.save_graphs(processed_path, graph)

                    pickle.dump(pred, open(f'{self.info_dir}/d{cnt}_g{j}_pred.pkl', 'wb'))
                    pickle.dump(succ, open(f'{self.info_dir}/d{cnt}_g{j}_succ.pkl', 'wb'))
                    pickle.dump(reads, open(f'{self.info_dir}/d{cnt}_g{j}_reads.pkl', 'wb'))

                    graphia_path = os.path.join(graphia_dir, f'd{cnt}_g{j}_graph.txt')
                    graph_parser.print_pairwise(graph, graphia_path)
                f.write(f'{cnt} - {fastq}\n')
