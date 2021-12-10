import argparse

import torch
import dgl

import graph_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Root directory path')
    parser.add_argument('-t', '--threads', type=int, default=32, help='Number of threads used in Raven')
    parser.add_argument('-f', '--filter', type=float, default=0.99, help='Overlap filter used in Raven')
    parser.add_argument('-o', '--out', type=str, defualt='assembly.fasta', help='Name of Raven assembly output file')
    args = parser.parse_args()
    specs = {
        'threads': args.threads,
        'filter': args.filter,
        'out': args.out
    }
    ds = graph_dataset.AssemblyGraphDataset(args.data, specs)
