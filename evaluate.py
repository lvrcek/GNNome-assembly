import os
import pickle

import torch
import dgl
from Bio import SeqIO


chr_lens = {
    'chr1' : 248387328,
    'chr2' : 242696752,
    'chr3' : 201105948,
    'chr4' : 193574945,
    'chr5' : 182045439,
    'chr6' : 172126628,
    'chr7' : 160567428,
    'chr8' : 146259331,
    'chr9' : 150617247,
    'chr10': 134758134,
    'chr11': 135127769,
    'chr12': 133324548,
    'chr13': 113566686,
    'chr14': 101161492,
    'chr15': 99753195,
    'chr16': 96330374,
    'chr17': 84276897,
    'chr18': 80542538,
    'chr19': 61707364,
    'chr20': 66210255,
    'chr21': 45090682,
    'chr22': 51324926,
    'chrX' : 154259566,
}


def walk_to_sequence(walks, graph, reads, edges):
    contigs = []
    for i, walk in enumerate(walks):
        sequence = ''
        prefixes = [(src, graph.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]
        sequences = [reads[src][:prefix] for (src, prefix) in prefixes]
        sequence = ''.join(map(str, sequences)) + reads[walk[-1]]
        sequence = SeqIO.SeqRecord(sequence)
        sequence.id = f'contig_{i+1}'
        sequence.description = f'length={len(sequence)}'
        contigs.append(sequence)
    return contigs


def save_assembly(contigs, data_path, idx, suffix='', dir_name='assembly'):
    assembly_dir = os.path.join(data_path, dir_name)
    if dir_name not in os.listdir(data_path):
        os.mkdir(assembly_dir)
    assembly_path = os.path.join(assembly_dir, f'{idx}_assembly{suffix}.fasta')
    SeqIO.write(contigs, assembly_path, 'fasta')


def calculate_N50(contigs):
    """Calculate N50 for contigs.
    Args:
        list_of_lengths (list): List of SeqRecord objects.
    Returns:
        float: N50 value.
    """
    lengths_list = [len(c.seq) for c in contigs]
    lengths_list.sort(reverse=True)
    total_length = sum(lengths_list)
    total_bps = 0
    for length in lengths_list:
        total_bps += length
        if total_bps >= total_length / 2:
            return length
    return -1


def calculate_NG50(contigs, ref_length):
    """Calculate NG50 for contigs.
    Args:
        list_of_lengths (list): List of SeqRecord objects.
    Returns:
        int: NG50 value.
    """
    if ref_length <= 0:
        return -1
    lengths_list = [len(c.seq) for c in contigs]
    lengths_list.sort(reverse=True)
    total_bps = 0
    for length in lengths_list:
        total_bps += length
        if total_bps >= ref_length / 2:
            return length
    return -1


def quick_evaluation(contigs, chrN):
    # contigs = walk_to_sequence(walks, graph, reads, edges)
    chr_len = chr_lens[chrN]
    lengths_list = [len(c.seq) for c in contigs]
    num_contigs = len(contigs)
    longest_contig = max(lengths_list)
    reconstructed = sum(lengths_list) / chr_len
    n50 = calculate_N50(contigs)
    ng50 = calculate_NG50(contigs, chr_len)
    return num_contigs, longest_contig, reconstructed, n50, ng50


def txt_output(f, txt):
    print(f'{txt}')
    f.write(f'{txt}\n')


def print_summary(data_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50):
    reports_dir = os.path.join(data_path, 'reports')
    if not os.path.isdir(reports_dir ):
        os.mkdir(reports_dir)
    with open(f'{reports_dir}/{idx}_report.txt', 'w') as f:
        txt_output(f, f'-'*80)
        txt_output(f, f'Report for graph {idx} in {data_path}')
        txt_output(f, f'Graph created from {chrN}')
        txt_output(f, f'Num contigs:\t{num_contigs}')
        txt_output(f, f'Longest contig:\t{longest_contig}')
        txt_output(f, f'Reconstructed:\t{reconstructed * 100:2f}%')
        txt_output(f, f'N50:\t{n50}')
        txt_output(f, f'NG50:\t{ng50}')
        
