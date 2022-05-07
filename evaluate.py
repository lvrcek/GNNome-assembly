import pickle

import torch
import dgl
from Bio import SeqIO


def walk_to_sequence(walks, graph, reads, edges):
    contigs = []
    for i, walk in enumerate(walks):
        sequence = ''
        for src, dst in zip(walk[:-1], walk[1:]):
            edge_id = edges[(src, dst)]
            prefix = graph.edata['prefix_length'][edge_id].item()
            sequence += reads[src][:prefix]
        sequence += reads[walk[-1]]
        sequence = SeqIO.SeqRecord(sequence)
        sequence.id = f'contig_{i+1}'
        sequence.description = f'length={len(sequence)}'
        contigs.append(sequence)
    return contigs


def save_assembly(contigs, data_path, idx):
    assembly_dir = os.path.join(data_path, 'assembly')
    if 'assembly' not in os.listdir(data_path):
        os.mkdir(assembly_dir)
    assembly_path = os.path.join(assembly_dir, '{idx}_assembly.fasta')
    SeqIO.write(contigs, assembly_path, 'fasta')


def calculate_N50(contigs):
    """Calculate N50 for a sequence of numbers.
    Args:
        list_of_lengths (list): List of numbers.
    Returns:
        float: N50 value.
    """
    list_of_lengths = [len(c) for c in contigs]
    list_of_length.sort(reverse=True)
    total_length = sum(list_of_lengths)
    total_bps = 0
    if total_length <= 0:
        return -1
    for length in list_of_lengths:
        total_bps += length
        if total_bps >= total_length/2:
            return length
    return -1


def calculate_N50_from_lengths(list_of_lengths):
    """Calculate N50 for a sequence of numbers.
    Args:
        list_of_lengths (list): List of numbers.
    Returns:
        float: N50 value.
    """
    tmp = []
    for tmp_number in set(list_of_lengths):
        tmp += [tmp_number] * list_of_lengths.count(tmp_number) * tmp_number
    tmp.sort()

    if (len(tmp) % 2) == 0:
        median = (tmp[int(len(tmp) / 2) - 1] + tmp[int(len(tmp) / 2)]) / 2
    else:
        median = tmp[int(len(tmp) / 2)]

    return median


def calculate_NG50(contigs, ref_length):
    """Calculate N50 for contigs.
    Args:
        list_of_lengths (list): List of SeqRecord objects.
    Returns:
        int: NG50 value.
    """
    list_of_lengths = [len(c.seq) for c in contigs]
    if ref_length <= 0:
        return -1
    list_of_lengths.sort(reverse=True)
    total_bps = 0
    for length in list_of_lengths:
        total_bps += length
        if total_bps >= ref_length/2:
            return length
    return -1



def calculate_NG50_from_lengths(list_of_lengths, ref_length):
    """Calculate N50 for a sequence of numbers.
    Args:
        list_of_lengths (list): List of numbers.
    Returns:
        float: N50 value.
    """
    if ref_length == 0:
        return -1
    list_of_lengths.sort(reverse=True)
    total_bps = 0
    for contig in list_of_lengths:
        total_bps += contig
        if total_bps > ref_length/2:
            return contig
    return -1


def quick_evaluation(walks, graph, reads, edges):
    contigs = walk_to_sequence(walks, graph, reads, edges)
    n50 = calculate_N50(contigs)
    ng50 = calculate_NG50(contigs, 0)  # How to pass which chromosome it is? Create another dictionary? ...
    return n50, ng50


def run_quest():
    ...

