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


def quick_evaluation(walks, graph, reads, edges, chrN):
    contigs = walk_to_sequence(walks, graph, reads, edges)
    chr_len = chr_lens[chrN]
    lengths_list = [len(c.seq) for c in contigs]
    num_contigs = len(contigs)
    longest_contig = max(lengths_list)
    reconstructed = sum(lengths_list) / chr_len
    n50 = calculate_N50(contigs)
    ng50 = calculate_NG50(contigs, chr_len)
    return num_contigs, longest_contig, reconstructed, n50, ng50


# def txt_output(f, txt):
#     print(f'\t{txt}')
#     f.write(f'\t{txt}\n')

# def analyze(graph, gnn_paths, greedy_paths, out, ref_length):
#     with open(f'{out}/analysis.txt', 'w') as f:
#         # f.write(f'Chromosome total length:\t\n')
#         #print(out.split("/"), out.split("/")[-2])
#         gnn_contig_lengths = []
#         for path in gnn_paths:
#             contig_len = graph.ndata["read_end"][path[-1]] - graph.ndata["read_start"][path[0]]
#             gnn_contig_lengths.append(abs(contig_len).item())
#         txt_output(f, 'GNN: ')
#         txt_output(f, f'Contigs: \t{gnn_contig_lengths}')
#         txt_output(f,f'Contigs amount:\t{len(gnn_contig_lengths)}')
#         txt_output(f,f'Longest Contig:\t{max(gnn_contig_lengths)}')
#         txt_output(f,f'Reconstructed:\t{sum(gnn_contig_lengths)}')
#         txt_output(f,f'Percentage:\t{sum(gnn_contig_lengths)/ref_length*100}')
#         n50_gnn = calculate_N50(gnn_contig_lengths)
#         txt_output(f,f'N50:\t{n50_gnn}')
#         ng50_gnn = calculate_NG50(gnn_contig_lengths, ref_length)
#         txt_output(f,f'NG50:\t{ng50_gnn}')


#         txt_output(f,f'Greedy paths:\t{len(greedy_paths)}\n')
#         greedy_contig_lengths = []
#         for path in greedy_paths:
#             contig_len = graph.ndata["read_end"][path[-1]] - graph.ndata["read_start"][path[0]]
#             greedy_contig_lengths.append(abs(contig_len).item())
#         txt_output(f, 'Greedy: ')
#         txt_output(f, f'Contigs: \t{greedy_contig_lengths}')
#         txt_output(f,f'Contigs amount:\t{len(greedy_contig_lengths)}')
#         txt_output(f,f'Longest Contig:\t{max(greedy_contig_lengths)}')
#         txt_output(f,f'Reconstructed:\t{sum(greedy_contig_lengths)}')
#         txt_output(f,f'Percentage:\t{sum(greedy_contig_lengths)/ref_length*100}')
#         n50_greedy = calculate_N50(greedy_contig_lengths)
#         txt_output(f,f'N50:\t{n50_greedy}')
#         ng50_greedy = calculate_NG50(greedy_contig_lengths, ref_length)
#         txt_output(f,f'NG50:\t{ng50_greedy}')


def run_quast():
    ...

