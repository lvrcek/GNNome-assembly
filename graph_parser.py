import os
import re
from collections import deque, defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
import dgl
import networkx as nx
from matplotlib import pyplot as plt


def draw_graph(graph_nx):
    nx.draw(graph_nx, node_size=6, width=.2, arrowsize=3)
    plt.show()


# TODO: Maybe put all these into a Graph class?
def get_neighbors(graph):
    neighbor_dict = defaultdict(list)
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        neighbor_dict[src.item()].append(dst.item())
    return neighbor_dict


def get_predecessors(graph):
    predecessor_dict = defaultdict(list)
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        predecessor_dict[dst.item()].append(src.item())
    return predecessor_dict


def find_edge_index(graph, src, dst):
    for idx, (node1, node2) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        if node1 == src and node2 == dst:
            return idx


# TODO: add overlap length from GFA to graph
# def translate_nodes_into_sequence(graph, node_tr):
#     seq = graph.read_sequence[node_tr[0]]
#     for src, dst in zip(node_tr[:-1], node_tr[1:]):
#         idx = find_edge_index(graph, src, dst)
#         overlap_length = graph.overlap_length[idx]
#         seq += graph.read_sequence[dst][overlap_length:]
#     return seq


def translate_nodes_into_sequence2(graph, node_tr):
    seq = ''
    for src, dst in zip(node_tr[:-1], node_tr[1:]):
        idx = find_edge_index(graph, src, dst)
        prefix_length = graph.prefix_length[idx]
        # In graph, len(graph.read_sequence) == num_nodes. Same if I take it out from dataset
        # But with DataLoader, len(graph.read_sequence) == 1. As if it was unsqueezed at some point during loading

        if not hasattr(graph, 'batch'):
            seq += graph.read_sequence[src][:prefix_length]
        else:
            seq += graph.read_sequence[0][src][:prefix_length]  # Why is this so?!

    if not hasattr(graph, 'batch'):
        seq += graph.read_sequence[node_tr[-1]]
    else:
        seq += graph.read_sequence[0][node_tr[-1]]
    return seq


def get_quality(hits, seq_len):
    # Returns the fraction of the best mapping
    # Could also include the number of mappings (-), mapping quality (+)
    # Maybe something else, a more sophisticated method of calculation
    return (hits[0].q_en - hits[0].q_st) / seq_len


def print_pairwise(graph, path):
    with open(path, 'w') as f:
        for src, dst in zip(graph.edges()[0], graph.edges()[1]):
            f.write(f'{src}\t{dst}\n')


def print_fasta(graph, path):
    for idx, seq in enumerate(graph.read_sequence):
        with open(f'{path}/{idx}.fasta', 'w') as f:
            f.write(f'>node_{idx}\n')
            f.write(str(seq + '\n'))


def from_gfa(graph_path, reads_path):
    read_sequences = deque()
    description_queue = deque()
    # TODO: Parsing of reads won't work for larger datasets nor gzipped files
    reads_list = list(SeqIO.parse(reads_path, 'fastq'))
    with open(graph_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 5:
                tag, id, sequence, length, count = line
                sequence = Seq(sequence)
                read_sequences.append(sequence)
                read_sequences.append(sequence.reverse_complement())
                try:
                    description = reads_list[int(id)].description
                except ValueError:
                    # TODO: The parsing above does not work for unitigs, find a better workaround
                    description = '0 idx=0, strand=+, start=0, end=0'
                description_queue.append(description)
            else:
                break
    return read_sequences, description_queue


def from_csv_dgl(graph_path, reads_path):
    graph_nx = nx.DiGraph()
    graph_nx_und = nx.Graph()
    read_length = {}
    node_data = {}
    read_idx, read_strand, read_start, read_end = {}, {}, {}, {}
    edge_ids, prefix_length, overlap_similarity = {}, {}, {}
    read_sequences, description_queue = from_gfa(graph_path[:-3] + 'gfa', reads_path)

    with open(graph_path) as f:
        for line in f.readlines():
            src, dst, flag, overlap = line.strip().split(',')
            src, dst = src.split(), dst.split()
            flag = int(flag)
            pattern = r':(\d+)'
            src_id, src_len = int(src[0]), int(re.findall(pattern, src[2])[0])
            dst_id, dst_len = int(dst[0]), int(re.findall(pattern, dst[2])[0])

            if flag == 0:
                
                description = description_queue.popleft()
                id, idx, strand, start, end = description.split()
                idx = int(re.findall(r'idx=(\d+)', idx)[0])
                strand = 1 if strand[-2] == '+' else -1
                start = int(re.findall(r'start=(\d+)', start)[0])
                end = int(re.findall(r'end=(\d+)', end)[0])
                if strand == -1:
                    start, end = end, start
                
                if src_id not in read_length.keys():
                    read_length[src_id] = src_len
                    node_data[src_id] = read_sequences.popleft()
                    read_idx[src_id] = idx
                    read_strand[src_id] = strand
                    read_start[src_id] = start
                    read_end[src_id] = end
                    graph_nx.add_node(src_id)
                    graph_nx_und.add_node(src_id)

                if dst_id not in read_length.keys():
                    read_length[dst_id] = dst_len
                    node_data[dst_id] = read_sequences.popleft()
                    read_idx[dst_id] = idx
                    read_strand[dst_id] = -strand
                    read_start[dst_id] = end
                    read_end[dst_id] = start
                    graph_nx.add_node(dst_id)
                    graph_nx_und.add_node(dst_id)

            else:
                # Overlap info: id, length, weight, similarity
                overlap = overlap.split()
                try:
                    [edge_id, prefix_len, _], similarity = map(int, overlap[:3]), float(overlap[3])
                except IndexError:
                    continue
                graph_nx.add_edge(src_id, dst_id)
                graph_nx_und.add_edge(src_id, dst_id)
                if (src_id, dst_id) not in prefix_length.keys():
                    edge_ids[(src_id, dst_id)] = edge_id
                    prefix_length[(src_id, dst_id)] = prefix_len
                    overlap_similarity[(src_id, dst_id)] = similarity
    
    nx.set_node_attributes(graph_nx, read_length, 'read_length')
    nx.set_node_attributes(graph_nx, read_idx, 'read_idx')
    nx.set_node_attributes(graph_nx, read_strand, 'read_strand')
    nx.set_node_attributes(graph_nx, read_start, 'read_start')
    nx.set_node_attributes(graph_nx, read_end, 'read_end')
    # nx.set_node_attributes(graph_nx, node_data, 'read_sequence')
    nx.set_edge_attributes(graph_nx, prefix_length, 'prefix_length')
    nx.set_edge_attributes(graph_nx, overlap_similarity, 'overlap_similarity')
    
    # This produces vector-like features (e.g. shape=(num_nodes,))
    graph_dgl = dgl.from_networkx(graph_nx, node_attrs=['read_length', 'read_strand', 'read_start', 'read_end'], 
                                  edge_attrs=['prefix_length', 'overlap_similarity'])
    predecessors = get_predecessors(graph_dgl)
    successors = get_neighbors(graph_dgl)

    return graph_dgl, predecessors, successors
