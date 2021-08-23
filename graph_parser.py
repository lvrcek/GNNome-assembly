import re
from collections import deque, defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
import dgl
import networkx as nx
from matplotlib import pyplot as plt


def draw_graph(graph_nx):
    """Visualize the graph in the NetworkX format - DEPRECATED.
    
    Parameters
    ----------
    graph_nx : networkx.DiGraph
        A graph to be visualized

    Returns
    -------
    None
    """
    nx.draw(graph_nx, node_size=6, width=.2, arrowsize=3)
    plt.show()


# TODO: Maybe put all these into a Graph class?
def get_neighbors(graph):
    """Return neighbors/successors for each node in the graph.
    
    Parameters
    ----------
    graph : graph.DGLGraph
        A DGLGraph for which neighbors will be determined for each
        node

    Returns
    -------
    dict
        a dictionary where nodes' ordinal numbers are keys and lists
        with all the nodes' neighbors are values
    """
    neighbor_dict = {i.item(): [] for i in graph.nodes()}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        neighbor_dict[src.item()].append(dst.item())
    return neighbor_dict


def get_predecessors(graph):
    """Return predecessors for each node in the graph.
    
    Parameters
    ----------
    graph : graph.DGLGraph
        A DGLGraph for which predecessors will be determined for each
        node

    Returns
    -------
    dict
        a dictionary where nodes' ordinal numbers are keys and lists
        with all the nodes' predecessors are values
    """
    predecessor_dict = {i.item(): [] for i in graph.nodes()}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        predecessor_dict[dst.item()].append(src.item())
    return predecessor_dict


def find_edge_index(graph, src, dst):
    """Return index for the edge in a graph.

    Given a DGLGraph object, and two nodes connected by an edge,
    return the index of that edge.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph in which the edge is searched
    src : int
        Index of the source node
    dst : int
        Index of the destination node

    Returns
    -------
    int
        index of the edge connecting src and dst nodes
    """
    for idx, (node1, node2) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        if node1 == src and node2 == dst:
            return idx


def translate_nodes_into_sequence(graph, reads, node_tr):
    """Concatenate reads associated with nodes in a walk.

    Each node is associated with a read (genomic sequence), and when
    there is an edge between two ndoes that means that the associated
    sequences are overlapping. Given a graph, reads, and a list of
    nodes, this function concatenates the sequences for all the nodes
    in a list in a prefix-suffix manner. For overlapping, this function
    relies on the overlap_length attribute.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph on which the walk was performed
    reads : dict
        A dictionary where each node is associated with a sequence
    node_tr : list
        A list of nodes depicting the walk

    Returns
    -------
    str
        a sequence of concatenated reads
    """
    seq = reads[node_tr[0]]
    for src, dst in zip(node_tr[:-1], node_tr[1:]):
        idx = find_edge_index(graph, src, dst)
        overlap_length = graph.edata['overlap_length'][idx]
        seq += reads[dst][overlap_length:]
    return seq


def translate_nodes_into_sequence2(graph, reads, node_tr):
    """Concatenate reads associated with nodes in a walk.

    Each node is associated with a read (genomic sequence), and when
    there is an edge between two ndoes that means that the associated
    sequences are overlapping. Given a graph, reads, and a list of
    nodes, this function concatenates the sequences for all the nodes
    in a list in a prefix-suffix manner. For overlapping, this function
    relies on the prefix_length attribute.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        A graph on which the walk was performed
    reads : dict
        A dictionary where each node is associated with a sequence
    node_tr : list
        A list of nodes depicting the walk

    Returns
    -------
    str
        a sequence of concatenated reads
    """
    seq = ''
    for src, dst in zip(node_tr[:-1], node_tr[1:]):
        idx = find_edge_index(graph, src, dst)
        prefix_length = graph.edata['prefix_length'][idx]
        seq += reads[src][:prefix_length]
    seq += reads[node_tr[-1]]
    return seq


def get_quality(hits, seq_len):
    """Returns the fraction of the best mapping - DEPRECATED
    
    Parameters
    ----------
    hits : list
        The list of mappy.Alignment objects resulting from aligning
        a sequence to the reference
    seq_len : int
        Length of the sequence aligned to the reference

    Returns
    -------
    float
        a fraction of how much sequence was aligned to the reference
    """
    return (hits[0].q_en - hits[0].q_st) / seq_len


def print_pairwise(graph, path):
    """Outputs the graph into a pairwise TXT format.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        The DGLGraph which is saved to the TXT file
    path : str
        The location where to save the TXT file

    Returns
    -------
    None
    """
    with open(path, 'w') as f:
        for src, dst in zip(graph.edges()[0], graph.edges()[1]):
            f.write(f'{src}\t{dst}\n')


def from_gfa(graph_path, reads_path):
    """Parse an assembly graph stored in a GFA format.

    Raven assemblers can store an assembly graph in a CSV or a GFA
    format. This function parses the GFA file and extracts the
    sequences from the FASTQ files by comparing read IDs. Returns
    deques of sequences and discriptions extracted from the FASTQ file.

    Parameters
    ----------
    graph_path : src
        The location of the CSV file
    reads_path : src
        The location of the FASTQ file with the associated reads

    Returns
    -------
    deque
       a deque of read sequences
    deque
        a deque of read discriptions
    """
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


def from_csv(graph_path, reads_path):
    """Parse an assembly graph stored in a CSV format.

    Raven assemblers can store an assembly graph in a CSV or a GFA
    format. This function parses the CSV file, creates a DGL graph
    and returns the DGL graph and its related information stored in
    dictionaries---neighbors, predecessors and sequences of each node.

    Parameters
    ----------
    graph_path : src
        The location of the CSV file
    reads_path : src
        The location of the FASTQ file with the associated reads

    Returns
    -------
    dgl.DGLGraph
        a DGLGraph constructed from the parsed CSV file
    dict
        a dictionary storing predecessors of each node
    dict
        a dictionary storing successors of each node
    dict
        a dictionary storing genomic sequence of each node
    """
    graph_nx = nx.DiGraph()
    graph_nx_und = nx.Graph()
    read_length = {}
    node_data = {}
    read_idx, read_strand, read_start, read_end = {}, {}, {}, {}
    edge_ids, prefix_length, overlap_similarity, overlap_length = {}, {}, {}, {}
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
                    # This is on purpose so that positive strand reads are 'forwards'
                    # While negative strand reads become 'backwards' (start < end)
                    read_start[dst_id] = end
                    read_end[dst_id] = start
                    graph_nx.add_node(dst_id)
                    graph_nx_und.add_node(dst_id)

            else:
                # Overlap info: id, length, weight, similarity
                overlap = overlap.split()
                try:
                    (edge_id, prefix_len), (weight, similarity) = map(int, overlap[:2]), map(float, overlap[2:])
                except IndexError:
                    continue
                graph_nx.add_edge(src_id, dst_id)
                graph_nx_und.add_edge(src_id, dst_id)
                if (src_id, dst_id) not in prefix_length.keys():
                    edge_ids[(src_id, dst_id)] = edge_id
                    prefix_length[(src_id, dst_id)] = prefix_len
                    overlap_length[(src_id, dst_id)] = read_length[src_id] - prefix_len
                    overlap_similarity[(src_id, dst_id)] = similarity
    
    nx.set_node_attributes(graph_nx, read_length, 'read_length')
    nx.set_node_attributes(graph_nx, read_idx, 'read_idx')
    nx.set_node_attributes(graph_nx, read_strand, 'read_strand')
    nx.set_node_attributes(graph_nx, read_start, 'read_start')
    nx.set_node_attributes(graph_nx, read_end, 'read_end')
    nx.set_edge_attributes(graph_nx, prefix_length, 'prefix_length')
    nx.set_edge_attributes(graph_nx, overlap_similarity, 'overlap_similarity')
    nx.set_edge_attributes(graph_nx, overlap_length, 'overlap_length')
    
    # This produces vector-like features (e.g. shape=(num_nodes,))
    graph_dgl = dgl.from_networkx(graph_nx,
                                  node_attrs=['read_length', 'read_idx', 'read_strand', 'read_start', 'read_end'], 
                                  edge_attrs=['prefix_length', 'overlap_similarity', 'overlap_length'])
    predecessors = get_predecessors(graph_dgl)
    successors = get_neighbors(graph_dgl)
    reads = {}
    for i, key in enumerate(sorted(node_data)):
        reads[i] = node_data[key]

    return graph_dgl, predecessors, successors, reads
