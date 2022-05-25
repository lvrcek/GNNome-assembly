import re
from collections import deque, defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
import dgl
import networkx as nx
import torch

import algorithms


def get_neighbors(graph):
    """Return neighbors/successors for each node in the graph.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
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
    graph : dgl.DGLGraph
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


def get_edges(graph):
    """Return edge index for each edge in the graph.

    Parameters
    ----------
    graph : dgl.DGLGraph
        A DGLGraph for which edge indices will be saved

    Returns
    -------
    dict
        a dictionary where keys are (source, destination) tuples of
        nodes, and corresponding edge indices are values
    """
    edges_dict = {}
    for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_dict[(src, dst)] = idx
    return edges_dict


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

    Raven assembler can store an assembly graph in a CSV or a GFA
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

    # ---------------------------------------------------------- READ THIS ----------------------------------------------------------------------------
    # I need both the GFA and the FASTQ because they hold different information.
    # GFA also serves as the "link" between different formats since it contains the read id taken from GFA.
    # First all the reads are stored in a list, together with their info (description)--the info contains read id, start, end, PRIOR TO TRIMMING!!!
    # Then I parse the GFA file. From here I get the read ids (indices in FASTQ) and the sequences.
    # Why the sequences from here? Because they are already trimmed, so it's easier than to take them from FASTQ and trim them manually
    # With the read ids I can access the reads in the FASTQ file (or rather in a list obtained from parsing the file).
    # This is simply: nth_fastq_read = read_list[id_from_gfa].
    # Note: Not all the reads are stored in the GFA. The GFA is created when the contained reads are already discarded.
    # --------------------------------------------------------------------------------------------------------------------------------------------------

    read_sequences = deque()
    description_queue = deque()
    # TODO: Parsing of reads won't work for larger datasets nor gzipped files
    if reads_path.endswith('fastq'):
        reads_list = {read.id: read.description for read in SeqIO.parse(reads_path, 'fastq')}
    else:
        reads_list = {read.id: read.description for read in SeqIO.parse(reads_path, 'fasta')}
    with open(graph_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 5:
                tag, id, sequence, length, count = line
                sequence = Seq(sequence)  # TODO: This sequence is already trimmed! Make sure that everything is matching
                read_sequences.append(sequence)
                # read_sequences.append(sequence.reverse_complement())
                try:
                    description = reads_list[id]
                except ValueError:
                    description = '0 strand=+, start=0, end=0'
                description_queue.append(description)
            else:
                break
    return read_sequences, description_queue


def from_csv(graph_path, reads_path):
    """Parse an assembly graph stored in a CSV format.

    Raven assembler can store an assembly graph in a CSV or a GFA
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
    read_length = {}  # Obtained from the CSV
    node_data = {}  # Obtained from the GFA
    read_idx, read_strand, read_start, read_end = {}, {}, {}, {}  # Obtained from the FASTA/Q headers
    edge_ids, prefix_length, overlap_similarity, overlap_length = {}, {}, {}, {}  # Obtained from the CSV

    # ---------------------------------------------------------------------------------------------------
    # Here I get the sequences (and their rcs) and the descriptions.
    # Descriptions contain read idx, strand, start, and read end info---obtained from the simulator.
    # This is crucial for the supervision signal and the ground-truth algorithms, that's why I need it.
    # ############## 0-based CSV node [] index == 0-based line ordinal number in GFA #################
    # e.g. in CSV 20 [10] ... 21 [10] ... == 10th (11th 1-based) line in GFA
    #
    # ############################ Sometimes CSV swallows some node ids ##############################
    # For eaxmple, this happens in graph 18 (chr11_58-60) where there are no nodes 824 and 825.
    # THe reason is that these nodes have no edges, so they are useless and can be omitted.
    # This means that all the node ids greater than 823 in the DGL will be reduced by one.
    # So, in CSV node ids will be: 822, 823, 826, 827, ..., 3028, 3029 <end>
    # And in the DGL they will be: 822, 823, 824, 825, ..., 3026, 3027 <end>
    # ---------------------------------------------------------------------------------------------------

    read_sequences, description_queue = from_gfa(graph_path[:-3] + 'gfa', reads_path)

    read_trim_start, read_trim_end = {}, {}  # Obtained from the CSV

    with open(graph_path) as f:
        for line in f.readlines():
            src, dst, flag, overlap = line.strip().split(',')
            src, dst = src.split(), dst.split()
            flag = int(flag)
            pattern = r':(\d+)'
            src_id, src_len = int(src[0]), int(re.findall(pattern, src[2])[0])
            dst_id, dst_len = int(dst[0]), int(re.findall(pattern, dst[2])[0])
            # --------------------------
            # src_len and dst_len are length of the trimmed reads!!
            # --------------------------

            if flag == 0:
                # Here overlap is actually trimming info! trim_begin trim_end
                description = description_queue.popleft()
                try:
                    id, strand, start, end = description.split()
                except ValueError:
                    id, idx, strand, start, end = description.split()
                # except:
                #     id, idx, strand, start, end = description.split()

                try:
                    idx = int(id)
                except ValueError:
                    idx = int(re.findall(r'[a-zA-Z0-9]*\.(\d+)', id)[0])

                strand = 1 if strand[-2] == '+' else -1  # strand[-1] == ','

                # -----------------------------------------
                # start and end values are UNTRIMMED!
                # -----------------------------------------
                start = int(re.findall(r'start=(\d+)', start)[0])  
                end = int(re.findall(r'end=(\d+)', end)[0])

                trimming = overlap
                if trimming == '-':
                    trim_start, trim_end = 0, end - start
                else:
                    trim_start, trim_end = trimming.split()
                    trim_start = int(trim_start)
                    trim_end = int(trim_end)
               
                end = start + trim_end
                start = start + trim_start

                read_sequence = read_sequences.popleft()
                node_data[src_id] = read_sequence
                node_data[dst_id] = read_sequence.reverse_complement()

                read_length[src_id], read_length[dst_id] = src_len, dst_len
                read_idx[src_id] = read_idx[dst_id] = idx
                read_strand[src_id], read_strand[dst_id] = strand, -strand
                read_start[src_id] = read_start[dst_id] = start
                read_end[src_id] = read_end[dst_id] = end
                read_trim_start[src_id] = read_trim_start[dst_id] = trim_start
                read_trim_end[src_id] = read_trim_end[dst_id] = trim_end

                graph_nx.add_node(src_id)
                graph_nx.add_node(dst_id)

            else:
                # Overlap info: id, length, weight, similarity
                overlap = overlap.split()
                try:
                    (edge_id, prefix_len), (weight, similarity) = map(int, overlap[:2]), map(float, overlap[2:])
                except IndexError:
                    print("Index ERROR occured!")
                    continue
                except ValueError:
                    (edge_id, prefix_len), weight, similarity = map(int, overlap[:2]), float(overlap[2]), 0
                graph_nx.add_edge(src_id, dst_id)
                if (src_id, dst_id) not in prefix_length.keys():  # TODO: This will always be true
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

    nx.set_node_attributes(graph_nx, read_trim_start, 'read_trim_start')
    nx.set_node_attributes(graph_nx, read_trim_end, 'read_trim_end')
    
    # This produces vector-like features (e.g. shape=(num_nodes,))
    graph_dgl = dgl.from_networkx(graph_nx,
                                  node_attrs=['read_length', 'read_idx', 'read_strand', 'read_start', 'read_end', 'read_trim_start', 'read_trim_end'], 
                                  edge_attrs=['prefix_length', 'overlap_similarity', 'overlap_length'])
    predecessors = get_predecessors(graph_dgl)
    successors = get_neighbors(graph_dgl)
    edges = get_edges(graph_dgl)
    reads = {}
    for i, key in enumerate(sorted(node_data)):
        reads[i] = node_data[key]

    gt_edges_pos, gt_edges_neg = algorithms.get_gt_graph(graph_dgl, successors, edges)
    labels = gt_edges_pos | gt_edges_neg
    graph_dgl.edata['y'] = torch.tensor([1 if i in labels else 0 for i in range(graph_dgl.num_edges())], dtype=torch.float)

    return graph_dgl, predecessors, successors, reads, edges, labels

