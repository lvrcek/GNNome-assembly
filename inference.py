import argparse
import os
import pickle
import random
from tqdm import tqdm 
import collections


import torch
import torch.nn.functional as F
import dgl

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import algorithms
from utils import load_graph_data


def predict_new(model, graph, succs, preds, edges, device):
    x = graph.ndata['x'].to(device)
    e = graph.edata['e'].to(device)
    edge_logits = model(graph, x, e)
    # TODO: Problem, my block model doesn't work on full graphs!
    # TODO: I can still iterate over the batches and append the predictions
    edge_logits= edge_logits.squeeze(-1)
    edge_p = F.sigmoid(edge_logits)
    walks = decode_new(graph, edge_p, succs, preds, edges)
    return walks
    # or (later) translate walks into sequences
    # what with the sequences? Store into FASTA ofc


def decode_neurips(graph, edges_p, neighbors, predecessors, edges):
    # Choose starting node for the first time
    walks = []
    visited = set()
    # ----- Modify this later ------
    all_nodes = {n.item() for n in graph.nodes()}
    correct_nodes = {n for n in range(graph.num_nodes()) if graph.ndata['y'][n] == 1}
    potential_nodes = correct_nodes
    # ------------------------------
    while True:
        potential_nodes = potential_nodes - visited
        start = get_random_start(potential_nodes)
        if start is None:
            break
        visited.add(start)
        visited.add(start ^ 1)
        walk_f, visited_f = walk_forwards(start, edges_p, neighbors, edges, visited)
        walk_b, visited_b = walk_backwards(start, edges_p, predecessors, edges, visited)
        walk = walk_b[:-1] + [start] + walk_f[1:]
        visited = visited | visited_f | visited_b
        walks.append(walk)
    walks = sorted(walks, key=lambda x: len(x))
    return walks



def decode_new(graph, edges_p, neighbors, predecessors, edges):
    # Choose starting node for the first time
    walks = []
    visited = set()
    # ----- Modify this later ------
    all_nodes = {n.item() for n in graph.nodes()}
    correct_nodes = {n for n in range(graph.num_nodes()) if graph.ndata['y'][n] == 1}
    potential_nodes = correct_nodes
    # ------------------------------
    while True:
        potential_nodes = potential_nodes - visited
        start = get_random_start(potential_nodes)
        if start is None:
            break
        visited.add(start)
        visited.add(start ^ 1)
        walk_f, visited_f = walk_forwards(start, edges_p, neighbors, edges, visited)
        walk_b, visited_b = walk_backwards(start, edges_p, predecessors, edges, visited)
        walk = walk_b[:-1] + [start] + walk_f[1:]
        visited = visited | visited_f | visited_b
        walks.append(walk)
    walks = sorted(walks, key=lambda x: len(x))
    return walks
    

def get_random_start(potential_nodes, nodes_p=None):
    # potential_nodes = {n.item() for n in graph.nodes()}
    if len(potential_nodes) < 10:
        return None
    potential_nodes = potential_nodes
    start = random.sample(potential_nodes, 1)[0]
    # start = max(potential_nodes_p)
    return start


def walk_forwards(start, edges_p, neighbors, edges, visited_old):
    current = start
    walk = []
    visited = set()
    while True:
        if current in visited | visited_old and walk:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        if len(neighbors[current]) == 0:
            break 
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue
        neighbor_edges = [edges[current, n] for n in neighbors[current] if n not in visited]
        if not neighbor_edges:
            break
        neighbor_p = edges_p[neighbor_edges]
        _, index = torch.topk(neighbor_p, k=1, dim=0)
        choice = neighbors[current][index]
        current = choice
    return walk, visited


def walk_backwards(start, edges_p, predecessors, edges, visited_old):
    current = start
    walk = []
    visited = set()
    while True:
        if current in visited | visited_old and walk:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        if len(predecessors[current]) == 0:
            break 
        if len(predecessors[current]) == 1:
            current = predecessors[current][0]
            continue
        neighbor_edges = [edges[n, current] for n in predecessors[current] if n not in visited]
        if not neighbor_edges:
            break
        neighbor_p = edges_p[neighbor_edges]
        _, index = torch.topk(neighbor_p, k=1, dim=0)
        choice = predecessors[current][index]
        current = choice
    walk = list(reversed(walk))
    return walk, visited


def predict_old(model, graph, pred, neighbors, reads, edges):
    starts = [k for k,v in pred.items() if len(v)==0 and graph.ndata['read_strand'][k]==1]
    
    components = algorithms.get_components(graph, neighbors, pred)
    # components = [c for c in components if len(c) >= 10]  # For some reason components are not split properly so I should leave this line out
    components = sorted(components, key=lambda x: -len(x))
    walks = []

    logits = model(graph, reads)

    for i, component in enumerate(components):
        try:
            start_nodes = [node for node in component if len(pred[node]) == 0 and graph.ndata['read_strand'][node] == 1]
            start = min(start_nodes, key=lambda x: graph.ndata['read_start'][x])  # TODO: Wait a sec, 'read_start' shouldn't be used!!
            walk = decode_old(neighbors, edges, start, logits)
            walks.append(walk)
        except ValueError:
            # Negative strand
            # TODO: Solve later
            pass

    walks = sorted(walks, key=lambda x: -len(x))
    final = [walks[0]]

    if len(walks) > 1:
        all_nodes = set(walks[0])
        for w in walks[1:]:
            if len(w) < 10:
                continue
            if len(set(w) & all_nodes) == 0:
                final.append(w)
                all_nodes = all_nodes | set(w)

    return final


def decode_old(neighbors, edges, start, logits):
    current = start
    visited = set()
    walk = []

    while True:
        if current in visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue

        neighbor_edges = [edges[current, n] for n in neighbors[current]]
        neighbor_logits = logits.squeeze(1)[neighbor_edges]

        _, index = torch.topk(neighbor_logits, k=1, dim=0)
        choice = neighbors[current][index]
        current = choice

    return walk


def calculate_N50(list_of_lengths):
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

def calculate_NG50(list_of_lengths, ref_length):
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

def txt_output(f, txt):
    print(f'\t{txt}')
    f.write(f'\t{txt}\n')

def analyze(graph, gnn_paths, greedy_paths, out, ref_length):
    with open(f'{out}/analysis.txt', 'w') as f:
        # f.write(f'Chromosome total length:\t\n')
        #print(out.split("/"), out.split("/")[-2])
        gnn_contig_lengths = []
        for path in gnn_paths:
            contig_len = graph.ndata["read_end"][path[-1]] - graph.ndata["read_start"][path[0]]
            gnn_contig_lengths.append(abs(contig_len).item())
        txt_output(f, 'GNN: ')
        txt_output(f, f'Contigs: \t{gnn_contig_lengths}')
        txt_output(f,f'Contigs amount:\t{len(gnn_contig_lengths)}')
        txt_output(f,f'Longest Contig:\t{max(gnn_contig_lengths)}')
        txt_output(f,f'Reconstructed:\t{sum(gnn_contig_lengths)}')
        txt_output(f,f'Percentage:\t{sum(gnn_contig_lengths)/ref_length*100}')
        n50_gnn = calculate_N50(gnn_contig_lengths)
        txt_output(f,f'N50:\t{n50_gnn}')
        ng50_gnn = calculate_NG50(gnn_contig_lengths, ref_length)
        txt_output(f,f'NG50:\t{ng50_gnn}')


        txt_output(f,f'Greedy paths:\t{len(greedy_paths)}\n')
        greedy_contig_lengths = []
        for path in greedy_paths:
            contig_len = graph.ndata["read_end"][path[-1]] - graph.ndata["read_start"][path[0]]
            greedy_contig_lengths.append(abs(contig_len).item())
        txt_output(f, 'Greedy: ')
        txt_output(f, f'Contigs: \t{greedy_contig_lengths}')
        txt_output(f,f'Contigs amount:\t{len(greedy_contig_lengths)}')
        txt_output(f,f'Longest Contig:\t{max(greedy_contig_lengths)}')
        txt_output(f,f'Reconstructed:\t{sum(greedy_contig_lengths)}')
        txt_output(f,f'Percentage:\t{sum(greedy_contig_lengths)/ref_length*100}')
        n50_greedy = calculate_N50(greedy_contig_lengths)
        txt_output(f,f'N50:\t{n50_greedy}')
        ng50_greedy = calculate_NG50(greedy_contig_lengths, ref_length)
        txt_output(f,f'NG50:\t{ng50_greedy}')



def test_walk_neurips(data_path, model_path, device):
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    #batch_size = hyperparameters['batch_size']
    batch_size_train = hyperparameters['batch_size_train']
    batch_size_eval = hyperparameters['batch_size_eval']
    nb_pos_enc = hyperparameters['nb_pos_enc']
    num_parts_metis_train = hyperparameters['num_parts_metis_train']
    num_parts_metis_eval = hyperparameters['num_parts_metis_eval']
    num_decoding_paths = hyperparameters['num_decoding_paths']
    num_contigs = hyperparameters['num_contigs']
    patience = hyperparameters['patience']
    lr = hyperparameters['lr']
    # device = hyperparameters['device']
    use_reads = hyperparameters['use_reads']
    use_amp = hyperparameters['use_amp']
    batch_norm = hyperparameters['batch_norm']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    hidden_edge_features = hyperparameters['hidden_edge_features']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']
    decay = hyperparameters['decay']
    pos_to_neg_ratio = hyperparameters['pos_to_neg_ratio']

    model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    ds = AssemblyGraphDataset(data_path, nb_pos_enc=nb_pos_enc)

    for idx, g in ds:
        with torch.no_grad():
            g = g.to(device)
            x = g.ndata['x'].to(device)
            e = g.edata['e'].to(device)
            pe = g.ndata['pe'].to(device)
            pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
            pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
            pe = torch.cat((pe_in, pe_out, pe), dim=1)
            edge_predictions = model(g, x, e, pe)
            g.edata['score'] = edge_predictions.squeeze()
        
        succs = pickle.load(open(f'{data_path}/info/{idx}_succ.pkl', 'rb'))
        preds = pickle.load(open(f'{data_path}/info/{idx}_pred.pkl', 'rb'))
        edges = pickle.load(open(f'{data_path}/info/{idx}_edges.pkl', 'rb'))

        g = dgl.remove_self_loop(g)
        all_contigs = []
        all_contigs_len = []
        nb_paths = 10

        n_original_g = g.num_nodes(); self_nodes = torch.arange(n_original_g, dtype=torch.int32).to(device)

        visited = set()

        for idx_contig in range(10):
            if not all_contigs:
                remove_node_idx = torch.LongTensor([])
            else:
                remove_node_idx = torch.LongTensor([item for sublist in all_contigs for item in sublist])
            list_node_idx = torch.arange(g.num_nodes())
            keep_node_idx = torch.ones(g.num_nodes())
            keep_node_idx[remove_node_idx] = 0
            keep_node_idx = list_node_idx[keep_node_idx==1].int().to(device)
            print(f'idx_contig: {idx_contig}, nb_processed_nodes: {n_original_g-keep_node_idx.size(0)}, nb_remaining_nodes: {keep_node_idx.size(0)}, nb_original_nodes: {n_original_g}')
            sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
            sub_g.ndata['idx_nodes'] = torch.arange(sub_g.num_nodes()).to(device)
            n_sub_g = sub_g.num_nodes()
            print(f'nb of nodes sub-graph: {n_sub_g}')
            map_subg_to_g = sub_g.ndata[dgl.NID]
            prob_edges = torch.sigmoid(sub_g.edata['score']).squeeze()
            prob_edges = prob_edges.masked_fill(prob_edges<1e-9, 1e-9)
            prob_edges = prob_edges/ prob_edges.sum()
            prob_edges_nb_paths = prob_edges.repeat(nb_paths, 1)
            idx_edges = torch.distributions.categorical.Categorical(prob_edges_nb_paths).sample()
            all_walks = []

            for idx in idx_edges:
                src_init_edges = sub_g.edges()[0][idx].item()
                dst_init_edges = sub_g.edges()[1][idx].item()
                print(src_init_edges, dst_init_edges, succs[src_init_edges], preds[dst_init_edges], (src_init_edges, dst_init_edges) in edges)
                src_init_edges = map_subg_to_g[src_init_edges].item()
                dst_init_edges = map_subg_to_g[dst_init_edges].item()

                print(src_init_edges, dst_init_edges, succs[src_init_edges], preds[dst_init_edges], (src_init_edges, dst_init_edges) in edges)
                # get forwards path
                walk_f, visited_f = walk_forwards(src_init_edges, g.edata['score'], succs, edges, visited)
                # get backwards path
                walk_b, visited_b = walk_backwards(dst_init_edges, g.edata['score'], preds, edges, visited)
                walk = list(reversed(walk_b)) + walk_f
                all_walks.append(walk)
            best_walk = max(all_walks, key=lambda x: len(x))
            all_contigs.append(best_walk)
            all_contigs_len.append(len(best_walk))
            print(all_contigs_len)
            visited |= set(best_walk)

        return all_contigs, all_contigs_len
                



   

        visited = set()
        edge_predictions = edge_predictions.squeeze(-1)
        value, idx = torch.topk(edge_predictions, k=1)
        # start = start.item()
        start_b, start_f = g.edges()[0][idx].item(), g.edges()[1][idx].item()
        print(value, idx, start_b, start_f)
        walk_f, visited_f = walk_forwards(start_f, edge_predictions, succs, edges, visited)
        walk_b, visited_b = walk_backwards(start_b, edge_predictions, preds, edges, visited_f)
        print(len(walk_f), len(walk_b))



def test_walk(data_path, model_path,  device):
    hyperparameters = get_hyperparameters()
    # device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    # use_reads = hyperparameters['use_reads']

    # node_dim = hyperparameters['node_features']
    # edge_dim = hyperparameters['edge_dim']

    # if model_path is None:
    #     model_path = 'pretrained/model_32d_8l.pt'  # Best performing model
    model = models.BlockGatedGCNModel(1, 2, 128, 4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    # model.eval()

    ds = AssemblyGraphDataset(data_path)

    # info_all = load_graph_data(len(ds), data_path, False)

    idx, g = ds[0]
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)
    graph_ids = torch.arange(g.num_edges()).int()
    dl = dgl.dataloading.EdgeDataLoader(g, graph_ids, sampler, batch_size=4096*10, shuffle=False, drop_last=False)
    logits = torch.tensor([]).to(device)
    with torch.no_grad():
        for input_nodes, edge_subgraph, blocks in tqdm(dl):
            blocks = [b.to(device) for b in blocks]
            edge_subgraph = edge_subgraph.to(device)
            x = blocks[0].srcdata['x']
            e_0 = blocks[0].edata['e']
            e_subgraph = edge_subgraph.edata['e']
            # print(x.squeeze(-1))
            # print(e_0)
            # print(e_subgraph)
            p = model(edge_subgraph, blocks, x, e_0, e_subgraph).squeeze(-1)
            # print(p)
            # print(p.sum())
            logits = torch.cat((logits, p), dim=0)
    return logits


def walk_to_sequence(data_path, walks, graph, reads, edges):
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
    if 'assembly' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'assembly'))
    assembly_path = os.path.join(data_path, 'assembly', 'assembly.fasta')
    SeqIO.write(contigs, path, 'fasta')


def inference(model_path=None, data_path=None):
    hyperparameters = get_hyperparameters()
    device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    use_reads = hyperparameters['use_reads']

    node_dim = hyperparameters['node_features']
    edge_dim = hyperparameters['edge_dim']

    # if model_path is None:
    #     model_path = 'pretrained/model_32d_8l.pt'  # Best performing model
    model = models.BlockGatedGCNModel(node_dim, edge_dim, dim_latent, num_gnn_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    if data_path is None:
        data_path = 'data/train'
    ds = AssemblyGraphDataset(data_path)

    info_all = load_graph_data(len(ds), data_path, use_reads)

    for i in range(len(ds)):
        idx, graph = ds[i]
        print(f'Graph index: {idx}')
        graph = graph.to(device)
        
        succ = info_all['succs'][idx]
        pred = info_all['preds'][idx]
        if use_reads:
            reads = info_all['reads'][idx]
        else:
            reads = None
        edges = info_all['edges'][idx]

        walks = predict_new(model, graph, pred, succ, reads, edges, device)

        inference_path = os.path.join(data_path, 'inference')
        if not os.path.isdir(inference_path):
            os.mkdir(inference_path)
        pickle.dump(walks, open(f'{inference_path}/{idx}_predict.pkl', 'wb'))

        start_nodes = [w[0] for w in walks]

        # TODO: Greedy will not be too relevant soon, most likely
        baselines = []
        for start in start_nodes:
            baseline = algorithms.greedy(graph, start, succ, pred, edges)
            baselines.append(baseline)
        pickle.dump(baselines, open(f'{inference_path}/{idx}_greedy.pkl', 'wb'))

        analyze(graph, walks, baselines, inference_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    model_path = args.model
    data_path = args.data
    inference(model_path, data_path)
