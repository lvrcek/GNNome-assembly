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
import evaluate

from datetime import datetime
from algorithms import parallel_greedy_decoding
import utils

# def test_walk(data_path, model_path,  device):
#     hyperparameters = get_hyperparameters()
#     # device = hyperparameters['device']
#     dim_latent = hyperparameters['dim_latent']
#     num_gnn_layers = hyperparameters['num_gnn_layers']
#     # use_reads = hyperparameters['use_reads']

#     # node_dim = hyperparameters['node_features']
#     # edge_dim = hyperparameters['edge_dim']

#     # if model_path is None:
#     #     model_path = 'pretrained/model_32d_8l.pt'  # Best performing model
#     model = models.BlockGatedGCNModel(1, 2, 128, 4).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
#     # model.eval()

#     ds = AssemblyGraphDataset(data_path)

#     # info_all = load_graph_data(len(ds), data_path, False)

#     idx, g = ds[0]
#     sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)
#     graph_ids = torch.arange(g.num_edges()).int()
#     dl = dgl.dataloading.EdgeDataLoader(g, graph_ids, sampler, batch_size=4096*10, shuffle=False, drop_last=False)
#     logits = torch.tensor([]).to(device)
#     with torch.no_grad():
#         for input_nodes, edge_subgraph, blocks in tqdm(dl):
#             blocks = [b.to(device) for b in blocks]
#             edge_subgraph = edge_subgraph.to(device)
#             x = blocks[0].srcdata['x']
#             e_0 = blocks[0].edata['e']
#             e_subgraph = edge_subgraph.edata['e']
#             # print(x.squeeze(-1))
#             # print(e_0)
#             # print(e_subgraph)
#             p = model(edge_subgraph, blocks, x, e_0, e_subgraph).squeeze(-1)
#             # print(p)
#             # print(p.sum())
#             logits = torch.cat((logits, p), dim=0)
#     return logits


# def predict_new(model, graph, succs, preds, edges, device):
#     x = graph.ndata['x'].to(device)
#     e = graph.edata['e'].to(device)
#     edge_logits = model(graph, x, e)
#     # TODO: Problem, my block model doesn't work on full graphs!
#     # TODO: I can still iterate over the batches and append the predictions
#     edge_logits= edge_logits.squeeze(-1)
#     edge_p = F.sigmoid(edge_logits)
#     walks = decode_new(graph, edge_p, succs, preds, edges)
#     return walks
#     # or (later) translate walks into sequences
#     # what with the sequences? Store into FASTA ofc


# def decode_new(graph, edges_p, neighbors, predecessors, edges):
#     # Choose starting node for the first time
#     walks = []
#     visited = set()
#     # ----- Modify this later ------
#     all_nodes = {n.item() for n in graph.nodes()}
#     correct_nodes = {n for n in range(graph.num_nodes()) if graph.ndata['y'][n] == 1}
#     potential_nodes = correct_nodes
#     # ------------------------------
#     while True:
#         potential_nodes = potential_nodes - visited
#         start = get_random_start(potential_nodes)
#         if start is None:
#             break
#         visited.add(start)
#         visited.add(start ^ 1)
#         walk_f, visited_f = walk_forwards(start, edges_p, neighbors, edges, visited)
#         walk_b, visited_b = walk_backwards(start, edges_p, predecessors, edges, visited)
#         walk = walk_b[:-1] + [start] + walk_f[1:]
#         visited = visited | visited_f | visited_b
#         walks.append(walk)
#     walks = sorted(walks, key=lambda x: len(x))
#     return walks
    

# def get_random_start(potential_nodes, nodes_p=None):
#     # potential_nodes = {n.item() for n in graph.nodes()}
#     if len(potential_nodes) < 10:
#         return None
#     potential_nodes = potential_nodes
#     start = random.sample(potential_nodes, 1)[0]
#     # start = max(potential_nodes_p)
#     return start


def walk_forwards(start, edges_p, neighbors, edges, visited_old):
    current = start
    walk = []
    visited = set()
    while True:
        all_visited = visited | visited_old
        if current in all_visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        if len(neighbors[current]) == 0:
            break 
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue
        neighbor_edges = [edges[current, n] for n in neighbors[current] if n not in all_visited]
        if not neighbor_edges:
            break
        neighbor_p = edges_p[neighbor_edges]
        _, index = torch.topk(neighbor_p, k=1, dim=0)
        current = neighbors[current][index]
    return walk, visited


def walk_backwards(start, edges_p, predecessors, edges, visited_old):
    current = start
    walk = []
    visited = set()
    while True:
        all_visited = visited | visited_old
        if current in all_visited:
            break
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        if len(predecessors[current]) == 0:
            break 
        if len(predecessors[current]) == 1:
            current = predecessors[current][0]
            continue
        neighbor_edges = [edges[n, current] for n in predecessors[current] if n not in all_visited]
        if not neighbor_edges:
            break
        neighbor_p = edges_p[neighbor_edges]
        _, index = torch.topk(neighbor_p, k=1, dim=0)
        current = predecessors[current][index]
    walk = list(reversed(walk))
    return walk, visited


def get_contigs_for_one_graph(g, succs, preds, edges, nb_paths=20, len_threshold=50, device='cpu'):
    # Get contigs for one graph
    g = dgl.remove_self_loop(g)
    g = g.to(device)
    all_contigs = []
    all_contigs_len = []
    nb_paths = 20
    visited = set()
    idx_contig = -1
    scores = g.edata['score'].to('cpu')

    while True:
        idx_contig += 1        
        sub_g, map_subg_to_g = get_subgraph(g, visited, device)
        idx_edges = sample_edges(sub_g.edata['score'], nb_paths)
        all_walks = []

        print(f'\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, ' \
              f'nb_remaining_nodes: {g.num_nodes() - len(visited)}, nb_original_nodes: {g.num_nodes()}')

        # Get nb_paths paths for a single iteration, then take the longest one
        for idx in idx_edges:
            src_init_edges = map_subg_to_g[sub_g.edges()[0][idx]].item()
            dst_init_edges = map_subg_to_g[sub_g.edges()[1][idx]].item()
            # print(src_init_edges, dst_init_edges, succs[src_init_edges], preds[dst_init_edges], (src_init_edges, dst_init_edges) in edges)

            # get forwards path
            walk_f, visited_f = walk_forwards(dst_init_edges, scores, succs, edges, visited)
            # get backwards path
            walk_b, visited_b = walk_backwards(src_init_edges, scores, preds, edges, visited | visited_f)
            # concatenate two paths
            walk = walk_b + walk_f
            all_walks.append(walk)
            print(f'src={src_init_edges} dst={dst_init_edges} len_f={len(walk_f)} len_b={(len(walk_b))}')

        best_walk = max(all_walks, key=lambda x: len(x))
        if len(best_walk) < len_threshold:
            break

        # If longest contig is longer than len_threshold, add it and continue, else break
        all_contigs.append(best_walk)
        visited = visited | set(best_walk) | set([n^1 for n in best_walk])
        all_contigs_len.append(len(best_walk))
        print(all_contigs_len)

    return all_contigs


def get_subgraph(g, visited, device):
    remove_node_idx = torch.LongTensor([item for item in visited])
    list_node_idx = torch.arange(g.num_nodes())
    keep_node_idx = torch.ones(g.num_nodes())
    keep_node_idx[remove_node_idx] = 0
    keep_node_idx = list_node_idx[keep_node_idx==1].int().to(device)

    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    sub_g.ndata['idx_nodes'] = torch.arange(sub_g.num_nodes()).to(device)
    map_subg_to_g = sub_g.ndata[dgl.NID]
    return sub_g, map_subg_to_g


def sample_edges(edge_scores, nb_paths):
    prob_edges = torch.sigmoid(edge_scores).squeeze()
    prob_edges = prob_edges.masked_fill(prob_edges<1e-9, 1e-9)
    prob_edges = prob_edges/ prob_edges.sum()
    prob_edges_nb_paths = prob_edges.repeat(nb_paths, 1)
    idx_edges = torch.distributions.categorical.Categorical(prob_edges_nb_paths).sample()
    return idx_edges


def inference(model_path, data_path, device='cpu'):
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    nb_pos_enc = hyperparameters['nb_pos_enc']
    num_decoding_paths = hyperparameters['num_decoding_paths']
    num_contigs = hyperparameters['num_contigs']
    # device = hyperparameters['device']
    batch_norm = hyperparameters['batch_norm']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    hidden_edge_features = hyperparameters['hidden_edge_features']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']   
    len_threshold = hyperparameters['len_threshold'] 
    num_greedy_paths = hyperparameters['num_greedy_paths']
    device = hyperparameters['device'] # for small graphs like chr19
    #device = 'cpu'                    # for large graphs
    
    # Paths 
    #   Model/network is saved at : pretrained/model_{out}
    #   DGL train greaph(s) are saved at : data/train_{out}/processed
    #   Contigs are saved at : data/train_{out}/assembly/{idx}_assembly.fasta
    #   Walks are save at : data/train_{out}/inference/{idx}_walks.pkl
    #   Reads at saved at : data/train_{out}/info/{idx}_reads.pkl
    #   E.g. model_path = 'pretrained/model_{out}.pt'
    #   E.g. data_path = 'data/train_{out}'
    print('model_path',model_path)
    print('data_path',data_path)
    
    # Add a folder inference/ to store results
    inference_dir = os.path.join(data_path, 'inference')
    if not os.path.isdir(inference_dir):
        os.mkdir(inference_dir)

    # Load model/network 
    model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    model.train() # DEBUG !! 

    # Add positional encoding to graph(s)
    ds = AssemblyGraphDataset(data_path, nb_pos_enc=nb_pos_enc)

    # Forward pass on the graph(s)
    walks_per_graph = []
    contigs_per_graph = []
    for idx, g in ds:
        # Get scores
        with torch.no_grad():
            time_start_forward = datetime.now()
            g = g.to(device)
            x = g.ndata['x'].to(device)
            e = g.edata['e'].to(device)
            pe = g.ndata['pe'].to(device)
            pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
            pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
            pe = torch.cat((pe_in, pe_out, pe), dim=1)
            edge_predictions = model(g, x, e, pe)
            g.edata['score'] = edge_predictions.squeeze()
        time_forward = datetime.now() - time_start_forward
        
        # Compute walks and contigs LV
        time_start_decoding = datetime.now()
        # Load info data
        # info_all = load_graph_data(len(ds), data_path, use_reads)  # Later can do it like this
        succs = pickle.load(open(f'{data_path}/info/{idx}_succ.pkl', 'rb'))
        preds = pickle.load(open(f'{data_path}/info/{idx}_pred.pkl', 'rb'))
        edges = pickle.load(open(f'{data_path}/info/{idx}_edges.pkl', 'rb'))
        reads = pickle.load(open(f'{data_path}/info/{idx}_reads.pkl', 'rb'))
        # Get contigs
        walks = get_contigs_for_one_graph(g, succs, preds, edges, num_greedy_paths, len_threshold, device='cpu')
        elapsed = utils.timedelta_to_str(time_forward + datetime.now() - time_start_decoding)
        print(f'\nelapsed time for forward pass + sequential greedy decoding : {elapsed}')
        inference_path = os.path.join(inference_dir, f'{idx}_walks.pkl')
        pickle.dump(walks, open(f'{inference_path}', 'wb'))
        contigs = evaluate.walk_to_sequence(walks, g, reads, edges)
        walks_len = [len(walk) for walk in walks]
        contigs_len = [len(contig) for contig in contigs]
        print(f'num_contigs: {len(contigs_len)}\nlengths of walks: {walks_len}\nlengths of contigs: {contigs_len}')
        evaluate.save_assembly(contigs, data_path, idx)
        g_to_chr = pickle.load(open(f'{data_path}/info/g_to_chr.pkl', 'rb'))
        chrN = g_to_chr[idx] # chromosome id
        num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs, chrN)
        print(f'{longest_contig=} {reconstructed=:.4f} {n50=} {ng50=}')
        walks_per_graph.append(walks)
        contigs_per_graph.append(contigs)
        elapsed = utils.timedelta_to_str(time_forward + datetime.now() - time_start_decoding)
        print(f'total time (sequential greedy decoding): {elapsed}, num_greedy_paths: {num_greedy_paths}, len_threshold: {len_threshold}, nb_contigs: {len(contigs)}\n')

        # Compute walks and contigs XB
        time_start_decoding = datetime.now()
        walks, walks_len = parallel_greedy_decoding(g, num_greedy_paths, len_threshold, device) 
        elapsed = utils.timedelta_to_str(time_forward + datetime.now() - time_start_decoding)
        print(f'\nelapsed time for forward pass + parallel greedy decoding : {elapsed}')
        inference_path = os.path.join(inference_dir, f'{idx}_walks.pkl')
        pickle.dump(walks, open(f'{inference_path}', 'wb'))
        reads = pickle.load(open(f'{data_path}/info/{idx}_reads.pkl', 'rb'))
        edges = pickle.load(open(f'{data_path}/info/{idx}_edges.pkl', 'rb'))
        contigs = evaluate.walk_to_sequence(walks, g, reads, edges)
        contigs_len = [len(contig) for contig in contigs]
        print(f'num_contigs: {len(contigs_len)}\nlengths of walks: {walks_len}\nlengths of contigs: {contigs_len}')
        evaluate.save_assembly(contigs, data_path, idx)
        g_to_chr = pickle.load(open(f'{data_path}/info/g_to_chr.pkl', 'rb'))
        chrN = g_to_chr[idx] # chromosome id
        num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs, chrN)
        print(f'{longest_contig=} {reconstructed=:.4f} {n50=} {ng50=}')
        walks_per_graph.append(walks)
        contigs_per_graph.append(contigs)
        elapsed = utils.timedelta_to_str(time_forward + datetime.now() - time_start_decoding)
        print(f'total time (parallel greedy decoding): {elapsed}, num_greedy_paths: {num_greedy_paths}, len_threshold: {len_threshold}, nb_contigs: {len(contigs)}\n')

    return walks_per_graph, contigs_per_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    model_path = args.model
    data_path = args.data
    inference(model_path, data_path)
