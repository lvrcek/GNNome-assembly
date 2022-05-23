import argparse
import os
import pickle
import random
from tqdm import tqdm 
import collections
from datetime import datetime

import torch
import torch.nn.functional as F
import dgl

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import evaluate
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


def get_contig_length(walk, graph, edges):
    total_length = 0
    for src, dst in zip(walk[:-1], walk[1:]):
        edge_id = edges[(src, dst)]
        prefix = graph.edata['prefix_length'][edge_id].item()
        total_length += prefix
    total_length += graph.ndata['read_length'][walk[-1]]
    return total_length


def walk_forwards(start, edges_p, neighbors, predecessors, edges, visited_old):
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
        masked_neighbors = [n for n in neighbors[current] if n not in all_visited]
        if not neighbor_edges:
            break
        neighbor_p = edges_p[neighbor_edges]
        _, index = torch.topk(neighbor_p, k=1, dim=0)
        # current = neighbors[current][index]
        next_n = masked_neighbors[index]
        #####
        # trans = set(neighbors[current]) & set(predecessors[next_n])
        # trans_rc = {t^1 for t in trans}
        # visited = visited | trans | trans_rc
        #####
        current = next_n
    return walk, visited


def walk_backwards(start, edges_p, predecessors, neighbors, edges, visited_old):
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
        masked_neighbors = [n for n in predecessors[current] if n not in all_visited]
        if not neighbor_edges:
            break
        neighbor_p = edges_p[neighbor_edges]
        _, index = torch.topk(neighbor_p, k=1, dim=0)
        # current = predecessors[current][index]
        # previous = current
        next_n = masked_neighbors[index]
        ####
        # trans = set(predecessors[current]) & set(neighbors[next_n])
        # trans_rc = {t^1 for t in trans}
        # visited = visited | trans | trans_rc
        ####
        current = next_n
    walk = list(reversed(walk))
    return walk, visited


def get_contigs_for_one_graph(g, succs, preds, edges, nb_paths=10, len_threshold=50, device='cpu'):
    # Get contigs for one graph
    g = dgl.remove_self_loop(g)
    g = g.to(device)
    all_contigs = []
    # all_contigs_lenghts = []
    ##############
    nb_paths = 50
    len_threshold = 20
    #############
    visited = set()
    idx_contig = -1

    scores = g.edata['score'].to('cpu')
    ol_lens = g.edata['overlap_length'].to('cpu')
    ol_sims = g.edata['overlap_similarity'].to('cpu')

    #################
    all_contigs_len = []
    all_contigs_sim = []
    #################

    while True:
        idx_contig += 1       
        time_start_sample_edges = datetime.now()
        sub_g, map_subg_to_g = get_subgraph(g, visited, device)
        idx_edges = sample_edges(sub_g.edata['score'], nb_paths)

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_sample_edges)
        print(f'elapsed time (get_candidates): {elapsed}')

        all_walks = []
        all_visited_iter = []

        #################
        all_walks_len = []
        all_walks_sim = []
        #################

        print(f'\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, ' \
              f'nb_remaining_nodes: {g.num_nodes() - len(visited)}, nb_original_nodes: {g.num_nodes()}')

        # Get nb_paths paths for a single iteration, then take the longest one
        time_start_get_candidates = datetime.now()
        for e, idx in enumerate(idx_edges):
            src_init_edges = map_subg_to_g[sub_g.edges()[0][idx]].item()
            dst_init_edges = map_subg_to_g[sub_g.edges()[1][idx]].item()
            # print(src_init_edges, dst_init_edges, succs[src_init_edges], preds[dst_init_edges], (src_init_edges, dst_init_edges) in edges)

            # get forwards path
            walk_f, visited_f = walk_forwards(dst_init_edges, scores, succs, preds, edges, visited)
            # get backwards path
            walk_b, visited_b = walk_backwards(src_init_edges, scores, preds, succs, edges, visited | visited_f)
            # concatenate two paths

            ###########################
            walk_f_len, visited_f_len = walk_forwards(dst_init_edges, ol_lens, succs, preds, edges, visited)
            walk_b_len, visited_b_len = walk_backwards(src_init_edges, ol_lens, preds, succs, edges, visited | visited_f_len)
            walk_len = walk_b_len + walk_f_len
            all_walks_len.append(walk_len)
            walk_f_sim, visited_f_sim = walk_forwards(dst_init_edges, ol_sims, succs, preds, edges, visited)
            walk_b_sim, visited_b_sim = walk_backwards(src_init_edges, ol_sims, preds, succs, edges, visited | visited_f_sim)
            walk_sim = walk_b_sim + walk_f_sim
            all_walks_sim.append(walk_sim)
            ###########################

            walk = walk_b + walk_f
            all_walks.append(walk)
            visited_iter = visited_f | visited_b
            all_visited_iter.append(visited_iter)
            # print(f'{e}: src={src_init_edges} dst={dst_init_edges} len_f={len(walk_f)} len_b={(len(walk_b))}')

            ######################
            # print(f'{e}: src={src_init_edges} dst={dst_init_edges} len_f={len(walk_f_ol)} len_b={(len(walk_b_ol))}')
            # print(f'{e}: src={src_init_edges} dst={dst_init_edges} len_f={len(walk_f_lab)} len_b={(len(walk_b_lab))}')
            # print()
            #####################

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_candidates)
        print(f'elapsed time (get_candidates): {elapsed}')

        best_walk = max(all_walks, key=lambda x: get_contig_length(x, g, edges))
        idxx = all_walks.index(best_walk)
        best_visited = all_visited_iter[idxx]
        # print(f'Len best walk: len(best_walk'))
        # print(f'Len best visited: len(best_visited)')
        ################# Add all transited nodes!!!
        time_start_get_visited = datetime.now()
        trans = set()
        for ss, dd in zip(best_walk[:-1], best_walk[1:]):
            t1 = set(succs[ss]) & set(preds[dd])
            t2 = {t^1 for t in t1}
            trans = trans | t1 | t2
        best_visited = best_visited | trans

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_visited)
        print(f'elapsed time (get_visited): {elapsed}')

        # print(len(t1)
        # print(len(best_visited))
        #################
        # idxx = all_walks_lab.index(best_walk)  # TODO: DEBUG!!! Remove later
        print(f'Chosen walk with index: {idxx} ; Length: {len(best_walk)}')
        if len(best_walk) < len_threshold:
            break

        #################
        best_walk_len = all_walks_len[idxx]
        best_walk_sim = all_walks_sim[idxx]
        #################

        # If longest contig is longer than len_threshold, add it and continue, else break
        all_contigs.append(best_walk)
        # visited = visited | set(best_walk) | set([n^1 for n in best_walk])
        visited |= best_visited
        # all_contigs_lenghts.append(len(best_walk))
        print([len(c) for c in all_contigs])

        #####################
        all_contigs_len.append(best_walk_len)
        all_contigs_sim.append(best_walk_sim)
        #####################

    return all_contigs, all_contigs_len, all_contigs_sim


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


def inference(data_path, model_path, device='cpu'):
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

    # TODO: Remove these hardcoded hyperparameters
    nb_paths = 50
    len_threshold = 20
    # device = 'cpu'
    # ######################
    time_start = datetime.now()

    model = models.GraphGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    ds = AssemblyGraphDataset(data_path, nb_pos_enc=nb_pos_enc)

    inference_dir = os.path.join(data_path, 'inference')
    if not os.path.isdir(inference_dir):
        os.mkdir(inference_dir)

    walks_per_graph = []
    contigs_per_graph = []

    ######################  TODO: Remove
    walks_per_graph_ol = []
    contigs_per_graph_ol = []
    walks_per_graph_lab = []
    contigs_per_graph_lab = []
    ######################
    g_to_chr = pickle.load(open(f'{data_path}/info/g_to_chr.pkl', 'rb'))

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'elapsed time (loading network and data): {elapsed}')

    for idx, g in ds:
        # Get scores
        chr_n = g_to_chr[idx]
        print(f'==== Processing graph {idx} : {chr_n} ====')
        with torch.no_grad():
            time_start_get_scores = datetime.now()
            g = g.to(device)
            x = g.ndata['x'].to(device)
            e = g.edata['e'].to(device)
            pe = g.ndata['pe'].to(device)
            pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
            pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
            pe = torch.cat((pe_in, pe_out, pe), dim=1)
            edge_predictions = model(g, x, e, pe)
            g.edata['score'] = edge_predictions.squeeze()

            edge_labels = g.edata['y'].squeeze()
            edge_predictions = edge_predictions.squeeze()
            print(edge_predictions)
            print(edge_labels)


            elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_scores)
            print(f'elapsed time (get_scores): {elapsed}')

       
            TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
            acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
            try:
                fp_rate = FP / (FP + TN)
            except ZeroDivisionError:
                fp_rate = 0.0
            try:
                fn_rate = FN / (FN + TP)
            except ZeroDivisionError:
                fn_rate = 0.0
            
            print(f'1: {(edge_labels==1).sum()} , 0:{(edge_labels==0).sum()}')
            print(f'==== METRICS for graph {idx} : {chr_n} ====')
            print(f'{acc=:.4f} {precision=:.4f} {recall=:.4f} {f1=:.4f}')
            print(f'{fp_rate=:.4f} {fn_rate=:.4f}\n')

        # Load info data
        # info_all = load_graph_data(len(ds), data_path, use_reads)  # Later can do it like this
        succs = pickle.load(open(f'{data_path}/info/{idx}_succ.pkl', 'rb'))
        preds = pickle.load(open(f'{data_path}/info/{idx}_pred.pkl', 'rb'))
        edges = pickle.load(open(f'{data_path}/info/{idx}_edges.pkl', 'rb'))
        reads = pickle.load(open(f'{data_path}/info/{idx}_reads.pkl', 'rb'))

        # Get contigs
        time_start_get_walks = datetime.now()
        walks, walks_ol, walks_lab = get_contigs_for_one_graph(g, succs, preds, edges, nb_paths, len_threshold, device='cpu')  # TODO: Remove walks_ol and walks_lab later
        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_walks)
        print(f'elapsed time (get_walks): {elapsed}')
        inference_path = os.path.join(inference_dir, f'{idx}_walks.pkl')
        pickle.dump(walks, open(f'{inference_path}', 'wb'))
        
        time_start_get_contigs = datetime.now()
        contigs = evaluate.walk_to_sequence(walks, g, reads, edges)
        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_contigs)
        print(f'elapsed time (get_contigs): {elapsed}')

        evaluate.save_assembly(contigs, data_path, idx)
        walks_per_graph.append(walks)
        contigs_per_graph.append(contigs)

        ###############
        walks_per_graph_ol.append(walks_ol)
        walks_per_graph_lab.append(walks_lab)
        contigs_ol = evaluate.walk_to_sequence(walks_ol, g, reads, edges)
        contigs_lab = evaluate.walk_to_sequence(walks_lab, g, reads, edges)
        contigs_per_graph_ol.append(contigs_ol)
        evaluate.save_assembly(contigs_ol, data_path, idx, suffix='_ol_len')
        contigs_per_graph_lab.append(contigs_lab)
        evaluate.save_assembly(contigs_lab, data_path, idx, suffix='_ol_sim')
        ###############
    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'elapsed time (total): {elapsed}')

    return walks_per_graph, contigs_per_graph, walks_per_graph_ol, contigs_per_graph_ol, walks_per_graph_lab, contigs_per_graph_lab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    model_path = args.model
    data_path = args.data
    inference(data_path, model_path)
