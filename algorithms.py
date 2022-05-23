import os
import math
import pickle
import subprocess
from collections import deque
from datetime import datetime

import dgl
import torch

 
def greedy_message_func(edges):
    s_ji = edges.data['score'] # j=>i or src=>dst
    idx_i = edges.dst['idx_nodes']   # i or dst
    idx_j = edges.src['idx_nodes']   # j or src
    return {'s_ji': s_ji, 'idx_i': idx_i, 'idx_j': idx_j}


def greedy_reduce_func(nodes):
    s_ji = nodes.mailbox['s_ji']
    idx_i = nodes.mailbox['idx_i']
    idx_j = nodes.mailbox['idx_j']
    j_max_score = torch.max(s_ji, dim=1)[1].squeeze()
    nb_nodes = s_ji.size(0)
    idx_range = torch.arange(nb_nodes)
    j_max_index = idx_j[idx_range, j_max_score]
    i_max_index = idx_i[idx_range, j_max_score]
    return {'src_greedy': i_max_index, 'dst_greedy': j_max_index}


def compute_tour_length(g, paths_nodes, device): 
    nb_paths = paths_nodes.shape[0]
    nb_nodes = paths_nodes.shape[1]
    path_tot_overlaps = torch.zeros(nb_paths, device=device)
    # idx_src = paths_nodes[:,0]
    # for i in range(1,nb_nodes):
    #     idx_dst = paths_nodes[:,i]
    #     path_tot_overlaps += g.edges[idx_src, idx_dst].data['overlap_length']
    #     idx_src = idx_dst
    path_node_lengths = torch.zeros(nb_paths, device=device).long()
    for k in range(nb_paths):
        path_nodes_unique = paths_nodes[k]
        path_node_lengths[k] = torch.unique(path_nodes_unique).size(0)
    return path_tot_overlaps, path_node_lengths


def parallel_greedy_decoding(original_g, nb_paths, num_contigs, device):

    with torch.no_grad():

        # Remove not used node/edge features
        # GatedGCN
        original_g.ndata.pop('x'); original_g.ndata.pop('pe'); original_g.ndata.pop('h');
        # original_g.ndata.pop('y')  # Not used anymore
        original_g.ndata.pop('A1h'); original_g.ndata.pop('A2h'); original_g.ndata.pop('A3h'); original_g.ndata.pop('B1h'); original_g.ndata.pop('B2h');
        original_g.ndata.pop('sum_sigma_h_f'); original_g.ndata.pop('sum_sigma_f'); original_g.ndata.pop('h_forward')
        original_g.edata.pop('e'); original_g.edata.pop('y'); original_g.edata.pop('B3e'); original_g.edata.pop('B12h'); original_g.edata.pop('e_ji'); original_g.edata.pop('sigma_f')
        # Reads
        # original_g.ndata.pop('read_trim_end'); original_g.ndata.pop('read_end'); original_g.ndata.pop('read_start'); original_g.ndata.pop('read_trim_start');
        # original_g.ndata.pop('read_idx'); original_g.ndata.pop('read_strand'); original_g.ndata.pop('read_length')
        # original_g.edata.pop('prefix_length')
        
        # Remove current self-loop and add new self-loop with -inf score
        original_g = dgl.remove_self_loop(original_g)
        n_original_g = original_g.num_nodes(); self_nodes = torch.arange(n_original_g, dtype=torch.int32).to(device)
        original_g.add_edges(self_nodes, self_nodes)
        original_g.edata['score'][-n_original_g:] = float('-inf')


        all_contigs = []
        all_contigs_len = []

        for idx_contig in range(num_contigs):

            # Extract sub-graph
            if not all_contigs:
                remove_node_idx = torch.LongTensor([]) # sub-graph = orginal graph 
            else:
                remove_node_idx = torch.LongTensor([item for sublist in all_contigs for item in sublist])
            #print('h1c',remove_node_idx.size(),remove_node_idx)
            list_node_idx = torch.arange(original_g.num_nodes())
            keep_node_idx = torch.ones(original_g.num_nodes())
            keep_node_idx[remove_node_idx] = 0
            keep_node_idx = list_node_idx[keep_node_idx==1].int().to(device)
            #keep_node_idx = torch.randperm(n_original_g)[:80000].int().to(device) # uniform sampling # DEBUG !!!!
            print(f'idx_contig: {idx_contig}, nb_processed_nodes: {n_original_g-keep_node_idx.size(0)}, nb_remaining_nodes: {keep_node_idx.size(0)}, nb_original_nodes: {n_original_g}')
            sub_g = dgl.node_subgraph(original_g, keep_node_idx, store_ids=True) 
            sub_g.ndata['idx_nodes'] = torch.arange(sub_g.num_nodes()).to(device) # index used for max score
            n_sub_g = sub_g.num_nodes()
            print(f'nb of nodes sub-graph: {n_sub_g}')

            # Mapping index from sub-graph to original graph
            map_subg_to_g = sub_g.ndata[dgl.NID]

            # Sample initial edges
            prob_edges = torch.sigmoid(sub_g.edata['score']).squeeze()
            prob_edges = prob_edges.masked_fill(prob_edges<1e-9, 1e-9) 
            prob_edges = prob_edges/ prob_edges.sum()
            prob_edges_nb_paths = prob_edges.repeat(nb_paths, 1)
            idx_edges = torch.distributions.categorical.Categorical(prob_edges_nb_paths).sample() # index in sub-graph
            # idx_edges = torch.randperm(sub_g.num_edges())[:nb_paths] # uniform sampling
            src_init_edges = sub_g.edges()[0][idx_edges].long() # index in sub-graph
            dst_init_edges = sub_g.edges()[1][idx_edges].long() # index in sub-graph

            # Forward paths
            g_reverse = dgl.reverse(sub_g, copy_ndata=True, copy_edata=True)
            g_reverse.update_all(greedy_message_func, greedy_reduce_func) 
            src_greedy_forward = g_reverse.ndata['src_greedy'] # index in sub-graph
            dst_greedy_forward = g_reverse.ndata['dst_greedy'] # index in sub-graph
            idx_cur_nodes_subg = dst_init_edges # dst # index in sub-graph
            idx_cur_nodes_g = map_subg_to_g[idx_cur_nodes_subg]
            paths_nodes_forward = [] # index in original graph 
            paths_nodes_forward.append(idx_cur_nodes_g) 
            flag_dead_ends = flag_cycles = True
            max_cycle_size = 30
            nb_steps = 0
            while flag_dead_ends and flag_cycles and nb_steps<n_sub_g//2: # positive or negative strand
                idx_next_nodes_subg = dst_greedy_forward[idx_cur_nodes_subg] # index in sub-graph
                idx_next_nodes_g = map_subg_to_g[idx_next_nodes_subg] # index in original graph
                paths_nodes_forward.append(idx_next_nodes_g)
                flag_dead_ends = (idx_cur_nodes_subg==idx_next_nodes_subg).long().sum()<nb_paths
                if not nb_steps%max_cycle_size:
                    idx_cycle_anchor1 = idx_next_nodes_subg 
                else:
                    idx_cycle_anchor2 = idx_next_nodes_subg 
                    flag_cycles = (idx_cycle_anchor1==idx_cycle_anchor2).long().sum()<nb_paths
                idx_cur_nodes_subg = idx_next_nodes_subg
                nb_steps += 1
            paths_nodes_forward = torch.stack(paths_nodes_forward,dim=1)
            print(f'Forward paths - nb_steps: {nb_steps}, nb_nodes_sub_g: {n_sub_g}, find_cycles: {not flag_cycles}')

            # Backward paths
            sub_g.update_all(greedy_message_func, greedy_reduce_func) 
            src_greedy_backward = sub_g.ndata['src_greedy'] # index in sub-graph
            dst_greedy_backward = sub_g.ndata['dst_greedy'] # index in sub-graph
            idx_cur_nodes_subg = src_init_edges # src # index in sub-graph
            idx_cur_nodes_g = map_subg_to_g[idx_cur_nodes_subg]
            paths_nodes_backward = [] # index in original graph
            paths_nodes_backward.append(idx_cur_nodes_g) 
            flag_dead_ends = flag_cycles = True
            max_cycle_size = 30
            nb_steps = 0
            while flag_dead_ends and flag_cycles and nb_steps<n_sub_g//2: # positive or negative strand
                idx_next_nodes_subg = dst_greedy_backward[idx_cur_nodes_subg] # index in sub-graph    
                idx_next_nodes_g = map_subg_to_g[idx_next_nodes_subg] # index in original graph
                paths_nodes_backward.insert(0, idx_next_nodes_g)
                flag_dead_ends = (idx_cur_nodes_subg==idx_next_nodes_subg).long().sum()<nb_paths
                if not nb_steps%max_cycle_size:
                    idx_cycle_anchor1 = idx_next_nodes_subg 
                else:
                    idx_cycle_anchor2 = idx_next_nodes_subg 
                    flag_cycles = (idx_cycle_anchor1==idx_cycle_anchor2).long().sum()<nb_paths
                idx_cur_nodes_subg = idx_next_nodes_subg  
                nb_steps += 1
            paths_nodes_backward = torch.stack(paths_nodes_backward,dim=1)
            print(f'Backward paths - nb_steps: {nb_steps}, nb_nodes_sub_g: {n_sub_g}, find_cycles: {not flag_cycles}')

            # Concatenate forward and backward paths
            paths_nodes = torch.cat( (paths_nodes_backward, paths_nodes_forward), dim=1 )

            # compute total Length of overlaps
            path_tot_overlaps, path_node_lengths = compute_tour_length(original_g, paths_nodes, device)
            #print(f'path_tot_overlaps: {path_tot_overlaps}')
            print(f'path_node_lengths: {path_node_lengths}')

            # Select the path with max total length of overlaps
            #idx_max = torch.argmax(path_tot_overlaps)
            idx_max = torch.argmax(path_node_lengths)
            selected_node_path = paths_nodes[idx_max]

            # Remove duplicate nodes at the beginning and the end of the sequence
            idx_start = ((selected_node_path[0]==selected_node_path).sum()) - 1
            idx_end = ((selected_node_path[-1]==selected_node_path)).sum() - 1
            selected_node_path = selected_node_path[idx_start:-idx_end].tolist()

            # Append to all contigs
            all_contigs.append(selected_node_path) 
            all_contigs_len.append(path_node_lengths[idx_max].item())
            print(f'idx of max path: {idx_max}, len of max path: {path_node_lengths[idx_max]}, tot overlaps of max path: {path_tot_overlaps[idx_max]}')
            print(f'idx_contig: {idx_contig}, len of selected contig: {path_node_lengths[idx_max]}\n')
            
    return all_contigs, all_contigs_len


def assert_strand(graph, walk):
    org_strand = graph.ndata['read_strand'][walk[0]].item()
    for idx, node in enumerate(walk[1:]):
        curr_strand = graph.ndata['read_strand'][node].item()
        if curr_strand != org_strand:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'node index: {node}')


def assert_overlap(graph, walk):
    for idx, (src, dst) in enumerate(zip(walk[:-1], walk[1:])):
        src_start = graph.ndata['read_start'][src].item()
        dst_start = graph.ndata['read_start'][dst].item()
        src_end = graph.ndata['read_end'][src].item()
        dst_end = graph.ndata['read_end'][dst].item()
        src_strand = graph.ndata['read_strand'][src].item()
        dst_strand = graph.ndata['read_strand'][dst].item()
        if src_strand == dst_strand == 1 and dst_start > src_end:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'end: {src_end}, start: {dst_start}')
        if src_strand == dst_strand == -1 and dst_end < src_start:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'end: {src_start}, start: {dst_end}')



# def to_csv(name, root, print_strand=True, save_dir='test_cases'):
#     graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     with open(f'{save_dir}/{name}_info.csv', 'w') as f:
#         if print_strand:
#             f.write('node_id,read_strand,read_start,read_end\n')
#         else:
#             f.write('node_id,read_start,read_end\n')
#         for n in range(graph.num_nodes()):
#             strand = graph.ndata['read_strand'][n].item()
#             start = graph.ndata['read_start'][n].item()
#             end = graph.ndata['read_end'][n].item()
#             if print_strand:
#                 f.write(f'{n},{strand},{start},{end}\n')
#             else:
#                 if strand == 1:
#                     f.write(f'{n},{start},{end}\n')
# 
# 
# def to_positive_pairwise(name, root, save_dir='test_cases'):
#     graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     with open(f'{save_dir}/{name}_edges.txt', 'w') as f:
#         f.write('src\tdst\n')
#         for src, dst in zip(graph.edges()[0], graph.edges()[1]):
#             src = src.item()
#             dst = dst.item()
#             if graph.ndata['read_strand'][src] == 1 and graph.ndata['read_strand'][dst] == 1:
#                 f.write(f'{src}\t{dst}\n')
#     
# 
# def get_solutions_for_all_cpp(root, save_dir='test_cases'):
#     processed_path = os.path.join(root, 'processed')
# 
#     for filename in os.listdir(processed_path):
#         name = filename[:-4]
#         print(f'Finding walk for... {name}')
#         to_csv(name, root, print_strand=False, save_dir=save_dir)
#         to_positive_pairwise(name, root, save_dir=save_dir)
#         subprocess.run(f'./longestContinuousSequence {name}_info.csv > {name}.out', shell=True, cwd=save_dir)
#         with open(f'{save_dir}/{name}.out') as f:
#            lines = f.readlines()
#            walk = lines[-1].strip().split(' -> ')
#            walk = list(map(int, walk))
#            pickle.dump(walk, open(f'{save_dir}/{name}_walk.pkl', 'wb'))


def interval_union(name, root):
    graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
    intervals = []
    for strand, start, end in zip(graph.ndata['read_strand'], graph.ndata['read_start'], graph.ndata['read_end']):
        if strand.item() == 1:
            intervals.append([start.item(), end.item()])
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]

    for interval in intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)

    return result


def dfs(graph, neighbors, start=None, avoid={}):
    # TODO: Take only those with in-degree 0
    if start is None:
        min_value, idx = torch.topk(graph.ndata['read_start'], k=1, largest=False)
        start = idx.item()

    # threshold, _ = torch.topk(graph.ndata['read_start'][graph.ndata['read_strand']==1], k=1)
    # threshold = threshold.item()

    stack = deque()
    stack.append(start)

    visited = [True if i in avoid else False for i in range(graph.num_nodes())]

    path = {start: None}
    max_node = start
    max_value = graph.ndata['read_end'][start]

    try:
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            
            # if graph.ndata['read_end'][current] == threshold:
            #     break

            if graph.ndata['read_end'][current] > max_value:
                max_value = graph.ndata['read_end'][current]
                max_node = current

            visited[current] = True
            tmp = []
            for node in neighbors.get(current, []):
                if visited[node]:
                    continue
                if graph.ndata['read_strand'][node] == -1:
                    continue
                if graph.ndata['read_start'][node] > graph.ndata['read_end'][current]:
                    continue
                if graph.ndata['read_start'][node] < graph.ndata['read_start'][current]:
                    continue
                tmp.append(node)

            if len(tmp) == 0:
                for node in neighbors.get(current, []):
                    if visited[node]:
                        continue
                    if graph.ndata['read_strand'][node] == -1:
                        continue
                    if graph.ndata['read_start'][node] < graph.ndata['read_start'][current]:
                        continue
                    if graph.ndata['read_start'][node] > graph.ndata['read_end'][current]:
                        tmp.append(node)

            tmp.sort(key=lambda x: -graph.ndata['read_start'][x])
            for node in tmp:
                stack.append(node)
                path[node] = current

    except KeyboardInterrupt:
        pass
    
    finally:
        walk = []
        current = max_node
        while current is not None:
            walk.append(current)
            current = path[current]
        walk.reverse()
        visited = {i for i in range(graph.num_nodes()) if visited[i]}
        return walk, visited


def get_correct_edges(graph, neighbors, edges, walk):
    pos_str_edges = set()
    neg_str_edges = set()
    for i, src in enumerate(walk[:-1]):
        for dst in walk[i+1:]:
            if dst in neighbors[src] and graph.ndata['read_start'][dst] < graph.ndata['read_end'][src]:
                try:
                    pos_str_edges.add(edges[(src, dst)])
                except KeyError:
                    print('Edge not found in the edge dictionary')
                    raise
                try:
                    neg_str_edges.add(edges[dst^1, src^1])
                except KeyError:
                    print('Negative strand edge not found in the edge dictionary')
                    raise
            else:
                break
    return pos_str_edges, neg_str_edges


def get_gt_graph(graph, neighbors, edges):
    all_nodes = {i for i in range(graph.num_nodes()) if graph.ndata['read_strand'][i] == 1}
    last_node = max(all_nodes, key=lambda x: graph.ndata['read_end'][x])

    largest_visited = -1
    all_walks = []
    pos_correct_edges, neg_correct_edges = set(), set()
    all_visited = set()

    while all_nodes:
        start = min(all_nodes, key=lambda x: graph.ndata['read_start'][x])
        walk, visited = dfs(graph, neighbors, start, avoid=all_visited)
        if graph.ndata['read_end'][walk[-1]] < largest_visited or len(walk) == 1:
            all_nodes = all_nodes - visited
            all_visited = all_visited | visited
            # print(f'\nDiscard component')
            # print(f'Start = {graph.ndata["read_start"][walk[0]]}\t Node = {walk[0]}')
            # print(f'End   = {graph.ndata["read_end"][walk[-1]]}\t Node = {walk[-1]}')
            # print(f'Walk length = {len(walk)}')
            continue
        else:
            largest_visited = graph.ndata['read_end'][walk[-1]]
            all_walks.append(walk)

        # print(f'\nInclude component')
        # print(f'Start = {graph.ndata["read_start"][walk[0]]}\t Node = {walk[0]}')
        # print(f'End   = {graph.ndata["read_end"][walk[-1]]}\t Node = {walk[-1]}')
        # print(f'Walk length = {len(walk)}')

        pos_str_edges, neg_str_edges = get_correct_edges(graph, neighbors, edges, walk)
        pos_correct_edges = pos_correct_edges | pos_str_edges
        neg_correct_edges = neg_correct_edges | neg_str_edges

        if largest_visited == graph.ndata['read_end'][last_node]:
            break
        all_nodes = all_nodes - visited
        all_visited = all_visited | visited

    return pos_correct_edges, neg_correct_edges

