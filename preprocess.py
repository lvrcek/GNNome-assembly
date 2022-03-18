import dgl
import torch

import utils


def preprocess_graph(g, data_path, idx):
    g = g.int()
    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = g.edata['overlap_length'].float()
    ol_sim = g.edata['overlap_similarity']
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    ol_sim = (ol_sim - ol_sim.mean()) / ol_sim.std()
    g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path)
    g.edata['y'] = torch.tensor([1 if i in edges_gt else 0 for i in range(g.num_edges())], dtype=torch.float)

    g = dgl.add_self_loop(g)    

    return g
