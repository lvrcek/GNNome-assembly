import dgl

import utils


def preprocess_graph(g, data_path, idx):
    g = g.int()
    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = (graph.edata['overlap_length'] - graph.edata['overlap_length'].mean() / graph.edata['overlap_length'].std()
    ol_sim = (graph.edata['overlap_similarity'] - graph.edata['overlap_similarity'].mean() / graph.edata['overlap_similarity'].std()
    g.edata['e'] = torch.cat((ol_len, ol_sim), dim=1)

    nodes_gt, edges_gt = utils.get_correct_ne(idx, data_path)
    g.edata['y'] = torch.tensor([1 if i in edges_gt else 0 for i in range(g.num_edges())], dtype=torch.float)

    g = dgl.add_self_loop(g)    

    return g