import dgl, torch
import models

import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
import dgl.function as fn

import graph_dataset
import models


class CustomGCN(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f)

    def forward(self, block, h):
        with block.local_scope():
            h_src = h
            h_dst = h[:block.num_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            return self.W(block.dstdata['h_neigh'])
            # return self.W(torch.cat([block.dstdata['h'], block.dstdata['h']], dim=1))


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # self.conv1 = dglnn.GraphConv(in_features, hidden_features)
        # self.conv2 = dglnn.GraphConv(hidden_features, out_features)
        self.conv1 = CustomGCN(in_features, hidden_features)
        self.conv2 = CustomGCN(hidden_features, out_features)

    def forward(self, blocks, x):
        print(x.shape)
        x = F.relu(self.conv1(blocks[0], x))
        print(x.shape)
        x = F.relu(self.conv2(blocks[1], x))
        print(x.shape)
        return x


class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(2 * in_features, num_classes)

    def apply_edges(self, edges):
        data = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        print(data.shape)
        return {'score': self.W(data)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes):
        super().__init__()
        self.gcn = StochasticTwoLayerGCN(
            in_features, hidden_features, out_features)
        self.predictor = ScorePredictor(num_classes, out_features)

    def forward(self, edge_subgraph, blocks, x):
        x = self.gcn(blocks, x)
        print(x.shape)
        return self.predictor(edge_subgraph, x)


def main_1():
    g = dgl.load_graphs(f'data/train_12-01-22/chr19/processed/0.dgl')[0][0]
    ids = torch.arange(g.num_edges())
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dl = dgl.dataloading.EdgeDataLoader(g, ids, sampler, shuffle=True)
    it = iter(dl)

    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = g.edata['overlap_length'].float()
    ol_sim = g.edata['overlap_similarity']
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    ol_sim = (ol_sim - ol_sim.mean()) / ol_sim.std()
    g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
    
    x = g.ndata['x']
    e = g.edata['e']

    print(x.shape)
    print(e.shape)

    input_nodes, edge_subgraph, blocks = next(it)
    x = blocks[0].srcdata['x']

    print(blocks)
    print(blocks[0].srcdata[dgl.NID])
    blocks[0].srcdata['ones'] = torch.ones((blocks[0].number_of_src_nodes(), 1))
    print(blocks[0].srcdata['ones'])
    print('ones' in blocks[0].dstdata.keys())
    blocks[0].update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.mean('m', 'hh'))
    print('hh' in blocks[0].dstdata.keys())
    print('hh' in blocks[0].srcdata.keys())
    print('m' in blocks[0].edata.keys())
    print('m' in blocks[0].srcdata.keys())

    blocks[0].dstdata['zeros'] = torch.zeros((blocks[0].number_of_dst_nodes(), 1))
    blocks[0].apply_edges(fn.u_add_v('ones', 'zeros', 'added'))
    print(blocks[0].edata['added'])

    print(blocks[0].edges())
    print(blocks[1].edges())
    # for i in range(4):
    #     assert (blocks[i+1].edges()[0] == blocks[i].edges()[0][:blocks[i+1].num_edges()]).all() 
    # print(blocks[2].edges())
    print('Done!')

    idxx = [blocks[-1].edge_ids(0, 1), 2, 3, 4]
    ids = [blocks[-1].edge_ids(src, dst) for src, dst in zip(*edge_subgraph.edges())]
    print(idxx)
    print(blocks[-1].edata['e'])
    print(blocks[-1].edata['e'][ids])

    print(edge_subgraph.edges())
    print(edge_subgraph.edge_ids(0, 1))
    print(edge_subgraph.edata['e'])

    model = Model(1, 4, 2, 1)
    out = model(edge_subgraph, blocks, x)


def main_2():
    ds = graph_dataset.AssemblyGraphDataset('data/train_12-01-22/chr19')
    model = models.BlockGatedGCNModel(1, 2, 4, 4)
    idx, g = ds[0]

    device = 'cpu'
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)
    graph_ids = torch.arange(g.num_edges()).int()
    dl = dgl.dataloading.EdgeDataLoader(g, graph_ids, sampler, batch_size=1024, shuffle=True)
    it = iter(dl)


    input_nodes, edge_subgraph, blocks = next(it)
    blocks = [b.to(device) for b in blocks]
    edge_subgraph = edge_subgraph.to(device)
    x = blocks[0].srcdata['x']
    # TODO: For GNN edge feature update, I need edge data from block[0]
    e_0 = blocks[0].edata['e'].to(device)
    e_subgraph = edge_subgraph.edata['e'].to(device)  # e = blocks[0].edata['e'].to(device)
    # TODO: What I said above, read your own comments moron
    edge_labels = edge_subgraph.edata['y'].to(device)
    edge_predictions = model(edge_subgraph, blocks, x, e_0, e_subgraph)


def main_3():
    g = dgl.load_graphs(f'data/train_12-01-22/chr19/processed/0.dgl')[0][0]
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = g.edata['overlap_length'].float()
    ol_sim = g.edata['overlap_similarity']
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    ol_sim = (ol_sim - ol_sim.mean()) / ol_sim.std()
    g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    bg = dgl.graph((torch.cat((g.edges()[0], g.edges()[1])), torch.cat((g.edges()[1], g.edges()[0]))))
    bg.ndata['x'] = g.ndata['x'].clone()
    bg.edata['e'] = torch.cat((g.edata['e'], g.edata['e']))

    assert g.num_nodes() == bg.num_nodes()
    assert (bg.edges()[0] == torch.cat((g.edges()[0], g.edges()[1]))).all()
    assert (bg.edges()[1] == torch.cat((g.edges()[1], g.edges()[0]))).all()
    assert (bg.edata['e'] == torch.cat((g.edata['e'], g.edata['e']))).all()

    ids = torch.arange(bg.num_edges())
    dl = dgl.dataloading.EdgeDataLoader(bg, ids, sampler, shuffle=True)
    it = iter(dl)

    input_nodes, edge_subgraph, blocks = next(it)
    print(input_nodes)
    print(edge_subgraph)
    print(blocks)
    print(blocks[0].edges())
    print(blocks[-1].srcnodes())
    print(blocks[-1].dstnodes())
    print(blocks[-1].edges()[0])
    print(blocks[-1].edges()[1])
    return
    x = blocks[0].srcdata['x']

    print(blocks)
    print(blocks[0].srcdata[dgl.NID])
    blocks[0].srcdata['ones'] = torch.ones((blocks[0].number_of_src_nodes(), 1))
    print(blocks[0].srcdata['ones'])
    print('ones' in blocks[0].dstdata.keys())
    blocks[0].update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.mean('m', 'hh'))
    print('hh' in blocks[0].dstdata.keys())
    print('hh' in blocks[0].srcdata.keys())
    print('m' in blocks[0].edata.keys())
    print('m' in blocks[0].srcdata.keys())



if __name__ == '__main__':
    main_3()

