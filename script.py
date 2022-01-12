import os, pickle, torch, dgl, sys

path = sys.argv[1]
proc_path = os.path.join(path, 'processed')
info_path = os.path.join(path, 'info')
sols_path = os.path.join(path, 'solutions')
infr_path = os.path.join(path, 'inference_32d_8l')

for i in [1, 3, 4, 5, 12, 15, 16, 25, 30, 40, 45, 48]:
    pred = pickle.load(open(f'{infr_path}/{i}_predict.pkl', 'rb'))
    base = pickle.load(open(f'{infr_path}/{i}_greedy.pkl', 'rb'))
    sols = pickle.load(open(f'{sols_path}/{i}_gt.pkl', 'rb'))
    graph = dgl.load_graphs(f'{proc_path}/{i}.dgl')[0][0]
    if i in (12, 48):
        pred = pred[:-1]
    pred_len = graph.ndata['read_end'][pred[-1]] - graph.ndata['read_start'][0]
    base_len = graph.ndata['read_end'][base[-1]] - graph.ndata['read_start'][0]
    sols_len = graph.ndata['read_end'][sols[-1]] - graph.ndata['read_start'][0]
    print(i, 'gnn:', pred_len.item(), 'greedy:', base_len.item(), 'g-t:', sols_len.item())


