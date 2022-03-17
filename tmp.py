import torch, dgl, pickle, importlib, models, train, graph_dataset

ds = graph_dataset.AssemblyGraphDataset('data/train_12-01-22/chr19')

idx, g = ds[0]

num_layers = 3
latent_dim = 5

g = dgl.add_self_loop(g)
g.ndata['x'] = torch.rand(g.num_nodes(), latent_dim)
g.edata['e'] = torch.rand(g.num_edges(), latent_dim)
g.edata['y'] = torch.ones(g.num_edges())

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)

dataloader = dgl.dataloading.EdgeDataLoader(g, torch.arange(g.num_edges()), sampler, batch_size=1024)

model = models.Model(latent_dim, latent_dim, num_layers, 1)

it = iter(dataloader)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
crit = torch.nn.BCEWithLogitsLoss()

for input_nodes, edge_subgraph, blocks in dataloader:

    input_features = blocks[0].srcdata['x']
    e = edge_subgraph.edata['e']
    edge_labels = edge_subgraph.edata['y']
    edge_predictions = model(edge_subgraph, blocks, input_features, e)

    print()
    print(input_features.shape)
    print(edge_subgraph)
    print(blocks[0])
    print(blocks[-1])
    print()
    
    loss = crit(edge_predictions.squeeze(), edge_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    # print(edge_predictions)


