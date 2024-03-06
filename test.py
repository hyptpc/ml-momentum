import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# pytouch
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

# torch_geometric
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, summary

def convert_graph_data(pos_data, features, n_edge = 2): # should be len(pos_data) > n_egde+1
    edge_from = []
    edge_to   = []
    edge_attr = []
    for i, point in enumerate(pos_data):
        distance = np.linalg.norm( pos_data - np.full_like(pos_data, point), axis = 1) # calc. Euclidean distance
        edge_from += [i]*n_edge
        edge_to   += np.argsort(distance)[1:n_edge+1].tolist()
        edge_attr += distance[ np.argsort(distance)[1:n_edge+1] ].tolist()
    graph_data = Data(
        x          = torch.tensor(features),             # node feature
        y          = None,                               # node label
        edge_index = torch.tensor([edge_from, edge_to]), # edge
        edge_attr  = torch.tensor(edge_attr)             # edge feature
    )
    return graph_data

pos_data = [
    [0, 0, 0],
    [10, 15, 30],
    [5, -2, 6],
    [1, 2, 3]
]

features = [
    [1],
    [10],
    [2],
    [5]
]

a = convert_graph_data(pos_data, features)
print(a.x)
print(a.edge_index)
print(a.edge_attr)

nxg = to_networkx(a)
nx.draw(nxg,
        node_color = 'w',
        edgecolors = 'k', # node border color
        with_labels = True,
        edge_color = a.edge_attr.mul(1/a.edge_attr.max()).tolist(),
        edge_cmap = plt.cm.plasma,
        alpha = 0.5)
plt.show()